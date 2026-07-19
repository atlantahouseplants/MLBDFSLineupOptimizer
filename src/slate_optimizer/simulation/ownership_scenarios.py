"""Ownership uncertainty scenarios for lineup robustness scoring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from slate_optimizer.optimizer.solver import LineupResult

from .contest_simulator import ContestSimResult
from .duplication import estimate_lineup_duplication


@dataclass(frozen=True)
class OwnershipScenario:
    name: str
    chalk_multiplier: float = 1.0
    low_owned_multiplier: float = 1.0
    value_multiplier: float = 1.0
    stack_steam_multiplier: float = 1.0


DEFAULT_OWNERSHIP_SCENARIOS: tuple[OwnershipScenario, ...] = (
    OwnershipScenario("Base"),
    OwnershipScenario("Chalk steams", chalk_multiplier=1.28, low_owned_multiplier=0.92),
    OwnershipScenario("Chalk comes in light", chalk_multiplier=0.86, low_owned_multiplier=1.08),
    OwnershipScenario("Value opens", value_multiplier=1.35, chalk_multiplier=1.08),
    OwnershipScenario("Stack steam", stack_steam_multiplier=1.35, chalk_multiplier=1.12),
)


def apply_ownership_scenario_metrics(
    contest_result: ContestSimResult,
    candidate_lineups: Sequence[LineupResult],
    optimizer_df: pd.DataFrame,
    field_size: int = 50_000,
    field_profile: Mapping | None = None,
    scenarios: Sequence[OwnershipScenario] = DEFAULT_OWNERSHIP_SCENARIOS,
) -> pd.DataFrame:
    """Mutate lineup sim results with robustness metrics across ownership worlds."""

    if not contest_result.lineup_results or not candidate_lineups:
        return pd.DataFrame()
    scenario_rows: list[dict[str, float | str | int]] = []
    slate_meta = _slate_meta(optimizer_df)

    for result in contest_result.lineup_results:
        if result.lineup_id >= len(candidate_lineups):
            continue
        base_lineup = candidate_lineups[result.lineup_id].dataframe
        scenario_ownerships: list[float] = []
        scenario_dups: list[float] = []
        for scenario in scenarios:
            adjusted = _apply_scenario(base_lineup, slate_meta, scenario)
            estimate = estimate_lineup_duplication(
                adjusted,
                field_size=field_size,
                salary_cap=35_000,
                field_profile=field_profile,
            )
            total_ownership = _ownership_decimal(adjusted.get("proj_fd_ownership")).sum()
            scenario_ownerships.append(float(total_ownership))
            scenario_dups.append(float(estimate.duplication_score))
            scenario_rows.append(
                {
                    "lineup_id": int(result.lineup_id),
                    "scenario": scenario.name,
                    "scenario_total_ownership": float(total_ownership),
                    "scenario_duplication_score": float(estimate.duplication_score),
                    "scenario_estimated_dupes": float(estimate.estimated_dupes),
                }
            )
        if scenario_dups:
            worst_dup = max(scenario_dups)
            avg_dup = float(np.mean(scenario_dups))
            own_spread = float(np.std(scenario_ownerships))
            result.worst_case_duplication_score = float(worst_dup)
            result.worst_case_ownership = float(max(scenario_ownerships))
            result.ownership_scenario_spread = own_spread
            result.ownership_scenario_score = float(
                np.clip(1.0 - 0.70 * worst_dup - 0.20 * avg_dup - 0.10 * min(1.0, own_spread), 0.0, 1.0)
            )
            result.field_duplication_rate = max(float(result.field_duplication_rate or 0.0), float(worst_dup))
    return pd.DataFrame(scenario_rows)


def _slate_meta(optimizer_df: pd.DataFrame) -> pd.DataFrame:
    if optimizer_df is None or optimizer_df.empty:
        return pd.DataFrame()
    meta = optimizer_df.copy()
    meta["_own"] = _ownership_decimal(meta.get("proj_fd_ownership")).reindex(meta.index, fill_value=0.0)
    proj = pd.to_numeric(
        meta["proj_fd_mean"] if "proj_fd_mean" in meta.columns else pd.Series(0.0, index=meta.index),
        errors="coerce",
    ).fillna(0.0)
    salary = pd.to_numeric(
        meta["salary"] if "salary" in meta.columns else pd.Series(np.nan, index=meta.index),
        errors="coerce",
    ).replace(0, np.nan)
    meta["_value"] = (proj / salary * 1000.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    meta["_is_chalk"] = meta["_own"] >= 0.18
    meta["_is_low_owned"] = meta["_own"] <= 0.06
    meta["_is_value"] = meta["_value"] >= float(meta["_value"].quantile(0.75)) if meta["_value"].notna().any() else False
    stack_teams = (
        meta[meta.get("player_type", "").astype(str).str.lower() != "pitcher"]
        .groupby("team_code")["_own"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
        .index.astype(str)
        .tolist()
        if "team_code" in meta.columns and "player_type" in meta.columns
        else []
    )
    meta["_is_steam_stack"] = meta.get("team_code", pd.Series("", index=meta.index)).astype(str).isin(stack_teams)
    return meta[["fd_player_id", "_is_chalk", "_is_low_owned", "_is_value", "_is_steam_stack"]].drop_duplicates("fd_player_id")


def _apply_scenario(lineup_df: pd.DataFrame, slate_meta: pd.DataFrame, scenario: OwnershipScenario) -> pd.DataFrame:
    adjusted = lineup_df.copy()
    if "proj_fd_ownership" not in adjusted.columns:
        adjusted["proj_fd_ownership"] = 0.0
    ownership = _ownership_decimal(adjusted.get("proj_fd_ownership"))
    scale = pd.Series(1.0, index=adjusted.index, dtype=float)
    if slate_meta is not None and not slate_meta.empty and "fd_player_id" in adjusted.columns:
        merged = adjusted[["fd_player_id"]].astype(str).merge(
            slate_meta.assign(fd_player_id=slate_meta["fd_player_id"].astype(str)),
            on="fd_player_id",
            how="left",
        )
        scale = scale.to_numpy()
        scale = np.where(merged.get("_is_chalk", False).fillna(False), scale * scenario.chalk_multiplier, scale)
        scale = np.where(merged.get("_is_low_owned", False).fillna(False), scale * scenario.low_owned_multiplier, scale)
        scale = np.where(merged.get("_is_value", False).fillna(False), scale * scenario.value_multiplier, scale)
        scale = np.where(merged.get("_is_steam_stack", False).fillna(False), scale * scenario.stack_steam_multiplier, scale)
        scale = pd.Series(scale, index=adjusted.index, dtype=float)
    adjusted_ownership = (ownership.reset_index(drop=True) * scale.reset_index(drop=True)).clip(lower=0.001, upper=0.75)
    original = pd.to_numeric(adjusted["proj_fd_ownership"], errors="coerce").fillna(0.0)
    if not original.empty and float(original.max()) > 1.5:
        adjusted["proj_fd_ownership"] = adjusted_ownership.values * 100.0
    else:
        adjusted["proj_fd_ownership"] = adjusted_ownership.values
    return adjusted


def _ownership_decimal(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    series = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0).astype(float)
    if series.empty:
        return series
    if float(series.max()) > 1.5:
        series = series / 100.0
    return series.clip(lower=0.0, upper=1.0)


__all__ = [
    "DEFAULT_OWNERSHIP_SCENARIOS",
    "OwnershipScenario",
    "apply_ownership_scenario_metrics",
]
