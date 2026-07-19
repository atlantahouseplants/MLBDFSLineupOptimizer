"""Automatic exposure recommendations from stack plans and player leverage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class ExposureTuningConfig:
    max_batter_cap: float = 0.45
    max_pitcher_cap: float = 0.55
    min_batter_cap: float = 0.05
    min_pitcher_cap: float = 0.10
    chalk_threshold: float = 0.20


def build_exposure_recommendations(
    optimizer_df: pd.DataFrame,
    stack_plan: pd.DataFrame | None = None,
    config: ExposureTuningConfig | None = None,
) -> pd.DataFrame:
    """Return player-level max exposure recommendations."""

    config = config or ExposureTuningConfig()
    if optimizer_df is None or optimizer_df.empty:
        return pd.DataFrame()
    df = optimizer_df.copy()
    for col in ("fd_player_id", "full_name", "team_code", "player_type", "position"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)
    for col in ("proj_fd_mean", "proj_fd_ownership", "player_leverage_score", "proj_fd_bust_rate"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if df["proj_fd_ownership"].max() > 1.5:
        df["proj_fd_ownership"] = df["proj_fd_ownership"] / 100.0

    stack_targets = _stack_target_map(stack_plan, "target_exposure")
    stack_max = _stack_target_map(stack_plan, "recommended_max_exposure")
    projection_rank = df["proj_fd_mean"].rank(pct=True)
    leverage = df["player_leverage_score"].clip(-1.0, 1.0)
    ownership = df["proj_fd_ownership"].clip(0.0, 1.0)
    bust = df["proj_fd_bust_rate"].clip(0.0, 1.0)
    is_pitcher = df["player_type"].str.lower() == "pitcher"
    team_target = df["team_code"].map(stack_targets).fillna(0.10)
    team_max = df["team_code"].map(stack_max).fillna(0.25)

    hitter_cap = (
        0.10
        + 0.18 * projection_rank
        + 0.35 * team_target
        + 0.12 * leverage.clip(lower=0)
        - 0.10 * ownership.clip(lower=config.chalk_threshold)
        - 0.08 * bust
    )
    pitcher_cap = (
        0.18
        + 0.30 * projection_rank
        + 0.10 * leverage.clip(lower=0)
        - 0.10 * ownership.clip(lower=0.30)
        - 0.05 * bust
    )
    caps = hitter_cap.where(~is_pitcher, pitcher_cap)
    caps = caps.where(~is_pitcher, caps.clip(config.min_pitcher_cap, config.max_pitcher_cap))
    caps = caps.where(is_pitcher, caps.clip(config.min_batter_cap, config.max_batter_cap))
    caps = caps.where(is_pitcher, caps.clip(upper=(team_max + 0.18).clip(upper=config.max_batter_cap)))

    result = df[["fd_player_id", "full_name", "team_code", "position", "player_type", "proj_fd_mean", "proj_fd_ownership", "player_leverage_score"]].copy()
    result["recommended_max_exposure"] = caps.round(3)
    result["team_target_exposure"] = team_target.round(3)
    result["reason"] = result.apply(_reason, axis=1)
    return result.sort_values(["recommended_max_exposure", "proj_fd_mean"], ascending=[False, False]).reset_index(drop=True)


def player_overrides_text(recommendations: pd.DataFrame, top_n: int = 80) -> str:
    if recommendations is None or recommendations.empty:
        return ""
    lines = []
    for _, row in recommendations.head(max(1, int(top_n))).iterrows():
        lines.append(f"{row['full_name']}:{float(row['recommended_max_exposure']):.3f}")
    return "\n".join(lines)


def stack_cap_maps(stack_plan: pd.DataFrame | None) -> tuple[Dict[str, float], Dict[str, float]]:
    if stack_plan is None or stack_plan.empty or "team_code" not in stack_plan.columns:
        return {}, {}
    max_caps = {}
    min_caps = {}
    for _, row in stack_plan.iterrows():
        team = str(row.get("team_code", ""))
        if not team:
            continue
        max_val = float(row.get("recommended_max_exposure", 0.0) or 0.0)
        min_val = float(row.get("recommended_min_exposure", 0.0) or 0.0)
        if max_val > 0:
            max_caps[team] = max_val
        if min_val > 0:
            min_caps[team] = min_val
    return min_caps, max_caps


def _stack_target_map(stack_plan: pd.DataFrame | None, column: str) -> pd.Series:
    if stack_plan is None or stack_plan.empty or "team_code" not in stack_plan.columns or column not in stack_plan.columns:
        return pd.Series(dtype=float)
    values = stack_plan[["team_code", column]].dropna().copy()
    values["team_code"] = values["team_code"].astype(str)
    values[column] = pd.to_numeric(values[column], errors="coerce").fillna(0.0)
    return values.drop_duplicates("team_code").set_index("team_code")[column]


def _reason(row: pd.Series) -> str:
    ownership = float(row.get("proj_fd_ownership", 0.0) or 0.0)
    leverage = float(row.get("player_leverage_score", 0.0) or 0.0)
    player_type = str(row.get("player_type", "")).lower()
    if player_type == "pitcher":
        return "Pitcher cap based on projection rank, ownership, and leverage."
    if ownership >= 0.20 and leverage <= 0:
        return "Chalk hitter capped unless needed as a stack piece."
    if leverage > 0.10:
        return "Positive-leverage hitter allowed above baseline exposure."
    return "Hitter cap follows team stack target and projection rank."


__all__ = [
    "ExposureTuningConfig",
    "build_exposure_recommendations",
    "player_overrides_text",
    "stack_cap_maps",
]
