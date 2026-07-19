"""Ownership-based lineup duplication and uniqueness estimates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DuplicationEstimate:
    ownership_sum: float
    average_ownership: float
    stack_ownership: float
    chalk_count: int
    salary_used: float
    salary_cap: float
    duplication_score: float
    estimated_dupes: float
    uniqueness_score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "ownership_sum": float(self.ownership_sum),
            "average_ownership": float(self.average_ownership),
            "stack_ownership": float(self.stack_ownership),
            "chalk_count": float(self.chalk_count),
            "salary_used": float(self.salary_used),
            "salary_cap": float(self.salary_cap),
            "duplication_score": float(self.duplication_score),
            "estimated_dupes": float(self.estimated_dupes),
            "uniqueness_score": float(self.uniqueness_score),
        }


def estimate_lineup_duplication(
    lineup_df: pd.DataFrame,
    field_size: int = 50_000,
    salary_cap: float = 35_000,
    field_profile: Mapping | None = None,
) -> DuplicationEstimate:
    """Estimate how duplicated a lineup is likely to be in a large-field MLB GPP.

    This is intentionally a heuristic, not a claim that we can know exact duplicate
    counts before lock. It gives the selector a second duplication signal beyond
    overlap with generated field lineups, which is especially useful when a lineup
    uses a chalky 4/5 stack plus popular spend patterns.
    """

    if lineup_df is None or lineup_df.empty:
        return DuplicationEstimate(0.0, 0.0, 0.0, 0, 0.0, float(salary_cap), 0.0, 0.0, 1.0)

    ownership = _ownership_decimal(lineup_df.get("proj_fd_ownership"))
    ownership_sum = float(ownership.sum())
    average_ownership = float(ownership.mean()) if len(ownership) else 0.0
    chalk_count = int((ownership >= 0.18).sum())
    salary_used = _salary_used(lineup_df)
    effective_cap = max(float(salary_cap or 0), salary_used, 1.0)
    salary_ratio = float(np.clip(salary_used / effective_cap, 0.0, 1.05))

    stack_ownership, max_stack_size = _stack_ownership(lineup_df, ownership)
    stack_size_signal = float(np.clip(max_stack_size / 5.0, 0.0, 1.2))

    raw_score = (
        -5.25
        + 1.25 * ownership_sum
        + 3.20 * average_ownership
        + 2.60 * stack_ownership
        + 0.32 * chalk_count
        + _salary_weight(field_profile) * max(0.0, salary_ratio - 0.92)
        + 0.85 * stack_size_signal
        + _profile_duplicate_signal(field_profile)
    )
    duplication_score = float(1.0 / (1.0 + np.exp(-raw_score)))

    entries = max(1, int(field_size or 1))
    estimated_dupes = float((duplication_score ** 2.25) * entries * _profile_dupe_multiplier(field_profile))
    uniqueness_score = float(np.clip(1.0 - duplication_score, 0.0, 1.0))

    return DuplicationEstimate(
        ownership_sum=ownership_sum,
        average_ownership=average_ownership,
        stack_ownership=float(stack_ownership),
        chalk_count=chalk_count,
        salary_used=salary_used,
        salary_cap=float(effective_cap),
        duplication_score=duplication_score,
        estimated_dupes=estimated_dupes,
        uniqueness_score=uniqueness_score,
    )


def estimate_duplication_dict(
    lineup_df: pd.DataFrame,
    field_size: int = 50_000,
    salary_cap: float = 35_000,
    field_profile: Mapping | None = None,
) -> Mapping[str, float]:
    return estimate_lineup_duplication(
        lineup_df,
        field_size=field_size,
        salary_cap=salary_cap,
        field_profile=field_profile,
    ).to_dict()


def _ownership_decimal(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    series = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0).astype(float)
    if series.empty:
        return series
    if float(series.max()) > 1.5:
        series = series / 100.0
    return series.clip(lower=0.002, upper=0.70)


def _salary_used(lineup_df: pd.DataFrame) -> float:
    if "salary" not in lineup_df.columns:
        return 0.0
    salary = pd.to_numeric(lineup_df["salary"], errors="coerce").fillna(0.0)
    return float(salary.sum())


def _stack_ownership(lineup_df: pd.DataFrame, ownership: pd.Series) -> tuple[float, int]:
    if "team_code" not in lineup_df.columns:
        return 0.0, 0
    working = lineup_df.reset_index(drop=True).copy()
    working["_own"] = ownership.reset_index(drop=True)
    if "player_type" in working.columns:
        working = working[working["player_type"].astype(str).str.lower() != "pitcher"]
    if working.empty:
        return 0.0, 0
    grouped = working.groupby("team_code").agg(stack_ownership=("_own", "sum"), stack_size=("_own", "size"))
    stacked = grouped[grouped["stack_size"] >= 3]
    if stacked.empty:
        best = grouped.sort_values(["stack_size", "stack_ownership"], ascending=False).iloc[0]
    else:
        best = stacked.sort_values(["stack_size", "stack_ownership"], ascending=False).iloc[0]
    return float(best["stack_ownership"]), int(best["stack_size"])


def _salary_weight(field_profile: Mapping | None) -> float:
    if not field_profile:
        return 1.10
    try:
        full_salary_pct = float(field_profile.get("full_salary_pct", 0.0) or 0.0)
    except (TypeError, ValueError):
        full_salary_pct = 0.0
    return float(0.85 + 1.20 * np.clip(full_salary_pct, 0.0, 1.0))


def _profile_duplicate_signal(field_profile: Mapping | None) -> float:
    if not field_profile:
        return 0.0
    try:
        duplicate_pct = float(field_profile.get("duplicate_lineup_pct", 0.0) or 0.0)
        top_stack_share = float(field_profile.get("top_stack_share", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return float(0.70 * np.clip(duplicate_pct, 0.0, 1.0) + 0.35 * np.clip(top_stack_share, 0.0, 1.0))


def _profile_dupe_multiplier(field_profile: Mapping | None) -> float:
    base = 0.012
    if not field_profile:
        return base
    try:
        avg_dupes = float(field_profile.get("avg_duplicate_count", 1.0) or 1.0)
    except (TypeError, ValueError):
        avg_dupes = 1.0
    return float(base * np.clip(np.sqrt(max(avg_dupes, 1.0)), 0.75, 3.5))


__all__ = [
    "DuplicationEstimate",
    "estimate_duplication_dict",
    "estimate_lineup_duplication",
]
