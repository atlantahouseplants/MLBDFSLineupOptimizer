"""Slate-shape heuristics for recommending optimizer/simulation presets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .stack_exposure import build_stack_exposure_plan


@dataclass
class PresetRecommendation:
    preset: str
    confidence: float
    metrics: dict[str, float]
    reasons: list[str]


def recommend_slate_preset(optimizer_df: pd.DataFrame, stack_plan: pd.DataFrame | None = None) -> PresetRecommendation:
    if optimizer_df is None or optimizer_df.empty:
        return PresetRecommendation("Balanced leverage build", 0.25, {}, ["Slate data is not loaded yet."])

    df = optimizer_df.copy()
    ownership = _ownership_decimal(df.get("proj_fd_ownership"))
    batters = df[df["player_type"].astype(str).str.lower() != "pitcher"] if "player_type" in df.columns else df
    batter_ownership = _ownership_decimal(batters.get("proj_fd_ownership"))
    team_count = int(batters.get("team_code", pd.Series(dtype=str)).dropna().astype(str).nunique())
    chalk_batters = int((batter_ownership >= 0.18).sum()) if not batter_ownership.empty else 0
    top5_own = float(batter_ownership.sort_values(ascending=False).head(5).sum()) if not batter_ownership.empty else 0.0
    own_std = float(batter_ownership.std()) if len(batter_ownership) > 1 else 0.0

    plan = stack_plan if stack_plan is not None and not stack_plan.empty else build_stack_exposure_plan(df)
    top_stack_own = 0.0
    top_stack_score = 0.0
    if plan is not None and not plan.empty:
        if "stack_ownership" in plan.columns:
            top_stack_own = float(pd.to_numeric(plan["stack_ownership"], errors="coerce").fillna(0.0).max())
        if "stack_score" in plan.columns:
            top_stack_score = float(pd.to_numeric(plan["stack_score"], errors="coerce").fillna(0.0).max())

    metrics = {
        "team_count": float(team_count),
        "chalk_batters": float(chalk_batters),
        "top5_batter_ownership": top5_own,
        "ownership_std": own_std,
        "top_stack_ownership": top_stack_own,
        "top_stack_score": top_stack_score,
        "average_ownership": float(ownership.mean()) if not ownership.empty else 0.0,
    }

    reasons: list[str] = []
    if top_stack_own >= 0.95 or top5_own >= 1.15 or chalk_batters >= 10:
        reasons.append("Ownership is concentrated around obvious chalk bats/stacks.")
        reasons.append("A more aggressive uniqueness penalty should help your 300-entry portfolio avoid duplicated builds.")
        return PresetRecommendation("Chalk fade build", 0.82, metrics, reasons)

    if team_count <= 8:
        reasons.append("The slate is small enough that projection matters more and pure fades get thin quickly.")
        return PresetRecommendation("Small-field GPP", 0.68, metrics, reasons)

    if own_std <= 0.055 and top_stack_own <= 0.70:
        reasons.append("Ownership looks fairly flat, so there is less need to force a heavy chalk fade.")
        return PresetRecommendation("Balanced leverage build", 0.70, metrics, reasons)

    if top_stack_score >= 1.0 and top_stack_own <= 0.85:
        reasons.append("The stack plan has at least one strong team that is not prohibitively owned.")
        return PresetRecommendation("300-entry large-field GPP", 0.74, metrics, reasons)

    reasons.append("Slate shape is normal for a large-field MLB GPP.")
    return PresetRecommendation("300-entry large-field GPP", 0.62, metrics, reasons)


def _ownership_decimal(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna().astype(float)
    if series.empty:
        return series
    if float(series.max()) > 1.5:
        series = series / 100.0
    return series.clip(lower=0.0, upper=1.0)


__all__ = ["PresetRecommendation", "recommend_slate_preset"]
