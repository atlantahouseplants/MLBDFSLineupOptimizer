"""Lineup-level explanation scorecards for selected portfolios."""
from __future__ import annotations

from typing import Optional

import pandas as pd


def build_lineup_scorecard(
    lineup_df: Optional[pd.DataFrame],
    portfolio_df: Optional[pd.DataFrame] = None,
    stack_plan: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if lineup_df is None or lineup_df.empty or "lineup_id" not in lineup_df.columns:
        return pd.DataFrame()

    df = lineup_df.copy()
    for col in ["proj_fd_mean", "proj_fd_ownership", "player_leverage_score", "salary"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in ["team_code", "player_type"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)

    base = df.groupby("lineup_id").agg(
        projection=("proj_fd_mean", "sum"),
        ownership=("proj_fd_ownership", "sum"),
        leverage=("player_leverage_score", "sum"),
        salary=("salary", "sum"),
    ).reset_index()
    stacks = _lineup_stack_summary(df, stack_plan)
    if not stacks.empty:
        base = base.merge(stacks, on="lineup_id", how="left")
    else:
        base["primary_stack"] = ""
        base["stack_upside_score"] = 0.0

    if portfolio_df is not None and not portfolio_df.empty and "lineup_id" in portfolio_df.columns:
        sim = portfolio_df.copy()
        sim["lineup_id"] = pd.to_numeric(sim["lineup_id"], errors="coerce")
        base_ids = pd.to_numeric(base["lineup_id"], errors="coerce")
        if sim["lineup_id"].notna().any() and base_ids.notna().any() and sim["lineup_id"].min() == 0 and base_ids.min() >= 1:
            sim["lineup_id"] = sim["lineup_id"] + 1
        keep = [
            col
            for col in [
                "lineup_id",
                "payout_ev",
                "expected_roi",
                "top_1pct_rate",
                "duplication_score",
                "estimated_dupes",
                "uniqueness_score",
                "ownership_scenario_score",
                "field_duplication_rate",
            ]
            if col in sim.columns
        ]
        base = base.merge(sim[keep].drop_duplicates("lineup_id"), on="lineup_id", how="left")

    for col in ["payout_ev", "expected_roi", "top_1pct_rate", "duplication_score", "field_duplication_rate", "uniqueness_score", "ownership_scenario_score"]:
        if col not in base.columns:
            base[col] = 0.0 if col != "uniqueness_score" and col != "ownership_scenario_score" else 1.0
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0.0 if col not in {"uniqueness_score", "ownership_scenario_score"} else 1.0)

    base["projection_component"] = _pct_rank(base["projection"])
    base["leverage_component"] = _pct_rank(base["leverage"])
    base["stack_component"] = _pct_rank(base["stack_upside_score"])
    base["duplication_component"] = 1.0 - _pct_rank(base[["duplication_score", "field_duplication_rate"]].max(axis=1))
    base["ownership_scenario_component"] = _pct_rank(base["ownership_scenario_score"])
    base["uniqueness_component"] = _pct_rank(base["uniqueness_score"])
    base["why_selected_score"] = (
        0.25 * base["projection_component"]
        + 0.20 * base["leverage_component"]
        + 0.18 * base["stack_component"]
        + 0.15 * base["duplication_component"]
        + 0.12 * base["ownership_scenario_component"]
        + 0.10 * base["uniqueness_component"]
    )
    base["why_selected"] = base.apply(_why_text, axis=1)
    return base.sort_values("why_selected_score", ascending=False).reset_index(drop=True)


def _lineup_stack_summary(lineup_df: pd.DataFrame, stack_plan: Optional[pd.DataFrame]) -> pd.DataFrame:
    hitters = lineup_df[lineup_df["player_type"].str.lower() != "pitcher"].copy()
    if hitters.empty:
        return pd.DataFrame()
    stacks = hitters.groupby(["lineup_id", "team_code"]).size().reset_index(name="hitters")
    stacks = stacks[stacks["hitters"] >= 3]
    if stacks.empty:
        return pd.DataFrame()
    if stack_plan is not None and not stack_plan.empty and "team_code" in stack_plan.columns:
        plan = stack_plan.copy()
        plan["team_code"] = plan["team_code"].astype(str)
        keep = [col for col in ["team_code", "stack_score", "sim_upside_score", "stack_leverage_score", "p_runs_ge_6"] if col in plan.columns]
        stacks = stacks.merge(plan[keep], on="team_code", how="left")
    for col in ["stack_score", "sim_upside_score", "stack_leverage_score", "p_runs_ge_6"]:
        if col not in stacks.columns:
            stacks[col] = 0.0
        stacks[col] = pd.to_numeric(stacks[col], errors="coerce").fillna(0.0)
    stacks["stack_upside_score"] = (
        0.45 * stacks["stack_score"]
        + 0.30 * stacks["sim_upside_score"]
        + 0.20 * stacks["stack_leverage_score"]
        + 0.05 * stacks["p_runs_ge_6"]
    )
    primary = stacks.sort_values(["lineup_id", "hitters", "stack_upside_score"], ascending=[True, False, False])
    primary = primary.drop_duplicates("lineup_id")
    return primary[["lineup_id", "team_code", "stack_upside_score"]].rename(columns={"team_code": "primary_stack"})


def _pct_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if values.nunique() <= 1:
        return pd.Series(0.5, index=values.index)
    return values.rank(pct=True)


def _why_text(row: pd.Series) -> str:
    parts = []
    components = {
        "projection": float(row.get("projection_component", 0.0)),
        "leverage": float(row.get("leverage_component", 0.0)),
        "stack": float(row.get("stack_component", 0.0)),
        "uniqueness": float(row.get("uniqueness_component", 0.0)),
        "duplication": float(row.get("duplication_component", 0.0)),
    }
    for label, _ in sorted(components.items(), key=lambda item: item[1], reverse=True)[:2]:
        parts.append(label)
    return " + ".join(parts)


__all__ = ["build_lineup_scorecard"]
