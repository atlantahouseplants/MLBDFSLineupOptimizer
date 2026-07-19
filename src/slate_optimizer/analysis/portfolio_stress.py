"""Portfolio stress tests for chalk, stack, and player dependency."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioStressReport:
    summary: dict[str, float | str]
    scenarios: pd.DataFrame
    recommendations: list[str]


def build_portfolio_stress_report(
    lineup_df: pd.DataFrame,
    optimizer_df: pd.DataFrame,
    stack_plan: pd.DataFrame | None = None,
) -> PortfolioStressReport:
    if lineup_df is None or lineup_df.empty:
        return PortfolioStressReport({}, pd.DataFrame(), [])
    df = lineup_df.copy()
    total_lineups = max(1, int(df["lineup_id"].nunique()))
    hitters = df[df["player_type"].astype(str).str.lower() != "pitcher"] if "player_type" in df.columns else df
    stack_counts = _lineup_stack_counts(hitters)
    top_chalk_stack = _top_chalk_stack(optimizer_df, stack_plan)
    top_owned_pitcher = _top_owned_pitcher(optimizer_df)
    top_owned_batter = _top_owned_batter(optimizer_df)
    low_owned_stack = _best_low_owned_stack(stack_plan)

    scenario_rows = []
    base_projection = _lineup_totals(df, "proj_fd_mean")
    base_ownership = _lineup_totals(df, "proj_fd_ownership")
    scenario_rows.append(
        _sensitivity_row(
            "Projection -5%",
            base_projection * 0.95,
            base_ownership,
            "Checks whether the portfolio still has enough raw projection when all means are haircut.",
        )
    )
    scenario_rows.append(
        _sensitivity_row(
            "Ownership +20%",
            base_projection,
            base_ownership * 1.20,
            "Shows how crowded the portfolio becomes if projected ownership steams upward.",
        )
    )
    if top_chalk_stack:
        stack_lineups = set(stack_counts.loc[stack_counts["team_code"] == top_chalk_stack, "lineup_id"].tolist())
        exposure = len(stack_lineups) / total_lineups
        scenario_rows.append(
            _scenario_row(
                "Chalk stack succeeds",
                exposure,
                1.0 - exposure,
                "High exposure benefits if the stack wins the slate; low exposure needs other teams to outscore it.",
            )
        )
        scenario_rows.append(
            _scenario_row(
                "Chalk stack fails",
                1.0 - exposure,
                exposure,
                "Fading or underweighting the chalk stack creates leverage when it fails.",
            )
        )
        scenario_rows.append(
            _team_projection_shift_row(
                df,
                top_chalk_stack,
                "Chalk stack smashes",
                boost=0.18,
                note="Boosts hitters from the highest-owned stack to show chalk-smash sensitivity.",
            )
        )
    if low_owned_stack:
        scenario_rows.append(
            _team_projection_shift_row(
                df,
                low_owned_stack,
                "Low-owned stack overperforms",
                boost=0.22,
                note="Boosts a low-owned upside stack to test whether the portfolio has leverage paths.",
            )
        )
    if top_owned_pitcher:
        pitcher_lineups = _lineups_with_player(df, top_owned_pitcher)
        exposure = len(pitcher_lineups) / total_lineups
        scenario_rows.append(
            _scenario_row(
                "Top pitcher fails",
                1.0 - exposure,
                exposure,
                "Pitcher concentration creates direct portfolio fragility because every lineup uses one pitcher.",
            )
        )
    if top_owned_batter:
        batter_lineups = _lineups_with_player(df, top_owned_batter)
        exposure = len(batter_lineups) / total_lineups
        scenario_rows.append(
            _scenario_row(
                "Top owned bat fails",
                1.0 - exposure,
                exposure,
                "High exposure to one chalk bat can quietly make many stacks share the same failure point.",
            )
        )

    scenarios = pd.DataFrame(scenario_rows)
    summary = {
        "lineups": float(total_lineups),
        "top_chalk_stack": top_chalk_stack or "",
        "top_pitcher": top_owned_pitcher or "",
        "top_batter": top_owned_batter or "",
        "max_stack_dependency": float(stack_counts.groupby("team_code")["lineup_id"].nunique().max() / total_lineups) if not stack_counts.empty else 0.0,
    }
    recommendations = _recommendations(scenarios, summary)
    return PortfolioStressReport(summary=summary, scenarios=scenarios, recommendations=recommendations)


def _scenario_row(name: str, alive_pct: float, fragile_pct: float, note: str) -> dict[str, float | str]:
    return {
        "scenario": name,
        "portfolio_alive_pct": float(np.clip(alive_pct, 0.0, 1.0)),
        "fragile_pct": float(np.clip(fragile_pct, 0.0, 1.0)),
        "note": note,
    }


def _sensitivity_row(name: str, projection: pd.Series, ownership: pd.Series, note: str) -> dict[str, float | str]:
    projection = pd.to_numeric(projection, errors="coerce").fillna(0.0)
    ownership = pd.to_numeric(ownership, errors="coerce").fillna(0.0)
    return {
        "scenario": name,
        "portfolio_alive_pct": float((projection >= projection.quantile(0.50)).mean()) if len(projection) else 0.0,
        "fragile_pct": float((ownership >= ownership.quantile(0.75)).mean()) if len(ownership) else 0.0,
        "avg_projection": float(projection.mean()) if len(projection) else 0.0,
        "avg_ownership": float(ownership.mean()) if len(ownership) else 0.0,
        "note": note,
    }


def _team_projection_shift_row(df: pd.DataFrame, team: str, name: str, boost: float, note: str) -> dict[str, float | str]:
    shifted = df.copy()
    if "proj_fd_mean" not in shifted.columns:
        shifted["proj_fd_mean"] = 0.0
    shifted["proj_fd_mean"] = pd.to_numeric(shifted["proj_fd_mean"], errors="coerce").fillna(0.0)
    is_team_hitter = (
        shifted.get("team_code", "").astype(str).eq(str(team))
        & (shifted.get("player_type", "").astype(str).str.lower() != "pitcher")
    )
    shifted.loc[is_team_hitter, "proj_fd_mean"] = shifted.loc[is_team_hitter, "proj_fd_mean"] * (1.0 + float(boost))
    totals = _lineup_totals(shifted, "proj_fd_mean")
    team_lineups = shifted.loc[is_team_hitter, "lineup_id"].nunique()
    total_lineups = max(1, shifted["lineup_id"].nunique())
    return {
        "scenario": name,
        "portfolio_alive_pct": float(team_lineups / total_lineups),
        "fragile_pct": float(1.0 - team_lineups / total_lineups),
        "avg_projection": float(totals.mean()) if len(totals) else 0.0,
        "avg_ownership": float(_lineup_totals(shifted, "proj_fd_ownership").mean()) if "proj_fd_ownership" in shifted.columns else 0.0,
        "note": note,
    }


def _lineup_totals(df: pd.DataFrame, column: str) -> pd.Series:
    if df is None or df.empty or "lineup_id" not in df.columns or column not in df.columns:
        return pd.Series(dtype=float)
    values = df.copy()
    values[column] = pd.to_numeric(values[column], errors="coerce").fillna(0.0)
    if column == "proj_fd_ownership" and values[column].max() > 1.5:
        values[column] = values[column] / 100.0
    return values.groupby("lineup_id")[column].sum()


def _lineup_stack_counts(hitters: pd.DataFrame) -> pd.DataFrame:
    if hitters.empty or not {"lineup_id", "team_code"}.issubset(hitters.columns):
        return pd.DataFrame(columns=["lineup_id", "team_code", "batters"])
    counts = hitters.groupby(["lineup_id", "team_code"]).size().reset_index(name="batters")
    return counts[counts["batters"] >= 3]


def _top_chalk_stack(optimizer_df: pd.DataFrame, stack_plan: pd.DataFrame | None) -> str:
    if stack_plan is not None and not stack_plan.empty and {"team_code", "stack_ownership"}.issubset(stack_plan.columns):
        row = stack_plan.assign(_own=pd.to_numeric(stack_plan["stack_ownership"], errors="coerce").fillna(0.0)).sort_values("_own", ascending=False).head(1)
        if not row.empty:
            return str(row.iloc[0]["team_code"])
    if optimizer_df is None or optimizer_df.empty or "team_code" not in optimizer_df.columns:
        return ""
    df = optimizer_df.copy()
    if "player_type" in df.columns:
        df = df[df["player_type"].astype(str).str.lower() != "pitcher"]
    own = _ownership_decimal(df.get("proj_fd_ownership"))
    df = df.reset_index(drop=True)
    df["_own"] = own.reset_index(drop=True)
    grouped = df.groupby("team_code")["_own"].sum().sort_values(ascending=False)
    return str(grouped.index[0]) if not grouped.empty else ""


def _best_low_owned_stack(stack_plan: pd.DataFrame | None) -> str:
    if stack_plan is None or stack_plan.empty or "team_code" not in stack_plan.columns:
        return ""
    plan = stack_plan.copy()
    for col in ["stack_ownership", "sim_upside_score", "stack_score"]:
        if col not in plan.columns:
            plan[col] = 0.0
        plan[col] = pd.to_numeric(plan[col], errors="coerce").fillna(0.0)
    if plan.empty:
        return ""
    low_owned = plan[plan["stack_ownership"] <= plan["stack_ownership"].median()]
    if low_owned.empty:
        low_owned = plan
    row = low_owned.sort_values(["sim_upside_score", "stack_score"], ascending=False).head(1)
    return str(row.iloc[0]["team_code"]) if not row.empty else ""


def _top_owned_pitcher(optimizer_df: pd.DataFrame) -> str:
    return _top_owned_player(optimizer_df, pitcher=True)


def _top_owned_batter(optimizer_df: pd.DataFrame) -> str:
    return _top_owned_player(optimizer_df, pitcher=False)


def _top_owned_player(optimizer_df: pd.DataFrame, pitcher: bool) -> str:
    if optimizer_df is None or optimizer_df.empty or "full_name" not in optimizer_df.columns:
        return ""
    df = optimizer_df.copy()
    if "player_type" in df.columns:
        is_pitcher = df["player_type"].astype(str).str.lower() == "pitcher"
        df = df[is_pitcher] if pitcher else df[~is_pitcher]
    if df.empty:
        return ""
    own = _ownership_decimal(df.get("proj_fd_ownership"))
    df = df.reset_index(drop=True)
    df["_own"] = own.reset_index(drop=True)
    return str(df.sort_values("_own", ascending=False).iloc[0]["full_name"])


def _lineups_with_player(lineup_df: pd.DataFrame, player_name: str) -> set[int]:
    if not player_name or "full_name" not in lineup_df.columns:
        return set()
    return set(lineup_df.loc[lineup_df["full_name"].astype(str) == str(player_name), "lineup_id"].astype(int).tolist())


def _recommendations(scenarios: pd.DataFrame, summary: dict[str, float | str]) -> list[str]:
    notes: list[str] = []
    if scenarios is not None and not scenarios.empty:
        fragile = pd.to_numeric(scenarios["fragile_pct"], errors="coerce").max()
        if fragile >= 0.45:
            notes.append("One stress path can damage at least 45% of the portfolio; reduce that player/stack dependency or increase marginal value weight.")
        elif fragile <= 0.25:
            notes.append("Stress exposure is well distributed across the major chalk failure paths.")
    if float(summary.get("max_stack_dependency", 0.0) or 0.0) >= 0.40:
        notes.append("One team stack is carrying a large share of the portfolio. This is fine only if it is an intentional overweight.")
    if not notes:
        notes.append("No obvious portfolio stress concentration detected.")
    return notes


def _ownership_decimal(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    series = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0).astype(float)
    if not series.empty and float(series.max()) > 1.5:
        series = series / 100.0
    return series.clip(lower=0.0, upper=1.0)


__all__ = ["PortfolioStressReport", "build_portfolio_stress_report"]
