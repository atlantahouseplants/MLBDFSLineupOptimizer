"""Post-slate attribution for why a portfolio won, lost, or stalled."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class FailureAttributionReport:
    summary: dict[str, float | str]
    factors: pd.DataFrame
    player_breakdown: pd.DataFrame
    stack_breakdown: pd.DataFrame
    lessons: list[str]


def build_failure_attribution_report(
    selected_lineups: pd.DataFrame,
    actuals: pd.DataFrame,
    optimizer_df: Optional[pd.DataFrame] = None,
    stack_plan: Optional[pd.DataFrame] = None,
    raw_optimizer_lineups: Optional[pd.DataFrame] = None,
    sim_portfolio: Optional[pd.DataFrame] = None,
) -> FailureAttributionReport:
    selected = _lineups_with_actuals(selected_lineups, actuals)
    raw = _lineups_with_actuals(raw_optimizer_lineups, actuals) if raw_optimizer_lineups is not None else pd.DataFrame()
    player_breakdown = _player_breakdown(selected, optimizer_df, actuals)
    stack_breakdown = _stack_breakdown(selected, optimizer_df, actuals, stack_plan)
    factors = _factor_rows(selected, raw, player_breakdown, stack_breakdown, actuals, sim_portfolio)
    summary = _summary(selected, factors, stack_breakdown)
    lessons = _lessons(factors, summary)
    return FailureAttributionReport(summary, factors, player_breakdown, stack_breakdown, lessons)


def _lineups_with_actuals(lineups: Optional[pd.DataFrame], actuals: pd.DataFrame) -> pd.DataFrame:
    if lineups is None or lineups.empty or actuals is None or actuals.empty:
        return pd.DataFrame()
    if "fd_player_id" not in lineups.columns:
        return pd.DataFrame()
    score_col = "actual_fd_points" if "actual_fd_points" in actuals.columns else "actual_points"
    actual_frame = actuals.copy()
    actual_frame["fd_player_id"] = actual_frame["fd_player_id"].astype(str)
    actual_map = actual_frame.set_index("fd_player_id")[score_col].astype(float)
    df = lineups.copy()
    df["fd_player_id"] = df["fd_player_id"].astype(str)
    df["actual_points"] = df["fd_player_id"].map(actual_map).fillna(0.0)
    if "proj_fd_mean" not in df.columns:
        df["proj_fd_mean"] = 0.0
    df["proj_fd_mean"] = pd.to_numeric(df["proj_fd_mean"], errors="coerce").fillna(0.0)
    if "proj_fd_ownership" in df.columns:
        df["ownership"] = _ownership_decimal(df["proj_fd_ownership"])
    else:
        df["ownership"] = 0.0
    if "player_leverage_score" not in df.columns:
        df["player_leverage_score"] = 0.0
    df["player_leverage_score"] = pd.to_numeric(df["player_leverage_score"], errors="coerce").fillna(0.0)
    df["actual_minus_projection"] = df["actual_points"] - df["proj_fd_mean"]
    return df


def _player_breakdown(
    selected: pd.DataFrame,
    optimizer_df: Optional[pd.DataFrame],
    actuals: pd.DataFrame,
) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    rows = (
        selected.groupby("fd_player_id")
        .agg(
            appearances=("lineup_id", "nunique"),
            player_name=("full_name", "first") if "full_name" in selected.columns else ("fd_player_id", "first"),
            team_code=("team_code", "first") if "team_code" in selected.columns else ("fd_player_id", "first"),
            player_type=("player_type", "first") if "player_type" in selected.columns else ("fd_player_id", "first"),
            projection=("proj_fd_mean", "mean"),
            actual=("actual_points", "mean"),
            actual_minus_projection=("actual_minus_projection", "mean"),
            ownership=("ownership", "mean"),
            leverage=("player_leverage_score", "mean"),
        )
        .reset_index()
    )
    total_lineups = max(1, int(selected["lineup_id"].nunique())) if "lineup_id" in selected.columns else 1
    rows["exposure"] = rows["appearances"] / total_lineups

    if actuals is not None and not actuals.empty and "actual_ownership_pct" in actuals.columns:
        own = actuals[["fd_player_id", "actual_ownership_pct"]].copy()
        own["fd_player_id"] = own["fd_player_id"].astype(str)
        own["actual_ownership"] = _ownership_decimal(own["actual_ownership_pct"])
        rows = rows.merge(own[["fd_player_id", "actual_ownership"]], on="fd_player_id", how="left")
        rows["ownership_error"] = rows["ownership"] - rows["actual_ownership"]
    if optimizer_df is not None and not optimizer_df.empty and "proj_fd_upside" in optimizer_df.columns:
        upside = optimizer_df[["fd_player_id", "proj_fd_upside", "proj_fd_bust_rate"]].drop_duplicates("fd_player_id")
        rows = rows.merge(upside, on="fd_player_id", how="left")
    return rows.sort_values(["exposure", "actual_minus_projection"], ascending=[False, True])


def _stack_breakdown(
    selected: pd.DataFrame,
    optimizer_df: Optional[pd.DataFrame],
    actuals: pd.DataFrame,
    stack_plan: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if selected.empty or "team_code" not in selected.columns or "player_type" not in selected.columns:
        return pd.DataFrame()
    hitters = selected[selected["player_type"].astype(str).str.lower() != "pitcher"].copy()
    if hitters.empty:
        return pd.DataFrame()
    lineup_totals = selected.groupby("lineup_id").agg(
        lineup_actual=("actual_points", "sum"),
        lineup_projection=("proj_fd_mean", "sum"),
    )
    stacks = hitters.groupby(["lineup_id", "team_code"]).size().reset_index(name="hitters")
    stacks = stacks[stacks["hitters"] >= 3]
    if stacks.empty:
        return pd.DataFrame()
    rows = stacks.merge(lineup_totals, on="lineup_id", how="left").groupby("team_code").agg(
        lineups=("lineup_id", "nunique"),
        avg_lineup_actual=("lineup_actual", "mean"),
        avg_lineup_projection=("lineup_projection", "mean"),
        avg_stack_size=("hitters", "mean"),
    ).reset_index()
    rows["actual_minus_projection"] = rows["avg_lineup_actual"] - rows["avg_lineup_projection"]
    rows["slate_actual_stack_rank"] = rows["team_code"].map(_actual_team_ranks(optimizer_df, actuals))
    if stack_plan is not None and not stack_plan.empty and "team_code" in stack_plan.columns:
        keep = [col for col in ["team_code", "tier", "target_exposure", "stack_score", "stack_ownership"] if col in stack_plan.columns]
        rows = rows.merge(stack_plan[keep], on="team_code", how="left")
    return rows.sort_values(["lineups", "actual_minus_projection"], ascending=[False, False])


def _factor_rows(
    selected: pd.DataFrame,
    raw: pd.DataFrame,
    player_breakdown: pd.DataFrame,
    stack_breakdown: pd.DataFrame,
    actuals: pd.DataFrame,
    sim_portfolio: Optional[pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if selected.empty:
        return pd.DataFrame()

    lineup_scores = selected.groupby("lineup_id").agg(
        actual=("actual_points", "sum"),
        projection=("proj_fd_mean", "sum"),
    )
    avg_delta = float((lineup_scores["actual"] - lineup_scores["projection"]).mean())
    rows.append(_factor("Projection miss", avg_delta, "Avg lineup actual minus projection", higher_is_good=True, high_bad=-18, med_bad=-9))

    pitchers = selected[selected["player_type"].astype(str).str.lower() == "pitcher"] if "player_type" in selected.columns else pd.DataFrame()
    if not pitchers.empty:
        pitcher_delta = float(pitchers["actual_minus_projection"].mean())
        rows.append(_factor("Pitcher outcome", pitcher_delta, "Selected pitcher actual minus projection", higher_is_good=True, high_bad=-8, med_bad=-4))

    hitters = selected[selected["player_type"].astype(str).str.lower() != "pitcher"] if "player_type" in selected.columns else selected
    chalk = hitters[hitters["ownership"] >= 0.18]
    if not chalk.empty:
        chalk_delta = float(chalk["actual_minus_projection"].mean())
        rows.append(_factor("Chalk bats", chalk_delta, "Selected chalk hitter actual minus projection", higher_is_good=True, high_bad=-5, med_bad=-2))

    leverage = hitters[hitters["player_leverage_score"] > 0.05]
    if not leverage.empty:
        leverage_delta = float(leverage["actual_minus_projection"].mean())
        rows.append(_factor("Leverage bats", leverage_delta, "Positive-leverage hitter actual minus projection", higher_is_good=True, high_bad=-5, med_bad=-2))

    if not stack_breakdown.empty:
        weak_stack_share = float((pd.to_numeric(stack_breakdown["actual_minus_projection"], errors="coerce") < -10).mean())
        rows.append(_factor("Stack selection", 1.0 - weak_stack_share, "Share of stacked teams avoiding a large miss", higher_is_good=True, high_bad=0.45, med_bad=0.65))

    if "ownership_error" in player_breakdown.columns:
        own_mae = float(pd.to_numeric(player_breakdown["ownership_error"], errors="coerce").abs().mean())
        rows.append(_factor("Ownership miss", own_mae, "Avg exposure-weighted projected vs actual ownership error", higher_is_good=False, high_bad=0.08, med_bad=0.045))

    if sim_portfolio is not None and not sim_portfolio.empty:
        dup = pd.to_numeric(sim_portfolio.get("estimated_dupes"), errors="coerce").dropna()
        if not dup.empty:
            rows.append(_factor("Duplication risk", float(dup.mean()), "Avg estimated duplicates in selected sim portfolio", higher_is_good=False, high_bad=18.0, med_bad=8.0))

    if not raw.empty:
        raw_scores = raw.groupby("lineup_id")["actual_points"].sum()
        selected_avg = float(lineup_scores["actual"].mean())
        raw_avg = float(raw_scores.mean())
        rows.append(_factor("Selector value", selected_avg - raw_avg, "Selected portfolio average actual minus raw pool", higher_is_good=True, high_bad=-8, med_bad=-3))

    return pd.DataFrame(rows)


def _factor(name: str, value: float, detail: str, higher_is_good: bool, high_bad: float, med_bad: float) -> dict[str, object]:
    if not np.isfinite(value):
        severity = "Unknown"
    elif higher_is_good:
        severity = "High" if value <= high_bad else "Medium" if value <= med_bad else "Low"
    else:
        severity = "High" if value >= high_bad else "Medium" if value >= med_bad else "Low"
    return {"factor": name, "value": value, "severity": severity, "detail": detail}


def _summary(selected: pd.DataFrame, factors: pd.DataFrame, stack_breakdown: pd.DataFrame) -> dict[str, float | str]:
    if selected.empty:
        return {}
    lineup_scores = selected.groupby("lineup_id")["actual_points"].sum()
    summary: dict[str, float | str] = {
        "lineups": int(lineup_scores.shape[0]),
        "avg_actual": float(lineup_scores.mean()),
        "best_actual": float(lineup_scores.max()),
        "high_severity_factors": int((factors.get("severity") == "High").sum()) if not factors.empty else 0,
    }
    if not stack_breakdown.empty:
        summary["most_used_stack"] = str(stack_breakdown.iloc[0]["team_code"])
        summary["most_used_stack_delta"] = float(stack_breakdown.iloc[0]["actual_minus_projection"])
    return summary


def _lessons(factors: pd.DataFrame, summary: dict[str, float | str]) -> list[str]:
    if factors is None or factors.empty:
        return []
    notes = []
    high = factors[factors["severity"] == "High"]
    if not high.empty:
        names = ", ".join(high["factor"].astype(str).tolist()[:3])
        notes.append(f"Primary failure area: {names}.")
    else:
        notes.append("No single high-severity failure driver; portfolio result was more likely normal MLB variance.")
    medium = factors[factors["severity"] == "Medium"]
    if not medium.empty:
        notes.append("Medium-severity items are good candidates for small calibration moves, not full strategy changes.")
    return notes


def _actual_team_ranks(optimizer_df: Optional[pd.DataFrame], actuals: pd.DataFrame) -> dict[str, int]:
    if optimizer_df is None or optimizer_df.empty or actuals is None or actuals.empty:
        return {}
    if "team_code" not in optimizer_df.columns or "fd_player_id" not in optimizer_df.columns:
        return {}
    score_col = "actual_fd_points" if "actual_fd_points" in actuals.columns else "actual_points"
    meta = optimizer_df[["fd_player_id", "team_code", "player_type"]].drop_duplicates("fd_player_id").copy()
    actual_frame = actuals[["fd_player_id", score_col]].copy()
    meta["fd_player_id"] = meta["fd_player_id"].astype(str)
    actual_frame["fd_player_id"] = actual_frame["fd_player_id"].astype(str)
    merged = meta.merge(actual_frame, on="fd_player_id", how="inner")
    if "player_type" in merged.columns:
        merged = merged[merged["player_type"].astype(str).str.lower() != "pitcher"]
    if merged.empty:
        return {}
    team_scores = pd.to_numeric(merged[score_col], errors="coerce").fillna(0.0).groupby(merged["team_code"].astype(str)).sum()
    ordered = team_scores.sort_values(ascending=False)
    return {str(team): rank + 1 for rank, team in enumerate(ordered.index.tolist())}


def _ownership_decimal(values) -> pd.Series:
    series = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0).astype(float)
    if not series.empty and float(series.max()) > 1.5:
        series = series / 100.0
    return series.clip(0.0, 1.0)


__all__ = ["FailureAttributionReport", "build_failure_attribution_report"]
