"""Post-slate learning reports for stack, leverage, and portfolio decisions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PostSlateLearningReport:
    summary: dict[str, float]
    stack_results: pd.DataFrame
    leverage_results: pd.DataFrame
    portfolio_comparison: pd.DataFrame
    lessons: pd.DataFrame


def build_post_slate_learning_report(
    active_lineups: pd.DataFrame,
    actuals: pd.DataFrame,
    optimizer_df: Optional[pd.DataFrame] = None,
    stack_plan: Optional[pd.DataFrame] = None,
    raw_optimizer_lineups: Optional[pd.DataFrame] = None,
) -> PostSlateLearningReport:
    actual_map = _actual_map(actuals)
    active = _lineups_with_actuals(active_lineups, actual_map)
    raw = _lineups_with_actuals(raw_optimizer_lineups, actual_map) if raw_optimizer_lineups is not None else pd.DataFrame()
    stack_results = _stack_results(active, stack_plan)
    leverage_results = _leverage_results(active, optimizer_df, actuals)
    portfolio_comparison = _portfolio_comparison(active, raw)
    summary = _summary(active, stack_results, portfolio_comparison)
    lessons = _lessons(summary, stack_results, leverage_results, portfolio_comparison)
    return PostSlateLearningReport(summary, stack_results, leverage_results, portfolio_comparison, lessons)


def _actual_map(actuals: pd.DataFrame) -> pd.Series:
    if actuals is None or actuals.empty or "fd_player_id" not in actuals.columns:
        return pd.Series(dtype=float)
    score_col = "actual_fd_points" if "actual_fd_points" in actuals.columns else "actual_points"
    return actuals.set_index("fd_player_id")[score_col].astype(float)


def _lineups_with_actuals(lineups: Optional[pd.DataFrame], actual_map: pd.Series) -> pd.DataFrame:
    if lineups is None or lineups.empty or actual_map.empty:
        return pd.DataFrame()
    df = lineups.copy()
    if "fd_player_id" not in df.columns or "lineup_id" not in df.columns:
        return pd.DataFrame()
    df["actual_points"] = df["fd_player_id"].astype(str).map(actual_map).fillna(0.0)
    if "proj_fd_mean" not in df.columns:
        df["proj_fd_mean"] = 0.0
    df["proj_fd_mean"] = pd.to_numeric(df["proj_fd_mean"], errors="coerce").fillna(0.0)
    return df


def _stack_results(active: pd.DataFrame, stack_plan: Optional[pd.DataFrame]) -> pd.DataFrame:
    if active.empty or "team_code" not in active.columns or "player_type" not in active.columns:
        return pd.DataFrame()
    hitters = active[active["player_type"].astype(str).str.lower() != "pitcher"].copy()
    if hitters.empty:
        return pd.DataFrame()
    lineup_totals = active.groupby("lineup_id").agg(
        lineup_actual=("actual_points", "sum"),
        lineup_projection=("proj_fd_mean", "sum"),
    )
    team_lineups = hitters.groupby(["lineup_id", "team_code"]).size().reset_index(name="hitters")
    stacks = team_lineups[team_lineups["hitters"] >= 3]
    if stacks.empty:
        return pd.DataFrame()
    joined = stacks.merge(lineup_totals, on="lineup_id", how="left")
    result = (
        joined.groupby("team_code")
        .agg(
            stack_lineups=("lineup_id", "nunique"),
            avg_lineup_actual=("lineup_actual", "mean"),
            avg_lineup_projection=("lineup_projection", "mean"),
            avg_stack_size=("hitters", "mean"),
        )
        .reset_index()
    )
    result["actual_minus_projection"] = result["avg_lineup_actual"] - result["avg_lineup_projection"]
    if stack_plan is not None and not stack_plan.empty and "team_code" in stack_plan.columns:
        keep = [
            col
            for col in ["team_code", "tier", "target_exposure", "stack_score", "stack_leverage_score"]
            if col in stack_plan.columns
        ]
        result = result.merge(stack_plan[keep], on="team_code", how="left")
    return result.sort_values("avg_lineup_actual", ascending=False)


def _leverage_results(active: pd.DataFrame, optimizer_df: Optional[pd.DataFrame], actuals: pd.DataFrame) -> pd.DataFrame:
    if optimizer_df is None or optimizer_df.empty or actuals is None or actuals.empty:
        return pd.DataFrame()
    score_col = "actual_fd_points" if "actual_fd_points" in actuals.columns else "actual_points"
    cols = [col for col in ["fd_player_id", "full_name", "team_code", "player_type", "proj_fd_mean", "proj_fd_ownership", "player_leverage_score", "proj_fd_bust_rate", "proj_fd_upside"] if col in optimizer_df.columns]
    merged = optimizer_df[cols].drop_duplicates("fd_player_id").merge(
        actuals[["fd_player_id", score_col]],
        on="fd_player_id",
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()
    merged["actual_points"] = pd.to_numeric(merged[score_col], errors="coerce").fillna(0.0)
    merged["proj_fd_mean"] = pd.to_numeric(merged.get("proj_fd_mean"), errors="coerce").fillna(0.0)
    merged["proj_fd_ownership"] = pd.to_numeric(merged.get("proj_fd_ownership"), errors="coerce").fillna(0.0)
    if merged["proj_fd_ownership"].max() > 1.5:
        merged["proj_fd_ownership"] = merged["proj_fd_ownership"] / 100.0
    merged["player_leverage_score"] = pd.to_numeric(merged.get("player_leverage_score"), errors="coerce").fillna(0.0)
    merged["bucket"] = "Neutral"
    merged.loc[merged["proj_fd_ownership"] >= 0.20, "bucket"] = "Chalk"
    merged.loc[(merged["proj_fd_ownership"] < 0.10) & (merged["player_leverage_score"] > 0), "bucket"] = "Positive leverage"
    merged["actual_minus_projection"] = merged["actual_points"] - merged["proj_fd_mean"]
    return (
        merged.groupby("bucket")
        .agg(
            players=("fd_player_id", "nunique"),
            avg_projection=("proj_fd_mean", "mean"),
            avg_actual=("actual_points", "mean"),
            avg_ownership=("proj_fd_ownership", "mean"),
            avg_actual_minus_projection=("actual_minus_projection", "mean"),
        )
        .reset_index()
        .sort_values("avg_actual_minus_projection", ascending=False)
    )


def _portfolio_comparison(active: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, frame in [("Selected portfolio", active), ("Raw optimizer pool", raw)]:
        if frame.empty:
            continue
        lineup_scores = frame.groupby("lineup_id").agg(
            lineups=("lineup_id", "size"),
            actual=("actual_points", "sum"),
            projection=("proj_fd_mean", "sum"),
        )
        rows.append(
            {
                "portfolio": label,
                "lineups": int(lineup_scores.shape[0]),
                "avg_actual": float(lineup_scores["actual"].mean()),
                "max_actual": float(lineup_scores["actual"].max()),
                "avg_projection": float(lineup_scores["projection"].mean()),
                "actual_minus_projection": float((lineup_scores["actual"] - lineup_scores["projection"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def _summary(active: pd.DataFrame, stack_results: pd.DataFrame, portfolio_comparison: pd.DataFrame) -> dict[str, float]:
    if active.empty:
        return {}
    lineup_scores = active.groupby("lineup_id")["actual_points"].sum()
    summary = {
        "lineups": int(lineup_scores.shape[0]),
        "avg_actual": float(lineup_scores.mean()),
        "max_actual": float(lineup_scores.max()),
        "min_actual": float(lineup_scores.min()),
    }
    if not stack_results.empty:
        best = stack_results.iloc[0]
        summary["best_stack_avg_actual"] = float(best["avg_lineup_actual"])
        summary["best_stack_team"] = str(best["team_code"])
    if len(portfolio_comparison) >= 2:
        selected = portfolio_comparison[portfolio_comparison["portfolio"] == "Selected portfolio"]
        raw = portfolio_comparison[portfolio_comparison["portfolio"] == "Raw optimizer pool"]
        if not selected.empty and not raw.empty:
            summary["selected_vs_raw_avg_delta"] = float(selected.iloc[0]["avg_actual"] - raw.iloc[0]["avg_actual"])
    return summary


def _lessons(
    summary: dict[str, float],
    stack_results: pd.DataFrame,
    leverage_results: pd.DataFrame,
    portfolio_comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    if "selected_vs_raw_avg_delta" in summary:
        delta = float(summary["selected_vs_raw_avg_delta"])
        rows.append(
            {
                "area": "Portfolio selection",
                "lesson": "Simulation-selected portfolio beat raw optimizer pool on average." if delta > 0 else "Raw optimizer pool beat selected portfolio on average.",
                "value": delta,
            }
        )
    if not stack_results.empty and "tier" in stack_results.columns:
        top = stack_results.iloc[0]
        rows.append(
            {
                "area": "Stacks",
                "lesson": f"Best stacked team was {top['team_code']} ({top.get('tier', 'unrated')}).",
                "value": float(top["avg_lineup_actual"]),
            }
        )
    if not leverage_results.empty:
        best_bucket = leverage_results.iloc[0]
        rows.append(
            {
                "area": "Leverage",
                "lesson": f"{best_bucket['bucket']} players outperformed projection the most.",
                "value": float(best_bucket["avg_actual_minus_projection"]),
            }
        )
    return pd.DataFrame(rows)


__all__ = ["PostSlateLearningReport", "build_post_slate_learning_report"]
