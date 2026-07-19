"""Historical slate replay diagnostics using saved pre-lock data and actuals."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from slate_optimizer.analysis.stack_exposure import build_stack_exposure_plan
from slate_optimizer.data.storage import SlateDatabase


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"


@dataclass
class SlateReplayReport:
    summary: pd.DataFrame
    daily: pd.DataFrame
    strategy_scores: pd.DataFrame
    lessons: list[str]


REPLAY_WEIGHT_PROFILES: dict[str, Mapping[str, float]] = {
    "Projection only": {"projection": 1.00},
    "Balanced leverage": {"projection": 0.55, "leverage": 0.18, "ownership": -0.10, "upside": 0.20, "bust": -0.10},
    "Chalk fade": {"projection": 0.42, "leverage": 0.30, "ownership": -0.28, "upside": 0.22, "bust": -0.12},
    "Stack upside": {"projection": 0.44, "leverage": 0.18, "ownership": -0.12, "upside": 0.18, "bust": -0.08, "stack": 0.28},
    "Boom/bust leverage": {"projection": 0.36, "leverage": 0.28, "ownership": -0.18, "upside": 0.34, "bust": -0.22},
}


def build_slate_replay_report(
    db_path: Path | str,
    start_date: str,
    end_date: str,
    data_dir: Path | str | None = None,
) -> SlateReplayReport:
    data_root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    db = SlateDatabase(Path(db_path))
    daily_rows: list[dict[str, object]] = []
    strategy_rows: list[dict[str, object]] = []
    try:
        slates = db.fetch_slate_results_range(start_date, end_date)
        if slates.empty:
            return SlateReplayReport(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [])
        for _, slate in slates.iterrows():
            date = str(slate.get("date", ""))
            actuals = db.fetch_actual_scores(date)
            lineups = db.fetch_lineup_results(date)
            optimizer = _load_optimizer_dataset(date, data_root)
            daily_rows.append(_daily_replay_row(date, slate, lineups, actuals, optimizer))
            strategy_rows.extend(_strategy_replay_rows(date, optimizer, actuals))
    finally:
        db.close()

    daily = pd.DataFrame(daily_rows).sort_values("date") if daily_rows else pd.DataFrame()
    strategy_scores = pd.DataFrame(strategy_rows)
    summary = _summary(strategy_scores, daily)
    lessons = _lessons(summary, daily, strategy_scores)
    return SlateReplayReport(summary=summary, daily=daily, strategy_scores=strategy_scores, lessons=lessons)


def _load_optimizer_dataset(date: str, data_dir: Path) -> pd.DataFrame:
    matches = list(Path(data_dir).glob(f"*{date}_optimizer_dataset.csv"))
    if not matches:
        return pd.DataFrame()
    try:
        return pd.read_csv(matches[0])
    except Exception:
        return pd.DataFrame()


def _daily_replay_row(
    date: str,
    slate: pd.Series,
    lineups: pd.DataFrame,
    actuals: pd.DataFrame,
    optimizer: pd.DataFrame,
) -> dict[str, object]:
    row: dict[str, object] = {
        "date": date,
        "contest_entries": int(slate.get("num_entries") or 0),
        "winning_score": _num(slate.get("winning_score")),
        "cash_line": _num(slate.get("cash_line")),
        "recorded_lineups": int(len(lineups)) if lineups is not None else 0,
        "optimizer_dataset_found": bool(optimizer is not None and not optimizer.empty),
    }
    if lineups is not None and not lineups.empty:
        payout = pd.to_numeric(lineups.get("payout"), errors="coerce").fillna(0.0)
        roi = pd.to_numeric(lineups.get("roi"), errors="coerce")
        score = pd.to_numeric(lineups.get("total_actual_points"), errors="coerce")
        row.update(
            {
                "avg_roi": float(roi.mean()) if roi.notna().any() else np.nan,
                "cash_rate": float((payout > 0).mean()) if len(lineups) else np.nan,
                "best_lineup_score": float(score.max()) if score.notna().any() else np.nan,
                "avg_lineup_score": float(score.mean()) if score.notna().any() else np.nan,
            }
        )
    merged = _merge_optimizer_actuals(optimizer, actuals)
    if not merged.empty:
        chalk = merged[merged["ownership"] >= 0.18]
        contrarian = merged[merged["ownership"] < 0.10]
        row.update(
            {
                "projection_actual_corr": _corr(merged["proj_fd_mean"], merged["actual_fd_points"]),
                "leverage_actual_corr": _corr(merged.get("player_leverage_score"), merged["actual_minus_projection"]),
                "chalk_outperformance": float(chalk["actual_minus_projection"].mean()) if not chalk.empty else np.nan,
                "contrarian_outperformance": float(contrarian["actual_minus_projection"].mean()) if not contrarian.empty else np.nan,
                "top_actual_stack": _top_team(merged, "actual_fd_points"),
                "top_projected_stack": _top_team(merged, "proj_fd_mean"),
                "top_owned_stack": _top_team(merged, "ownership"),
            }
        )
        plan = build_stack_exposure_plan(optimizer)
        if not plan.empty and "team_code" in plan.columns:
            row["top_stack_plan_team"] = str(plan.iloc[0]["team_code"])
            row["stack_plan_hit"] = row.get("top_stack_plan_team") == row.get("top_actual_stack")
    return row


def _strategy_replay_rows(date: str, optimizer: pd.DataFrame, actuals: pd.DataFrame) -> list[dict[str, object]]:
    merged = _merge_optimizer_actuals(optimizer, actuals)
    if merged.empty:
        return []
    hitters = merged[merged["player_type"].astype(str).str.lower() != "pitcher"].copy()
    if hitters.empty:
        return []
    plan = build_stack_exposure_plan(optimizer)
    stack_weights = {
        str(row["team_code"]): float(row.get("stack_weight", row.get("stack_score", 0.0)) or 0.0)
        for _, row in plan.iterrows()
    } if not plan.empty and "team_code" in plan.columns else {}
    actual_top_stack = _top_team(merged, "actual_fd_points")

    rows: list[dict[str, object]] = []
    take = min(24, max(8, int(np.ceil(len(hitters) * 0.20))))
    for profile, weights in REPLAY_WEIGHT_PROFILES.items():
        scored = hitters.copy()
        scored["replay_selection_score"] = _weighted_score(scored, weights, stack_weights)
        selected = scored.sort_values("replay_selection_score", ascending=False).head(take)
        top_stack = _top_team(selected, "replay_selection_score")
        rows.append(
            {
                "date": date,
                "strategy_profile": profile,
                "players_selected": int(len(selected)),
                "avg_actual": float(selected["actual_fd_points"].mean()),
                "avg_projection": float(selected["proj_fd_mean"].mean()),
                "avg_actual_minus_projection": float(selected["actual_minus_projection"].mean()),
                "boom_hit_rate": float((selected["actual_minus_projection"] >= 5.0).mean()),
                "chalk_selected_pct": float((selected["ownership"] >= 0.18).mean()),
                "avg_leverage": float(pd.to_numeric(selected.get("player_leverage_score"), errors="coerce").fillna(0.0).mean()),
                "top_selected_stack": top_stack,
                "actual_top_stack": actual_top_stack,
                "stack_hit": bool(top_stack and actual_top_stack and top_stack == actual_top_stack),
                "replay_score": _profile_replay_score(selected, top_stack == actual_top_stack),
            }
        )
    return rows


def _merge_optimizer_actuals(optimizer: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    if optimizer is None or optimizer.empty or actuals is None or actuals.empty:
        return pd.DataFrame()
    if "fd_player_id" not in optimizer.columns or "fd_player_id" not in actuals.columns:
        return pd.DataFrame()
    cols = [
        col
        for col in [
            "fd_player_id",
            "full_name",
            "team_code",
            "player_type",
            "proj_fd_mean",
            "proj_fd_ownership",
            "player_leverage_score",
            "proj_fd_upside",
            "proj_fd_bust_rate",
        ]
        if col in optimizer.columns
    ]
    actual_cols = [col for col in ["fd_player_id", "actual_fd_points", "actual_ownership_pct"] if col in actuals.columns]
    left = optimizer[cols].drop_duplicates("fd_player_id").copy()
    right = actuals[actual_cols].copy()
    left["fd_player_id"] = left["fd_player_id"].astype(str)
    right["fd_player_id"] = right["fd_player_id"].astype(str)
    merged = left.merge(right, on="fd_player_id", how="inner")
    if merged.empty:
        return pd.DataFrame()
    for col in ["proj_fd_mean", "proj_fd_ownership", "player_leverage_score", "proj_fd_upside", "proj_fd_bust_rate", "actual_fd_points"]:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    merged["ownership"] = merged["proj_fd_ownership"]
    if merged["ownership"].max() > 1.5:
        merged["ownership"] = merged["ownership"] / 100.0
    merged["ownership"] = merged["ownership"].clip(0.0, 1.0)
    merged["proj_fd_upside"] = merged["proj_fd_upside"].where(merged["proj_fd_upside"] > 0, merged["proj_fd_mean"])
    merged["actual_minus_projection"] = merged["actual_fd_points"] - merged["proj_fd_mean"]
    return merged


def _weighted_score(df: pd.DataFrame, weights: Mapping[str, float], stack_weights: Mapping[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=df.index, dtype=float)
    components = {
        "projection": df["proj_fd_mean"],
        "leverage": df["player_leverage_score"],
        "ownership": df["ownership"],
        "upside": df["proj_fd_upside"],
        "bust": df["proj_fd_bust_rate"],
        "stack": df.get("team_code", pd.Series("", index=df.index)).astype(str).map(stack_weights).fillna(0.0),
    }
    for name, weight in weights.items():
        score += float(weight) * _rank01(components.get(name, pd.Series(0.0, index=df.index)))
    return score


def _profile_replay_score(selected: pd.DataFrame, stack_hit: bool) -> float:
    actual_delta = float(selected["actual_minus_projection"].mean()) if not selected.empty else 0.0
    boom_rate = float((selected["actual_minus_projection"] >= 5.0).mean()) if not selected.empty else 0.0
    chalk_pct = float((selected["ownership"] >= 0.18).mean()) if not selected.empty else 0.0
    return float(actual_delta + 4.0 * boom_rate + (2.5 if stack_hit else 0.0) - 0.75 * chalk_pct)


def _summary(strategy_scores: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    if strategy_scores is None or strategy_scores.empty:
        return pd.DataFrame()
    grouped = (
        strategy_scores.groupby("strategy_profile")
        .agg(
            slates=("date", "nunique"),
            avg_replay_score=("replay_score", "mean"),
            avg_actual_minus_projection=("avg_actual_minus_projection", "mean"),
            boom_hit_rate=("boom_hit_rate", "mean"),
            stack_hit_rate=("stack_hit", "mean"),
            chalk_selected_pct=("chalk_selected_pct", "mean"),
        )
        .reset_index()
        .sort_values(["avg_replay_score", "boom_hit_rate"], ascending=False)
    )
    if daily is not None and not daily.empty and "avg_roi" in daily.columns:
        grouped["recorded_portfolio_avg_roi"] = float(pd.to_numeric(daily["avg_roi"], errors="coerce").mean())
    return grouped


def _lessons(summary: pd.DataFrame, daily: pd.DataFrame, strategy_scores: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    if summary is not None and not summary.empty:
        leader = summary.iloc[0]
        notes.append(
            f"Best replay profile: {leader['strategy_profile']} "
            f"({float(leader['avg_replay_score']):+.2f} replay score across {int(leader['slates'])} slates)."
        )
    if daily is not None and not daily.empty:
        corr = pd.to_numeric(daily.get("projection_actual_corr"), errors="coerce").dropna()
        if not corr.empty and float(corr.mean()) < 0.15:
            notes.append("Projection-to-actual correlation has been weak in this range; lean more on portfolio construction than raw median.")
        stack_hit = daily.get("stack_plan_hit")
        if stack_hit is not None and len(stack_hit.dropna()):
            rate = float(pd.Series(stack_hit).astype(float).mean())
            if rate < 0.25:
                notes.append("Stack plan top team has not often been the actual best stack; keep stack exposure spread wider.")
    if not notes and strategy_scores is not None and not strategy_scores.empty:
        notes.append("Replay data loaded; compare profiles before changing defaults.")
    return notes


def _top_team(df: pd.DataFrame, column: str) -> str:
    if df is None or df.empty or "team_code" not in df.columns or column not in df.columns:
        return ""
    hitters = df[df["player_type"].astype(str).str.lower() != "pitcher"] if "player_type" in df.columns else df
    if hitters.empty:
        return ""
    values = pd.to_numeric(hitters[column], errors="coerce").fillna(0.0)
    grouped = values.groupby(hitters["team_code"].astype(str)).sum()
    return str(grouped.sort_values(ascending=False).index[0]) if not grouped.empty else ""


def _rank01(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if len(values) <= 1 or values.nunique() <= 1:
        return pd.Series(0.5, index=values.index, dtype=float)
    return values.rank(pct=True)


def _corr(a, b) -> float:
    if a is None or b is None:
        return np.nan
    left = pd.to_numeric(pd.Series(a), errors="coerce")
    right = pd.to_numeric(pd.Series(b), errors="coerce")
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 3 or frame["left"].nunique() <= 1 or frame["right"].nunique() <= 1:
        return np.nan
    return float(frame["left"].corr(frame["right"]))


def _num(value) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


__all__ = ["REPLAY_WEIGHT_PROFILES", "SlateReplayReport", "build_slate_replay_report"]
