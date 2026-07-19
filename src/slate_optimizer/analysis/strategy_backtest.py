"""Historical strategy performance summaries."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from slate_optimizer.data.storage import SlateDatabase


@dataclass
class StrategyBacktestReport:
    by_strategy: pd.DataFrame
    daily: pd.DataFrame
    lessons: list[str]


def build_strategy_backtest_report(
    db_path: Path | str,
    start_date: str,
    end_date: str,
) -> StrategyBacktestReport:
    db = SlateDatabase(Path(db_path))
    try:
        slates = db.fetch_slate_results_range(start_date, end_date)
        rows: list[pd.DataFrame] = []
        for date in slates.get("date", pd.Series(dtype=str)).astype(str).unique():
            lineups = db.fetch_lineup_results(date)
            if lineups.empty:
                continue
            rows.append(_lineup_strategy_rows(date, lineups))
    finally:
        db.close()

    daily = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if daily.empty:
        return StrategyBacktestReport(pd.DataFrame(), pd.DataFrame(), [])

    by_strategy = (
        daily.groupby("strategy_label", dropna=False)
        .agg(
            slates=("date", "nunique"),
            entries=("lineups", "sum"),
            avg_roi=("avg_roi", "mean"),
            median_roi=("avg_roi", "median"),
            cash_rate=("cash_rate", "mean"),
            total_payout=("total_payout", "sum"),
            avg_score=("avg_score", "mean"),
            best_score=("best_score", "max"),
        )
        .reset_index()
        .sort_values(["avg_roi", "cash_rate"], ascending=False)
    )
    lessons = _strategy_lessons(by_strategy)
    return StrategyBacktestReport(by_strategy=by_strategy, daily=daily, lessons=lessons)


def _lineup_strategy_rows(date: str, lineups: pd.DataFrame) -> pd.DataFrame:
    working = lineups.copy()
    if "strategy_config_json" not in working.columns:
        working["strategy_config_json"] = ""
    working["strategy_label"] = working["strategy_config_json"].map(_strategy_label)
    rows = []
    for label, group in working.groupby("strategy_label", dropna=False):
        payout = pd.to_numeric(group.get("payout"), errors="coerce").fillna(0.0)
        roi = pd.to_numeric(group.get("roi"), errors="coerce")
        score = pd.to_numeric(group.get("total_actual_points"), errors="coerce")
        rows.append(
            {
                "date": date,
                "strategy_label": str(label or "Unknown"),
                "lineups": int(len(group)),
                "avg_roi": float(roi.mean()) if roi.notna().any() else np.nan,
                "cash_rate": float((payout > 0).mean()) if len(group) else np.nan,
                "total_payout": float(payout.sum()),
                "avg_score": float(score.mean()) if score.notna().any() else np.nan,
                "best_score": float(score.max()) if score.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _strategy_label(raw: Any) -> str:
    payload = _parse_json(raw)
    if not payload:
        return "Unknown"
    for key in ("active_strategy_build", "slate_run_preset"):
        value = payload.get(key)
        if value:
            return str(value)
    simulation = payload.get("simulation") or {}
    if isinstance(simulation, dict):
        metric = simulation.get("selection_metric")
        leverage = _num(simulation.get("selection_leverage_weight"))
        ownership = _num(simulation.get("selection_ownership_weight"))
        duplication = _num(simulation.get("selection_duplication_weight"))
        if duplication >= 0.35:
            return "Chalk fade unique"
        if leverage >= 0.30:
            return "Aggressive leverage"
        if metric:
            return f"{metric} build"
    optimizer = payload.get("optimizer") or {}
    if isinstance(optimizer, dict):
        leverage_weight = _num(optimizer.get("leverage_weight"))
        if leverage_weight >= 22:
            return "Chalk fade build"
        if leverage_weight >= 16:
            return "Balanced leverage build"
    return "Legacy strategy"


def _parse_json(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _num(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _strategy_lessons(summary: pd.DataFrame) -> list[str]:
    if summary.empty:
        return []
    notes: list[str] = []
    leader = summary.iloc[0]
    notes.append(
        f"Best recorded strategy in this range: {leader['strategy_label']} "
        f"({leader['avg_roi']:.2f} avg ROI across {int(leader['slates'])} slates)."
    )
    negative = summary[pd.to_numeric(summary["avg_roi"], errors="coerce") < 0]
    if not negative.empty:
        notes.append("Strategies with negative ROI should be treated as slate-specific, not default presets.")
    return notes


__all__ = ["StrategyBacktestReport", "build_strategy_backtest_report"]
