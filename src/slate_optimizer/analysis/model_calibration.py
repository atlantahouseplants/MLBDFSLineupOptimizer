"""Cross-slate calibration summaries for the optimizer and simulator."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from slate_optimizer.data.storage import SlateDatabase


@dataclass
class ModelCalibrationReport:
    daily: pd.DataFrame
    summary: pd.DataFrame
    accuracy_metrics: pd.DataFrame
    lessons: list[str]


def build_model_calibration_report(
    db_path: Path | str,
    start_date: str,
    end_date: str,
) -> ModelCalibrationReport:
    """Build a calibration report from recorded slate and post-slate data."""

    db = SlateDatabase(Path(db_path))
    try:
        slate_df = db.fetch_slate_results_range(start_date, end_date)
        rows: list[dict[str, float | str | int]] = []
        accuracy_rows: list[pd.DataFrame] = []
        if slate_df.empty:
            return ModelCalibrationReport(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [])

        for _, slate in slate_df.iterrows():
            date = str(slate.get("date", ""))
            lineups = db.fetch_lineup_results(date)
            actuals = db.fetch_actual_scores(date)
            accuracy = db.fetch_simulation_accuracy(date)
            if not accuracy.empty:
                accuracy_rows.append(accuracy.assign(date=date))
            row = _daily_row(date, slate, lineups, actuals, accuracy)
            rows.append(row)
    finally:
        db.close()

    daily = pd.DataFrame(rows).sort_values("date") if rows else pd.DataFrame()
    accuracy_metrics = pd.concat(accuracy_rows, ignore_index=True) if accuracy_rows else pd.DataFrame()
    summary = _summary_table(daily)
    lessons = _lessons(daily)
    return ModelCalibrationReport(
        daily=daily,
        summary=summary,
        accuracy_metrics=accuracy_metrics,
        lessons=lessons,
    )


def _daily_row(
    date: str,
    slate: pd.Series,
    lineups: pd.DataFrame,
    actuals: pd.DataFrame,
    accuracy: pd.DataFrame,
) -> dict[str, float | str | int]:
    row: dict[str, float | str | int] = {
        "date": date,
        "entries": int(slate.get("num_entries") or 0),
        "entry_fee": float(slate.get("entry_fee") or 0.0),
        "winning_score": _to_float(slate.get("winning_score")),
        "cash_line": _to_float(slate.get("cash_line")),
        "lineups_recorded": int(len(lineups)) if lineups is not None else 0,
    }
    if lineups is not None and not lineups.empty:
        payouts = pd.to_numeric(lineups.get("payout"), errors="coerce").fillna(0.0)
        roi = pd.to_numeric(lineups.get("roi"), errors="coerce")
        scores = pd.to_numeric(lineups.get("total_actual_points"), errors="coerce")
        row.update(
            {
                "avg_roi": float(roi.mean()) if roi.notna().any() else np.nan,
                "cash_rate": float((payouts > 0).mean()),
                "avg_lineup_score": float(scores.mean()) if scores.notna().any() else np.nan,
                "best_lineup_score": float(scores.max()) if scores.notna().any() else np.nan,
            }
        )
    if actuals is not None and not actuals.empty:
        ownership = _ownership_decimal(actuals.get("actual_ownership_pct"))
        actual_scores = pd.to_numeric(actuals.get("actual_fd_points"), errors="coerce")
        row.update(
            {
                "actual_player_count": int(len(actuals)),
                "actual_top5_ownership": float(ownership.sort_values(ascending=False).head(5).sum()) if not ownership.empty else np.nan,
                "actual_chalk_count": int((ownership >= 0.18).sum()) if not ownership.empty else 0,
                "actual_score_std": float(actual_scores.std()) if actual_scores.notna().any() else np.nan,
            }
        )
    if accuracy is not None and not accuracy.empty:
        latest = accuracy.sort_values("created_at").drop_duplicates("metric_name", keep="last")
        for _, metric in latest.iterrows():
            row[str(metric.get("metric_name"))] = _to_float(metric.get("metric_value"))
    if {"field_winning_score_predicted", "field_winning_score_actual"}.issubset(row):
        row["winning_score_error"] = float(row["field_winning_score_predicted"]) - float(row["field_winning_score_actual"])
    if {"dist_calibration_p10", "dist_calibration_p25", "dist_calibration_p50", "dist_calibration_p75", "dist_calibration_p90"}.intersection(row):
        row["distribution_calibration_error"] = _distribution_error(row)
    return row


def _summary_table(daily: pd.DataFrame) -> pd.DataFrame:
    if daily is None or daily.empty:
        return pd.DataFrame()
    numeric_cols = [
        col
        for col in daily.columns
        if col != "date" and pd.api.types.is_numeric_dtype(pd.to_numeric(daily[col], errors="coerce"))
    ]
    rows = []
    for col in numeric_cols:
        values = pd.to_numeric(daily[col], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric": col,
                "slates": int(len(values)),
                "average": float(values.mean()),
                "latest": float(values.iloc[-1]),
                "best": float(values.max()),
                "worst": float(values.min()),
                "trend_per_slate": _slope(values),
            }
        )
    return pd.DataFrame(rows).sort_values("metric")


def _lessons(daily: pd.DataFrame) -> list[str]:
    if daily is None or daily.empty:
        return []
    notes: list[str] = []
    roi = _mean_if_present(daily, "avg_roi")
    if roi is not None and roi < 0:
        notes.append("Portfolio ROI is negative over this range; compare strategy builds before lock and lower exposure to duplicated chalk.")
    brier = _mean_if_present(daily, "brier_score_15")
    if brier is not None and brier > 0.25:
        notes.append("Lineup ceiling probabilities are miscalibrated; reduce volatility or revisit upside/bust inputs for the next slate.")
    dist_error = _mean_if_present(daily, "distribution_calibration_error")
    if dist_error is not None and dist_error > 0.12:
        notes.append("Actual lineup outcomes are landing away from simulated quantiles; the score distribution shape needs attention.")
    win_error = _mean_if_present(daily, "winning_score_error")
    if win_error is not None and abs(win_error) >= 8:
        direction = "high" if win_error > 0 else "low"
        notes.append(f"Winning-score simulations have been about {abs(win_error):.1f} points too {direction} on average.")
    corr_error = _mean_if_present(daily, "teammate_corr_error")
    if corr_error is not None and abs(corr_error) >= 0.08:
        notes.append("Observed team correlation is meaningfully different from the model; tune teammate correlation before the next large-field run.")
    if not notes:
        notes.append("No major calibration drift detected in the recorded range.")
    return notes


def _distribution_error(row: dict[str, float | str | int]) -> float:
    expected = {
        "dist_calibration_p10": 0.10,
        "dist_calibration_p25": 0.25,
        "dist_calibration_p50": 0.50,
        "dist_calibration_p75": 0.75,
        "dist_calibration_p90": 0.90,
    }
    errors = []
    for key, target in expected.items():
        value = row.get(key)
        if value is None:
            continue
        try:
            errors.append(abs(float(value) - target))
        except (TypeError, ValueError):
            continue
    return float(np.mean(errors)) if errors else np.nan


def _ownership_decimal(values) -> pd.Series:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna().astype(float)
    if series.empty:
        return series
    if float(series.max()) > 1.5:
        series = series / 100.0
    return series.clip(lower=0.0, upper=1.0)


def _mean_if_present(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _slope(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=float)
    return float(np.polyfit(x, arr, 1)[0])


def _to_float(value) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


__all__ = ["ModelCalibrationReport", "build_model_calibration_report"]
