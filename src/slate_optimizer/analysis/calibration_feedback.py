"""Turn recorded calibration drift into engine setting adjustments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .model_calibration import build_model_calibration_report


@dataclass
class CalibrationFeedback:
    adjustments: pd.DataFrame
    adjusted_state: dict[str, Any]
    notes: list[str]


def build_calibration_feedback(
    db_path: Path | str,
    start_date: str,
    end_date: str,
    current_state: dict[str, Any],
    strength: float = 1.0,
) -> CalibrationFeedback:
    report = build_model_calibration_report(db_path, start_date, end_date)
    adjusted = dict(current_state or {})
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    if report.daily.empty:
        return CalibrationFeedback(pd.DataFrame(), adjusted, ["No recorded calibration data in the selected window."])

    strength = float(np.clip(strength, 0.0, 1.0))
    daily = report.daily
    win_error = _mean(daily, "winning_score_error")
    if win_error is not None and abs(win_error) >= 6:
        current = float(adjusted.get("volatility_scale", 1.0) or 1.0)
        delta = (-0.06 if win_error > 0 else 0.06) * strength
        new_value = float(np.clip(current + delta, 0.65, 1.60))
        _record(rows, "volatility_scale", current, new_value, f"Winning-score sim error {win_error:+.1f}.")
        adjusted["volatility_scale"] = new_value

    p90 = _mean(daily, "dist_calibration_p90")
    if p90 is not None and (p90 > 0.94 or p90 < 0.86):
        current = float(adjusted.get("volatility_scale", 1.0) or 1.0)
        delta = (-0.04 if p90 > 0.94 else 0.04) * strength
        new_value = float(np.clip(current + delta, 0.65, 1.60))
        _record(rows, "volatility_scale", current, new_value, f"P90 calibration landed at {p90:.1%}.")
        adjusted["volatility_scale"] = new_value

    corr_error = _mean(daily, "teammate_corr_error")
    if corr_error is not None and abs(corr_error) >= 0.06:
        current = float(adjusted.get("teammate_corr", 0.25) or 0.25)
        new_value = float(np.clip(current + corr_error * 0.40 * strength, 0.05, 0.50))
        _record(rows, "teammate_corr", current, new_value, f"Observed teammate correlation drift {corr_error:+.2f}.")
        adjusted["teammate_corr"] = new_value

    brier = _mean(daily, "brier_score_15")
    if brier is not None and brier >= 0.24:
        current = float(adjusted.get("selection_scenario_weight", 0.10) or 0.10)
        new_value = float(np.clip(current + 0.05 * strength, 0.0, 0.50))
        _record(rows, "selection_scenario_weight", current, new_value, f"Lineup ceiling Brier is elevated at {brier:.3f}.")
        adjusted["selection_scenario_weight"] = new_value

    dup = _mean(daily, "avg_duplication_score_predicted")
    if dup is not None and dup >= 0.42:
        current = float(adjusted.get("selection_duplication_weight", 0.15) or 0.15)
        new_value = float(np.clip(current + 0.06 * strength, 0.0, 0.60))
        _record(rows, "selection_duplication_weight", current, new_value, f"Predicted duplication has averaged {dup:.2f}.")
        adjusted["selection_duplication_weight"] = new_value

    if rows:
        notes.append("Calibration feedback found actionable drift and prepared adjusted Step 4 settings.")
    else:
        notes.append("No automatic calibration changes recommended from the selected window.")
    return CalibrationFeedback(pd.DataFrame(rows), adjusted, notes)


def _record(rows: list[dict[str, Any]], setting: str, old, new, reason: str) -> None:
    if abs(float(old) - float(new)) < 1e-9:
        return
    rows.append({"setting": setting, "current": old, "recommended": new, "reason": reason})


def _mean(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


__all__ = ["CalibrationFeedback", "build_calibration_feedback"]
