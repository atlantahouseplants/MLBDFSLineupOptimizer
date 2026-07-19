"""Post-slate calibration checks for boom/bust, median, and upside inputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class BoomBustCalibrationReport:
    summary: dict[str, float]
    by_bust_bucket: pd.DataFrame
    player_results: pd.DataFrame
    lessons: pd.DataFrame


def build_boom_bust_calibration_report(
    optimizer_df: Optional[pd.DataFrame],
    actuals: Optional[pd.DataFrame],
) -> BoomBustCalibrationReport:
    if optimizer_df is None or actuals is None or optimizer_df.empty or actuals.empty:
        return BoomBustCalibrationReport({}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    score_col = "actual_fd_points" if "actual_fd_points" in actuals.columns else "actual_points"
    if score_col not in actuals.columns or "fd_player_id" not in actuals.columns:
        return BoomBustCalibrationReport({}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    keep_cols = [
        col
        for col in [
            "fd_player_id",
            "full_name",
            "team_code",
            "player_type",
            "proj_fd_mean",
            "proj_fd_median",
            "proj_fd_bust_rate",
            "proj_fd_upside",
            "proj_fd_ownership",
        ]
        if col in optimizer_df.columns
    ]
    players = optimizer_df[keep_cols].drop_duplicates("fd_player_id").copy()
    merged = players.merge(actuals[["fd_player_id", score_col]], on="fd_player_id", how="inner")
    if merged.empty:
        return BoomBustCalibrationReport({}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    for col in ["proj_fd_mean", "proj_fd_median", "proj_fd_bust_rate", "proj_fd_upside", "proj_fd_ownership", score_col]:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    if merged["proj_fd_bust_rate"].max() > 1.5:
        merged["proj_fd_bust_rate"] = merged["proj_fd_bust_rate"] / 100.0
    if merged["proj_fd_ownership"].max() > 1.5:
        merged["proj_fd_ownership"] = merged["proj_fd_ownership"] / 100.0

    merged["actual_points"] = merged[score_col]
    merged["actual_minus_mean"] = merged["actual_points"] - merged["proj_fd_mean"]
    merged["actual_under_median"] = merged["actual_points"] < merged["proj_fd_median"]
    merged["actual_upside_hit"] = merged["actual_points"] >= merged["proj_fd_upside"]
    merged["projected_bust_bucket"] = _bucket_bust(merged["proj_fd_bust_rate"])

    by_bucket = (
        merged.groupby("projected_bust_bucket", observed=True)
        .agg(
            players=("fd_player_id", "nunique"),
            avg_projected_bust=("proj_fd_bust_rate", "mean"),
            actual_under_median_rate=("actual_under_median", "mean"),
            upside_hit_rate=("actual_upside_hit", "mean"),
            avg_projection=("proj_fd_mean", "mean"),
            avg_actual=("actual_points", "mean"),
            avg_actual_minus_mean=("actual_minus_mean", "mean"),
        )
        .reset_index()
    )
    by_bucket["bust_calibration_error"] = by_bucket["actual_under_median_rate"] - by_bucket["avg_projected_bust"]

    summary = {
        "matched_players": int(merged["fd_player_id"].nunique()),
        "avg_projected_bust": float(merged["proj_fd_bust_rate"].mean()),
        "actual_under_median_rate": float(merged["actual_under_median"].mean()),
        "bust_calibration_error": float(merged["actual_under_median"].mean() - merged["proj_fd_bust_rate"].mean()),
        "upside_hit_rate": float(merged["actual_upside_hit"].mean()),
        "avg_projection_error": float(merged["actual_minus_mean"].mean()),
        "mean_absolute_error": float(merged["actual_minus_mean"].abs().mean()),
    }
    lessons = _lessons(summary, by_bucket)
    return BoomBustCalibrationReport(summary, by_bucket, merged.sort_values("actual_minus_mean"), lessons)


def _bucket_bust(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    try:
        return pd.qcut(values.rank(method="first"), q=min(5, max(1, values.nunique())), labels=False, duplicates="drop").astype("Int64").astype(str)
    except ValueError:
        return pd.Series(["0"] * len(values), index=values.index)


def _lessons(summary: dict[str, float], by_bucket: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not summary:
        return pd.DataFrame()
    error = float(summary.get("bust_calibration_error", 0.0))
    if error > 0.12:
        rows.append({"area": "Bust calibration", "lesson": "Actual bust rate was materially higher than projected.", "value": error})
    elif error < -0.12:
        rows.append({"area": "Bust calibration", "lesson": "Projected bust rates were too pessimistic versus actual outcomes.", "value": error})
    else:
        rows.append({"area": "Bust calibration", "lesson": "Projected bust rates were reasonably close to actual under-median outcomes.", "value": error})
    if not by_bucket.empty and "bust_calibration_error" in by_bucket.columns:
        worst = by_bucket.iloc[by_bucket["bust_calibration_error"].abs().argmax()]
        rows.append(
            {
                "area": "Bucket check",
                "lesson": f"Largest bust calibration miss was bucket {worst['projected_bust_bucket']}.",
                "value": float(worst["bust_calibration_error"]),
            }
        )
    return pd.DataFrame(rows)


__all__ = ["BoomBustCalibrationReport", "build_boom_bust_calibration_report"]
