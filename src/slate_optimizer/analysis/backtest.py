"""Backtesting and results analysis utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from slate_optimizer.data.storage import SlateDatabase
from slate_optimizer.simulation import fit_player_distributions

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "slates.db"


def _load_optimizer_dataset(date: str) -> pd.DataFrame:
    pattern = f"{date}_optimizer_dataset.csv"
    path = next((p for p in DATA_DIR.glob(f"*{pattern}")), None)
    if path is None:
        raise FileNotFoundError(f"Optimizer dataset for {date} not found in {DATA_DIR}")
    return pd.read_csv(path)


def _load_actual_data(db: SlateDatabase, date: str) -> pd.DataFrame:
    actual = db.fetch_actual_scores(date)
    if actual.empty:
        raise ValueError(f"No actual scores stored for {date}")
    return actual


def calculate_ownership_accuracy(date: str, db_path: Path = DEFAULT_DB) -> Dict[str, float]:
    db = SlateDatabase(db_path)
    projections = _load_optimizer_dataset(date)
    actual = _load_actual_data(db, date)
    merged = projections.merge(
        actual[["fd_player_id", "actual_ownership_pct"]],
        on="fd_player_id",
        how="inner",
    )
    merged["abs_diff"] = (
        merged["proj_fd_ownership"].astype(float) - merged["actual_ownership_pct"].astype(float)
    ).abs()
    mae = float(merged["abs_diff"].mean()) if not merged.empty else float("nan")
    merged["projected_bucket"] = (merged["proj_fd_ownership"].astype(float) // 5).clip(lower=0)
    merged["actual_bucket"] = (merged["actual_ownership_pct"].astype(float) // 5).clip(lower=0)
    db.close()
    bucketed = merged.groupby("projected_bucket")["abs_diff"].mean().rename("mae") if not merged.empty else pd.Series()
    bucket_dict = {int(bucket * 5): float(value) for bucket, value in bucketed.items()}
    return {"mae": mae, "count": int(len(merged)), "bucket_mae": bucket_dict}


def calculate_projection_accuracy(date: str, db_path: Path = DEFAULT_DB) -> Dict[str, float]:
    db = SlateDatabase(db_path)
    projections = _load_optimizer_dataset(date)
    actual = _load_actual_data(db, date)
    merged = projections.merge(
        actual[["fd_player_id", "actual_fd_points"]],
        on="fd_player_id",
        how="inner",
    )
    merged["abs_diff"] = (
        merged["proj_fd_mean"].astype(float) - merged["actual_fd_points"].astype(float)
    ).abs()
    mae = float(merged["abs_diff"].mean()) if not merged.empty else float("nan")
    db.close()
    return {"mae": mae, "count": int(len(merged))}



def calculate_brier_score(
    date: str,
    db_path: Path = DEFAULT_DB,
    threshold: float = 15.0,
    volatility_scale: float = 1.0,
) -> Dict[str, float]:
    """Compute Brier score for the P(score > threshold)."""

    db = SlateDatabase(db_path)
    projections = _load_optimizer_dataset(date)
    actual = _load_actual_data(db, date)
    db.close()

    distributions = fit_player_distributions(projections, volatility_scale)
    actual = actual[["fd_player_id", "actual_fd_points"]].dropna()

    preds = []
    outcomes = []
    threshold_val = float(threshold)
    for _, row in actual.iterrows():
        player_id = str(row.get("fd_player_id"))
        dist = distributions.get(player_id)
        if dist is None:
            continue
        actual_points = float(row.get("actual_fd_points", 0.0) or 0.0)
        event = 1.0 if actual_points >= threshold_val else 0.0
        prob = 1.0 - float(dist.cdf(threshold_val))
        preds.append(prob)
        outcomes.append(event)

    if not preds:
        return {
            "brier_score": float("nan"),
            "num_players": 0,
            "pct_above_threshold": float("nan"),
            "threshold": threshold_val,
        }

    preds_arr = np.asarray(preds, dtype=float)
    outcomes_arr = np.asarray(outcomes, dtype=float)
    brier = float(np.mean((preds_arr - outcomes_arr) ** 2))
    pct_above = float(outcomes_arr.mean())
    return {
        "brier_score": brier,
        "num_players": int(len(preds_arr)),
        "pct_above_threshold": pct_above,
        "threshold": threshold_val,
    }


def calculate_leverage_roi(date: str, db_path: Path = DEFAULT_DB) -> Dict[str, float]:
    db = SlateDatabase(db_path)
    projections = _load_optimizer_dataset(date)
    actual = _load_actual_data(db, date)
    merged = projections.merge(actual[["fd_player_id", "actual_fd_points", "actual_ownership_pct"]], on="fd_player_id", how="inner")
    merged = merged.dropna(subset=["player_leverage_score"])
    positive = merged[merged["player_leverage_score"] > 0]
    negative = merged[merged["player_leverage_score"] <= 0]
    def _avg_outperformance(df: pd.DataFrame) -> float:
        if df.empty:
            return float("nan")
        projected = df["proj_fd_mean"].astype(float)
        actual_pts = df["actual_fd_points"].astype(float)
        return float((actual_pts - projected).mean())
    result = {
        "positive_leverage_outperformance": _avg_outperformance(positive),
        "negative_leverage_outperformance": _avg_outperformance(negative),
        "sample_size": len(merged),
    }
    db.close()
    return result


def calculate_lineup_performance(date: str, db_path: Path = DEFAULT_DB) -> Dict[str, float]:
    db = SlateDatabase(db_path)
    lineups = db.fetch_lineup_results(date)
    if lineups.empty:
        db.close()
        return {"cash_rate": float("nan"), "avg_roi": float("nan"), "max_score": float("nan"), "min_score": float("nan")}
    payout_mask = lineups.get("payout", 0).astype(float) > 0
    cash_rate = float(payout_mask.mean())
    avg_roi = float(lineups.get("roi", 0).astype(float).mean())
    max_score = float(lineups.get("total_actual_points", 0).astype(float).max())
    min_score = float(lineups.get("total_actual_points", 0).astype(float).min())
    db.close()
    return {
        "cash_rate": cash_rate,
        "avg_roi": avg_roi,
        "max_score": max_score,
        "min_score": min_score,
    }


def calculate_chalk_analysis(date: str, chalk_threshold: float = 20.0, db_path: Path = DEFAULT_DB) -> Dict[str, float]:
    db = SlateDatabase(db_path)
    projections = _load_optimizer_dataset(date)
    actual = _load_actual_data(db, date)
    merged = projections.merge(actual[["fd_player_id", "actual_fd_points"]], on="fd_player_id", how="inner")
    merged["is_chalk"] = merged["proj_fd_ownership"].astype(float) >= chalk_threshold
    chalk_pts = merged[merged["is_chalk"]]["actual_fd_points"].astype(float)
    contrarian_pts = merged[~merged["is_chalk"]]["actual_fd_points"].astype(float)
    result = {
        "chalk_avg": float(chalk_pts.mean()) if not chalk_pts.empty else float("nan"),
        "contrarian_avg": float(contrarian_pts.mean()) if not contrarian_pts.empty else float("nan"),
    }
    db.close()
    return result


def get_cumulative_metrics(start_date: str, end_date: str, db_path: Path = DEFAULT_DB) -> Dict[str, float]:
    db = SlateDatabase(db_path)
    slate_results = db.fetch_slate_results_range(start_date, end_date)
    db.close()
    if slate_results.empty:
        return {"total_slates": 0, "overall_roi": float("nan"), "overall_cash_rate": float("nan")}

    roi_values = []
    cash_values = []
    for date in slate_results["date"]:
        perf = calculate_lineup_performance(date, db_path)
        roi = perf.get("avg_roi")
        cash = perf.get("cash_rate")
        if roi is not None and not np.isnan(roi):
            roi_values.append(roi)
        if cash is not None and not np.isnan(cash):
            cash_values.append(cash)
    metrics = {
        "total_slates": int(len(slate_results)),
        "overall_roi": float(np.mean(roi_values)) if roi_values else float("nan"),
        "overall_cash_rate": float(np.mean(cash_values)) if cash_values else float("nan"),
    }
    return metrics


__all__ = [
    "calculate_ownership_accuracy",
    "calculate_projection_accuracy",
    "calculate_brier_score",
    "calculate_leverage_roi",
    "calculate_lineup_performance",
    "calculate_chalk_analysis",
    "get_cumulative_metrics",
    "calculate_variance_metrics",
]


def calculate_variance_metrics(date: str) -> Dict[str, float]:
    dataset = _load_optimizer_dataset(date)
    if dataset.empty:
        return {"avg_volatility": float("nan"), "high_leverage_count": 0, "high_volatility_leverage": float("nan")}

    ceiling = pd.to_numeric(dataset.get("proj_fd_ceiling"), errors="coerce")
    floor = pd.to_numeric(dataset.get("proj_fd_floor"), errors="coerce")
    volatility = (ceiling - floor).clip(lower=0)
    leverage = pd.to_numeric(dataset.get("player_leverage_score"), errors="coerce")
    ownership = pd.to_numeric(dataset.get("proj_fd_ownership"), errors="coerce")

    avg_vol = float(volatility.mean()) if volatility.notna().any() else float("nan")
    leverage_mask = (leverage > 0.05) & (ownership <= 0.15)
    high_leverage_count = int(leverage_mask.sum()) if leverage_mask.notna().any() else 0
    high_volatility = float(volatility[leverage_mask].mean()) if leverage_mask.any() else float("nan")
    return {
        "avg_volatility": avg_vol,
        "high_leverage_count": high_leverage_count,
        "high_volatility_leverage": high_volatility,
    }
