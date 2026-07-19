"""Fit the structural ownership model against recorded actual ownership.

Reads the per-player training snapshots that Step 6 writes to
``ownership_history`` in slates.db (features + actual ownership from the
FanDuel contest export), fits pool-specific feature weights with a ridge
regression on log-ownership (slate fixed effects), and writes the learned
parameters to ``config/ownership_model.json`` — but only when the refit
model beats the currently active parameters on historical MAE.

The structural model is scale-invariant in temperature (softmax(w·z / T)
== softmax((w/T)·z)), so fitted weights absorb temperature and it is
stored as 1.0.

Usage:
    python scripts/fit_ownership_model.py [--min-slates 10] [--force] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slate_optimizer.data.storage import SlateDatabase
from slate_optimizer.projection.ownership_model import (
    DEFAULT_HITTER_WEIGHTS,
    DEFAULT_PARAMS_PATH,
    DEFAULT_PITCHER_WEIGHTS,
    StructuralOwnershipParams,
    estimate_structural_ownership,
    load_ownership_params,
    save_ownership_params,
)

DEFAULT_DB_PATH = ROOT / "data" / "slates.db"
MIN_ACTUAL_OWN = 0.002  # log floor for players the field barely rostered


def load_training_frame(db_path: Path) -> pd.DataFrame:
    db = SlateDatabase(db_path)
    history = db.fetch_ownership_history()
    db.close()
    if history.empty:
        return pd.DataFrame()
    rows = []
    for _, row in history.iterrows():
        try:
            features = json.loads(row["features"]) if row.get("features") else {}
        except Exception:
            continue
        record = {
            "date": row["date"],
            "fd_player_id": str(row["fd_player_id"]),
            "full_name": row.get("player_name") or "",
            "player_type": row.get("player_type") or "batter",
            "predicted_own": row.get("predicted_own"),
            "actual_own": row.get("actual_own"),
        }
        record.update({k: v for k, v in features.items() if isinstance(v, (int, float))})
        rows.append(record)
    return pd.DataFrame(rows)


def _zscore_within(df: pd.DataFrame, columns: list[str], group_col: str) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        values = pd.to_numeric(out.get(col), errors="coerce").fillna(0.0)
        grouped = values.groupby(out[group_col])
        mean = grouped.transform("mean")
        std = grouped.transform(lambda s: s.std(ddof=0)).replace(0, np.nan)
        out[col] = ((values - mean) / std).fillna(0.0)
    return out


def fit_pool_weights(pool_df: pd.DataFrame, feature_names: list[str], ridge: float = 1.0) -> dict[str, float] | None:
    """Ridge regression of log(actual ownership) on z-features with slate fixed effects."""
    data = pool_df.dropna(subset=["actual_own"]).copy()
    data = data[pd.to_numeric(data["actual_own"], errors="coerce").notna()]
    if "viable" in data.columns:
        # Fit on the players the softmax actually allocates; bench slivers are fixed.
        data = data[pd.to_numeric(data["viable"], errors="coerce").fillna(0.0) > 0]
    if len(data) < 60:
        return None
    data["_y"] = np.log(pd.to_numeric(data["actual_own"], errors="coerce").clip(lower=MIN_ACTUAL_OWN))
    data = _zscore_within(data, feature_names, "date")
    # Within transformation: demean y and X per slate to absorb slate-level effects.
    data["_y"] = data["_y"] - data.groupby("date")["_y"].transform("mean")
    X = data[feature_names].to_numpy(dtype=float)
    X = X - data.groupby("date")[feature_names].transform("mean").to_numpy(dtype=float)
    y = data["_y"].to_numpy(dtype=float)
    lam = ridge * max(1.0, len(data) / 500.0)
    coef = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    if not np.all(np.isfinite(coef)):
        return None
    return {name: float(value) for name, value in zip(feature_names, coef)}


def evaluate_params(frame: pd.DataFrame, params: StructuralOwnershipParams) -> float:
    """Mean absolute error of the model against recorded actual ownership."""
    errors = []
    for _, slate in frame.groupby("date"):
        features = slate.reset_index(drop=True)
        actual = pd.to_numeric(features["actual_own"], errors="coerce")
        if actual.notna().sum() < 20:
            continue
        predicted = estimate_structural_ownership(
            pd.DataFrame({"fd_player_id": features["fd_player_id"]}),
            pd.DataFrame({"fd_player_id": features["fd_player_id"], "proj_fd_mean": features.get("proj_mean", 0.0)}),
            params=params,
            features=features,
        )
        merged = pd.DataFrame(
            {
                "predicted": features["fd_player_id"].astype(str).map(predicted.to_dict()),
                "actual": actual,
            }
        ).dropna()
        if not merged.empty:
            errors.append(float((merged["predicted"] - merged["actual"]).abs().mean()))
    return float(np.mean(errors)) if errors else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--out", type=Path, default=DEFAULT_PARAMS_PATH)
    ap.add_argument("--min-slates", type=int, default=10)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--force", action="store_true", help="Write params even if MAE did not improve")
    ap.add_argument("--dry-run", action="store_true", help="Fit and report, but never write")
    args = ap.parse_args()

    frame = load_training_frame(args.db)
    if frame.empty:
        print("No ownership history recorded yet. Process slate results in Step 6 first.")
        return

    with_actuals = frame[pd.to_numeric(frame["actual_own"], errors="coerce").notna()]
    slates = sorted(with_actuals["date"].unique())
    print(f"Ownership history: {len(frame)} player-rows, {len(slates)} slates with actuals: {slates}")
    if len(slates) < args.min_slates:
        print(f"Need at least {args.min_slates} slates with actual ownership to fit (have {len(slates)}).")
        print("Keep processing results in Step 6 — every slate makes the fit better.")
        return

    hitter_features = list(DEFAULT_HITTER_WEIGHTS.keys())
    pitcher_features = list(DEFAULT_PITCHER_WEIGHTS.keys())

    hitters = with_actuals[with_actuals["player_type"] != "pitcher"]
    pitchers = with_actuals[with_actuals["player_type"] == "pitcher"]
    hitter_weights = fit_pool_weights(hitters, hitter_features, ridge=args.ridge)
    pitcher_weights = fit_pool_weights(pitchers, pitcher_features, ridge=args.ridge)
    if hitter_weights is None or pitcher_weights is None:
        print("Not enough rows with actual ownership in one of the pools; aborting fit.")
        return

    fitted = StructuralOwnershipParams(
        hitter_weights=hitter_weights,
        pitcher_weights=pitcher_weights,
        hitter_temperature=1.0,
        pitcher_temperature=1.0,
        fitted_at=datetime.now().isoformat(timespec="seconds"),
        slates_used=len(slates),
    )

    current = load_ownership_params(args.out)
    current_mae = evaluate_params(frame, current)
    fitted_mae = evaluate_params(frame, fitted)
    print(f"\nHitter weights:  {json.dumps(hitter_weights, indent=2)}")
    print(f"Pitcher weights: {json.dumps(pitcher_weights, indent=2)}")
    print(f"\nHistorical ownership MAE — current params: {current_mae:.4f}, refit: {fitted_mae:.4f}")

    if args.dry_run:
        print("Dry run: nothing written.")
        return
    if not args.force and not (fitted_mae < current_mae):
        print("Refit did not beat current params; keeping existing configuration (use --force to override).")
        return
    save_ownership_params(fitted, args.out)
    print(f"Wrote learned parameters to {args.out}")


if __name__ == "__main__":
    main()
