#!/usr/bin/env python3
"""Calibrate score_distributions.py against Ballpark Pal DFS Optimizer exports.

Ground truth: the website-only DFS export (Bust / Median / Upside per player),
e.g. data/uploads/Ballpark DFS  Ballpark Pal (88).xlsx (batters) and (89) (pitchers).

Model inputs: the matching BallparkPal_Batters / _Pitchers Excel exports
(official-API-compatible fields: PointsFD, HomeRunProbability, HitProbability,
StolenBaseProbability, Innings, WinPct, Strikeouts, Walks, RunsAllowed).

Stages:
  A. Batters are right-skewed (mean > median) -> lognormal. Per-player truth
     sigma = sqrt(2*ln(mean/median)) is exact under the lognormal assumption.
  B. Choose the global ceiling percentile (Upside) and bust threshold
     (x median) minimizing batter MAE.
  C. Pitchers are near-symmetric (median ~= mean) -> normal. Per-pitcher truth
     sigma = (Upside - mean) / z(ceiling_pct) from stage B.
  D. Grid-search CV proxy factors to match per-player truth CV.
  E. End-to-end MAE report: Median / Upside / Bust, model vs truth.

Usage:
    python scripts/calibrate_boom_bust.py \
        --batters-truth "data/uploads/Ballpark DFS  Ballpark Pal (88).xlsx" \
        --pitchers-truth "data/uploads/Ballpark DFS  Ballpark Pal (89).xlsx" \
        --batters-bpp data/uploads/BallparkPal_Batters-*.xlsx \
        --pitchers-bpp data/uploads/BallparkPal_Pitchers-*.xlsx
"""

from __future__ import annotations

import argparse
import glob
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import lognorm, norm


# ---------------------------------------------------------------------------
# Loading / matching
# ---------------------------------------------------------------------------

def _load_dfs_export(path: str) -> pd.DataFrame:
    """Website DFS export: title in row 1, real headers in row 2."""
    df = pd.read_excel(path, header=1)
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    for col in ("Points", "Bust", "Median", "Upside"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Players", "Points", "Median"])
    df["Bust"] = df["Bust"].fillna(0.0)
    df["Upside"] = df["Upside"].fillna(df["Points"])
    return df[["Tm", "Pos", "Players", "Points", "Bust", "Median", "Upside"]]


def _split_dfs_name(raw: str):
    """'F. Tatis Jr. 1' -> ('F', 'Tatis Jr.') ; 'M. Meyer' -> ('M', 'Meyer')."""
    name = re.sub(r"\s+\d+$", "", str(raw)).strip()
    parts = name.split(". ", 1)
    if len(parts) == 2:
        return parts[0].strip().upper(), parts[1].strip().lower()
    bits = name.split(" ", 1)
    if len(bits) == 2 and len(bits[0]) <= 2:
        return bits[0].replace(".", "").upper(), bits[1].strip().lower()
    return "", name.lower()


def _join_truth(truth: pd.DataFrame, bpp: pd.DataFrame) -> pd.DataFrame:
    t = truth.copy()
    t[["init", "last"]] = t["Players"].apply(lambda s: pd.Series(_split_dfs_name(s)))
    t["Tm"] = t["Tm"].astype(str).str.upper()
    b = bpp.copy()
    b["last"] = b["LastName"].astype(str).str.lower()
    b["init"] = b["FullName"].astype(str).str[0].str.upper()
    b["Team"] = b["Team"].astype(str).str.upper()
    merged = t.merge(
        b, left_on=["Tm", "init", "last"], right_on=["Team", "init", "last"], how="left"
    )
    # Retry unmatched on last name + team only (initials sometimes differ)
    miss = merged["FullName"].isna()
    if miss.any():
        retry = (
            t[miss]
            .drop(columns=["FullName"], errors="ignore")
            .merge(b, left_on=["Tm", "last"], right_on=["Team", "last"], how="left")
        )
        merged.loc[miss, retry.columns] = retry.values
    return merged.dropna(subset=["FullName"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Calibration stages
# ---------------------------------------------------------------------------

def _batter_sigma_truth(mean: pd.Series, median: pd.Series) -> pd.Series:
    ratio = (mean / median.clip(lower=0.5)).clip(lower=1.0 + 1e-6)
    return np.sqrt(2.0 * np.log(ratio)).clip(lower=0.05)


def _best_ceiling_pct(mean, median, upside, dist="lognormal", sigma=None):
    best_p, best_mae = 0.90, float("inf")
    for p in np.arange(0.86, 0.9701, 0.01):
        if dist == "lognormal":
            mu = np.log(median)
            pred = lognorm.ppf(p, s=sigma, scale=np.exp(mu))
        else:
            pred = mean + norm.ppf(p) * sigma
        mae = float(np.mean(np.abs(pred - upside)))
        if mae < best_mae:
            best_mae, best_p = mae, p
    return best_p, best_mae


def _best_bust_threshold(mean, median, bust, dist="lognormal", sigma=None):
    best_t, best_mae = 0.50, float("inf")
    for t in np.arange(0.35, 0.651, 0.025):
        if dist == "lognormal":
            mu = np.log(median)
            pred = lognorm.cdf(t * median, s=sigma, scale=np.exp(mu))
        else:
            pred = norm.cdf((t * median - mean) / sigma.clip(lower=1e-3))
        mae = float(np.mean(np.abs(pred - bust)))
        if mae < best_mae:
            best_mae, best_t = mae, t
    return best_t, best_mae


def _fit_batter_factors(df: pd.DataFrame, cv_truth: pd.Series):
    """Least-squares: cv ~= base + hr_f*HRprob + sb_f*SBprob - hit_f*Hitprob."""
    X = np.column_stack(
        [
            np.ones(len(df)),
            df["HomeRunProbability"].fillna(0).values,
            df["StolenBaseProbability"].fillna(0).values,
            -df["HitProbability"].fillna(0).values,
        ]
    )
    coef, *_ = np.linalg.lstsq(X, cv_truth.values, rcond=None)
    pred = X @ coef
    rmse = float(np.sqrt(np.mean((pred - cv_truth.values) ** 2)))
    return coef, rmse


def _fit_pitcher_factors(df: pd.DataFrame, ratio_truth: pd.Series):
    """Least-squares: sigma/mean ~= base + k_f*(K/IP) + bb_f*(BB/IP) - ip_f*IP + win_f*WinPct."""
    ip = df["Innings"].fillna(5).clip(lower=0.5).values
    X = np.column_stack(
        [
            np.ones(len(df)),
            (df["Strikeouts"].fillna(0).values / ip),
            (df["Walks"].fillna(0).values / ip),
            -df["Innings"].fillna(5).values,
            df["WinPct"].fillna(0.3).values,
        ]
    )
    coef, *_ = np.linalg.lstsq(X, ratio_truth.values, rcond=None)
    pred = X @ coef
    rmse = float(np.sqrt(np.mean((pred - ratio_truth.values) ** 2)))
    return coef, rmse


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batters-truth", required=True)
    ap.add_argument("--pitchers-truth", required=True)
    ap.add_argument("--batters-bpp", required=True)
    ap.add_argument("--pitchers-bpp", required=True)
    args = ap.parse_args()

    bat_bpp_path = sorted(glob.glob(args.batters_bpp))[-1]
    pit_bpp_path = sorted(glob.glob(args.pitchers_bpp))[-1]

    bat = _join_truth(_load_dfs_export(args.batters_truth), pd.read_excel(bat_bpp_path))
    pit = _join_truth(_load_dfs_export(args.pitchers_truth), pd.read_excel(pit_bpp_path))
    print(f"matched: {len(bat)} batters, {len(pit)} pitchers")

    # --- Stage A/B: batters, lognormal ---
    bat_sigma = _batter_sigma_truth(bat["Points"], bat["Median"])
    bat_cv_truth = np.sqrt(np.expm1(bat_sigma**2))
    p_ceil, mae_upside_shape = _best_ceiling_pct(
        bat["Points"], bat["Median"], bat["Upside"], "lognormal", bat_sigma
    )
    t_bust, mae_bust_shape = _best_bust_threshold(
        bat["Points"], bat["Median"], bat["Bust"], "lognormal", bat_sigma
    )
    print(f"[batters] ceiling pct = {p_ceil:.2f} (Upside MAE {mae_upside_shape:.2f} pts)")
    print(f"[batters] bust threshold = {t_bust:.3f} x median (Bust MAE {mae_bust_shape:.3f})")

    bat_coef, bat_rmse = _fit_batter_factors(bat, bat_cv_truth)
    print(
        f"[batters] CV fit: base={bat_coef[0]:.3f} hr={bat_coef[1]:.3f} "
        f"sb={bat_coef[2]:.3f} hit={bat_coef[3]:.3f} (RMSE {bat_rmse:.3f})"
    )

    # --- Stage C/D: pitchers, normal ---
    z = norm.ppf(p_ceil)
    pit_sigma = ((pit["Upside"] - pit["Points"]) / z).clip(lower=1.0)
    pit_ratio = pit_sigma / pit["Points"].clip(lower=1.0)
    pit_coef, pit_rmse = _fit_pitcher_factors(pit, pit_ratio)
    print(
        f"[pitchers] sigma/mean fit: base={pit_coef[0]:.3f} k={pit_coef[1]:.3f} "
        f"bb={pit_coef[2]:.3f} ip={pit_coef[3]:.4f} win={pit_coef[4]:.3f} (RMSE {pit_rmse:.3f})"
    )

    # --- Stage E: end-to-end MAE with calibrated model ---
    def batter_row(r):
        cv = np.clip(
            bat_coef[0]
            + bat_coef[1] * (r.HomeRunProbability or 0)
            + bat_coef[2] * (r.StolenBaseProbability or 0)
            - bat_coef[3] * (r.HitProbability or 0),
            0.10,
            1.20,
        )
        sigma = math.sqrt(math.log(cv**2 + 1))
        mu = math.log(max(r.Points, 0.5)) - sigma**2 / 2
        med = math.exp(mu)
        return pd.Series(
            {
                "median": med,
                "upside": float(lognorm.ppf(p_ceil, s=sigma, scale=med)),
                "bust": float(lognorm.cdf(t_bust * med, s=sigma, scale=med)),
            }
        )

    def pitcher_row(r):
        ip = max(r.Innings or 5.0, 0.5)
        ratio = np.clip(
            pit_coef[0]
            + pit_coef[1] * ((r.Strikeouts or 0) / ip)
            + pit_coef[2] * ((r.Walks or 0) / ip)
            - pit_coef[3] * (r.Innings or 5.0)
            + pit_coef[4] * (r.WinPct or 0.3),
            0.10,
            0.80,
        )
        sig = ratio * max(r.Points, 1.0)
        return pd.Series(
            {
                "median": r.Points,
                "upside": r.Points + z * sig,
                "bust": float(norm.cdf((t_bust * r.Points - r.Points) / max(sig, 1e-3))),
            }
        )

    bat_pred = bat.join(bat.apply(batter_row, axis=1))
    pit_pred = pit.join(pit.apply(pitcher_row, axis=1))

    print("\n=== End-to-end MAE (calibrated model vs BPP truth) ===")
    for label, d in (("batters", bat_pred), ("pitchers", pit_pred)):
        print(
            f"{label:9s} n={len(d):3d}  "
            f"Median MAE {np.mean(np.abs(d.median - d.Median)):5.2f} pts | "
            f"Upside MAE {np.mean(np.abs(d.upside - d.Upside)):5.2f} pts | "
            f"Bust MAE {np.mean(np.abs(d.bust - d.Bust)):5.3f}"
        )
    both = pd.concat([bat_pred, pit_pred])
    print(
        f"{'ALL':9s} n={len(both):3d}  "
        f"Median MAE {np.mean(np.abs(both.median - both.Median)):5.2f} pts | "
        f"Upside MAE {np.mean(np.abs(both.upside - both.Upside)):5.2f} pts | "
        f"Bust MAE {np.mean(np.abs(both.bust - both.Bust)):5.3f}"
    )

    print("\n=== Suggested constants for score_distributions.py ===")
    print(f"CEILING_PERCENTILE = {p_ceil:.2f}")
    print(f"BUST_THRESHOLD = {t_bust:.3f}")
    print(
        f"BATTER_BASE_CV = {bat_coef[0]:.3f} / BATTER_HR_CV_FACTOR = {bat_coef[1]:.3f} / "
        f"BATTER_SB_CV_FACTOR = {bat_coef[2]:.3f} / BATTER_HIT_CV_FACTOR = {bat_coef[3]:.3f}"
    )
    print(
        f"PITCHER_BASE_CV = {pit_coef[0]:.3f} / PITCHER_K_CV_FACTOR = {pit_coef[1]:.3f} / "
        f"PITCHER_BB_CV_FACTOR = {pit_coef[2]:.3f} / PITCHER_IP_CV_FACTOR = {abs(pit_coef[3]):.4f} / "
        f"PITCHER_WIN_CV_FACTOR = {pit_coef[4]:.3f}"
    )
    print("NOTE: pitchers should use a normal (not lognormal) path: median ~= mean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
