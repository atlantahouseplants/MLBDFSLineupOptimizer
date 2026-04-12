"""Enhanced ownership projection model using BallparkPal simulation data.

Uses richer signals than the legacy model:
  Batters:  value_score, implied_total, upside_ratio, batting_position,
            hit_probability, hr_probability, bust_pct, park_hr_factor,
            salary_tier (the mid-range $4-5K trap)
  Pitchers: win_pct, projected_points, strikeouts, quality_start_pct, salary

Also includes a calibration utility to compare predictions vs. actual
post-contest ownership and tune model weights over time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Ownership range targets (FD main-slate GPP) ───────────────────────────────
BATTER_MIN_OWN  = 0.03   # floor: 3%
BATTER_MAX_OWN  = 0.45   # cap:  45%
PITCHER_MIN_OWN = 0.05   # floor: 5%
PITCHER_MAX_OWN = 0.55   # cap:  55% (aces get stacked hard)

# Batting order position ownership multiplier (leadoff premium is real)
_ORDER_OWN_MULT = {
    1: 1.12, 2: 1.05, 3: 1.08, 4: 1.04, 5: 1.00,
    6: 0.96, 7: 0.93, 8: 0.90, 9: 0.88,
}


def _pct_rank(series: pd.Series) -> pd.Series:
    """Percentile rank 0-1, NaN → 0.5."""
    return series.rank(pct=True, method="average").fillna(0.5)


def _sigmoid(x: pd.Series, k: float = 8.0, x0: float = 0.5) -> pd.Series:
    """Logistic sigmoid centred at x0."""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def _scale(series: pd.Series, lo: float, hi: float) -> pd.Series:
    """Min-max scale to [lo, hi]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series((lo + hi) / 2.0, index=series.index)
    return lo + (hi - lo) * (series - mn) / (mx - mn)


def estimate_batter_ownership(batters: pd.DataFrame) -> pd.Series:
    """Return ownership fractions (0-1) for each batter row.

    Required columns (all from BPP Batters export + Vegas merge):
      PointsFD, salary, BattingPosition, HomeRunProbability,
      HitProbability, vegas_team_total (optional but important)

    Optional BPP DFS columns (merge from dfs_projections if available):
      bust_pct, upside, dfs_avg
    """
    df = batters.copy().reset_index(drop=True)

    # ── 1. Value score (pts per $1000 salary) — strongest driver ─────────────
    pts = pd.to_numeric(df.get("PointsFD", df.get("proj_fd_mean", 0)), errors="coerce").fillna(0)
    sal = pd.to_numeric(df.get("salary", 0), errors="coerce").replace(0, np.nan)
    value = (pts / sal * 1000).fillna(0)
    value_pct = _pct_rank(value)                    # 0-1

    # ── 2. Vegas implied team total ────────────────────────────────────────────
    vegas = pd.to_numeric(df.get("vegas_team_total", np.nan), errors="coerce")
    if vegas.notna().any():
        vegas_pct = _pct_rank(vegas)
    else:
        vegas_pct = pd.Series(0.5, index=df.index)

    # ── 3. Upside ratio (upside / avg) — boom potential ───────────────────────
    avg = pd.to_numeric(df.get("dfs_avg", df.get("PointsFD", np.nan)), errors="coerce")
    ups = pd.to_numeric(df.get("upside", np.nan), errors="coerce")
    if ups.notna().any() and avg.notna().any():
        upside_ratio = (ups / avg.replace(0, np.nan)).fillna(1.0).clip(1.0, 3.0)
        upside_pct = _pct_rank(upside_ratio)
    else:
        upside_pct = pd.Series(0.5, index=df.index)

    # ── 4. Salary tier — mid-range $4-5K is most overowned tier ──────────────
    sal_num = sal.fillna(sal.median())
    # Bell-curve penalty for the $4k-5k sweet spot (overowned by field)
    # Players at $3k (cheapo) and $6k+ (elite) are actually *less* predictable
    sal_pct = _pct_rank(sal_num)

    # ── 5. Batting order position ─────────────────────────────────────────────
    order = pd.to_numeric(df.get("BattingPosition", df.get("batting_order_position", np.nan)), errors="coerce")
    order_mult = order.map(_ORDER_OWN_MULT).fillna(1.0)

    # ── 6. Hit/HR probability ─────────────────────────────────────────────────
    hit_prob = pd.to_numeric(df.get("HitProbability", np.nan), errors="coerce").fillna(0.5)
    hr_prob  = pd.to_numeric(df.get("HomeRunProbability", np.nan), errors="coerce").fillna(0.0)
    hit_signal = _pct_rank(hit_prob * 0.6 + hr_prob * 0.4)

    # ── 7. Bust% (inverted) — safe floors are popular in cash, fade-worthy in GPP
    #    High bust = volatile = contrarian. Low bust = safe = chalk.
    bust = pd.to_numeric(df.get("bust_pct", np.nan), errors="coerce")
    if bust.notna().any():
        safe_signal = _pct_rank(1.0 - bust)   # invert: low bust → high rank
    else:
        safe_signal = pd.Series(0.5, index=df.index)

    # ── Weighted combination ──────────────────────────────────────────────────
    # Weights tuned for large-field GPP ownership behavior
    raw = (
        0.30 * value_pct       +   # value is the #1 driver
        0.22 * vegas_pct       +   # run environment drives stacks
        0.15 * sal_pct         +   # name recognition / salary tier
        0.12 * hit_signal      +   # accessible upside
        0.10 * upside_pct      +   # boom ceiling
        0.08 * safe_signal     +   # chalk safety signal
        0.03 * (order_mult - 1.0)  # leadoff premium (small but real)
    )
    raw = raw.fillna(0.5)

    # Apply order multiplier
    raw = (raw * order_mult).clip(lower=0.0)

    # Scale to realistic ownership range
    ownership = _scale(_sigmoid(raw, k=6.0), BATTER_MIN_OWN, BATTER_MAX_OWN)
    return ownership.clip(BATTER_MIN_OWN, BATTER_MAX_OWN)


def estimate_pitcher_ownership(pitchers: pd.DataFrame) -> pd.Series:
    """Return ownership fractions (0-1) for each pitcher row.

    Required columns (from BPP Pitchers export):
      PointsFD, salary, WinPct, Strikeouts, QualityStart
    """
    df = pitchers.copy().reset_index(drop=True)

    pts = pd.to_numeric(df.get("PointsFD", df.get("proj_fd_mean", 0)), errors="coerce").fillna(0)
    sal = pd.to_numeric(df.get("salary", 0), errors="coerce").fillna(0)
    win = pd.to_numeric(df.get("WinPct", np.nan), errors="coerce").fillna(0.33)
    ks  = pd.to_numeric(df.get("Strikeouts", np.nan), errors="coerce").fillna(0)
    qs  = pd.to_numeric(df.get("QualityStart", np.nan), errors="coerce").fillna(0)

    # ── Signals ───────────────────────────────────────────────────────────────
    pts_pct = _pct_rank(pts)       # projection rank is the #1 pitcher signal
    sal_pct = _pct_rank(sal)       # salary = name recognition
    win_pct = _pct_rank(win)       # win probability
    k_pct   = _pct_rank(ks)        # strikeout upside (aces get used)
    qs_pct  = _pct_rank(qs)        # consistency / floor

    raw = (
        0.35 * pts_pct  +
        0.25 * sal_pct  +
        0.20 * win_pct  +
        0.12 * k_pct    +
        0.08 * qs_pct
    )
    raw = raw.fillna(0.5)

    ownership = _scale(_sigmoid(raw, k=7.0), PITCHER_MIN_OWN, PITCHER_MAX_OWN)
    return ownership.clip(PITCHER_MIN_OWN, PITCHER_MAX_OWN)


def estimate_ownership(players_df: pd.DataFrame) -> pd.Series:
    """Estimate ownership for a mixed batter/pitcher DataFrame.

    Works with both the raw BPP export format and the normalized pipeline format.
    Returns a Series of ownership fractions indexed like players_df.

    Detects player type via 'player_type' or 'position' column.
    """
    df = players_df.copy().reset_index(drop=True)

    if "player_type" in df.columns:
        is_pitcher = df["player_type"].astype(str).str.lower() == "pitcher"
    elif "position" in df.columns:
        is_pitcher = df["position"].astype(str).str.upper().str.contains("^P$", regex=True)
    else:
        is_pitcher = pd.Series(False, index=df.index)

    ownership = pd.Series(np.nan, index=df.index, dtype=float)

    if is_pitcher.any():
        p_own = estimate_pitcher_ownership(df[is_pitcher].reset_index(drop=True))
        ownership.loc[is_pitcher] = p_own.values

    if (~is_pitcher).any():
        b_own = estimate_batter_ownership(df[~is_pitcher].reset_index(drop=True))
        ownership.loc[~is_pitcher] = b_own.values

    return ownership.fillna(0.10)


# ── Calibration utilities ──────────────────────────────────────────────────────

def build_calibration_record(
    players_df: pd.DataFrame,
    predicted_ownership: pd.Series,
    slate_date: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Save predicted ownership for later comparison to actuals.

    Call this before each slate. After contest closes, load the actual
    ownership CSV from FanDuel and call evaluate_calibration().

    Args:
        players_df: Player pool with full_name, team, position, salary, proj_fd_mean
        predicted_ownership: Series of predicted ownership fractions (same index)
        slate_date: YYYY-MM-DD string
        output_path: Optional path to save CSV. Default: data/calibration/YYYY-MM-DD_predicted.csv

    Returns:
        DataFrame with player info + predicted_ownership column.
    """
    df = players_df.copy().reset_index(drop=True)
    df["predicted_ownership"] = predicted_ownership.values
    df["slate_date"] = slate_date

    keep = [c for c in ["full_name", "team", "position", "salary", "proj_fd_mean",
                         "player_type", "predicted_ownership", "slate_date"] if c in df.columns]
    out = df[keep]

    if output_path is None:
        output_path = f"data/calibration/{slate_date}_predicted.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def evaluate_calibration(
    predicted_path: str,
    actual_path: str,
    name_col: str = "full_name",
    actual_own_col: str = "ownership",
) -> pd.DataFrame:
    """Compare predicted vs. actual ownership after contest closes.

    FanDuel shows actual ownership in contest results. Export that CSV and
    pass it here. Outputs a report with MAE, correlation, and worst misses.

    Args:
        predicted_path: Path to CSV saved by build_calibration_record()
        actual_path: Path to FanDuel contest results CSV with ownership column
        name_col: Column with player names in actual CSV
        actual_own_col: Column with actual ownership % in actual CSV

    Returns:
        DataFrame with predicted vs. actual per player + error metrics.
    """
    predicted = pd.read_csv(predicted_path)
    actual = pd.read_csv(actual_path)

    # Normalize names
    actual["_name"] = actual[name_col].astype(str).str.strip().str.lower()
    predicted["_name"] = predicted["full_name"].astype(str).str.strip().str.lower()

    # Clean actual ownership
    actual[actual_own_col] = pd.to_numeric(
        actual[actual_own_col].astype(str).str.replace("%", "", regex=False),
        errors="coerce"
    )
    max_val = actual[actual_own_col].max()
    if max_val > 1.5:
        actual[actual_own_col] = actual[actual_own_col] / 100.0

    merged = predicted.merge(
        actual[["_name", actual_own_col]].rename(columns={actual_own_col: "actual_ownership"}),
        on="_name", how="inner"
    )

    if merged.empty:
        print("No matching players found — check name_col and file format.")
        return pd.DataFrame()

    merged["error"] = merged["predicted_ownership"] - merged["actual_ownership"]
    merged["abs_error"] = merged["error"].abs()

    mae = merged["abs_error"].mean()
    corr = merged["predicted_ownership"].corr(merged["actual_ownership"])
    print(f"\n=== Ownership Calibration Report ===")
    print(f"Players matched: {len(merged)}")
    print(f"MAE:             {mae:.3f} ({mae*100:.1f} pp)")
    print(f"Correlation:     {corr:.3f}")
    print(f"\nBiggest misses (predicted too HIGH):")
    print(merged.nlargest(5, "error")[["full_name", "predicted_ownership", "actual_ownership", "error"]].to_string(index=False))
    print(f"\nBiggest misses (predicted too LOW):")
    print(merged.nsmallest(5, "error")[["full_name", "predicted_ownership", "actual_ownership", "error"]].to_string(index=False))

    return merged.sort_values("abs_error", ascending=False)


__all__ = [
    "estimate_ownership",
    "estimate_batter_ownership",
    "estimate_pitcher_ownership",
    "build_calibration_record",
    "evaluate_calibration",
]
