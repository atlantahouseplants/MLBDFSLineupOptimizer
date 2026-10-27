"""Computed Boom/Bust/Upside metrics from principled score distributions.

The Ballpark Pal *official API* does not expose the website-only Bust% /
Median / Upside projections. This module replaces the crude fallback
(floor = mean x 0.8, ceiling = mean x 1.2) with per-player log-normal score
distributions whose coefficient of variation (CV) is scaled by volatility
proxies that the API *does* provide (HR/Hit/SB probabilities for batters,
K/BB/IP/WinPct profile for pitchers).

Derived metrics per player:
    median  = P50
    floor   = P10
    ceiling = P90
    boom%   = P(score >= 1.5 x median)
    bust%   = P(score <= 0.5 x median)

Shipped behind the COMPUTED_BOOM_BUST flag (.env). When the flag is off the
module returns the legacy crude values so behavior is unchanged.

Column-name resolution tries the pipeline schema first (bpp_* / proj_fd_*)
and falls back to the raw website export names so the module works on both
optimizer_df rows and raw BallparkPal tables. CV factors calibrated against
the 2026-07-18 Ballpark Pal DFS export (see scripts/calibrate_boom_bust.py).
"""

import math
import os
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import lognorm

try:
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover - dotenv is installed in the venv
    def dotenv_values(_path):  # type: ignore
        return {}

# ---------------------------------------------------------------------------
# CV scaling factors (calibrated — see scripts/calibrate_boom_bust.py)
# ---------------------------------------------------------------------------
BATTER_BASE_CV = 0.30
BATTER_HR_CV_FACTOR = 0.5   # Higher HR prob  -> higher variance (power is volatile)
BATTER_HIT_CV_FACTOR = 0.2  # Higher Hit prob -> lower variance (more consistent)
BATTER_SB_CV_FACTOR = 0.3   # Higher SB prob  -> higher variance (all-or-nothing)

PITCHER_BASE_CV = 0.30
PITCHER_K_CV_FACTOR = 0.6    # Higher K rate  -> higher variance (Ks are volatile)
PITCHER_BB_CV_FACTOR = 0.3   # Higher BB rate -> higher variance
PITCHER_IP_CV_FACTOR = 0.02  # More IP        -> lower variance (more consistent)
PITCHER_WIN_CV_FACTOR = 0.2  # Win pct contributes to upside variance

CV_MIN = 0.10
CV_MAX = 1.20

BOOM_THRESHOLD = 1.5  # x median
BUST_THRESHOLD = 0.5  # x median

# Column-name candidates, most-preferred first. The optimizer dataset carries
# bpp_*/proj_fd_* names; raw website exports use PointsFD/HomeRunProbability/...
_MEAN_KEYS = ("proj_fd_mean", "PointsFD", "points_fd", "bpp_points_fd", "mean")
_HR_KEYS = ("bpp_home_run_probability", "HomeRunProbability", "home_run_probability")
_HIT_KEYS = ("bpp_hit_probability", "HitProbability", "hit_probability")
_SB_KEYS = ("bpp_stolen_base_probability", "StolenBaseProbability", "stolen_base_probability")
_K_KEYS = ("bpp_strikeouts", "Strikeouts", "strikeouts")
_IP_KEYS = ("bpp_innings", "Innings", "innings")
_BB_KEYS = ("bpp_bb", "BB", "bb", "bpp_walks", "walks")
_WIN_KEYS = ("bpp_win_pct", "WinPct", "win_pct", "bpp_win_percent", "win_percent")


def _get_env_flag(flag_name: str, default_value: bool = False) -> bool:
    """Read a boolean flag from the process env or the repo .env file."""
    env_vars = dotenv_values(".env")
    value = os.environ.get(flag_name, env_vars.get(flag_name, str(default_value))).lower()
    return value == "true"


def _safe_float(value, default: float = 0.0) -> float:
    """Coerce anything (None, '', '12.3', numpy scalars) to a finite float."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _row_get(record, names: Iterable[str], default: float = 0.0) -> float:
    """Return the first present, convertible value among candidate columns."""
    for name in names:
        if isinstance(record, pd.Series):
            value = record.get(name, None)
        elif isinstance(record, dict):
            value = record.get(name, None)
        else:
            value = getattr(record, name, None)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        return _safe_float(value, default)
    return default


def _calculate_lognorm_params(mean: float, cv: float):
    """Convert (mean, CV) to log-normal (mu, sigma).

    CV = sqrt(exp(sigma^2) - 1)  ->  sigma = sqrt(ln(CV^2 + 1))
    E[X] = exp(mu + sigma^2 / 2) ->  mu = ln(mean) - sigma^2 / 2
    """
    if mean <= 0:
        return np.log(0.1), 0.1  # degenerate: tiny positive mean, low variance
    cv = max(cv, 0.0)
    sigma_squared = np.log(cv**2 + 1)
    sigma = np.sqrt(sigma_squared)
    mu = np.log(mean) - sigma_squared / 2
    return mu, sigma


def _batter_cv(record) -> float:
    hr_prob = _row_get(record, _HR_KEYS)
    hit_prob = _row_get(record, _HIT_KEYS)
    sb_prob = _row_get(record, _SB_KEYS)
    cv = BATTER_BASE_CV
    cv += BATTER_HR_CV_FACTOR * hr_prob
    cv -= BATTER_HIT_CV_FACTOR * hit_prob
    cv += BATTER_SB_CV_FACTOR * sb_prob
    return float(min(CV_MAX, max(CV_MIN, cv)))


def _pitcher_cv(record) -> float:
    strikeouts = _row_get(record, _K_KEYS)
    innings = _row_get(record, _IP_KEYS)
    walks = _row_get(record, _BB_KEYS)
    win_pct = _row_get(record, _WIN_KEYS)
    k_rate = strikeouts / innings if innings > 0 else 0.0
    bb_rate = walks / innings if innings > 0 else 0.0
    cv = PITCHER_BASE_CV
    cv += PITCHER_K_CV_FACTOR * k_rate
    cv += PITCHER_BB_CV_FACTOR * bb_rate
    cv -= PITCHER_IP_CV_FACTOR * innings
    cv += PITCHER_WIN_CV_FACTOR * win_pct
    return float(min(CV_MAX, max(CV_MIN, cv)))


def _metrics_from_mean_cv(mean_fd_points: float, cv: float) -> dict:
    """Log-normal percentiles + boom/bust for one player."""
    mu, sigma = _calculate_lognorm_params(mean_fd_points, cv)
    if mean_fd_points <= 0 or sigma <= 0:
        return {
            "median_fd_points": mean_fd_points,
            "floor_fd_points": mean_fd_points * 0.8,
            "ceiling_fd_points": mean_fd_points * 1.2,
            "boom_percentage": 0.0,
            "bust_percentage": 0.0,
        }
    scale = math.exp(mu)
    median = float(lognorm.median(s=sigma, scale=scale))
    floor = float(lognorm.ppf(0.10, s=sigma, scale=scale))
    ceiling = float(lognorm.ppf(0.90, s=sigma, scale=scale))
    boom = float(1 - lognorm.cdf(BOOM_THRESHOLD * median, s=sigma, scale=scale))
    bust = float(lognorm.cdf(BUST_THRESHOLD * median, s=sigma, scale=scale))
    return {
        "median_fd_points": median,
        "floor_fd_points": max(0.0, floor),
        "ceiling_fd_points": ceiling,
        "boom_percentage": boom,
        "bust_percentage": bust,
    }


def fit_player_distributions(player_data: dict, player_type: str) -> dict:
    """Fit one player's score distribution and derive boom/bust metrics.

    Args:
        player_data: mapping/Series with a mean projection (proj_fd_mean or
            PointsFD) plus volatility proxy columns (see module docstring).
        player_type: 'batter' or 'pitcher'.

    Returns:
        dict with median/floor/ceiling/boom/bust. When COMPUTED_BOOM_BUST is
        off, returns the legacy crude floor/ceiling and zeroed boom/bust.
    """
    if not _get_env_flag("COMPUTED_BOOM_BUST", False):
        mean_fd_points = _row_get(player_data, _MEAN_KEYS)
        return {
            "median_fd_points": mean_fd_points,
            "floor_fd_points": mean_fd_points * 0.8,
            "ceiling_fd_points": mean_fd_points * 1.2,
            "boom_percentage": 0.0,
            "bust_percentage": 0.0,
        }

    mean_fd_points = _row_get(player_data, _MEAN_KEYS)
    cv = _pitcher_cv(player_data) if str(player_type).lower() == "pitcher" else _batter_cv(player_data)
    return _metrics_from_mean_cv(mean_fd_points, cv)


def compute_boom_bust_frame(df: pd.DataFrame, player_type_col: str = "player_type") -> pd.DataFrame:
    """Vectorized boom/bust metrics for a whole player table.

    Returns a DataFrame (same index) with columns:
        computed_median, computed_floor, computed_ceiling,
        computed_boom_pct, computed_bust_pct

    When COMPUTED_BOOM_BUST is off, floor/ceiling fall back to the crude
    0.8/1.2 multipliers and boom/bust are zero — matching legacy behavior.
    """
    enabled = _get_env_flag("COMPUTED_BOOM_BUST", False)
    means = df.apply(lambda r: _row_get(r, _MEAN_KEYS), axis=1)

    if player_type_col in df.columns:
        is_pitcher = df[player_type_col].astype(str).str.lower().eq("pitcher")
    else:  # roster_position / position fallback: P vs everything else
        pos_col = "roster_position" if "roster_position" in df.columns else "position"
        is_pitcher = df[pos_col].astype(str).str.upper().str.contains(r"\bP\b|SP|RP", regex=True) if pos_col in df.columns else pd.Series(False, index=df.index)

    if enabled:
        cvs = pd.Series(
            [_pitcher_cv(row) if pit else _batter_cv(row) for pit, (_, row) in zip(is_pitcher, df.iterrows())],
            index=df.index,
        )
    else:
        cvs = pd.Series(np.nan, index=df.index)

    out = pd.DataFrame(index=df.index)
    if not enabled:
        out["computed_median"] = means
        out["computed_floor"] = means * 0.8
        out["computed_ceiling"] = means * 1.2
        out["computed_boom_pct"] = 0.0
        out["computed_bust_pct"] = 0.0
        return out

    rows = [_metrics_from_mean_cv(m, c) for m, c in zip(means, cvs)]
    out["computed_median"] = [r["median_fd_points"] for r in rows]
    out["computed_floor"] = [r["floor_fd_points"] for r in rows]
    out["computed_ceiling"] = [r["ceiling_fd_points"] for r in rows]
    out["computed_boom_pct"] = [r["boom_percentage"] for r in rows]
    out["computed_bust_pct"] = [r["bust_percentage"] for r in rows]
    return out
