"""Baseline projection helpers built on top of BallparkPal sims."""
from __future__ import annotations

import pandas as pd

PROJECTION_COLUMNS = [
    "fd_player_id",
    "full_name",
    "team",
    "opponent",
    "position",
    "player_type",
    "salary",
    "proj_fd_mean",
    "proj_fd_floor",
    "proj_fd_ceiling",
    "proj_fd_ownership",
    "park_adjustment",
    "pitch_adjustment",
    "weather_score",
]


def _team_run_multiplier(df: pd.DataFrame) -> pd.Series:
    if "bpp_runs" not in df.columns:
        return pd.Series(1.0, index=df.index)
    runs = pd.to_numeric(df["bpp_runs"], errors="coerce")
    temp = pd.DataFrame({"team": df["team"], "runs": runs}).dropna()
    if temp.empty:
        return pd.Series(1.0, index=df.index)
    team_means = temp.groupby("team")["runs"].mean()
    baseline = team_means.mean()
    if pd.isna(baseline) or baseline == 0:
        return pd.Series(1.0, index=df.index)
    multipliers = (team_means / baseline).clip(lower=0.75, upper=1.25)
    return df["team"].map(multipliers).fillna(1.0)


def _pitcher_win_multiplier(df: pd.DataFrame) -> pd.Series:
    if "bpp_win_percent" not in df.columns:
        return pd.Series(1.0, index=df.index)
    win_pct = pd.to_numeric(df["bpp_win_percent"], errors="coerce")
    adjustment = ((win_pct - 0.5) * 0.6).clip(-0.25, 0.25)
    return (1 + adjustment).fillna(1.0)


def _park_adjustment(df: pd.DataFrame) -> pd.Series:
    if "bpp_runs" not in df.columns:
        return pd.Series(0.0, index=df.index)
    runs = pd.to_numeric(df["bpp_runs"], errors="coerce")
    delta = runs - runs.mean()
    return delta.clip(-2, 2).fillna(0.0)


def _pitch_adjustment(df: pd.DataFrame) -> pd.Series:
    win_pct = pd.to_numeric(df.get("bpp_win_percent"), errors="coerce").fillna(0.5)
    runs_allowed = pd.to_numeric(df.get("bpp_runs_allowed"), errors="coerce")
    fills = runs_allowed.mean() if runs_allowed.notna().any() else 4.0
    runs_allowed = runs_allowed.fillna(fills)
    return ((0.5 - runs_allowed / 6) * 2.5 + (win_pct - 0.5) * 5).fillna(0.0)


def _weather_score(df: pd.DataFrame) -> pd.Series:
    weather = pd.to_numeric(df.get("bpp_runs_first_inning_pct"), errors="coerce")
    if weather.isna().all():
        weather = pd.to_numeric(df.get("bpp_runs_first5away"), errors="coerce")
    if weather.isna().all():
        return pd.Series(0.0, index=df.index)
    weather = (weather - weather.mean()).fillna(0.0)
    return weather.clip(-0.5, 0.5)



def compute_baseline_projections(players_df: pd.DataFrame) -> pd.DataFrame:
    """Generate naive mean/floor/ceiling projections from available columns."""
    df = players_df.copy()
    fd_points = pd.to_numeric(df.get("bpp_points_fd"), errors="coerce")
    fallback = pd.to_numeric(df.get("fppg"), errors="coerce")
    df["proj_fd_mean"] = fd_points.fillna(fallback).fillna(0.0)

    if "player_type" not in df.columns:
        df["player_type"] = ""

    park_adj = _park_adjustment(df)
    weather_score = _weather_score(df)
    hitters_mask = df["player_type"].str.lower() == "batter"
    df.loc[hitters_mask, "proj_fd_mean"] += park_adj.loc[hitters_mask] * 0.5 + weather_score.loc[hitters_mask]

    pitch_adj = _pitch_adjustment(df)
    pitchers_mask = df["player_type"].str.lower() == "pitcher"
    win_multiplier = _pitcher_win_multiplier(df)
    df.loc[pitchers_mask, "proj_fd_mean"] = (
        df.loc[pitchers_mask, "proj_fd_mean"] * win_multiplier.loc[pitchers_mask]
        + pitch_adj.loc[pitchers_mask]
    )

    df["park_adjustment"] = park_adj.fillna(0.0)
    df["pitch_adjustment"] = pitch_adj.fillna(0.0)
    df["weather_score"] = weather_score.fillna(0.0)

    team_multiplier = _team_run_multiplier(df)
    df.loc[hitters_mask, "proj_fd_mean"] *= team_multiplier.loc[hitters_mask]

    df["proj_fd_floor"] = df["proj_fd_mean"] * 0.8
    df["proj_fd_ceiling"] = df["proj_fd_mean"] * 1.2

    for col in ("team", "opponent", "position", "full_name"):
        if col not in df.columns:
            df[col] = ""

    missing_cols = [col for col in PROJECTION_COLUMNS if col not in df.columns]
    for col in missing_cols:
        df[col] = 0.0 if col in ("park_adjustment", "pitch_adjustment", "weather_score", "proj_fd_ownership") else ""

    output = df[PROJECTION_COLUMNS].copy()
    output["salary"] = pd.to_numeric(output["salary"], errors="coerce").fillna(0).astype(int)
    return output


__all__ = ["compute_baseline_projections", "PROJECTION_COLUMNS"]
