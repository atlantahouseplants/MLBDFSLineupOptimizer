"""Ingest BallparkPal DFS Optimizer Excel exports.

These files contain the distilled simulation output: Points, Bust%, Median,
Upside, and value metrics for every player on the FanDuel slate.  They are
the most important projection input — they should override any derived
projections when available.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .text_utils import canonicalize_series

# BPP DFS Optimizer uses abbreviated names like "F. Lindor 1" where the
# trailing number is the batting order position.  We need to strip that
# and the leading initial format to match against FanDuel full names.

def _clean_bpp_name(raw: str) -> str:
    """Strip batting order number and normalize name format."""
    parts = raw.strip().rsplit(" ", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0].strip()
    return raw.strip()


def _expand_initial(name: str) -> str:
    """Best-effort expansion: 'F. Lindor' stays as-is for canonical matching."""
    return name


def load_bpp_dfs_optimizer(
    batter_path: Optional[Path] = None,
    pitcher_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load and combine BPP DFS Optimizer batter + pitcher files.

    Returns a DataFrame with columns:
        bpp_dfs_name, bpp_dfs_team, bpp_dfs_position, bpp_dfs_salary,
        bpp_dfs_points, bpp_dfs_bust, bpp_dfs_median, bpp_dfs_upside,
        bpp_dfs_pts_per_k, bpp_dfs_ups_per_k, bpp_dfs_player_type,
        bpp_dfs_batting_order, bpp_dfs_game
    """
    frames = []

    for path, player_type in [(batter_path, "batter"), (pitcher_path, "pitcher")]:
        if path is None:
            continue
        path = Path(path)
        if not path.exists():
            continue

        # Read with header on row 1 (row 0 is a title row in BPP DFS Optimizer)
        df = pd.read_excel(path, header=1)

        # Auto-detect: if "Players" column is missing, try header=0
        if "Players" not in df.columns:
            df = pd.read_excel(path, header=0)
        if "Players" not in df.columns:
            # Not a BPP DFS Optimizer file — skip silently
            continue

        # Extract batting order from player name (e.g. "F. Lindor 1" -> order=1)
        raw_names = df["Players"].astype(str)
        orders = raw_names.str.extract(r'\s+(\d+)\s*$')[0]
        clean_names = raw_names.apply(_clean_bpp_name)

        result = pd.DataFrame({
            "bpp_dfs_name": clean_names,
            "bpp_dfs_canonical": canonicalize_series(clean_names),
            "bpp_dfs_team": df["Tm"].astype(str).str.upper().str.strip(),
            "bpp_dfs_position": df.get("Pos", "").astype(str),
            "bpp_dfs_salary": pd.to_numeric(df.get("$"), errors="coerce").fillna(0) * 1000,
            "bpp_dfs_points": pd.to_numeric(df.get("Points"), errors="coerce").fillna(0),
            "bpp_dfs_bust": pd.to_numeric(df.get("Bust"), errors="coerce").fillna(0),
            "bpp_dfs_median": pd.to_numeric(df.get("Median"), errors="coerce").fillna(0),
            "bpp_dfs_upside": pd.to_numeric(df.get("Upside"), errors="coerce").fillna(0),
            "bpp_dfs_pts_per_k": pd.to_numeric(df.get("Pts/$"), errors="coerce").fillna(0),
            "bpp_dfs_ups_per_k": pd.to_numeric(df.get("Ups/$"), errors="coerce").fillna(0),
            "bpp_dfs_player_type": player_type,
            "bpp_dfs_batting_order": pd.to_numeric(orders, errors="coerce"),
            "bpp_dfs_game": df.get("GameDescription", "").astype(str) if "GameDescription" in df.columns else "",
        })
        frames.append(result)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return combined


def merge_bpp_dfs_projections(
    player_df: pd.DataFrame,
    bpp_dfs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge BPP DFS Optimizer data into the player dataset.

    Matches on team_code + last_name (canonical).  When matched, overrides:
    - proj_fd_mean with bpp_dfs_points
    - proj_fd_ceiling with bpp_dfs_upside
    - proj_fd_floor with a floor derived from bpp_dfs_median and bpp_dfs_bust
    - Adds bust_rate column

    Players without a BPP DFS match keep their existing projections.
    """
    if bpp_dfs_df is None or bpp_dfs_df.empty:
        return player_df

    df = player_df.copy()

    # Build canonical last name for matching
    # BPP uses "F. Lindor" format — last name is the second part
    bpp = bpp_dfs_df.copy()
    bpp["_last"] = canonicalize_series(
        bpp["bpp_dfs_name"].str.split().str[-1]
    )

    # Player dataset last names
    if "last_name" in df.columns:
        df["_last"] = canonicalize_series(df["last_name"])
    elif "full_name" in df.columns:
        df["_last"] = canonicalize_series(df["full_name"].str.split().str[-1])
    else:
        return df

    if "team_code" not in df.columns:
        return df

    # Deduplicate BPP by team+last_name (keep first = highest projected)
    bpp_dedup = bpp.sort_values("bpp_dfs_points", ascending=False).drop_duplicates(
        ["bpp_dfs_team", "_last"]
    )

    # Merge
    merge_cols = [
        "bpp_dfs_points", "bpp_dfs_bust", "bpp_dfs_median", "bpp_dfs_upside",
        "bpp_dfs_pts_per_k", "bpp_dfs_ups_per_k", "bpp_dfs_batting_order",
        "bpp_dfs_team", "_last",
    ]

    df = df.merge(
        bpp_dedup[merge_cols],
        left_on=["team_code", "_last"],
        right_on=["bpp_dfs_team", "_last"],
        how="left",
        suffixes=("", "_bppdfs"),
    )

    matched = df["bpp_dfs_points"].notna() & (df["bpp_dfs_points"] > 0)

    # Override projections where BPP DFS data exists
    if matched.any():
        # Mean: use BPP Points directly (this is from 10K sims)
        df.loc[matched, "proj_fd_mean"] = df.loc[matched, "bpp_dfs_points"]

        # Ceiling: use BPP Upside directly (this is ~90th percentile from 10K sims)
        df.loc[matched, "proj_fd_ceiling"] = df.loc[matched, "bpp_dfs_upside"]

        # Floor: derive from bust rate and median
        bust = df.loc[matched, "bpp_dfs_bust"].fillna(0.15)
        median = df.loc[matched, "bpp_dfs_median"].fillna(df.loc[matched, "bpp_dfs_points"])
        # Floor ~ halfway between 0 and median, scaled by reliability (1 - bust)
        floor = (median * (1 - bust)).clip(lower=0)
        df.loc[matched, "proj_fd_floor"] = floor

        # Bust rate: add as new column
        df["bust_rate"] = 0.15  # default for unmatched
        df.loc[matched, "bust_rate"] = df.loc[matched, "bpp_dfs_bust"]
    else:
        df["bust_rate"] = 0.15

    # Clean up merge columns
    drop_cols = ["_last", "bpp_dfs_team"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df
