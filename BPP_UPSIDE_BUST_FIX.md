# FIX: Use BallparkPal's Actual Simulation Data (Points, Bust, Median, Upside)

Read this entire document before coding. This is the most important data fix in the system.

## The Problem

BallparkPal runs 10,000 game simulations and provides per-player:
- **Points**: Mean projected FanDuel points
- **Bust**: Probability of a terrible game (decimal, e.g. 0.087 = 8.7%)
- **Median**: 50th percentile FD points
- **Upside**: ~90th percentile FD points (ceiling)
- **Pts/$**: Points per $1K salary
- **Ups/$**: Upside per $1K salary

These come from the **Ballpark DFS Optimizer** file (named like `Ballpark_DFS_Optimizer__Ballpark_Pal__17_.xlsx` for batters and `__18_.xlsx` for pitchers).

The system currently does NOT ingest this file. It only reads the BallparkPal_Batters file (which has raw PointsFD) and then **derives its own ceiling using crude multipliers of 1.2-1.5x the mean**. 

The actual BPP Upside is 2.0-2.4x the mean. Examples:
- Judge: Mean=11.5, System ceiling=13.8-17.3, BPP Upside=27.9
- Lindor: Mean=13.2, System ceiling=15.8-19.8, BPP Upside=28.2
- Ohtani: Mean=39.6, System ceiling=47-59, BPP Upside=58

The system has been working with HALF the real ceiling values. This breaks the entire leverage calculation because the boom_weight and ceiling_weight in the GPP objective are based on a ceiling that's wildly understated.

## Solution Overview

1. Add ingestion for the BPP DFS Optimizer file (both batter and pitcher versions)
2. Map BPP fields directly to projection columns: Points→proj_fd_mean, Upside→proj_fd_ceiling, Bust→bust_rate
3. Compute floor from Bust and Median rather than multipliers
4. Use bust_rate in the GPP score calculation
5. Update the dashboard to accept these files

---

## Step 1: Create BPP DFS Optimizer Ingestion

**New file: `src/slate_optimizer/ingestion/bpp_dfs_optimizer.py`**

```python
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
        
        # Read with header on row 1 (row 0 is a title row)
        df = pd.read_excel(path, header=1)
        
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
    match_count = int(matched.sum())
    
    # Override projections where BPP DFS data exists
    if matched.any():
        # Mean: use BPP Points directly (this is from 10K sims — much better than our derived mean)
        df.loc[matched, "proj_fd_mean"] = df.loc[matched, "bpp_dfs_points"]
        
        # Ceiling: use BPP Upside directly (this is ~90th percentile from 10K sims)
        df.loc[matched, "proj_fd_ceiling"] = df.loc[matched, "bpp_dfs_upside"]
        
        # Floor: derive from bust rate and median
        # If bust rate is 10%, floor is roughly the 10th percentile
        # Approximate: floor = median * (1 - bust_rate) * 0.5
        # Or simpler: floor = points * (1 - bust_rate * 2), clamped to >= 0
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
```

## Step 2: Wire BPP DFS Optimizer into the Dashboard

**File: `dashboard/daily_workflow.py`**

In the Step 1 data upload section, find where BallparkPal files are uploaded (search for "BallparkPal" file uploaders). Add two new file uploaders for the DFS Optimizer files:

```python
    bpp_dfs_batter_file = st.file_uploader(
        "BPP DFS Optimizer — Batters (optional but recommended)",
        type=["xlsx"],
        key="bpp_dfs_batter",
        help="The 'Ballpark DFS Optimizer' export with Points, Bust%, Median, Upside for batters. This gives much more accurate projections than derived estimates.",
    )
    bpp_dfs_pitcher_file = st.file_uploader(
        "BPP DFS Optimizer — Pitchers (optional but recommended)",
        type=["xlsx"],
        key="bpp_dfs_pitcher",
        help="The 'Ballpark DFS Optimizer' export for pitchers.",
    )
```

Then in the data processing function (the one that builds the combined player dataset), after the existing `combined` DataFrame is built but BEFORE projections are computed, add:

```python
    # Merge BPP DFS Optimizer data (Points, Bust, Median, Upside)
    from slate_optimizer.ingestion.bpp_dfs_optimizer import load_bpp_dfs_optimizer, merge_bpp_dfs_projections
    
    bpp_dfs_batter_path = _save_uploaded_file(bpp_dfs_batter_file, temp_dir) if bpp_dfs_batter_file else None
    bpp_dfs_pitcher_path = _save_uploaded_file(bpp_dfs_pitcher_file, temp_dir) if bpp_dfs_pitcher_file else None
    
    if bpp_dfs_batter_path or bpp_dfs_pitcher_path:
        bpp_dfs_df = load_bpp_dfs_optimizer(
            batter_path=bpp_dfs_batter_path,
            pitcher_path=bpp_dfs_pitcher_path,
        )
        if not bpp_dfs_df.empty:
            combined = merge_bpp_dfs_projections(combined, bpp_dfs_df)
            bpp_dfs_matched = int((combined.get("bpp_dfs_points", 0) > 0).sum()) if "bpp_dfs_points" in combined.columns else 0
            optional_messages.append(
                f"BPP DFS Optimizer: matched {bpp_dfs_matched} players with simulation-grade projections (Points, Bust%, Upside)"
            )
```

Place this AFTER the `build_player_dataset` call and the `_merge_optional_sources` call, but BEFORE the projection computation. The BPP DFS data should override the baseline projections.

## Step 3: Update the GPP Score to Use Bust Rate

**File: `src/slate_optimizer/optimizer/solver.py`**

In `_compute_gpp_score()`, add bust_rate awareness. Players with LOW bust rates and HIGH upside are the best GPP plays. Players with high bust rates are riskier but offer more leverage when they hit.

Add after the existing boom_pct calculation:

```python
    # Bust rate factor: for GPP, we actually WANT some bust risk because
    # high-bust players are under-owned (the field avoids them).
    # But we need the upside to justify the risk.
    # "GPP attractiveness" = upside-to-bust ratio
    bust_rate = float(pool.loc[idx, "bust_rate"]) if "bust_rate" in pool.columns else 0.15
    if bust_rate > 0 and ceiling > 0:
        # Upside efficiency: how much ceiling do you get per unit of bust risk?
        upside_efficiency = ceiling / (bust_rate * 100)  # e.g. 28 / (8.7) = 3.2
        # Small bonus for high upside efficiency (good risk/reward)
        score += leverage_config.boom_weight * min(upside_efficiency * 0.02, 0.5)
```

## Step 4: Update Distribution Fitting to Use Real BPP Numbers

**File: `src/slate_optimizer/simulation/distributions.py`**

The `fit_player_distributions` function fits lognormal/normal distributions from mean/floor/ceiling. Now that floor and ceiling come from BPP's actual simulation (not our crude multipliers), the distributions will automatically be more accurate. No code change needed here — it already uses `proj_fd_floor` and `proj_fd_ceiling`, which we're now populating with real BPP data.

However, add bust_rate to the PlayerDistribution dataclass so it can be used in contest simulation:

In the `PlayerDistribution` dataclass, add:
```python
    bust_rate: float = 0.15
```

In `fit_player_distributions`, when building each PlayerDistribution, add:
```python
        bust_rate = float(record.get("bust_rate", 0.15) or 0.15)
```
And pass it to the constructor:
```python
        dists[player_id] = PlayerDistribution(
            ...
            bust_rate=bust_rate,
        )
```

## Step 5: Add the BPP DFS Optimizer files to the upload cache

**File: `dashboard/daily_workflow.py`**

In the file caching/manifest section (where it saves uploaded files for one-click reload), add the BPP DFS files to the manifest so they persist across sessions. Look for where `manifest["batting_orders"]` is set and add:

```python
    if bpp_dfs_batter_file:
        manifest["bpp_dfs_batters"] = _cache(bpp_dfs_batter_file)
    if bpp_dfs_pitcher_file:
        manifest["bpp_dfs_pitchers"] = _cache(bpp_dfs_pitcher_file)
```

And in the reload section, restore them.

---

## Step 6: Update the `__init__.py` exports

**File: `src/slate_optimizer/ingestion/__init__.py`**

Add:
```python
from .bpp_dfs_optimizer import load_bpp_dfs_optimizer, merge_bpp_dfs_projections
```

---

## What This Changes Functionally

Before this fix:
- Ohtani: mean=39.6, ceiling=47-59 (derived), bust=unknown
- Lindor: mean=13.2, ceiling=15.8-19.8 (derived), bust=unknown
- Doyle: mean=8.75, ceiling=10.5-13.1 (derived), bust=unknown

After this fix:
- Ohtani: mean=39.6, ceiling=58 (BPP actual), bust=3.5%
- Lindor: mean=13.2, ceiling=28.2 (BPP actual), bust=8.7%
- Doyle: mean=8.75, ceiling=22.2 (BPP actual), bust=27.9%

Now the GPP score correctly sees that Lindor has 2.1x upside with only 8.7% bust risk — a fantastic GPP play. Doyle has 2.5x upside but 27.9% bust risk — a boom-or-bust leverage play that should appear occasionally, not in 65% of lineups.

## After all changes:

1. Run `python -m pytest tests/ -v`
2. Commit: "Add BPP DFS Optimizer ingestion: use real Upside/Bust/Median from BallparkPal simulations"
3. Push to main
