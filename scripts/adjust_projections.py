"""Apply stack/pitcher adjustments to an optimizer dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Optimizer dataset CSV to adjust.")
    parser.add_argument("--output", required=True, help="Where to write the adjusted CSV.")
    parser.add_argument("--stack-weight", type=float, default=0.5, help="Multiplier for (team runs - league mean) applied to hitters.")
    parser.add_argument("--pitcher-weight", type=float, default=5.0, help="Multiplier for (win% - 0.5) applied to pitcher projections.")
    return parser.parse_args()

def adjust(dataset: pd.DataFrame, stack_weight: float, pitcher_weight: float) -> pd.DataFrame:
    df = dataset.copy()
    df["proj_fd_mean"] = pd.to_numeric(df["proj_fd_mean"], errors="coerce").fillna(0.0)

    if "bpp_runs" in df.columns:
        runs = pd.to_numeric(df["bpp_runs"], errors="coerce")
        league_mean = runs.mean()
        hitters = df["player_type"].str.lower() == "batter"
        run_delta = runs.fillna(league_mean) - league_mean
        df.loc[hitters, "proj_fd_mean"] += stack_weight * run_delta.loc[hitters]

    if "bpp_win_percent" in df.columns:
        win_percent = pd.to_numeric(df["bpp_win_percent"], errors="coerce").fillna(0.5)
        pitchers = df["player_type"].str.lower() == "pitcher"
        win_delta = win_percent - 0.5
        df.loc[pitchers, "proj_fd_mean"] += pitcher_weight * win_delta.loc[pitchers]

    df["proj_fd_mean"] = df["proj_fd_mean"].clip(lower=0)
    return df

def main() -> None:
    args = parse_args()
    df = pd.read_csv(Path(args.dataset))
    adjusted = adjust(df, args.stack_weight, args.pitcher_weight)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    adjusted.to_csv(Path(args.output), index=False)
    print(f"Adjusted projections saved to {args.output}")

if __name__ == "__main__":
    main()
