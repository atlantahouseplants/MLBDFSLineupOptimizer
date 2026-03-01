"""Compute leverage metrics from an optimizer dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Optimizer dataset CSV (from prepare_optimizer_dataset or pipeline).")
    parser.add_argument("--top", type=int, default=10, help="How many hitters/pitchers to show.")
    parser.add_argument("--output", default=None, help="Optional CSV to write the full leverage table.")
    return parser.parse_args()

def compute_leverage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["proj_fd_mean"] = pd.to_numeric(df.get("proj_fd_mean"), errors="coerce").fillna(0.0)
    df["proj_fd_ownership"] = pd.to_numeric(df.get("proj_fd_ownership"), errors="coerce").fillna(0.0)
    df["projection_rank"] = df["proj_fd_mean"].rank(pct=True)
    df["ownership_rank"] = df["proj_fd_ownership"].rank(pct=True)
    df["leverage_score"] = df["projection_rank"] - df["ownership_rank"]
    return df

def print_sections(leverage_df: pd.DataFrame, top: int) -> None:
    hitters = leverage_df[leverage_df["player_type"].str.lower() == "batter"].sort_values("leverage_score", ascending=False)
    pitchers = leverage_df[leverage_df["player_type"].str.lower() == "pitcher"].sort_values("leverage_score", ascending=False)
    print("Top hitter leverage:")
    for _, row in hitters.head(top).iterrows():
        print(
            f"  {row['full_name']} ({row['team_code']}): leverage {row['leverage_score']:.2f}, proj {row['proj_fd_mean']:.1f}, own {row['proj_fd_ownership']:.2f}"
        )
    print("\nTop pitcher leverage:")
    for _, row in pitchers.head(top).iterrows():
        print(
            f"  {row['full_name']} ({row['team_code']}): leverage {row['leverage_score']:.2f}, proj {row['proj_fd_mean']:.1f}, win% {row.get('bpp_win_percent', float('nan')):.2f}"
        )

def main() -> None:
    args = parse_args()
    df = pd.read_csv(Path(args.dataset))
    leverage_df = compute_leverage(df)
    print_sections(leverage_df, args.top)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        leverage_df.to_csv(args.output, index=False)
        print(f"\nLeverage table saved to {args.output}")

if __name__ == "__main__":
    main()
