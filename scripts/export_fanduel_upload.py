"""Convert solver lineups to FanDuel upload format."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.optimizer import LineupResult
from slate_optimizer.optimizer.export import lineups_to_fanduel_upload

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineups", required=True, help="CSV created by run_optimizer.py or run_daily_pipeline.py.")
    parser.add_argument("--output", required=True, help="Destination FanDuel upload CSV.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    lineups_df = pd.read_csv(Path(args.lineups))
    if "lineup_id" not in lineups_df.columns:
        raise SystemExit("Input lineups CSV must include a lineup_id column")

    lineups: list[LineupResult] = []
    for lineup_id, group in lineups_df.groupby("lineup_id"):
        group = group.copy().reset_index(drop=True)
        total_salary = int(pd.to_numeric(group.get("salary"), errors="coerce").fillna(0).sum())
        total_projection = float(pd.to_numeric(group.get("proj_fd_mean"), errors="coerce").fillna(0).sum())
        lineups.append(
            LineupResult(
                dataframe=group,
                total_salary=total_salary,
                total_projection=total_projection,
            )
        )

    upload_df = lineups_to_fanduel_upload(lineups)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    upload_df.to_csv(output_path, index=False)
    print(f"Saved FanDuel upload file to {args.output}")

if __name__ == "__main__":
    main()
