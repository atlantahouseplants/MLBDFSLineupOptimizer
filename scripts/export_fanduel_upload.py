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

from slate_optimizer.exporters.fanduel import build_upload_dataframe

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineups", required=True, help="CSV created by run_optimizer.py or run_daily_pipeline.py.")
    parser.add_argument("--output", required=True, help="Destination FanDuel upload CSV.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    lineups_df = pd.read_csv(Path(args.lineups))
    upload_df = build_upload_dataframe(lineups_df)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    upload_df.to_csv(Path(args.output), index=False)
    print(f"Saved FanDuel upload file to {args.output}")

if __name__ == "__main__":
    main()
