"""Builds a combined slate dataset linking FanDuel salaries with BallparkPal sims."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.data.storage import SlateDatabase
from slate_optimizer.ingestion.aliases import load_alias_map
from slate_optimizer.ingestion.ballparkpal import BallparkPalLoader
from slate_optimizer.ingestion.fanduel import FanduelCSVLoader
from slate_optimizer.ingestion.slate_builder import build_player_dataset

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bpp-source",
        required=True,
        help="Directory containing BallparkPal Excel exports.",
    )
    parser.add_argument(
        "--fanduel-csv",
        required=True,
        help="FanDuel players list CSV for the slate (downloaded from FanDuel).",
    )
    parser.add_argument(
        "--output",
        default="data/processed",
        help="Directory for processed slate datasets.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional label for the processed files (defaults to current date).",
    )
    parser.add_argument(
        "--alias-file",
        default=None,
        help="Optional JSON map of canonical name overrides.",
    )
    parser.add_argument(
        "--db-path",
        default="data/slates.db",
        help="SQLite database file for persisting slate snapshots.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    tag = args.tag or datetime.now().strftime("%Y%m%d")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpp_loader = BallparkPalLoader(Path(args.bpp_source))
    bundle = bpp_loader.load_bundle()

    fd_loader = FanduelCSVLoader(Path(args.fanduel_csv))
    fd_players = fd_loader.load()

    alias_map = None
    if args.alias_file:
        alias_map = load_alias_map(Path(args.alias_file))

    combined, diagnostics = build_player_dataset(bundle, fd_players.players, alias_map=alias_map)

    output_path = output_dir / f"{tag}_slate_players.csv"
    combined.to_csv(output_path, index=False)

    unmatched = combined[combined["bpp_full_name"].isna()]
    unmatched_path = None
    if not unmatched.empty:
        unmatched_path = output_dir / f"{tag}_unmatched_players.csv"
        unmatched.to_csv(unmatched_path, index=False)

    db = SlateDatabase(Path(args.db_path))
    slate_record = db.insert_slate(tag, Path(args.fanduel_csv), Path(args.bpp_source))
    db.write_players(slate_record.slate_id, combined)
    db.close()

    print(f"Saved merged slate dataset to {output_path}")
    if unmatched_path:
        print(
            f"Warning: {len(unmatched)} players without BallparkPal match. Details: {unmatched_path}"
        )
    print("Diagnostics:", diagnostics.as_dict())
    print(f"Persisted slate #{slate_record.slate_id} into {args.db_path}")

if __name__ == "__main__":
    main()
