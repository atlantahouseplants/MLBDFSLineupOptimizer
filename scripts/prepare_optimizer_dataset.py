"""Build optimizer-ready dataset for a stored slate."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.data.storage import SlateDatabase
from slate_optimizer.optimizer import build_optimizer_dataset
from slate_optimizer.projection import compute_baseline_projections

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        default="data/slates.db",
        help="SQLite database containing persisted slates.",
    )
    parser.add_argument(
        "--slate-id",
        type=int,
        default=None,
        help="Specific slate_id to process (defaults to latest).",
    )
    parser.add_argument(
        "--output",
        default="data/processed",
        help="Directory for optimizer dataset CSV.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    db = SlateDatabase(Path(args.db_path))
    if args.slate_id is not None:
        slate = db.get_slate(args.slate_id)
        if slate is None:
            raise SystemExit(f"Slate {args.slate_id} not found in {args.db_path}")
    else:
        slate = db.get_latest_slate()
        if slate is None:
            raise SystemExit("Database is empty; run build_slate_dataset.py first.")

    players_df = db.fetch_players(slate.slate_id)
    db.close()

    if players_df.empty:
        raise SystemExit(f"Slate {slate.slate_id} has no stored players.")

    projections = compute_baseline_projections(players_df)
    optimizer_df = build_optimizer_dataset(players_df, projections)

    output_path = output_dir / f"{slate.tag}_optimizer_dataset.csv"
    optimizer_df.to_csv(output_path, index=False)

    print(f"Wrote {len(optimizer_df)} optimizer rows to {output_path}")

if __name__ == "__main__":
    main()
