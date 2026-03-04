"""Generate baseline projections for a stored slate."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.data.storage import SlateDatabase
from slate_optimizer.projection import PROJECTION_COLUMNS, compute_baseline_projections

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        default="data/slates.db",
        help="SQLite database that holds slate snapshots.",
    )
    parser.add_argument(
        "--slate-id",
        type=int,
        default=None,
        help="Specific slate_id to project (defaults to latest).",
    )
    parser.add_argument(
        "--output",
        default="data/processed",
        help="Directory to write the projection CSV.",
    )
    parser.add_argument(
        "--platoon-opposite-boost",
        type=float,
        default=None,
        help="Multiplier for hitters vs opposite-hand pitchers (default 1.06).",
    )
    parser.add_argument(
        "--platoon-same-penalty",
        type=float,
        default=None,
        help="Multiplier for same-hand matchups (default 0.95).",
    )
    parser.add_argument(
        "--platoon-switch-boost",
        type=float,
        default=None,
        help="Multiplier for switch hitters (default 1.03).",
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
        raise SystemExit(f"Slate {slate.slate_id} has no stored players to project.")

    projections = compute_baseline_projections(
        players_df,
        platoon_opposite_boost=args.platoon_opposite_boost,
        platoon_same_penalty=args.platoon_same_penalty,
        platoon_switch_boost=args.platoon_switch_boost,
    )
    output_path = output_dir / f"{slate.tag}_baseline_projections.csv"
    projections.to_csv(output_path, index=False)

    print(
        f"Wrote {len(projections)} baseline projections to {output_path} (columns: {', '.join(PROJECTION_COLUMNS)})"
    )

if __name__ == "__main__":
    main()
