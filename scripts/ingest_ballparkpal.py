"""CLI to ingest BallparkPal Excel exports into normalized CSV files."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.ingestion.ballparkpal import BallparkPalLoader

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="Directory containing the BallparkPal Excel exports.",
    )
    parser.add_argument(
        "--output",
        default="data/raw",
        help="Directory where normalized CSV files will be written.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional identifier for output filenames (defaults to current date).",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = BallparkPalLoader(source_dir)
    bundle = loader.load_bundle()

    tag = args.tag or datetime.now().strftime("%Y%m%d")
    for name, frame in bundle.frames().items():
        output_path = output_dir / f"{tag}_ballparkpal_{name}.csv"
        frame.to_csv(output_path, index=False)
        print(f"Wrote {len(frame)} rows to {output_path}")

    print("Summary:", bundle.summary())

if __name__ == "__main__":
    main()
