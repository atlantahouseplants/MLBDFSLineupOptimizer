"""Run the full DFS pipeline (synthetic or real) end-to-end."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"

def run(cmd: List[str]) -> None:
    print("\n==>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bpp-source", required=True)
    parser.add_argument("--fanduel-csv", required=True)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--db-path", default="data/slates.db")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--num-lineups", type=int, default=20)
    parser.add_argument("--stack-templates", default=None)
    parser.add_argument("--max-lineup-ownership", type=float, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--adjust", action="store_true", help="Apply projection adjustments before optimizing.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    tag = args.tag or "run"

    pipeline_cmd = [
        "python", str(SCRIPTS / "run_daily_pipeline.py"),
        "--bpp-source", args.bpp_source,
        "--fanduel-csv", args.fanduel_csv,
        "--tag", tag,
        "--db-path", args.db_path,
        "--output-dir", args.output_dir,
        "--num-lineups", str(args.num_lineups),
        "--write-intermediate",
    ]
    if args.stack_templates:
        pipeline_cmd += ["--stack-templates", args.stack_templates]
    if args.max_lineup_ownership is not None:
        pipeline_cmd += ["--max-lineup-ownership", str(args.max_lineup_ownership)]
    if args.config:
        pipeline_cmd += ["--config", args.config]

    run(pipeline_cmd)

    optimizer_dataset = Path(args.output_dir) / f"{tag}_optimizer_dataset.csv"

    if args.adjust:
        adjusted = Path(args.output_dir) / f"{tag}_optimizer_dataset_adj.csv"
        run([
            "python", str(SCRIPTS / "adjust_projections.py"),
            "--dataset", str(optimizer_dataset),
            "--output", str(adjusted),
        ])
        optimizer_dataset = adjusted

    run([
        "python", str(SCRIPTS / "compute_leverage.py"),
        "--dataset", str(optimizer_dataset),
        "--output", str(Path(args.output_dir) / f"{tag}_leverage.csv"),
        "--top", "10",
    ])

    run([
        "python", str(SCRIPTS / "run_optimizer.py"),
        "--dataset", str(optimizer_dataset),
        "--num-lineups", str(args.num_lineups),
        "--output", str(Path(args.output_dir) / f"{tag}_lineups_final.csv"),
    ])

    run([
        "python", str(SCRIPTS / "export_fanduel_upload.py"),
        "--lineups", str(Path(args.output_dir) / f"{tag}_lineups_final.csv"),
        "--output", str(Path(args.output_dir) / f"{tag}_fanduel_upload.csv"),
    ])

if __name__ == "__main__":
    main()
