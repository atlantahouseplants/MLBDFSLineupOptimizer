"""Full pipeline runner: ingest, project, prepare optimizer dataset, solve."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.data.storage import SlateDatabase
from slate_optimizer.ingestion.aliases import load_alias_map
from slate_optimizer.ingestion.ballparkpal import BallparkPalLoader
from slate_optimizer.ingestion.fanduel import FanduelCSVLoader
from slate_optimizer.ingestion.slate_builder import build_player_dataset
from slate_optimizer.optimizer import (
    LineupResult,
    OptimizerConfig,
    build_optimizer_dataset,
    generate_lineups,
)
from slate_optimizer.projection import compute_baseline_projections

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bpp-source", required=True, help="Directory with BallparkPal Excel files.")
    parser.add_argument("--fanduel-csv", required=True, help="FanDuel player list CSV for the slate.")
    parser.add_argument("--alias-file", default=None, help="Optional alias JSON for name overrides.")
    parser.add_argument("--tag", default=None, help="Slate tag (defaults to YYYYMMDD).")
    parser.add_argument("--db-path", default="data/slates.db", help="SQLite DB to persist the slate.")
    parser.add_argument("--output-dir", default="data/processed", help="Directory for projections/datasets/lineups.")
    parser.add_argument("--num-lineups", type=int, default=20, help="Lineups to generate.")
    parser.add_argument("--salary-cap", type=int, default=35000, help="Salary cap override.")
    parser.add_argument("--min-stack-size", type=int, default=4, help="Min hitters from one team.")
    parser.add_argument("--stack-templates", default=None, help="Comma-separated stack sizes (e.g., 4,3).")
    parser.add_argument("--max-lineup-ownership", type=float, default=None, help="Optional cap on total lineup ownership.")
    parser.add_argument("--chalk-threshold", type=float, default=None, help="Ownership percentile defining chalk.")
    parser.add_argument("--chalk-exposure-cap", type=float, default=None, help="Max exposure for chalk players.")
    parser.add_argument("--config", default=None, help="Optional optimizer config JSON.")
    parser.add_argument(
        "--write-intermediate",
        action="store_true",
        help="Write merged slate + projections CSVs alongside optimizer outputs.",
    )
    return parser.parse_args()

def _combine_lineups(lineups: List[LineupResult]) -> pd.DataFrame:
    rows = []
    for idx, lineup in enumerate(lineups, start=1):
        temp = lineup.dataframe.copy()
        temp.insert(0, "lineup_id", idx)
        rows.append(temp)
    return pd.concat(rows, ignore_index=True)

def _print_exposure_summary(lineups: List[LineupResult]) -> None:
    if not lineups:
        return
    combined = _combine_lineups(lineups)
    player_counts = (
        combined.groupby(["fd_player_id", "full_name"]) ["lineup_id"]
        .count()
        .reset_index(name="lineups")
        .sort_values("lineups", ascending=False)
    )
    player_counts["exposure_pct"] = player_counts["lineups"] / len(lineups)
    print("Top player exposures:")
    for _, row in player_counts.head(10).iterrows():
        print(
            f"  {row['full_name']}: {row['lineups']} lineups ({row['exposure_pct']:.0%})"
        )

def _print_stack_summary(lineups: List[LineupResult]) -> None:
    if not lineups:
        return
    combined = _combine_lineups(lineups)
    hitters = combined[combined["player_type"].str.lower() == "batter"]
    if hitters.empty:
        return
    stack_counts = (
        hitters.groupby("team_code")["lineup_id"]
        .count()
        .reset_index(name="appearances")
        .sort_values("appearances", ascending=False)
    )
    print("Stack exposure by team:")
    for _, row in stack_counts.head(10).iterrows():
        pct = row["appearances"] / (len(lineups) * 9)
        print(f"  {row['team_code']}: {row['appearances']} spots ({pct:.0%} of hitters)")

def main() -> None:
    args = parse_args()
    tag = args.tag or datetime.now().strftime("%Y%m%d")

    cli_stack_templates = None
    if args.stack_templates:
        try:
            cli_stack_templates = [int(s.strip()) for s in args.stack_templates.split(',') if s.strip()]
        except ValueError:
            raise SystemExit("Invalid --stack-templates format (use comma-separated integers).")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpp_loader = BallparkPalLoader(Path(args.bpp_source))
    bundle = bpp_loader.load_bundle()

    fd_loader = FanduelCSVLoader(Path(args.fanduel_csv))
    fd_players = fd_loader.load()

    alias_map = None
    if args.alias_file:
        alias_map = load_alias_map(Path(args.alias_file))

    combined, diagnostics = build_player_dataset(bundle, fd_players.players, alias_map=alias_map)

    db = SlateDatabase(Path(args.db_path))
    slate = db.insert_slate(tag, Path(args.fanduel_csv), Path(args.bpp_source))
    db.write_players(slate.slate_id, combined)
    db.close()
    print(f"Persisted slate #{slate.slate_id} ({tag}) with diagnostics: {diagnostics.as_dict()}")

    projections = compute_baseline_projections(combined)
    optimizer_df = build_optimizer_dataset(combined, projections)

    config = OptimizerConfig.load(Path(args.config)) if args.config else OptimizerConfig()
    optimizer_df = config.apply_exposure_overrides(optimizer_df)

    salary_cap = config.salary_cap if config.salary_cap is not None else args.salary_cap
    min_stack_size = config.min_stack_size if config.min_stack_size is not None else args.min_stack_size
    max_lineup_ownership = config.max_lineup_ownership if config.max_lineup_ownership is not None else args.max_lineup_ownership

    chalk_threshold = args.chalk_threshold if args.chalk_threshold is not None else config.chalk_threshold
    chalk_cap = args.chalk_exposure_cap if args.chalk_exposure_cap is not None else config.chalk_exposure_cap
    if chalk_threshold is not None or config.player_ownership_caps:
        temp_cfg = OptimizerConfig(
            chalk_threshold=chalk_threshold,
            chalk_exposure_cap=(chalk_cap if chalk_cap is not None else config.chalk_exposure_cap),
            player_ownership_caps=config.player_ownership_caps,
        )
        optimizer_df = temp_cfg.apply_ownership_strategy(optimizer_df)

    stack_templates = cli_stack_templates if cli_stack_templates else config.stack_templates
    if stack_templates:
        stack_templates = [size for size in stack_templates if isinstance(size, int) and size > 0]

    if args.write_intermediate:
        combined.to_csv(output_dir / f"{tag}_slate_players.csv", index=False)
        projections.to_csv(output_dir / f"{tag}_baseline_projections.csv", index=False)
        optimizer_df.to_csv(output_dir / f"{tag}_optimizer_dataset.csv", index=False)

    lineups = generate_lineups(
        optimizer_df,
        num_lineups=args.num_lineups,
        salary_cap=salary_cap,
        min_stack_size=max(0, min_stack_size),
        stack_player_types=(config.stack_player_types if config.stack_player_types else ("batter",)),
        stack_templates=stack_templates,
        max_lineup_ownership=max_lineup_ownership,
    )

    if not lineups:
        print("No feasible lineups generated.")
        return

    print(f"Generated {len(lineups)} lineups. Top lineup proj={lineups[0].total_projection:.2f}")
    _print_exposure_summary(lineups)
    _print_stack_summary(lineups)

    lineups_path = output_dir / f"{tag}_lineups.csv"
    _combine_lineups(lineups).to_csv(lineups_path, index=False)
    print(f"Saved lineup breakdown to {lineups_path}")

if __name__ == "__main__":
    main()



