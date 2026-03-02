"""Run the lineup solver on a prepared optimizer dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.optimizer import LineupResult, OptimizerConfig, generate_lineups
from slate_optimizer.optimizer.export import write_fanduel_upload

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to optimizer dataset CSV (output of prepare_optimizer_dataset.py).",
    )
    parser.add_argument(
        "--num-lineups",
        type=int,
        default=20,
        help="Number of lineups to generate.",
    )
    parser.add_argument(
        "--salary-cap",
        type=int,
        default=35000,
        help="Salary cap for each lineup.",
    )
    parser.add_argument(
        "--min-stack-size",
        type=int,
        default=4,
        help="Minimum hitters from the same team (set 0 to disable stack constraint).",
    )
    parser.add_argument(
        "--stack-templates",
        default=None,
        help="Comma-separated stack sizes (e.g., 4,3) to enforce per lineup.",
    )
    parser.add_argument(
        "--max-lineup-ownership",
        type=float,
        default=None,
        help="Optional cap on total lineup ownership (sum of proj_fd_ownership).",
    )
    parser.add_argument(
        "--bring-back",
        action="store_true",
        help="Enable bring-back requirement (include hitters from opposing team).",
    )
    parser.add_argument(
        "--bring-back-count",
        type=int,
        default=None,
        help="Number of opposing hitters required when bring-back is enabled.",
    )
    parser.add_argument(
        "--min-game-total",
        type=float,
        default=None,
        help="Only stack teams from games with Vegas total at or above this value.",
    )
    parser.add_argument(
        "--chalk-threshold",
        type=float,
        default=None,
        help="Ownership percentile above which players are considered chalk.",
    )
    parser.add_argument(
        "--chalk-exposure-cap",
        type=float,
        default=None,
        help="Max exposure for chalk players (0-1 range).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON config file for overrides (salary, exposures, stacks).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV to write all lineups (with lineup_id column).",
    )
    parser.add_argument(
        "--fd-upload",
        default=None,
        help="Optional FanDuel upload CSV (positions with player IDs).",
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

    hitters = combined[combined["player_type"].str.lower() == "batter"]
    if not hitters.empty:
        stack_counts = (
            hitters.groupby("team_code") ["lineup_id"]
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
    dataset_path = Path(args.dataset)
    df = pd.read_csv(dataset_path)

    cli_stack_templates = None
    if args.stack_templates:
        try:
            cli_stack_templates = [int(s.strip()) for s in args.stack_templates.split(',') if s.strip()]
        except ValueError:
            raise SystemExit("Invalid --stack-templates format (use comma-separated integers).")

    config = OptimizerConfig.load(Path(args.config)) if args.config else OptimizerConfig()
    df = config.apply_exposure_overrides(df)

    salary_cap = config.salary_cap if config.salary_cap is not None else args.salary_cap
    min_stack_size = config.min_stack_size if config.min_stack_size is not None else args.min_stack_size
    max_lineup_ownership = config.max_lineup_ownership if config.max_lineup_ownership is not None else args.max_lineup_ownership

    chalk_threshold = args.chalk_threshold if args.chalk_threshold is not None else config.chalk_threshold
    chalk_cap = args.chalk_exposure_cap if args.chalk_exposure_cap is not None else config.chalk_exposure_cap
    ownership_caps = config.player_ownership_caps or {}
    if chalk_threshold is not None or ownership_caps:
        temp_cfg = OptimizerConfig(
            chalk_threshold=chalk_threshold,
            chalk_exposure_cap=(chalk_cap if chalk_cap is not None else config.chalk_exposure_cap),
            player_ownership_caps=ownership_caps,
        )
        df = temp_cfg.apply_ownership_strategy(df)

    stack_templates = cli_stack_templates if cli_stack_templates else config.stack_templates
    if stack_templates:
        stack_templates = [size for size in stack_templates if isinstance(size, int) and size > 0]

    bring_back_enabled = config.bring_back_enabled or args.bring_back
    bring_back_count = config.bring_back_count if config.bring_back_count else 1
    if args.bring_back_count is not None:
        bring_back_count = args.bring_back_count
    min_game_total = config.min_game_total_for_stacks if config.min_game_total_for_stacks is not None else args.min_game_total

    lineups = generate_lineups(
        df,
        num_lineups=args.num_lineups,
        salary_cap=salary_cap,
        min_stack_size=max(0, min_stack_size),
        stack_player_types=(config.stack_player_types if config.stack_player_types else ("batter",)),
        stack_templates=stack_templates,
        max_lineup_ownership=max_lineup_ownership,
        bring_back_enabled=bring_back_enabled,
        bring_back_count=bring_back_count,
        min_game_total_for_stacks=min_game_total,
    )

    if not lineups:
        print("No feasible lineups generated. Check constraints or dataset.")
        return

    print(f"Generated {len(lineups)} lineups from {len(df)} players.")
    for idx, lineup in enumerate(lineups, start=1):
        print(f"Lineup {idx}: salary={lineup.total_salary}, proj={lineup.total_projection:.2f}")

    _print_exposure_summary(lineups)

    lineups_df = _combine_lineups(lineups)

    upload_target = None

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lineups_df.to_csv(output_path, index=False)
        print(f"Saved lineup breakdown to {output_path}")
        upload_target = output_path.with_name(f"{output_path.stem}_fanduel_upload.csv")

    if args.fd_upload:
        upload_target = Path(args.fd_upload)

    if upload_target is not None:
        write_fanduel_upload(lineups, upload_target)
        print(f"Saved FanDuel upload file to {upload_target}")

if __name__ == "__main__":
    main()




