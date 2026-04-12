"""Full pipeline runner: ingest, project, prepare optimizer dataset, solve."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.data.storage import SlateDatabase
from slate_optimizer.ingestion.aliases import load_alias_map
from slate_optimizer.ingestion.ballparkpal import BallparkPalLoader
from slate_optimizer.ingestion.batting_orders import BattingOrderLoader
from slate_optimizer.ingestion.fanduel import FanduelCSVLoader
from slate_optimizer.ingestion.handedness import HandednessLoader
from slate_optimizer.ingestion.recent_stats import RecentStatsLoader
from slate_optimizer.ingestion.vegas import VegasLoader
from slate_optimizer.ingestion.slate_builder import build_player_dataset
from slate_optimizer.optimizer import (
    LineupResult,
    OptimizerConfig,
    build_optimizer_dataset,
    generate_lineups,
)
from slate_optimizer.optimizer.export import write_fanduel_upload
from slate_optimizer.projection import compute_baseline_projections, compute_ownership_series

def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        add_help=add_help,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bpp-source", required=True, help="Directory with BallparkPal Excel files.")
    parser.add_argument("--fanduel-csv", required=True, help="FanDuel player list CSV for the slate.")
    parser.add_argument("--alias-file", default=None, help="Optional alias JSON for name overrides.")
    parser.add_argument("--tag", default=None, help="Slate tag (defaults to YYYYMMDD).")
    parser.add_argument(
        "--vegas-csv",
        default=None,
        help="Optional Vegas lines CSV (game,total,home_ml,away_ml).",
    )
    parser.add_argument(
        "--batting-orders-csv",
        default=None,
        help="Optional batting orders CSV (team,order_position,player_name).",
    )
    parser.add_argument(
        "--handedness-csv",
        default=None,
        help="Optional handedness reference CSV (player_name,team,bats,throws).",
    )
    parser.add_argument(
        "--recent-stats-csv",
        default=None,
        help="Optional recent performance CSV (player_name,team,last_7_fppg,last_14_fppg,season_fppg).",
    )
    parser.add_argument(
        "--recency-blend",
        default=None,
        help="Season vs recent weighting (e.g., 0.7,0.3).",
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
    parser.add_argument(
        "--ownership-sources",
        default=None,
        help="Comma-separated ownership CSV files to blend (fd_player_id + ownership).",
    )
    parser.add_argument(
        "--ownership-weights",
        default=None,
        help="Comma-separated weights for ownership sources (defaults to equal weights).",
    )
    parser.add_argument("--db-path", default="data/slates.db", help="SQLite DB to persist the slate.")
    parser.add_argument("--output-dir", default="data/processed", help="Directory for projections/datasets/lineups.")
    parser.add_argument("--num-lineups", type=int, default=20, help="Lineups to generate.")
    parser.add_argument("--salary-cap", type=int, default=35000, help="Salary cap override.")
    parser.add_argument("--min-stack-size", type=int, default=4, help="Min hitters from one team.")
    parser.add_argument("--stack-templates", default=None, help="Comma-separated stack sizes (e.g., 4,3).")
    parser.add_argument("--max-lineup-ownership", type=float, default=None, help="Optional cap on total lineup ownership.")
    parser.add_argument(
        "--bring-back",
        action="store_true",
        help="Enable bring-back requirement for stacks.",
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
    parser.add_argument("--chalk-threshold", type=float, default=None, help="Ownership percentile defining chalk.")
    parser.add_argument("--chalk-exposure-cap", type=float, default=None, help="Max exposure for chalk players.")
    parser.add_argument("--config", default=None, help="Optional optimizer config JSON.")
    parser.add_argument(
        "--write-intermediate",
        action="store_true",
        help="Write merged slate + projections CSVs alongside optimizer outputs.",
    )
    parser.add_argument(
        "--auto-fetch",
        action="store_true",
        help="Auto-fetch batting orders + Vegas lines before running pipeline.",
    )
    parser.add_argument(
        "--live-data-dir",
        default="data/live",
        help="Directory for auto-fetched live data files (default: data/live).",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)

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

def run_pipeline(args: argparse.Namespace) -> dict:
    tag = args.tag or datetime.now().strftime("%Y%m%d")

    def _parse_list(value: str | None) -> List[str]:
        if not value:
            return []
        return [item.strip() for item in value.split(',') if item.strip()]

    cli_stack_templates = None
    if args.stack_templates:
        try:
            cli_stack_templates = [int(s.strip()) for s in args.stack_templates.split(',') if s.strip()]
        except ValueError:
            raise SystemExit("Invalid --stack-templates format (use comma-separated integers).")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ownership_source_paths = _parse_list(args.ownership_sources)
    ownership_paths: List[Path] = [Path(path).expanduser() for path in ownership_source_paths]
    ownership_weights = None
    if args.ownership_weights:
        tokens = _parse_list(args.ownership_weights)
        try:
            ownership_weights = [float(token) for token in tokens]
        except ValueError:
            raise SystemExit("Invalid value in --ownership-weights (must be numeric)")
        if not ownership_paths:
            raise SystemExit("--ownership-weights requires --ownership-sources")
        if len(ownership_weights) != len(ownership_paths):
            raise SystemExit("Number of ownership weights must match ownership sources")

    recency_blend = None
    if args.recency_blend:
        tokens = _parse_list(args.recency_blend)
        if len(tokens) != 2:
            raise SystemExit("--recency-blend must have two comma-separated numbers (season,recent)")
        try:
            recency_blend = (float(tokens[0]), float(tokens[1]))
        except ValueError:
            raise SystemExit("Invalid --recency-blend (must be numeric)")

    platoon_opposite_boost = args.platoon_opposite_boost
    platoon_same_penalty = args.platoon_same_penalty
    platoon_switch_boost = args.platoon_switch_boost

    bpp_loader = BallparkPalLoader(Path(args.bpp_source))
    bundle = bpp_loader.load_bundle()

    fd_loader = FanduelCSVLoader(Path(args.fanduel_csv))
    fd_players = fd_loader.load()

    vegas_lines = None
    if args.vegas_csv:
        vegas_loader = VegasLoader(Path(args.vegas_csv))
        vegas_lines = vegas_loader.load()
        summary = vegas_lines.summary()
        print(
            f"Loaded Vegas lines for {summary['games']} games from {args.vegas_csv}"
        )

    alias_map = None
    if args.alias_file:
        alias_map = load_alias_map(Path(args.alias_file))

    combined, diagnostics = build_player_dataset(bundle, fd_players.players, alias_map=alias_map)

    combined["is_confirmed_lineup"] = False
    combined["batting_order_position"] = pd.Series(pd.NA, index=combined.index, dtype="Int64")
    combined["batter_hand"] = ""
    combined["pitcher_hand"] = ""
    combined["recent_last7_fppg"] = 0.0
    combined["recent_last14_fppg"] = 0.0
    combined["recent_season_fppg"] = 0.0

    if args.batting_orders_csv:
        order_loader = BattingOrderLoader(Path(args.batting_orders_csv))
        orders = order_loader.load(alias_map=alias_map)
        combined = combined.merge(
            orders.entries.rename(
                columns={"batting_order_position": "_batting_order_position"}
            ),
            on=["team_code", "canonical_name"],
            how="left",
        )
        combined["batting_order_position"] = combined[
            "_batting_order_position"
        ].combine_first(combined["batting_order_position"])
        combined.drop(columns=["_batting_order_position"], inplace=True)
        combined["is_confirmed_lineup"] = combined["batting_order_position"].notna()
        combined["batting_order_position"] = pd.to_numeric(
            combined["batting_order_position"], errors="coerce"
        ).astype("Int64")
        summary = orders.summary()
        print(
            f"Loaded batting orders for {summary['teams']} teams from {args.batting_orders_csv}"
        )

    if args.handedness_csv:
        hand_loader = HandednessLoader(Path(args.handedness_csv))
        handed = hand_loader.load(alias_map=alias_map)
        hand_entries = handed.entries.rename(
            columns={
                "batter_hand": "_batter_hand",
                "pitcher_hand": "_pitcher_hand",
            }
        )
        combined = combined.merge(
            hand_entries,
            on=["team_code", "canonical_name"],
            how="left",
        )
        combined["batter_hand"] = (
            combined["batter_hand"].astype(str).str.upper().str.strip()
        )
        combined["pitcher_hand"] = (
            combined["pitcher_hand"].astype(str).str.upper().str.strip()
        )
        combined["batter_hand"] = combined["batter_hand"].where(
            combined["batter_hand"].isin(["L", "R", "S"]), ""
        )
        combined["pitcher_hand"] = combined["pitcher_hand"].where(
            combined["pitcher_hand"].isin(["L", "R"]), ""
        )
        combined["batter_hand"] = combined["batter_hand"].where(
            combined["batter_hand"].ne(""), combined["_batter_hand"].fillna("")
        )
        combined["pitcher_hand"] = combined["pitcher_hand"].where(
            combined["pitcher_hand"].ne(""), combined["_pitcher_hand"].fillna("")
        )
        combined.drop(columns=["_batter_hand", "_pitcher_hand"], inplace=True)
        summary = handed.summary()
        print(
            f"Loaded handedness reference: {summary['total']} players from {args.handedness_csv}"
        )

    if args.recent_stats_csv:
        stats_loader = RecentStatsLoader(Path(args.recent_stats_csv))
        stats = stats_loader.load(alias_map=alias_map)
        combined = combined.merge(
            stats.entries,
            on=["team_code", "canonical_name"],
            how="left",
        )
        combined["recent_last7_fppg"] = pd.to_numeric(
            combined["recent_last7_fppg"], errors="coerce"
        ).fillna(0.0)
        combined["recent_last14_fppg"] = pd.to_numeric(
            combined["recent_last14_fppg"], errors="coerce"
        ).fillna(0.0)
        combined["recent_season_fppg"] = pd.to_numeric(
            combined["recent_season_fppg"], errors="coerce"
        ).fillna(0.0)
        summary = stats.summary()
        print(
            f"Loaded recent stats for {summary['total']} players from {args.recent_stats_csv}"
        )

    if vegas_lines is not None:
        vegas_totals = vegas_lines.team_totals.drop_duplicates(
            subset=["team_code", "opponent_code"]
        )
        combined = combined.merge(
            vegas_totals,
            on=["team_code", "opponent_code"],
            how="left",
        )

    db = SlateDatabase(Path(args.db_path))
    slate = db.insert_slate(tag, Path(args.fanduel_csv), Path(args.bpp_source))
    db.write_players(slate.slate_id, combined)
    db.close()
    print(f"Persisted slate #{slate.slate_id} ({tag}) with diagnostics: {diagnostics.as_dict()}")

    projections = compute_baseline_projections(
        combined,
        recency_blend=recency_blend,
        platoon_opposite_boost=platoon_opposite_boost,
        platoon_same_penalty=platoon_same_penalty,
        platoon_switch_boost=platoon_switch_boost,
    )

    ownership_result = compute_ownership_series(
        combined,
        projections,
        source_paths=ownership_paths,
        weights=ownership_weights,
    )
    ownership_map = ownership_result.ownership.to_dict()
    projections["proj_fd_ownership"] = (
        projections["fd_player_id"].astype(str).map(ownership_map).fillna(0.0)
    )
    if ownership_result.source_count:
        total_players = len(ownership_result.ownership)
        print(
            f"Blended ownership from {ownership_result.source_count} source(s); coverage {ownership_result.covered_players}/{total_players} players."
        )
    else:
        print(
            "Using fallback ownership estimator (no external ownership sources provided)."
        )
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

    bring_back_enabled = config.bring_back_enabled or args.bring_back
    bring_back_count = config.bring_back_count if config.bring_back_count else 1
    if args.bring_back_count is not None:
        bring_back_count = args.bring_back_count
    min_game_total = config.min_game_total_for_stacks if config.min_game_total_for_stacks is not None else args.min_game_total

    if args.write_intermediate:
        combined.to_csv(output_dir / f"{tag}_slate_players.csv", index=False)
        projections.to_csv(output_dir / f"{tag}_baseline_projections.csv", index=False)
        optimizer_df.to_csv(output_dir / f"{tag}_optimizer_dataset.csv", index=False)
        if vegas_lines is not None:
            vegas_lines.games.to_csv(output_dir / f"{tag}_vegas_lines.csv", index=False)

    try:
        lineups = generate_lineups(
            optimizer_df,
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
    except ValueError as exc:
        print(f"Optimizer failed: {exc}")
        return {"tag": tag, "output_dir": output_dir, "optimizer_df": optimizer_df, "lineups": [], "lineup_df": pd.DataFrame(), "salary_cap": salary_cap, "lineups_path": None, "upload_path": None}

    if not lineups:
        print("No feasible lineups generated.")
        return {
            "tag": tag,
            "output_dir": output_dir,
            "optimizer_df": optimizer_df,
            "lineups": [],
            "lineup_df": pd.DataFrame(),
            "salary_cap": salary_cap,
            "lineups_path": None,
            "upload_path": None,
        }

    print(f"Generated {len(lineups)} lineups. Top lineup proj={lineups[0].total_projection:.2f}")
    _print_exposure_summary(lineups)
    _print_stack_summary(lineups)

    combined_lineups = _combine_lineups(lineups)
    lineups_path = output_dir / f"{tag}_lineups.csv"
    combined_lineups.to_csv(lineups_path, index=False)
    print(f"Saved lineup breakdown to {lineups_path}")

    upload_path = output_dir / f"{tag}_fanduel_upload.csv"
    write_fanduel_upload(lineups, upload_path)
    print(f"Saved FanDuel upload file to {upload_path}")

    return {
        "tag": tag,
        "output_dir": output_dir,
        "optimizer_df": optimizer_df,
        "lineups": lineups,
        "lineup_df": combined_lineups,
        "salary_cap": salary_cap,
        "lineups_path": lineups_path,
        "upload_path": upload_path,
    }


def _run_auto_fetch(args: argparse.Namespace) -> None:
    """Run fetch_live_data.py logic inline if --auto-fetch is set."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from datetime import date as _date
    from slate_optimizer.ingestion.mlb_api import fetch_mlb_lineups
    from slate_optimizer.ingestion.odds_api import fetch_vegas_lines

    live_dir = Path(args.live_data_dir)
    live_dir.mkdir(parents=True, exist_ok=True)
    d = _date.today().strftime("%Y-%m-%d")

    print("[auto-fetch] Fetching batting orders from MLB Stats API...")
    batting_df, pitchers_df = fetch_mlb_lineups(date_str=d)

    orders_path = live_dir / f"batting_orders_{d}.csv"
    if not batting_df.empty:
        batting_df[["team_code", "order_position", "player_name"]].rename(
            columns={"team_code": "team"}
        ).to_csv(orders_path, index=False)
        print(f"  {len(batting_df)} lineup slots across {batting_df['team_code'].nunique()} teams → {orders_path}")
        if not args.batting_orders_csv:
            args.batting_orders_csv = str(orders_path)
    else:
        print("  No confirmed lineups yet.")

    pitchers_path = live_dir / f"probable_pitchers_{d}.csv"
    if not pitchers_df.empty:
        out = pitchers_df.rename(columns={"team_code": "team", "pitcher_hand": "throws"})
        out["bats"] = ""
        out[["player_name", "team", "bats", "throws"]].to_csv(pitchers_path, index=False)
        print(f"  {len(pitchers_df)} probable pitchers → {pitchers_path}")
        if not args.handedness_csv:
            args.handedness_csv = str(pitchers_path)

    print("[auto-fetch] Fetching Vegas lines from Odds API...")
    vegas = fetch_vegas_lines()
    if vegas is None:
        print("  ODDS_API_KEY not set — skipping Vegas auto-fetch.")
    elif not vegas.games.empty:
        vegas_path = live_dir / f"vegas_lines_{d}.csv"
        save_cols = [c for c in ["game", "total", "home_ml", "away_ml"] if c in vegas.games.columns]
        vegas.games[save_cols].to_csv(vegas_path, index=False)
        print(f"  {len(vegas.games)} games → {vegas_path}")
        if not args.vegas_lines_csv:
            args.vegas_lines_csv = str(vegas_path)
    print()


def main() -> None:
    args = parse_args()
    if getattr(args, "auto_fetch", False):
        _run_auto_fetch(args)
    run_pipeline(args)

if __name__ == "__main__":
    main()



