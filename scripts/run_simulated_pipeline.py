"""Run the daily pipeline followed by the Monte Carlo simulation stack."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.optimizer.export import lineups_to_fanduel_upload
from slate_optimizer.simulation import (
    FieldQualityMix,
    SimulationConfig,
    build_correlation_matrix,
    fit_player_distributions,
    select_portfolio,
    simulate_contest,
    simulate_field,
    simulate_slate,
)

import run_daily_pipeline as daily_pipeline


def build_parser() -> argparse.ArgumentParser:
    parent = daily_pipeline.build_parser(add_help=False)
    parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[parent],
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=500,
        help="Number of relaxed candidate lineups to simulate.",
    )
    parser.add_argument("--sim-config", default=None, help="Simulation config JSON path.")
    parser.add_argument("--num-sims", type=int, default=None, help="Override number of Monte Carlo iterations.")
    parser.add_argument("--field-size", type=int, default=None, help="Override simulated opponent lineup count.")
    parser.add_argument(
        "--selection-metric",
        default=None,
        choices=["top_1pct_rate", "win_rate", "cash_rate", "expected_roi"],
        help="Portfolio selection metric override.",
    )
    parser.add_argument("--diversity-weight", type=float, default=None, help="Override diversity penalty weight.")
    parser.add_argument("--max-player-exposure", type=float, default=None, help="Override max player exposure.")
    parser.add_argument("--volatility-scale", type=float, default=None, help="Override volatility scale.")
    parser.add_argument("--copula-nu", type=int, default=None, help="Override Student-t degrees of freedom.")
    parser.add_argument(
        "--teammate-corr",
        type=float,
        default=None,
        help="Override base teammate correlation.",
    )
    parser.add_argument(
        "--pitcher-vs-opposing",
        type=float,
        default=None,
        help="Override pitcher vs opposing hitter correlation.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--sim-report",
        default=None,
        help="Path for per-lineup simulation metrics CSV (defaults to <tag>_sim_results.csv).",
    )
    parser.add_argument(
        "--portfolio-report",
        default=None,
        help="Path for selected portfolio summary CSV (defaults to <tag>_sim_portfolio.csv).",
    )
    parser.add_argument(
        "--sim-upload",
        default=None,
        help="Path for FanDuel upload CSV built from the simulated portfolio (defaults to <tag>_simulated_upload.csv).",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


def _apply_sim_overrides(args: argparse.Namespace, config: SimulationConfig) -> SimulationConfig:
    if args.num_sims is not None:
        config.num_simulations = int(args.num_sims)
    if args.field_size is not None:
        config.num_field_lineups = int(args.field_size)
    if args.selection_metric:
        config.selection_metric = args.selection_metric
    if args.diversity_weight is not None:
        config.diversity_weight = float(args.diversity_weight)
    if args.max_player_exposure is not None:
        config.max_batter_exposure = float(args.max_player_exposure)
        config.max_pitcher_exposure = float(args.max_player_exposure)
    if args.volatility_scale is not None:
        config.volatility_scale = float(args.volatility_scale)
    if args.copula_nu is not None:
        config.correlation.copula_nu = int(args.copula_nu)
    if args.teammate_corr is not None:
        config.correlation.teammate_base = float(args.teammate_corr)
    if args.pitcher_vs_opposing is not None:
        config.correlation.pitcher_vs_opposing = float(args.pitcher_vs_opposing)
    if args.seed is not None:
        config.seed = int(args.seed)
    return config


def _selected_lineups(lineups, selected_ids) -> list:
    if not selected_ids:
        return []
    selections = []
    for lineup_id in sorted(set(int(idx) for idx in selected_ids)):
        if 0 <= lineup_id < len(lineups):
            selections.append(lineups[lineup_id])
    return selections


def main() -> None:
    args = parse_args()
    if args.num_candidates < args.num_lineups:
        raise SystemExit("--num-candidates must be >= --num-lineups for simulation selection.")

    pipeline_namespace = argparse.Namespace(**vars(args))
    pipeline_namespace.num_lineups = args.num_candidates
    pipeline_result = daily_pipeline.run_pipeline(pipeline_namespace)
    optimizer_df: pd.DataFrame = pipeline_result.get("optimizer_df", pd.DataFrame())
    lineups = pipeline_result.get("lineups", [])
    if not lineups:
        raise SystemExit("Pipeline did not generate any lineups; cannot run simulation.")

    sim_config = (
        SimulationConfig.load(Path(args.sim_config)) if args.sim_config else SimulationConfig()
    )
    sim_config.num_candidates = len(lineups)
    sim_config = _apply_sim_overrides(args, sim_config)

    distributions = fit_player_distributions(optimizer_df, sim_config.volatility_scale)
    correlation_model = build_correlation_matrix(optimizer_df, sim_config.correlation)
    slate_sim = simulate_slate(
        distributions,
        correlation_model,
        num_simulations=sim_config.num_simulations,
        seed=sim_config.seed,
        use_antithetic=sim_config.use_antithetic,
    )
    quality_mix = FieldQualityMix(
        shark_pct=sim_config.field_quality_shark_pct,
        rec_pct=sim_config.field_quality_rec_pct,
        random_pct=sim_config.field_quality_random_pct,
    )
    salary_cap = pipeline_result.get("salary_cap", 35000) or 35000
    field_sim = simulate_field(
        optimizer_df,
        num_opponent_lineups=sim_config.num_field_lineups,
        salary_cap=int(salary_cap),
        seed=sim_config.seed,
        position_constraints=True,
        quality_mix=quality_mix,
    )

    contest_result = simulate_contest(
        lineups,
        slate_sim,
        field_sim,
        entry_fee=sim_config.entry_fee,
        payout_structure=sim_config.payout_structure,
    )
    contest_df = contest_result.to_dataframe().sort_values(
        sim_config.selection_metric,
        ascending=False,
    )
    final_count = int(args.num_lineups)
    # max_stack_exposure: no single team as primary stack in >35% of lineups
    # Prevents the whole portfolio being one team's stack
    _max_stack_exp = getattr(sim_config, "max_stack_exposure", 0.35)
    portfolio = select_portfolio(
        contest_result,
        num_lineups=final_count,
        selection_metric=sim_config.selection_metric,
        max_overlap=sim_config.max_overlap,
        max_batter_exposure=sim_config.max_batter_exposure,
        max_pitcher_exposure=sim_config.max_pitcher_exposure,
        diversity_weight=sim_config.diversity_weight,
        max_stack_exposure=_max_stack_exp,
    )
    portfolio_df = portfolio.to_dataframe()
    selected_ids = [res.lineup_id for res in portfolio.selected]
    selected_lineups = _selected_lineups(lineups, selected_ids)

    tag = pipeline_result.get("tag", "slate")
    output_dir = Path(pipeline_result.get("output_dir", Path(args.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_report_path = Path(args.sim_report) if args.sim_report else output_dir / f"{tag}_sim_results.csv"
    portfolio_report_path = (
        Path(args.portfolio_report)
        if args.portfolio_report
        else output_dir / f"{tag}_sim_portfolio.csv"
    )
    upload_path = Path(args.sim_upload) if args.sim_upload else output_dir / f"{tag}_simulated_upload.csv"

    contest_df.to_csv(sim_report_path, index=False)
    print(f"Wrote simulation metrics to {sim_report_path}")
    if not portfolio_df.empty:
        portfolio_df.to_csv(portfolio_report_path, index=False)
        print(f"Wrote portfolio summary to {portfolio_report_path}")
    else:
        print("Portfolio selection did not return any lineups.")

    if selected_lineups:
        fan_duel_df = lineups_to_fanduel_upload(selected_lineups)
        fan_duel_df.to_csv(upload_path, index=False)
        print(f"Wrote simulated FanDuel upload to {upload_path}")
    else:
        print("No lineups available for FanDuel upload from the simulated portfolio.")

    print(
        "Portfolio win rate: "
        f"{portfolio.portfolio_win_rate:.2%}, top1: {portfolio.portfolio_top1pct_rate:.2%}, "
        f"cash: {portfolio.portfolio_cash_rate:.2%}, expected ROI: {portfolio.portfolio_expected_roi:.2f}x"
    )


if __name__ == "__main__":
    main()
