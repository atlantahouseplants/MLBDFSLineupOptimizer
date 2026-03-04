"""Run the Monte Carlo simulation stack on optimizer outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.optimizer import LineupResult, generate_lineups
from slate_optimizer.simulation import (
    CorrelationConfig,
    SimulationConfig,
    build_correlation_matrix,
    fit_player_distributions,
    simulate_slate,
)
from slate_optimizer.simulation.contest_simulator import simulate_contest
from slate_optimizer.simulation.field_simulator import FieldQualityMix, simulate_field
from slate_optimizer.simulation.lineup_selector import select_portfolio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Optimizer dataset CSV path.")
    parser.add_argument(
        "--candidates",
        default=None,
        help="Optional candidate lineups CSV (output from run_optimizer).",
    )
    parser.add_argument("--num-candidates", type=int, default=500, help="Lineups to generate if --candidates missing.")
    parser.add_argument("--num-select", type=int, default=20, help="Final lineups to select.")
    parser.add_argument("--num-sims", type=int, default=None, help="Override number of Monte Carlo simulations.")
    parser.add_argument("--num-field", type=int, default=None, help="Override opponent lineup count.")
    parser.add_argument("--metric", default=None, help="Selection metric (top_1pct_rate, win_rate, cash_rate, expected_roi).")
    parser.add_argument("--output", default=None, help="Optional CSV to write per-lineup simulation results.")
    parser.add_argument("--portfolio-output", default=None, help="Optional CSV for portfolio summary (selected lineups).")
    parser.add_argument("--config", default=None, help="Simulation config JSON.")
    parser.add_argument("--volatility-scale", type=float, default=None, help="Override volatility scale.")
    parser.add_argument("--copula-nu", type=int, default=None, help="Override Student-t nu.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser.parse_args()


def _load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "fd_player_id" not in df.columns:
        raise ValueError("Optimizer dataset missing fd_player_id column")
    return df


def _load_candidates_from_csv(path: Path) -> List[LineupResult]:
    df = pd.read_csv(path)
    if "lineup_id" not in df.columns:
        raise ValueError("Candidate CSV must include lineup_id column")
    lineups: List[LineupResult] = []
    for lineup_id, group in df.groupby("lineup_id"):
        lineup_df = group.reset_index(drop=True)
        salary = int(pd.to_numeric(lineup_df.get("salary"), errors="coerce").fillna(0).sum())
        projection = float(pd.to_numeric(lineup_df.get("proj_fd_mean"), errors="coerce").fillna(0.0).sum())
        lineups.append(LineupResult(dataframe=lineup_df, total_salary=salary, total_projection=projection))
    return lineups


def _generate_candidates(dataset: pd.DataFrame, num_candidates: int) -> List[LineupResult]:
    print(f"Generating {num_candidates} candidate lineups from dataset...")
    lineups = generate_lineups(
        dataset,
        num_lineups=num_candidates,
        salary_cap=35000,
        min_stack_size=2,
        stack_player_types=("batter",),
        stack_templates=None,
        max_lineup_ownership=None,
        bring_back_enabled=False,
        bring_back_count=1,
        min_game_total_for_stacks=None,
    )
    if not lineups:
        raise RuntimeError("Failed to generate candidates; relax constraints and retry")
    return lineups


def _quality_mix(config: SimulationConfig) -> FieldQualityMix:
    return FieldQualityMix(
        shark_pct=config.field_quality_shark_pct,
        rec_pct=config.field_quality_rec_pct,
        random_pct=config.field_quality_random_pct,
    )


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    dataset = _load_dataset(dataset_path)

    config = SimulationConfig.load(Path(args.config)) if args.config else SimulationConfig()
    if args.num_sims is not None:
        config.num_simulations = args.num_sims
    if args.num_field is not None:
        config.num_field_lineups = args.num_field
    if args.metric:
        config.selection_metric = args.metric
    if args.volatility_scale is not None:
        config.volatility_scale = args.volatility_scale
    if args.copula_nu is not None:
        config.correlation.copula_nu = args.copula_nu
    if args.seed is not None:
        config.seed = args.seed

    if args.candidates:
        candidates = _load_candidates_from_csv(Path(args.candidates))
    else:
        candidates = _generate_candidates(dataset, args.num_candidates)

    print(f"Loaded {len(dataset)} players, {len(candidates)} candidate lineups")

    distributions = fit_player_distributions(dataset, volatility_scale=config.volatility_scale)
    correlation_model = build_correlation_matrix(dataset, config.correlation)
    slate_sim = simulate_slate(
        distributions,
        correlation_model,
        num_simulations=config.num_simulations,
        seed=config.seed,
        use_antithetic=config.use_antithetic,
    )
    print(f"Simulated slate with {slate_sim.num_simulations} runs")

    field_sim = simulate_field(
        dataset,
        num_opponent_lineups=config.num_field_lineups,
        salary_cap=35000,
        seed=config.seed,
        position_constraints=True,
        quality_mix=_quality_mix(config),
    )
    print(f"Simulated field with {field_sim.num_lineups} lineups")

    contest_result = simulate_contest(
        candidates,
        slate_sim,
        field_sim,
        entry_fee=config.entry_fee,
        payout_structure=config.payout_structure,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        contest_result.to_dataframe().to_csv(output_path, index=False)
        print(f"Wrote simulation results to {output_path}")

    selection_metric = config.selection_metric
    portfolio = select_portfolio(
        contest_result,
        num_lineups=args.num_select,
        selection_metric=selection_metric,
        max_overlap=config.max_overlap,
        max_player_exposure=config.max_player_exposure,
        diversity_weight=config.diversity_weight,
    )

    print(f"Selected {portfolio.num_selected} lineups using metric={selection_metric}")
    print(f"Portfolio win rate: {portfolio.portfolio_win_rate:.2%}, top1%: {portfolio.portfolio_top1pct_rate:.2%}, cash: {portfolio.portfolio_cash_rate:.2%}")
    print(f"Expected ROI (sum across entries): {portfolio.portfolio_expected_roi:.2f}x")

    if args.portfolio_output and portfolio.selected:
        portfolio_path = Path(args.portfolio_output)
        portfolio_path.parent.mkdir(parents=True, exist_ok=True)
        portfolio.to_dataframe().to_csv(portfolio_path, index=False)
        print(f"Wrote selected lineup summary to {portfolio_path}")


if __name__ == "__main__":
    main()
