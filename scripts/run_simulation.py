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
from slate_optimizer.analysis.portfolio_audit import audit_report_to_csv, build_portfolio_audit
from slate_optimizer.analysis.exposure_tuner import stack_cap_maps
from slate_optimizer.analysis.stack_exposure import build_stack_exposure_plan, stack_plan_weights
from slate_optimizer.simulation import (
    CorrelationConfig,
    SimulationConfig,
    build_correlation_matrix,
    fit_player_distributions,
    parse_payout_ladder_text,
    payout_ladder_to_dicts,
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
    parser.add_argument("--contest-entries", type=int, default=None, help="Actual contest entry count for rank-based payout EV.")
    parser.add_argument("--entry-fee", type=float, default=None, help="Contest entry fee.")
    parser.add_argument("--payout-ladder", default=None, help="Path to rank,payout ladder text/CSV for contest EV.")
    parser.add_argument("--metric", default=None, help="Selection metric (top_1pct_rate, win_rate, cash_rate, expected_roi).")
    parser.add_argument("--output", default=None, help="Optional CSV to write per-lineup simulation results.")
    parser.add_argument("--portfolio-output", default=None, help="Optional CSV for portfolio summary (selected lineups).")
    parser.add_argument("--stack-plan-output", default=None, help="Optional CSV for team stack exposure recommendations.")
    parser.add_argument("--audit-output", default=None, help="Optional CSV-like text report for final portfolio audit.")
    parser.add_argument("--config", default=None, help="Simulation config JSON.")
    parser.add_argument("--volatility-scale", type=float, default=None, help="Override volatility scale.")
    parser.add_argument("--copula-nu", type=int, default=None, help="Override Student-t nu.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--no-portfolio-optimizer", action="store_true", help="Use the legacy greedy final selector.")
    parser.add_argument("--no-stack-exposure-engine", action="store_true", help="Do not use team stack recommendations during final selection.")
    parser.add_argument("--no-stack-auto-caps", action="store_true", help="Do not apply stack plan team max caps during final selection.")
    parser.add_argument("--stack-auto-mins", action="store_true", help="Apply stack plan team minimums during final selection.")
    parser.add_argument("--stack-target-weight", type=float, default=None, help="Override stack exposure plan weight.")
    parser.add_argument("--portfolio-time-limit", type=int, default=None, help="ILP portfolio optimizer time limit in seconds.")
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
    if args.contest_entries is not None:
        config.contest_entries = int(args.contest_entries)
    if args.entry_fee is not None:
        config.entry_fee = float(args.entry_fee)
    if args.payout_ladder:
        text = Path(args.payout_ladder).read_text(encoding="utf-8")
        bands = parse_payout_ladder_text(text)
        config.payout_structure = payout_ladder_to_dicts(bands) if bands else None
    if args.volatility_scale is not None:
        config.volatility_scale = args.volatility_scale
    if args.copula_nu is not None:
        config.correlation.copula_nu = args.copula_nu
    if args.seed is not None:
        config.seed = args.seed
    if args.no_portfolio_optimizer:
        config.use_portfolio_optimizer = False
    if args.no_stack_exposure_engine:
        config.use_stack_exposure_engine = False
    if args.no_stack_auto_caps:
        config.use_stack_auto_caps = False
    if args.stack_auto_mins:
        config.use_stack_auto_mins = True
    if args.stack_target_weight is not None:
        config.stack_target_weight = float(args.stack_target_weight)
    if args.portfolio_time_limit is not None:
        config.portfolio_time_limit_seconds = int(args.portfolio_time_limit)

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
        use_stratified=config.use_stratified,
        num_strata=config.num_strata,
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
        contest_entries=config.contest_entries,
    )

    stack_plan = build_stack_exposure_plan(dataset)
    stack_min_caps, stack_max_caps = stack_cap_maps(stack_plan)
    if args.stack_plan_output:
        stack_plan_path = Path(args.stack_plan_output)
        stack_plan_path.parent.mkdir(parents=True, exist_ok=True)
        stack_plan.to_csv(stack_plan_path, index=False)
        print(f"Wrote stack exposure plan to {stack_plan_path}")

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
        max_batter_exposure=config.max_batter_exposure,
        max_pitcher_exposure=config.max_pitcher_exposure,
        pitcher_ids=set(
            dataset.loc[
                dataset["player_type"].astype(str).str.lower() == "pitcher",
                "fd_player_id",
            ].astype(str)
        ),
        diversity_weight=config.diversity_weight,
        selection_leverage_weight=config.selection_leverage_weight,
        selection_ownership_weight=config.selection_ownership_weight,
        selection_duplication_weight=config.selection_duplication_weight,
        stack_team_weights=stack_plan_weights(stack_plan) if config.use_stack_exposure_engine else None,
        stack_target_weight=config.stack_target_weight,
        stack_team_min_exposures=stack_min_caps if config.use_stack_auto_mins else None,
        stack_team_max_exposures=stack_max_caps if config.use_stack_auto_caps else None,
        use_portfolio_optimizer=config.use_portfolio_optimizer,
        portfolio_time_limit_seconds=config.portfolio_time_limit_seconds,
    )

    print(f"Selected {portfolio.num_selected} lineups using metric={selection_metric}")
    print(f"Portfolio win rate: {portfolio.portfolio_win_rate:.2%}, top1%: {portfolio.portfolio_top1pct_rate:.2%}, cash: {portfolio.portfolio_cash_rate:.2%}")
    print(f"Expected ROI (sum across entries): {portfolio.portfolio_expected_roi:.2f}x")

    if args.portfolio_output and portfolio.selected:
        portfolio_path = Path(args.portfolio_output)
        portfolio_path.parent.mkdir(parents=True, exist_ok=True)
        portfolio.to_dataframe().to_csv(portfolio_path, index=False)
        print(f"Wrote selected lineup summary to {portfolio_path}")
    if args.audit_output and portfolio.selected:
        selected_rows = []
        for idx, lineup in enumerate(portfolio.selected, start=1):
            temp = lineup.dataframe.copy()
            temp.insert(0, "lineup_id", idx)
            selected_rows.append(temp)
        selected_df = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
        audit = build_portfolio_audit(selected_df, optimizer_df=dataset, stack_plan=stack_plan, salary_cap=35000)
        audit_path = Path(args.audit_output)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(audit_report_to_csv(audit), encoding="utf-8")
        print(f"Wrote final portfolio audit to {audit_path}")


if __name__ == "__main__":
    main()
