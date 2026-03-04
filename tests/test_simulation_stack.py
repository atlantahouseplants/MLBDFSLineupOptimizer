from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.optimizer.solver import LineupResult
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


POSITIONS = [
    "C/1B",
    "2B",
    "3B",
    "SS",
    "OF",
    "OF",
    "OF",
    "OF",
    "1B",
    "2B",
]


def _build_optimizer_df() -> pd.DataFrame:
    rows = []
    rows.append(
        {
            "fd_player_id": "P_AAA",
            "full_name": "Ace Alpha",
            "player_type": "pitcher",
            "position": "P",
            "team_code": "AAA",
            "opponent_code": "BBB",
            "game_key": "AAA@BBB",
            "stack_key": "AAA",
            "stack_priority": "high",
            "proj_fd_mean": 35.0,
            "proj_fd_floor": 20.0,
            "proj_fd_ceiling": 55.0,
            "salary": 9300,
            "proj_fd_ownership": 25.0,
            "player_leverage_score": 2.5,
            "batting_order_position": pd.NA,
            "vegas_game_total": 8.5,
        }
    )
    rows.append(
        {
            "fd_player_id": "P_BBB",
            "full_name": "Ace Bravo",
            "player_type": "pitcher",
            "position": "P",
            "team_code": "BBB",
            "opponent_code": "AAA",
            "game_key": "AAA@BBB",
            "stack_key": "BBB",
            "stack_priority": "high",
            "proj_fd_mean": 33.0,
            "proj_fd_floor": 18.0,
            "proj_fd_ceiling": 50.0,
            "salary": 8800,
            "proj_fd_ownership": 20.0,
            "player_leverage_score": 2.1,
            "batting_order_position": pd.NA,
            "vegas_game_total": 8.5,
        }
    )
    hitter_id = 0
    for team, opponent in (("AAA", "BBB"), ("BBB", "AAA")):
        for order in range(1, 7):
            rows.append(
                {
                    "fd_player_id": f"H_{team}_{order}",
                    "full_name": f"{team} Batter {order}",
                    "player_type": "batter",
                    "position": POSITIONS[hitter_id % len(POSITIONS)],
                    "team_code": team,
                    "opponent_code": opponent,
                    "game_key": "AAA@BBB",
                    "stack_key": team,
                    "stack_priority": "mid" if order > 2 else "high",
                    "proj_fd_mean": 12.0 + order,
                    "proj_fd_floor": 6.0 + order / 2,
                    "proj_fd_ceiling": 18.0 + order,
                    "salary": 2600 + order * 120,
                    "proj_fd_ownership": 10.0 + order,
                    "player_leverage_score": 1.2,
                    "batting_order_position": order,
                    "vegas_game_total": 8.5,
                }
            )
            hitter_id += 1
    return pd.DataFrame(rows)


def _build_candidate_lineups(df: pd.DataFrame) -> list[LineupResult]:
    pitchers = df[df["player_type"] == "pitcher"].reset_index(drop=True)
    hitters = df[df["player_type"] == "batter"].reset_index(drop=True)
    lineups: list[LineupResult] = []
    for offset in range(3):
        pitcher = pitchers.iloc[offset % len(pitchers)]
        hitter_block = hitters[offset : offset + 8]
        if len(hitter_block) < 8:
            extra = hitters.iloc[: 8 - len(hitter_block)]
            hitter_block = pd.concat([hitter_block, extra], ignore_index=True)
        lineup_df = pd.concat([pitcher.to_frame().T, hitter_block], ignore_index=True)
        lineup_df = lineup_df.reset_index(drop=True)
        total_salary = int(pd.to_numeric(lineup_df["salary"], errors="coerce").fillna(0).sum())
        total_projection = float(pd.to_numeric(lineup_df["proj_fd_mean"], errors="coerce").fillna(0).sum())
        lineups.append(
            LineupResult(
                dataframe=lineup_df,
                total_salary=total_salary,
                total_projection=total_projection,
            )
        )
    return lineups


class TestSimulationStack(unittest.TestCase):
    def test_simulation_stack_runs_end_to_end(self) -> None:
        dataset = _build_optimizer_df()
        lineups = _build_candidate_lineups(dataset)
        config = SimulationConfig(
            num_simulations=200,
            num_field_lineups=30,
            selection_metric="win_rate",
            diversity_weight=0.2,
            max_player_exposure=0.7,
        )
        distributions = fit_player_distributions(dataset, config.volatility_scale)
        correlation_model = build_correlation_matrix(dataset, config.correlation)
        slate_sim = simulate_slate(
            distributions,
            correlation_model,
            num_simulations=config.num_simulations,
            seed=42,
            use_antithetic=config.use_antithetic,
        )
        field_sim = simulate_field(
            dataset,
            num_opponent_lineups=config.num_field_lineups,
            salary_cap=35000,
            seed=42,
            position_constraints=False,
            quality_mix=FieldQualityMix(),
        )
        contest = simulate_contest(
            lineups,
            slate_sim,
            field_sim,
            entry_fee=config.entry_fee,
            payout_structure=config.payout_structure,
        )
        contest_df = contest.to_dataframe()
        self.assertFalse(contest_df.empty)
        self.assertTrue({"lineup_id", "mean_score", "win_rate", "expected_roi"}.issubset(contest_df.columns))

        portfolio = select_portfolio(
            contest,
            num_lineups=2,
            selection_metric=config.selection_metric,
            max_player_exposure=config.max_player_exposure,
            diversity_weight=config.diversity_weight,
        )
        self.assertGreater(portfolio.num_selected, 0)
        self.assertGreaterEqual(portfolio.portfolio_win_rate, 0.0)
        self.assertLessEqual(portfolio.portfolio_win_rate, 1.0)
        portfolio_df = portfolio.to_dataframe()
        self.assertFalse(portfolio_df.empty)
        self.assertEqual(portfolio_df.shape[0], portfolio.num_selected)


if __name__ == "__main__":
    unittest.main()
