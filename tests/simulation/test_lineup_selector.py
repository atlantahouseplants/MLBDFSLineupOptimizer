from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import unittest

import numpy as np

from slate_optimizer.simulation import PortfolioSelection
from slate_optimizer.simulation.lineup_selector import select_portfolio
from slate_optimizer.simulation.contest_simulator import LineupSimResult, ContestSimResult


class TestLineupSelector(unittest.TestCase):
    def _contest_result(self) -> ContestSimResult:
        lineup_results = []
        rng = np.random.default_rng(0)
        for lineup_id in range(5):
            lineup_results.append(
                LineupSimResult(
                    lineup_id=lineup_id,
                    player_ids=[f"P{lineup_id}_{i}" for i in range(9)],
                    mean_score=110 + lineup_id,
                    median_score=105 + lineup_id,
                    std_score=10,
                    p10_score=80,
                    p25_score=95,
                    p75_score=120,
                    p90_score=130,
                    p99_score=150,
                    max_score=160,
                    win_rate=0.02 * (lineup_id + 1),
                    top_1pct_rate=0.05 * (lineup_id + 1),
                    top_10pct_rate=0.1 * (lineup_id + 1),
                    cash_rate=0.3,
                    expected_roi=0.5 * (lineup_id + 1),
                    total_ownership=50 + lineup_id,
                    leverage_score=1.0,
                    field_duplication_rate=rng.uniform(0, 0.1),
                    stack_teams=["TEAM_A"] if lineup_id % 2 == 0 else ["TEAM_B"],
                )
            )
        return ContestSimResult(
            lineup_results=lineup_results,
            num_simulations=100,
            num_field_lineups=500,
            num_candidates=len(lineup_results),
            entry_fee=20.0,
        )

    def test_selector_respects_limits(self) -> None:
        contest = self._contest_result()
        portfolio = select_portfolio(
            contest,
            num_lineups=3,
            max_player_exposure=0.5,
            max_overlap=4,
            diversity_weight=0.3,
            selection_metric="top_1pct_rate",
        )
        self.assertIsInstance(portfolio, PortfolioSelection)
        self.assertLessEqual(portfolio.num_selected, 3)
        self.assertTrue(0.0 <= portfolio.portfolio_win_rate <= 1.0)

    def test_ilp_selector_returns_full_feasible_portfolio(self) -> None:
        contest = self._contest_result()
        portfolio = select_portfolio(
            contest,
            num_lineups=4,
            max_player_exposure=1.0,
            max_overlap=8,
            selection_metric="top_1pct_rate",
            use_portfolio_optimizer=True,
        )
        self.assertEqual(portfolio.num_selected, 4)
        self.assertGreaterEqual(portfolio.portfolio_top1pct_rate, 0.0)

    def test_stack_team_weights_can_break_close_ties(self) -> None:
        contest = self._contest_result()
        for lineup in contest.lineup_results:
            lineup.top_1pct_rate = 0.10
            lineup.win_rate = 0.01
            lineup.expected_roi = 0.0
            lineup.leverage_score = 0.0
            lineup.total_ownership = 0.10
            lineup.field_duplication_rate = 0.0
        portfolio = select_portfolio(
            contest,
            num_lineups=1,
            max_player_exposure=1.0,
            max_overlap=9,
            selection_metric="top_1pct_rate",
            stack_team_weights={"TEAM_B": 1.0, "TEAM_A": 0.0},
            stack_target_weight=1.0,
            use_portfolio_optimizer=True,
        )

        self.assertEqual(portfolio.num_selected, 1)
        self.assertEqual(portfolio.selected[0].stack_teams, ["TEAM_B"])

    def test_stack_team_caps_limit_team_exposure(self) -> None:
        contest = self._contest_result()
        portfolio = select_portfolio(
            contest,
            num_lineups=4,
            max_player_exposure=1.0,
            max_overlap=9,
            selection_metric="top_1pct_rate",
            stack_team_max_exposures={"TEAM_A": 0.25},
            use_portfolio_optimizer=True,
        )

        team_a_count = sum(1 for lineup in portfolio.selected if "TEAM_A" in lineup.stack_teams)
        self.assertLessEqual(team_a_count, 1)

    def test_marginal_value_weight_prefers_new_paths(self) -> None:
        contest = self._contest_result()
        contest.lineup_results[0].player_ids = [f"A{i}" for i in range(9)]
        contest.lineup_results[1].player_ids = [f"A{i}" for i in range(8)] + ["B1"]
        contest.lineup_results[2].player_ids = [f"C{i}" for i in range(9)]
        for lineup in contest.lineup_results:
            lineup.top_1pct_rate = 0.10
            lineup.expected_roi = 0.0
            lineup.field_duplication_rate = 0.0
            lineup.duplication_score = 0.0
            lineup.ownership_scenario_score = 1.0
        portfolio = select_portfolio(
            contest,
            num_lineups=2,
            max_player_exposure=1.0,
            max_overlap=9,
            selection_metric="top_1pct_rate",
            use_portfolio_optimizer=False,
            selection_marginal_value_weight=1.0,
        )

        selected_ids = {lineup.lineup_id for lineup in portfolio.selected}
        self.assertIn(0, selected_ids)
        self.assertNotIn(1, selected_ids)


if __name__ == "__main__":
    unittest.main()
