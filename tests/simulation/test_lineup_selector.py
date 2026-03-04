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


if __name__ == "__main__":
    unittest.main()
