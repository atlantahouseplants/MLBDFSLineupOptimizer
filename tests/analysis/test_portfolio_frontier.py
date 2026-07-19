from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.analysis.portfolio_frontier import build_portfolio_frontier_report


class TestPortfolioFrontier(unittest.TestCase):
    def test_builds_multiple_frontier_profiles(self) -> None:
        contest = pd.DataFrame(
            [
                {
                    "lineup_id": i,
                    "payout_ev": 1.0 + i * 0.01,
                    "expected_roi": 0.1 + i * 0.01,
                    "top_1pct_rate": 0.02 + i * 0.001,
                    "leverage_score": i * 0.1,
                    "total_ownership": 2.0 - i * 0.05,
                    "duplication_score": 0.10 + i * 0.01,
                    "field_duplication_rate": 0.08 + i * 0.01,
                    "uniqueness_score": 0.9 - i * 0.02,
                    "ownership_scenario_score": 0.7 + i * 0.01,
                }
                for i in range(10)
            ]
        )

        report = build_portfolio_frontier_report(contest, target_lineups=4)

        self.assertEqual(len(report.summary), 5)
        self.assertIn("Balanced GPP", set(report.summary["profile"]))
        self.assertTrue(report.recommendation)


if __name__ == "__main__":
    unittest.main()
