from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.analysis.portfolio_stress import build_portfolio_stress_report


class TestPortfolioStress(unittest.TestCase):
    def test_reports_chalk_stack_stress(self) -> None:
        lineup_rows = []
        for lineup_id in range(1, 4):
            for idx in range(8):
                lineup_rows.append(
                    {
                        "lineup_id": lineup_id,
                        "full_name": f"AAA H{idx}" if lineup_id <= 2 and idx < 4 else f"BBB H{lineup_id}{idx}",
                        "team_code": "AAA" if lineup_id <= 2 and idx < 4 else "BBB",
                        "player_type": "batter",
                    }
                )
            lineup_rows.append({"lineup_id": lineup_id, "full_name": "Pitcher A", "team_code": "CCC", "player_type": "pitcher"})
        optimizer = pd.DataFrame(
            [{"full_name": f"AAA H{i}", "team_code": "AAA", "player_type": "batter", "proj_fd_ownership": 30.0} for i in range(8)]
            + [{"full_name": "Pitcher A", "team_code": "CCC", "player_type": "pitcher", "proj_fd_ownership": 45.0}]
        )
        report = build_portfolio_stress_report(pd.DataFrame(lineup_rows), optimizer)

        self.assertFalse(report.scenarios.empty)
        self.assertEqual(report.summary["top_chalk_stack"], "AAA")
        self.assertIn("Projection -5%", set(report.scenarios["scenario"]))
        self.assertIn("Ownership +20%", set(report.scenarios["scenario"]))
        self.assertTrue(report.recommendations)


if __name__ == "__main__":
    unittest.main()
