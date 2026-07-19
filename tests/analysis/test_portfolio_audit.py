from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.analysis.portfolio_audit import build_portfolio_audit


class TestPortfolioAudit(unittest.TestCase):
    def test_audit_flags_pitcher_conflict_and_stack_overage(self) -> None:
        lineup_rows = [
            {
                "lineup_id": 1,
                "fd_player_id": "P1",
                "full_name": "Pitcher One",
                "team_code": "AAA",
                "opponent_code": "BBB",
                "position": "P",
                "player_type": "pitcher",
                "salary": 9000,
                "proj_fd_mean": 35,
                "proj_fd_ownership": 0.38,
                "player_leverage_score": -0.1,
                "is_confirmed_lineup": True,
            }
        ]
        for i in range(8):
            team = "BBB" if i < 4 else "CCC"
            lineup_rows.append(
                {
                    "lineup_id": 1,
                    "fd_player_id": f"H{i}",
                    "full_name": f"Hitter {i}",
                    "team_code": team,
                    "opponent_code": "AAA",
                    "position": "OF",
                    "player_type": "batter",
                    "salary": 3000,
                    "proj_fd_mean": 9,
                    "proj_fd_ownership": 0.10,
                    "player_leverage_score": 0.1,
                    "is_confirmed_lineup": i >= 2,
                }
            )
        stack_plan = pd.DataFrame(
            [
                {
                    "team_code": "BBB",
                    "target_exposure": 0.20,
                    "recommended_max_exposure": 0.30,
                    "stack_score": 0.8,
                    "tier": "Neutral",
                }
            ]
        )

        report = build_portfolio_audit(pd.DataFrame(lineup_rows), stack_plan=stack_plan)
        categories = set(report.issues["category"])

        self.assertIn("Pitcher conflicts", categories)
        self.assertIn("Stack exposure", categories)
        self.assertEqual(report.summary["lineups_with_unconfirmed_hitters"], 1)
        self.assertEqual(int(report.lineup_checks.loc[0, "hitters_vs_pitcher"]), 4)


if __name__ == "__main__":
    unittest.main()
