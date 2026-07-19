from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.analysis.lineup_scorecard import build_lineup_scorecard


class TestLineupScorecard(unittest.TestCase):
    def test_scorecard_explains_lineup_components(self) -> None:
        rows = []
        for lineup_id in [1, 2]:
            for idx in range(9):
                rows.append(
                    {
                        "lineup_id": lineup_id,
                        "fd_player_id": f"{lineup_id}-{idx}",
                        "team_code": "AAA" if idx < 4 else "BBB",
                        "player_type": "pitcher" if idx == 8 else "batter",
                        "salary": 3000,
                        "proj_fd_mean": 10 + lineup_id + idx,
                        "proj_fd_ownership": 0.10,
                        "player_leverage_score": 0.05 * lineup_id,
                    }
                )
        portfolio = pd.DataFrame(
            [
                {"lineup_id": 0, "payout_ev": 1.2, "duplication_score": 0.05, "uniqueness_score": 0.9, "ownership_scenario_score": 0.8},
                {"lineup_id": 1, "payout_ev": 1.0, "duplication_score": 0.10, "uniqueness_score": 0.7, "ownership_scenario_score": 0.6},
            ]
        )
        stack_plan = pd.DataFrame(
            [{"team_code": "AAA", "stack_score": 0.8, "sim_upside_score": 0.7, "stack_leverage_score": 0.6, "p_runs_ge_6": 0.25}]
        )

        scorecard = build_lineup_scorecard(pd.DataFrame(rows), portfolio, stack_plan)

        self.assertEqual(set(scorecard["lineup_id"]), {1, 2})
        self.assertIn("why_selected_score", scorecard.columns)
        self.assertIn("why_selected", scorecard.columns)
        self.assertTrue((scorecard["why_selected_score"] > 0).all())


if __name__ == "__main__":
    unittest.main()
