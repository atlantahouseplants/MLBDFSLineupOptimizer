from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.analysis.exposure_tuner import build_exposure_recommendations, stack_cap_maps


class TestExposureTuner(unittest.TestCase):
    def test_positive_leverage_stack_player_gets_more_room_than_chalk(self) -> None:
        optimizer = pd.DataFrame(
            [
                {
                    "fd_player_id": "A1",
                    "full_name": "Leverage Hitter",
                    "team_code": "AAA",
                    "position": "OF",
                    "player_type": "batter",
                    "proj_fd_mean": 12,
                    "proj_fd_ownership": 0.06,
                    "player_leverage_score": 0.25,
                    "proj_fd_bust_rate": 0.10,
                },
                {
                    "fd_player_id": "B1",
                    "full_name": "Chalk Hitter",
                    "team_code": "BBB",
                    "position": "OF",
                    "player_type": "batter",
                    "proj_fd_mean": 12,
                    "proj_fd_ownership": 0.30,
                    "player_leverage_score": -0.10,
                    "proj_fd_bust_rate": 0.10,
                },
            ]
        )
        stack_plan = pd.DataFrame(
            [
                {"team_code": "AAA", "target_exposure": 0.25, "recommended_max_exposure": 0.35},
                {"team_code": "BBB", "target_exposure": 0.08, "recommended_max_exposure": 0.15},
            ]
        )

        recs = build_exposure_recommendations(optimizer, stack_plan).set_index("full_name")
        mins, maxs = stack_cap_maps(stack_plan)

        self.assertGreater(
            recs.loc["Leverage Hitter", "recommended_max_exposure"],
            recs.loc["Chalk Hitter", "recommended_max_exposure"],
        )
        self.assertEqual(maxs["AAA"], 0.35)
        self.assertEqual(mins, {})


if __name__ == "__main__":
    unittest.main()
