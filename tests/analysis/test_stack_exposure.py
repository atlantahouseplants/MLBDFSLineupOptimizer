from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.analysis.stack_exposure import build_stack_exposure_plan, stack_plan_weights


class TestStackExposure(unittest.TestCase):
    def test_plan_rewards_low_owned_sim_upside(self) -> None:
        rows = []
        for team, runs, ownership, p6, hr2 in [
            ("AAA", 5.8, 0.06, 0.34, 0.28),
            ("BBB", 5.6, 0.22, 0.31, 0.24),
            ("CCC", 3.6, 0.04, 0.08, 0.05),
        ]:
            for i in range(5):
                rows.append(
                    {
                        "fd_player_id": f"{team}{i}",
                        "full_name": f"{team} Hitter {i}",
                        "team_code": team,
                        "player_type": "batter",
                        "proj_fd_mean": 11 - i,
                        "proj_fd_upside": 28 - i,
                        "proj_fd_bust_rate": 0.15,
                        "proj_fd_ownership": ownership,
                        "team_leverage_score": 0.2 if team == "AAA" else -0.1,
                        "vegas_team_total": runs,
                        "bpp_runs": runs,
                        "bpp_home_runs": 1.4,
                        "bpp_runs0": 0.02,
                        "bpp_runs1": 0.04,
                        "bpp_runs2": 0.07,
                        "bpp_runs3": 0.10,
                        "bpp_runs4": 0.14,
                        "bpp_runs5": 0.18,
                        "bpp_runs6": p6,
                        "bpp_home_runs0": 0.25,
                        "bpp_home_runs1": 0.40,
                        "bpp_home_runs2": hr2,
                    }
                )
        plan = build_stack_exposure_plan(pd.DataFrame(rows))
        by_team = plan.set_index("team_code")

        self.assertGreater(by_team.loc["AAA", "target_exposure"], by_team.loc["BBB", "target_exposure"])
        self.assertGreater(by_team.loc["AAA", "target_exposure"], by_team.loc["CCC", "target_exposure"])
        self.assertIn(by_team.loc["AAA", "tier"], {"Core overweight", "Leverage overweight"})
        self.assertGreater(stack_plan_weights(plan)["AAA"], stack_plan_weights(plan)["CCC"])


if __name__ == "__main__":
    unittest.main()
