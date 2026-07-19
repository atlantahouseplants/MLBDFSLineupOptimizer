from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.analysis.post_slate_learning import build_post_slate_learning_report


class TestPostSlateLearning(unittest.TestCase):
    def test_compares_selected_to_raw_optimizer_pool(self) -> None:
        selected = []
        raw = []
        actuals = []
        for lineup_id, team, actual in [(1, "AAA", 15.0), (2, "BBB", 5.0)]:
            for idx in range(9):
                pid = f"{team}{idx}"
                selected.append(
                    {
                        "lineup_id": lineup_id,
                        "fd_player_id": pid,
                        "team_code": team,
                        "player_type": "batter" if idx else "pitcher",
                        "proj_fd_mean": 8.0,
                    }
                )
                actuals.append({"fd_player_id": pid, "actual_fd_points": actual})
        for lineup_id, team, actual in [(1, "CCC", 3.0), (2, "DDD", 4.0)]:
            for idx in range(9):
                pid = f"{team}{idx}"
                raw.append(
                    {
                        "lineup_id": lineup_id,
                        "fd_player_id": pid,
                        "team_code": team,
                        "player_type": "batter" if idx else "pitcher",
                        "proj_fd_mean": 8.0,
                    }
                )
                actuals.append({"fd_player_id": pid, "actual_fd_points": actual})
        stack_plan = pd.DataFrame([{"team_code": "AAA", "tier": "Core overweight", "target_exposure": 0.25}])

        report = build_post_slate_learning_report(
            pd.DataFrame(selected),
            pd.DataFrame(actuals),
            stack_plan=stack_plan,
            raw_optimizer_lineups=pd.DataFrame(raw),
        )

        self.assertGreater(report.summary["selected_vs_raw_avg_delta"], 0)
        self.assertFalse(report.stack_results.empty)
        self.assertFalse(report.lessons.empty)


if __name__ == "__main__":
    unittest.main()
