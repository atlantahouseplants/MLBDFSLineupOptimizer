from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.analysis.failure_attribution import build_failure_attribution_report


class TestFailureAttribution(unittest.TestCase):
    def test_identifies_projection_and_pitcher_failure(self) -> None:
        lineups = []
        actuals = []
        for lineup_id in (1, 2):
            lineups.append(
                {
                    "lineup_id": lineup_id,
                    "fd_player_id": f"P{lineup_id}",
                    "full_name": f"Pitcher {lineup_id}",
                    "team_code": "PIT",
                    "player_type": "pitcher",
                    "proj_fd_mean": 35.0,
                    "proj_fd_ownership": 0.25,
                    "player_leverage_score": 0.0,
                }
            )
            actuals.append({"fd_player_id": f"P{lineup_id}", "actual_fd_points": 12.0, "actual_ownership_pct": 28.0})
            for idx in range(8):
                pid = f"H{lineup_id}_{idx}"
                lineups.append(
                    {
                        "lineup_id": lineup_id,
                        "fd_player_id": pid,
                        "full_name": pid,
                        "team_code": "AAA" if idx < 4 else "BBB",
                        "player_type": "batter",
                        "proj_fd_mean": 10.0,
                        "proj_fd_ownership": 0.20 if idx < 2 else 0.06,
                        "player_leverage_score": 0.10 if idx >= 4 else 0.0,
                    }
                )
                actuals.append({"fd_player_id": pid, "actual_fd_points": 5.0, "actual_ownership_pct": 8.0})

        report = build_failure_attribution_report(pd.DataFrame(lineups), pd.DataFrame(actuals))

        self.assertFalse(report.factors.empty)
        self.assertIn("Projection miss", set(report.factors["factor"]))
        self.assertGreaterEqual(report.summary["high_severity_factors"], 1)
        self.assertTrue(report.lessons)


if __name__ == "__main__":
    unittest.main()
