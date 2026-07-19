from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.analysis.boom_bust_calibration import build_boom_bust_calibration_report


class TestBoomBustCalibration(unittest.TestCase):
    def test_report_compares_projected_bust_to_actual_under_median(self) -> None:
        optimizer = pd.DataFrame(
            [
                {"fd_player_id": "A", "full_name": "A", "proj_fd_mean": 10, "proj_fd_median": 8, "proj_fd_bust_rate": 0.20, "proj_fd_upside": 20},
                {"fd_player_id": "B", "full_name": "B", "proj_fd_mean": 12, "proj_fd_median": 9, "proj_fd_bust_rate": 0.40, "proj_fd_upside": 24},
            ]
        )
        actuals = pd.DataFrame(
            [
                {"fd_player_id": "A", "actual_fd_points": 7},
                {"fd_player_id": "B", "actual_fd_points": 25},
            ]
        )

        report = build_boom_bust_calibration_report(optimizer, actuals)

        self.assertEqual(report.summary["matched_players"], 2)
        self.assertAlmostEqual(report.summary["actual_under_median_rate"], 0.5)
        self.assertAlmostEqual(report.summary["upside_hit_rate"], 0.5)
        self.assertFalse(report.by_bust_bucket.empty)
        self.assertFalse(report.lessons.empty)


if __name__ == "__main__":
    unittest.main()
