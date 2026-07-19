from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.analysis.model_calibration import build_model_calibration_report
from slate_optimizer.data.storage import SlateDatabase


class TestModelCalibration(unittest.TestCase):
    def test_report_summarizes_recorded_slate(self) -> None:
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "slates.db"
            db = SlateDatabase(db_path)
            db.insert_slate_result(
                slate_tag="20260424-main",
                date="20260424",
                contest_type="gpp",
                entry_fee=20.0,
                num_entries=300,
                winning_score=230.0,
                cash_line=145.0,
            )
            db.insert_lineup_results(
                "20260424",
                pd.DataFrame(
                    [
                        {"lineup_id": 1, "total_actual_points": 160.0, "rank": 100, "payout": 40.0, "roi": 1.0},
                        {"lineup_id": 2, "total_actual_points": 120.0, "rank": 1000, "payout": 0.0, "roi": -1.0},
                    ]
                ),
            )
            db.insert_actual_scores(
                "20260424",
                pd.DataFrame(
                    [
                        {"fd_player_id": "1", "player_name": "A", "actual_fd_points": 18.0, "actual_ownership_pct": 25.0},
                        {"fd_player_id": "2", "player_name": "B", "actual_fd_points": 4.0, "actual_ownership_pct": 5.0},
                    ]
                ),
            )
            db.insert_simulation_accuracy(
                "20260424",
                {
                    "brier_score_15": 0.20,
                    "field_winning_score_predicted": 235.0,
                    "field_winning_score_actual": 230.0,
                    "dist_calibration_p50": 0.60,
                },
                num_players=2,
            )
            db.close()

            report = build_model_calibration_report(db_path, "20260424", "20260424")

        self.assertFalse(report.daily.empty)
        self.assertFalse(report.summary.empty)
        self.assertIn("winning_score_error", report.daily.columns)
        self.assertTrue(report.lessons)


if __name__ == "__main__":
    unittest.main()
