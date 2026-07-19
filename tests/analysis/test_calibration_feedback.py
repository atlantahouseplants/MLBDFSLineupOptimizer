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

from slate_optimizer.analysis.calibration_feedback import build_calibration_feedback
from slate_optimizer.data.storage import SlateDatabase


class TestCalibrationFeedback(unittest.TestCase):
    def test_recommends_adjustment_from_win_score_error(self) -> None:
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "slates.db"
            db = SlateDatabase(db_path)
            db.insert_slate_result("slate", "20260424", "GPP", 20.0, 300, 220.0, 140.0)
            db.insert_simulation_accuracy(
                "20260424",
                {
                    "field_winning_score_predicted": 240.0,
                    "field_winning_score_actual": 220.0,
                    "brier_score_15": 0.30,
                },
                num_players=100,
            )
            db.close()

            feedback = build_calibration_feedback(
                db_path,
                "20260424",
                "20260424",
                {"volatility_scale": 1.0, "selection_scenario_weight": 0.10},
                strength=1.0,
            )

        self.assertFalse(feedback.adjustments.empty)
        self.assertLess(feedback.adjusted_state["volatility_scale"], 1.0)
        self.assertGreater(feedback.adjusted_state["selection_scenario_weight"], 0.10)


if __name__ == "__main__":
    unittest.main()
