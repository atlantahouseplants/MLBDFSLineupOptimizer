from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.analysis.strategy_backtest import build_strategy_backtest_report
from slate_optimizer.data.storage import SlateDatabase


class TestStrategyBacktest(unittest.TestCase):
    def test_groups_lineups_by_saved_strategy(self) -> None:
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "slates.db"
            db = SlateDatabase(db_path)
            db.insert_slate_result("slate", "20260424", "GPP", 20.0, 300, 220.0, 140.0)
            db.insert_lineup_results(
                "20260424",
                pd.DataFrame(
                    [
                        {
                            "lineup_id": 1,
                            "total_actual_points": 160.0,
                            "rank": 10,
                            "payout": 100.0,
                            "roi": 4.0,
                            "strategy_config_json": json.dumps({"active_strategy_build": "Chalk fade unique"}),
                        },
                        {
                            "lineup_id": 2,
                            "total_actual_points": 120.0,
                            "rank": 2000,
                            "payout": 0.0,
                            "roi": -1.0,
                            "strategy_config_json": json.dumps({"active_strategy_build": "Chalk fade unique"}),
                        },
                    ]
                ),
            )
            db.close()

            report = build_strategy_backtest_report(db_path, "20260424", "20260424")

        self.assertFalse(report.by_strategy.empty)
        self.assertEqual(report.by_strategy.iloc[0]["strategy_label"], "Chalk fade unique")
        self.assertEqual(int(report.by_strategy.iloc[0]["entries"]), 2)


if __name__ == "__main__":
    unittest.main()
