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

from slate_optimizer.analysis.optimizer_tuning import build_optimizer_weight_tuning_report
from slate_optimizer.analysis.slate_replay import build_slate_replay_report
from slate_optimizer.data.storage import SlateDatabase


class TestSlateReplayAndTuning(unittest.TestCase):
    def test_replay_scores_historical_profiles(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db_path = root / "slates.db"
            data_dir = root / "processed"
            data_dir.mkdir()
            optimizer = _optimizer_dataset()
            optimizer.to_csv(data_dir / "main_20260424_optimizer_dataset.csv", index=False)

            db = SlateDatabase(db_path)
            db.insert_slate_result("slate", "20260424", "GPP", 20.0, 50000, 230.0, 145.0)
            db.insert_actual_scores("20260424", _actuals())
            db.insert_lineup_results(
                "20260424",
                pd.DataFrame(
                    [
                        {"lineup_id": 1, "total_actual_points": 155.0, "rank": 500, "payout": 50.0, "roi": 1.5},
                        {"lineup_id": 2, "total_actual_points": 120.0, "rank": 5000, "payout": 0.0, "roi": -1.0},
                    ]
                ),
            )
            db.close()

            replay = build_slate_replay_report(db_path, "20260424", "20260424", data_dir=data_dir)
            tuning = build_optimizer_weight_tuning_report(db_path, "20260424", "20260424", data_dir=data_dir)

        self.assertFalse(replay.summary.empty)
        self.assertFalse(replay.strategy_scores.empty)
        self.assertTrue(replay.lessons)
        self.assertFalse(tuning.candidates.empty)
        self.assertTrue(tuning.recommended_state)


def _optimizer_dataset() -> pd.DataFrame:
    rows = []
    for idx in range(12):
        team = "AAA" if idx < 6 else "BBB"
        rows.append(
            {
                "fd_player_id": f"H{idx}",
                "full_name": f"Hitter {idx}",
                "team_code": team,
                "player_type": "batter",
                "proj_fd_mean": 9.0 + idx * 0.2,
                "proj_fd_ownership": 0.08 + idx * 0.01,
                "player_leverage_score": 0.20 if team == "AAA" else -0.05,
                "proj_fd_upside": 18.0 + idx,
                "proj_fd_bust_rate": 0.25,
            }
        )
    for idx in range(3):
        rows.append(
            {
                "fd_player_id": f"P{idx}",
                "full_name": f"Pitcher {idx}",
                "team_code": f"P{idx}",
                "player_type": "pitcher",
                "proj_fd_mean": 32.0 + idx,
                "proj_fd_ownership": 0.20,
                "player_leverage_score": 0.0,
                "proj_fd_upside": 45.0,
                "proj_fd_bust_rate": 0.20,
            }
        )
    return pd.DataFrame(rows)


def _actuals() -> pd.DataFrame:
    rows = []
    for idx in range(12):
        team_boost = 8.0 if idx < 6 else -3.0
        rows.append(
            {
                "fd_player_id": f"H{idx}",
                "player_name": f"Hitter {idx}",
                "actual_fd_points": 10.0 + team_boost + idx * 0.1,
                "actual_ownership_pct": 10.0 + idx,
            }
        )
    for idx in range(3):
        rows.append(
            {
                "fd_player_id": f"P{idx}",
                "player_name": f"Pitcher {idx}",
                "actual_fd_points": 30.0 + idx,
                "actual_ownership_pct": 20.0,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
