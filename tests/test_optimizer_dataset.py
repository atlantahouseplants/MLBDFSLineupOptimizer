from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.optimizer.dataset import build_optimizer_dataset


class TestOptimizerDataset(unittest.TestCase):
    def test_game_start_time_survives_dataset_build(self) -> None:
        players = pd.DataFrame(
            [
                {
                    "fd_player_id": "H1",
                    "full_name": "Leadoff Hitter",
                    "position": "OF",
                    "roster_position": "OF",
                    "player_type": "batter",
                    "team": "AAA",
                    "team_code": "AAA",
                    "opponent": "BBB",
                    "opponent_code": "BBB",
                    "salary": 3200,
                    "game": "AAA@BBB 4/23 7:05PM",
                    "bpp_runs6": 0.14,
                    "bpp_home_runs2": 0.22,
                    "bpp_home_runs": 1.35,
                }
            ]
        )
        projections = pd.DataFrame(
            [
                {
                    "fd_player_id": "H1",
                    "proj_fd_mean": 10.0,
                    "proj_fd_floor": 5.0,
                    "proj_fd_ceiling": 18.0,
                    "proj_fd_median": 8.0,
                    "proj_fd_bust_rate": 0.18,
                    "proj_fd_upside": 28.0,
                    "proj_fd_pts_per_salary": 3.1,
                    "proj_fd_median_per_salary": 2.5,
                    "proj_fd_upside_per_salary": 8.5,
                    "bpp_stack_count": 1,
                    "proj_fd_ownership": 0.12,
                }
            ]
        )

        dataset = build_optimizer_dataset(players, projections)

        self.assertNotEqual(dataset.loc[0, "game_start_time"], "0.0")
        parsed = pd.to_datetime(dataset.loc[0, "game_start_time"], errors="coerce", utc=True)
        self.assertFalse(pd.isna(parsed))
        self.assertAlmostEqual(dataset.loc[0, "proj_fd_bust_rate"], 0.18)
        self.assertAlmostEqual(dataset.loc[0, "proj_fd_median"], 8.0)
        self.assertAlmostEqual(dataset.loc[0, "proj_fd_upside"], 28.0)
        self.assertAlmostEqual(dataset.loc[0, "bpp_runs6"], 0.14)
        self.assertAlmostEqual(dataset.loc[0, "bpp_home_runs2"], 0.22)
        self.assertAlmostEqual(dataset.loc[0, "bpp_home_runs"], 1.35)


if __name__ == "__main__":
    unittest.main()
