from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.simulation import fit_player_distributions


class TestDistributions(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = pd.DataFrame(
            [
                {
                    "fd_player_id": "BAT1",
                    "full_name": "Hitter One",
                    "player_type": "batter",
                    "position": "OF",
                    "team_code": "AAA",
                    "opponent_code": "BBB",
                    "proj_fd_mean": 12.0,
                    "proj_fd_floor": 8.0,
                    "proj_fd_ceiling": 20.0,
                    "salary": 3500,
                },
                {
                    "fd_player_id": "BAT2",
                    "full_name": "Value Hitter",
                    "player_type": "batter",
                    "position": "2B",
                    "team_code": "AAA",
                    "opponent_code": "BBB",
                    "proj_fd_mean": 8.0,
                    "proj_fd_floor": 4.0,
                    "proj_fd_ceiling": 15.0,
                    "salary": 2400,
                },
                {
                    "fd_player_id": "PIT1",
                    "full_name": "Pitcher One",
                    "player_type": "pitcher",
                    "position": "P",
                    "team_code": "BBB",
                    "opponent_code": "AAA",
                    "proj_fd_mean": 34.0,
                    "proj_fd_floor": 20.0,
                    "proj_fd_ceiling": 55.0,
                    "salary": 10200,
                },
            ]
        )

    def test_distribution_means_close(self) -> None:
        dists = fit_player_distributions(self.dataset)
        for _, row in self.dataset.iterrows():
            dist = dists[row["fd_player_id"]]
            expected = float(row["proj_fd_mean"])
            self.assertAlmostEqual(dist.mean(), expected, delta=0.6)

    def test_samples_non_negative(self) -> None:
        dists = fit_player_distributions(self.dataset)
        for dist in dists.values():
            samples = dist.sample(500, np.random.default_rng(0))
            self.assertTrue(np.all(samples >= -1e-6))


if __name__ == "__main__":
    unittest.main()
