from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import unittest

import numpy as np
import pandas as pd

from slate_optimizer.simulation import (
    CorrelationConfig,
    build_correlation_matrix,
    fit_player_distributions,
    simulate_slate,
)


class TestSlateSimulator(unittest.TestCase):
    def _dataset(self) -> pd.DataFrame:
        rows = []
        for order in range(1, 5):
            rows.append(
                {
                    "fd_player_id": f"AAA_{order}",
                    "full_name": f"AAA Batter {order}",
                    "player_type": "batter",
                    "position": "OF",
                    "team_code": "AAA",
                    "opponent_code": "BBB",
                    "stack_key": "AAA",
                    "game_key": "AAA@BBB",
                    "stack_priority": "high",
                    "proj_fd_mean": 10.0 + order,
                    "proj_fd_floor": 6.0 + order / 2,
                    "proj_fd_ceiling": 16.0 + order,
                    "salary": 3500 + order * 100,
                }
            )
        rows.append(
            {
                "fd_player_id": "PIT_AAA",
                "full_name": "AAA Pitcher",
                "player_type": "pitcher",
                "position": "P",
                "team_code": "AAA",
                "opponent_code": "BBB",
                "stack_key": "AAA",
                "game_key": "AAA@BBB",
                "stack_priority": "high",
                "proj_fd_mean": 35.0,
                "proj_fd_floor": 22.0,
                "proj_fd_ceiling": 55.0,
                "salary": 10200,
            }
        )
        return pd.DataFrame(rows)

    def test_simulated_scores(self) -> None:
        dataset = self._dataset()
        dists = fit_player_distributions(dataset)
        corr = build_correlation_matrix(dataset, CorrelationConfig(teammate_base=0.25))
        slate = simulate_slate(dists, corr, num_simulations=500, seed=123)
        self.assertEqual(slate.scores.shape, (500, len(dataset)))
        self.assertTrue(np.all(slate.scores >= -1e-6))
        lineup_ids = dataset["fd_player_id"].tolist()[:4]
        lineup_scores = slate.lineup_scores(lineup_ids)
        self.assertEqual(len(lineup_scores), 500)
        self.assertGreater(lineup_scores.mean(), 0.0)


if __name__ == "__main__":
    unittest.main()
