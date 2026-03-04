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

from slate_optimizer.simulation import CorrelationConfig, build_correlation_matrix


class TestCorrelation(unittest.TestCase):
    def _dataset(self) -> pd.DataFrame:
        rows = []
        for order in range(1, 5):
            rows.append(
                {
                    "fd_player_id": f"AAA_{order}",
                    "full_name": f"AAA Batter {order}",
                    "player_type": "batter",
                    "team_code": "AAA",
                    "opponent_code": "BBB",
                    "stack_key": "AAA",
                    "game_key": "AAA@BBB",
                    "stack_priority": "high",
                    "batting_order_position": order,
                }
            )
        for order in range(1, 5):
            rows.append(
                {
                    "fd_player_id": f"BBB_{order}",
                    "full_name": f"BBB Batter {order}",
                    "player_type": "batter",
                    "team_code": "BBB",
                    "opponent_code": "AAA",
                    "stack_key": "BBB",
                    "game_key": "AAA@BBB",
                    "stack_priority": "mid",
                    "batting_order_position": order,
                }
            )
        rows.append(
            {
                "fd_player_id": "PIT_AAA",
                "full_name": "AAA Pitcher",
                "player_type": "pitcher",
                "team_code": "AAA",
                "opponent_code": "BBB",
                "stack_key": "AAA",
                "game_key": "AAA@BBB",
                "stack_priority": "high",
            }
        )
        rows.append(
            {
                "fd_player_id": "PIT_BBB",
                "full_name": "BBB Pitcher",
                "player_type": "pitcher",
                "team_code": "BBB",
                "opponent_code": "AAA",
                "stack_key": "BBB",
                "game_key": "AAA@BBB",
                "stack_priority": "mid",
            }
        )
        return pd.DataFrame(rows)

    def test_correlation_matrix_properties(self) -> None:
        dataset = self._dataset()
        config = CorrelationConfig(teammate_base=0.3, same_game_opponent=0.05, cross_game=0.0)
        model = build_correlation_matrix(dataset, config)
        matrix = model.matrix
        self.assertEqual(matrix.shape[0], matrix.shape[1])
        self.assertEqual(matrix.shape[0], len(dataset))
        eigvals = np.linalg.eigvalsh(matrix)
        self.assertTrue(np.all(eigvals >= -1e-6))
        hitter_idx = dataset.index[(dataset["team_code"] == "AAA") & (dataset["player_type"] == "batter")][:2]
        for idx in hitter_idx:
            other = idx + 1
            self.assertGreater(matrix[idx, other], 0.0)
        pitcher_idx = dataset.index[dataset["player_type"] == "pitcher"].tolist()
        self.assertLess(matrix[pitcher_idx[0], pitcher_idx[1]], 0.2)


if __name__ == "__main__":
    unittest.main()
