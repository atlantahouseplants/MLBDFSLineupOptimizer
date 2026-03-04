from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import unittest

import pandas as pd

from slate_optimizer.simulation import FieldQualityMix, simulate_field


class TestFieldSimulator(unittest.TestCase):
    def _dataset(self) -> pd.DataFrame:
        rows = []
        # two pitchers
        for team in ("AAA", "BBB"):
            rows.append(
                {
                    "fd_player_id": f"P_{team}",
                    "full_name": f"Pitcher {team}",
                    "player_type": "pitcher",
                    "position": "P",
                    "team_code": team,
                    "opponent_code": "BBB" if team == "AAA" else "AAA",
                    "proj_fd_mean": 35.0,
                    "proj_fd_ownership": 20.0,
                    "salary": 9000,
                }
            )
        # hitters
        positions = ['C/1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF', 'OF', 'UTIL', '2B', '1B', 'OF']
        idx = 0
        for team in ("AAA", "BBB"):
            for pos in positions:
                rows.append(
                    {
                        "fd_player_id": f"{team}_H{idx}",
                        "full_name": f"{team} Hitter {idx}",
                        "player_type": "batter",
                        "position": pos,
                        "team_code": team,
                        "opponent_code": "BBB" if team == "AAA" else "AAA",
                        "stack_key": team,
                        "game_key": "AAA@BBB",
                        "stack_key": team,
                        "stack_priority": "high" if team == "AAA" else "mid",
                        "proj_fd_mean": 10.0 + (idx % 4),
                        "proj_fd_ownership": 8.0 + idx,
                        "salary": 2600 + (idx * 30),
                    }
                )
                idx += 1
        return pd.DataFrame(rows)

    def test_field_lineup_generation(self) -> None:
        dataset = self._dataset()
        field = simulate_field(
            dataset,
            num_opponent_lineups=50,
            salary_cap=35000,
            seed=42,
            position_constraints=True,
            quality_mix=FieldQualityMix(shark_pct=0.2, rec_pct=0.6, random_pct=0.2),
        )
        self.assertEqual(field.lineups.shape[0], 50)
        self.assertEqual(field.lineups.shape[1], 9)
        self.assertTrue((field.lineups >= 0).all())


if __name__ == "__main__":
    unittest.main()
