from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.ingestion.ballparkpal import BallparkPalBundle
from slate_optimizer.ingestion.slate_builder import build_player_dataset


class TestSlateBuilder(unittest.TestCase):
    def test_duplicate_ballparkpal_player_rows_do_not_duplicate_fanduel_players(self) -> None:
        bundle = BallparkPalBundle(
            batters=pd.DataFrame(
                [
                    {
                        "full_name": "Lefty Leadoff",
                        "team": "AAA",
                        "opponent": "BBB",
                        "points_fd": 10.0,
                    },
                    {
                        "full_name": "Lefty Leadoff",
                        "team": "AAA",
                        "opponent": "BBB",
                        "points_fd": 12.0,
                    },
                ]
            ),
            pitchers=pd.DataFrame(
                [
                    {
                        "full_name": "Pitcher One",
                        "team": "BBB",
                        "opponent": "AAA",
                        "points_fd": 30.0,
                    }
                ]
            ),
            games=pd.DataFrame(),
            teams=pd.DataFrame(),
        )
        fanduel_players = pd.DataFrame(
            [
                {
                    "fd_player_id": "H1",
                    "full_name": "Lefty Leadoff",
                    "first_name": "Lefty",
                    "last_name": "Leadoff",
                    "position": "OF",
                    "team": "AAA",
                    "opponent": "BBB",
                    "salary": 3600,
                }
            ]
        )

        combined, diagnostics = build_player_dataset(bundle, fanduel_players)

        self.assertEqual(len(combined), 1)
        self.assertEqual(combined["fd_player_id"].astype(str).nunique(), 1)
        self.assertEqual(diagnostics.hitters_matched, 1)
        self.assertAlmostEqual(float(combined.loc[0, "bpp_points_fd"]), 11.0)


if __name__ == "__main__":
    unittest.main()
