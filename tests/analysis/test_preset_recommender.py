from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.analysis.preset_recommender import recommend_slate_preset


class TestPresetRecommender(unittest.TestCase):
    def test_recommends_chalk_fade_when_ownership_is_condensed(self) -> None:
        rows = []
        for team in ("AAA", "BBB", "CCC"):
            for idx in range(6):
                rows.append(
                    {
                        "fd_player_id": f"{team}_{idx}",
                        "full_name": f"{team} {idx}",
                        "player_type": "batter",
                        "team_code": team,
                        "proj_fd_ownership": 30.0 if team == "AAA" else 5.0,
                        "proj_fd_mean": 10.0,
                    }
                )
        rec = recommend_slate_preset(pd.DataFrame(rows))

        self.assertEqual(rec.preset, "Chalk fade build")
        self.assertGreater(rec.confidence, 0.7)


if __name__ == "__main__":
    unittest.main()
