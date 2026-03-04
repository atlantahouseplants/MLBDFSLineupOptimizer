from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.projection import compute_baseline_projections


class TestProjectionControls(unittest.TestCase):
    def setUp(self) -> None:
        self.players = pd.DataFrame(
            [
                {
                    "fd_player_id": "H1",
                    "full_name": "Lefty Leadoff",
                    "player_type": "batter",
                    "position": "OF",
                    "team": "AAA",
                    "team_code": "AAA",
                    "opponent": "BBB",
                    "opponent_code": "BBB",
                    "batter_hand": "L",
                    "bpp_points_fd": 12.0,
                    "fppg": 11.0,
                    "recent_last7_fppg": 9.0,
                    "recent_last14_fppg": 8.0,
                    "recent_season_fppg": 11.0,
                    "bpp_runs": 5.2,
                    "bpp_win_percent": 0.58,
                    "bpp_runs_first_inning_pct": 0.55,
                    "bpp_runs_allowed": 4.2,
                },
                {
                    "fd_player_id": "H2",
                    "full_name": "Righty Cleanup",
                    "player_type": "batter",
                    "position": "1B",
                    "team": "CCC",
                    "team_code": "CCC",
                    "opponent": "DDD",
                    "opponent_code": "DDD",
                    "batter_hand": "R",
                    "bpp_points_fd": 9.0,
                    "fppg": 8.5,
                    "recent_last7_fppg": 11.0,
                    "recent_last14_fppg": 10.0,
                    "recent_season_fppg": 8.5,
                    "bpp_runs": 4.1,
                    "bpp_win_percent": 0.46,
                    "bpp_runs_first_inning_pct": 0.42,
                    "bpp_runs_allowed": 4.5,
                },
                {
                    "fd_player_id": "P_BBB",
                    "full_name": "Opp Pitcher",
                    "player_type": "pitcher",
                    "position": "P",
                    "team": "BBB",
                    "team_code": "BBB",
                    "opponent": "AAA",
                    "opponent_code": "AAA",
                    "pitcher_hand": "R",
                    "bpp_points_fd": 0.0,
                    "fppg": 0.0,
                    "bpp_runs": 0.0,
                    "bpp_win_percent": 0.52,
                    "bpp_runs_first_inning_pct": 0.35,
                    "bpp_runs_allowed": 3.8,
                },
                {
                    "fd_player_id": "P_DDD",
                    "full_name": "Same Pitcher",
                    "player_type": "pitcher",
                    "position": "P",
                    "team": "DDD",
                    "team_code": "DDD",
                    "opponent": "CCC",
                    "opponent_code": "CCC",
                    "pitcher_hand": "R",
                    "bpp_points_fd": 0.0,
                    "fppg": 0.0,
                    "bpp_runs": 0.0,
                    "bpp_win_percent": 0.48,
                    "bpp_runs_first_inning_pct": 0.31,
                    "bpp_runs_allowed": 4.6,
                },
            ]
        )

    def test_platoon_multiplier_overrides(self) -> None:
        base = compute_baseline_projections(self.players)
        custom = compute_baseline_projections(
            self.players,
            platoon_opposite_boost=1.2,
            platoon_same_penalty=0.8,
            platoon_switch_boost=1.05,
        )

        base_map = base.set_index("fd_player_id")
        custom_map = custom.set_index("fd_player_id")

        self.assertAlmostEqual(base_map.loc["H1", "platoon_factor"], 1.06, places=2)
        self.assertAlmostEqual(custom_map.loc["H1", "platoon_factor"], 1.20, places=2)
        self.assertAlmostEqual(base_map.loc["H2", "platoon_factor"], 0.95, places=2)
        self.assertAlmostEqual(custom_map.loc["H2", "platoon_factor"], 0.80, places=2)
        self.assertGreater(custom_map.loc["H1", "proj_fd_mean"], base_map.loc["H1", "proj_fd_mean"])
        self.assertLess(custom_map.loc["H2", "proj_fd_mean"], base_map.loc["H2", "proj_fd_mean"])

    def test_recency_blend_adjustment(self) -> None:
        conservative = compute_baseline_projections(self.players, recency_blend=(0.8, 0.2))
        aggressive = compute_baseline_projections(self.players, recency_blend=(0.2, 0.8))

        cons_map = conservative.set_index("fd_player_id")
        aggr_map = aggressive.set_index("fd_player_id")

        self.assertGreater(cons_map.loc["H1", "recency_factor"], aggr_map.loc["H1", "recency_factor"])
        self.assertLess(cons_map.loc["H2", "recency_factor"], aggr_map.loc["H2", "recency_factor"])


if __name__ == "__main__":
    unittest.main()
