"""Tests for the enhanced ownership model (Section 3)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.projection import (
    OwnershipModelConfig,
    compute_baseline_projections,
    compute_ownership_series,
)


def _make_players() -> pd.DataFrame:
    """Build a realistic mini-slate for ownership testing."""
    return pd.DataFrame([
        {"fd_player_id": "B1", "full_name": "Star Slugger", "player_type": "batter",
         "position": "OF", "salary": 4200, "team": "AAA", "team_code": "AAA",
         "opponent": "BBB", "opponent_code": "BBB", "batter_hand": "R",
         "bpp_points_fd": 14.0, "fppg": 13.5, "vegas_team_total": 5.5,
         "bpp_runs": 5.2, "bpp_win_percent": 0.55, "bpp_runs_first_inning_pct": 0.50, "bpp_runs_allowed": 4.2},
        {"fd_player_id": "B2", "full_name": "Value Play", "player_type": "batter",
         "position": "SS", "salary": 2800, "team": "AAA", "team_code": "AAA",
         "opponent": "BBB", "opponent_code": "BBB", "batter_hand": "L",
         "bpp_points_fd": 8.0, "fppg": 7.5, "vegas_team_total": 5.5,
         "bpp_runs": 5.2, "bpp_win_percent": 0.55, "bpp_runs_first_inning_pct": 0.48, "bpp_runs_allowed": 4.0},
        {"fd_player_id": "B3", "full_name": "Chalk Bat", "player_type": "batter",
         "position": "C", "salary": 3800, "team": "CCC", "team_code": "CCC",
         "opponent": "DDD", "opponent_code": "DDD", "batter_hand": "R",
         "bpp_points_fd": 11.0, "fppg": 10.5, "vegas_team_total": 4.0,
         "bpp_runs": 4.0, "bpp_win_percent": 0.48, "bpp_runs_first_inning_pct": 0.42, "bpp_runs_allowed": 4.5},
        {"fd_player_id": "B4", "full_name": "Deep Value", "player_type": "batter",
         "position": "2B", "salary": 2500, "team": "CCC", "team_code": "CCC",
         "opponent": "DDD", "opponent_code": "DDD", "batter_hand": "S",
         "bpp_points_fd": 6.0, "fppg": 5.5, "vegas_team_total": 4.0,
         "bpp_runs": 4.0, "bpp_win_percent": 0.48, "bpp_runs_first_inning_pct": 0.40, "bpp_runs_allowed": 4.6},
        {"fd_player_id": "B5", "full_name": "Cheap OF", "player_type": "batter",
         "position": "OF", "salary": 2200, "team": "EEE", "team_code": "EEE",
         "opponent": "FFF", "opponent_code": "FFF", "batter_hand": "L",
         "bpp_points_fd": 5.5, "fppg": 5.0, "vegas_team_total": 3.5,
         "bpp_runs": 3.5, "bpp_win_percent": 0.42, "bpp_runs_first_inning_pct": 0.35, "bpp_runs_allowed": 3.8},
        {"fd_player_id": "P1", "full_name": "Ace Pitcher", "player_type": "pitcher",
         "position": "P", "salary": 10000, "team": "BBB", "team_code": "BBB",
         "opponent": "AAA", "opponent_code": "AAA", "pitcher_hand": "R",
         "bpp_points_fd": 0.0, "fppg": 0.0, "bpp_runs": 0.0, "bpp_win_percent": 0.62, "bpp_runs_first_inning_pct": 0.35, "bpp_runs_allowed": 3.8},
        {"fd_player_id": "P2", "full_name": "Mid Pitcher", "player_type": "pitcher",
         "position": "P", "salary": 7500, "team": "DDD", "team_code": "DDD",
         "opponent": "CCC", "opponent_code": "CCC", "pitcher_hand": "L",
         "bpp_points_fd": 0.0, "fppg": 0.0, "bpp_runs": 0.0, "bpp_win_percent": 0.50, "bpp_runs_first_inning_pct": 0.32, "bpp_runs_allowed": 4.6},
        {"fd_player_id": "P3", "full_name": "Cheap Pitcher", "player_type": "pitcher",
         "position": "P", "salary": 6000, "team": "FFF", "team_code": "FFF",
         "opponent": "EEE", "opponent_code": "EEE", "pitcher_hand": "R",
         "bpp_points_fd": 0.0, "fppg": 0.0, "bpp_runs": 0.0, "bpp_win_percent": 0.44, "bpp_runs_first_inning_pct": 0.30, "bpp_runs_allowed": 4.8},
    ])


class TestEnhancedOwnershipModel(unittest.TestCase):
    def setUp(self) -> None:
        self.players = _make_players()
        self.projections = compute_baseline_projections(self.players)

    def test_ownership_sums_to_reasonable_total(self) -> None:
        result = compute_ownership_series(self.players, self.projections)
        total = result.ownership.astype(float).sum()
        n = len(result.ownership)
        avg = total / n
        # Average ownership per player should be roughly 7-12%
        self.assertGreater(avg, 0.03)
        self.assertLess(avg, 0.20)

    def test_high_salary_gets_higher_ownership_than_low(self) -> None:
        result = compute_ownership_series(self.players, self.projections)
        series = result.ownership.astype(float)
        # B1 (salary 4200) should have more ownership than B5 (salary 2200)
        self.assertGreater(series.loc["B1"], series.loc["B5"])

    def test_pitcher_ownership_is_sharper(self) -> None:
        result = compute_ownership_series(self.players, self.projections)
        series = result.ownership.astype(float)
        # Top pitcher (P1, $10K) should have significantly more ownership than cheap pitcher (P3, $6K)
        self.assertGreater(series.loc["P1"], series.loc["P3"])

    def test_no_player_below_min_pct(self) -> None:
        cfg = OwnershipModelConfig(min_pct=0.02)
        result = compute_ownership_series(self.players, self.projections, model_config=cfg)
        self.assertTrue((result.ownership.astype(float) >= 0.02).all())

    def test_no_player_above_max_pct(self) -> None:
        cfg = OwnershipModelConfig(max_pct=0.30)
        result = compute_ownership_series(self.players, self.projections, model_config=cfg)
        self.assertTrue((result.ownership.astype(float) <= 0.30 + 1e-9).all())

    def test_backward_compatibility_with_custom_weights(self) -> None:
        """Legacy weight configs should still produce valid ownership."""
        cfg = OwnershipModelConfig(
            salary_weight=1.0,
            projection_weight=0.0,
            value_weight=0.0,
            team_weight=0.0,
            name_weight=0.0,
            position_weight=0.0,
            min_pct=0.01,
            max_pct=0.5,
        )
        result = compute_ownership_series(self.players, self.projections, model_config=cfg)
        series = result.ownership.astype(float)
        # Should produce non-zero results
        self.assertGreater(series.sum(), 0)

    def test_ownership_tier_classification(self) -> None:
        from slate_optimizer.optimizer.dataset import _classify_ownership_tier
        own = pd.Series([0.30, 0.15, 0.05, 0.02])
        tiers = _classify_ownership_tier(own)
        self.assertEqual(tiers.iloc[0], "chalk")
        self.assertEqual(tiers.iloc[1], "mid")
        self.assertEqual(tiers.iloc[2], "leverage")
        self.assertEqual(tiers.iloc[3], "deep")


if __name__ == "__main__":
    unittest.main()
