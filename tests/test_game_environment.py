"""Tests for game environment scoring (Section 4)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.projection.game_environment import (
    GameEnvironment,
    compute_game_environments,
    merge_game_environment_columns,
)


def _make_optimizer_df() -> pd.DataFrame:
    """Build a mini optimizer-style DataFrame with 3 games."""
    rows = []
    # Game 1: AAA vs BBB — high total, low ownership → prime candidate
    for pid, name, team, opp, sal, own in [
        ("B1", "A Player", "AAA", "BBB", 3500, 0.04),
        ("B2", "B Player", "AAA", "BBB", 3200, 0.03),
        ("B3", "C Player", "BBB", "AAA", 3000, 0.05),
        ("B4", "D Player", "BBB", "AAA", 2800, 0.03),
    ]:
        rows.append({
            "fd_player_id": pid, "full_name": name, "player_type": "batter",
            "team_code": team, "opponent_code": opp, "salary": sal,
            "proj_fd_ownership": own, "vegas_game_total": 10.0,
            "vegas_team_total": 5.0 if team == "AAA" else 5.0,
            "game_key": "AAA_vs_BBB", "proj_fd_mean": 10.0,
        })
    # Game 2: CCC vs DDD — high total, HIGH ownership → chalky
    for pid, name, team, opp, sal, own in [
        ("B5", "E Player", "CCC", "DDD", 4000, 0.20),
        ("B6", "F Player", "CCC", "DDD", 3800, 0.18),
        ("B7", "G Player", "DDD", "CCC", 3500, 0.15),
        ("B8", "H Player", "DDD", "CCC", 3200, 0.12),
    ]:
        rows.append({
            "fd_player_id": pid, "full_name": name, "player_type": "batter",
            "team_code": team, "opponent_code": opp, "salary": sal,
            "proj_fd_ownership": own, "vegas_game_total": 9.5,
            "vegas_team_total": 4.75 if team == "CCC" else 4.75,
            "game_key": "CCC_vs_DDD", "proj_fd_mean": 10.0,
        })
    # Game 3: EEE vs FFF — low total → avoid
    for pid, name, team, opp, sal, own in [
        ("B9", "I Player", "EEE", "FFF", 2500, 0.02),
        ("B10", "J Player", "FFF", "EEE", 2200, 0.01),
    ]:
        rows.append({
            "fd_player_id": pid, "full_name": name, "player_type": "batter",
            "team_code": team, "opponent_code": opp, "salary": sal,
            "proj_fd_ownership": own, "vegas_game_total": 6.0,
            "vegas_team_total": 3.0,
            "game_key": "EEE_vs_FFF", "proj_fd_mean": 5.0,
        })
    # Pitchers
    for pid, name, team, opp in [
        ("P1", "Pitcher A", "BBB", "AAA"),
        ("P2", "Pitcher B", "DDD", "CCC"),
        ("P3", "Pitcher C", "FFF", "EEE"),
    ]:
        rows.append({
            "fd_player_id": pid, "full_name": name, "player_type": "pitcher",
            "team_code": team, "opponent_code": opp, "salary": 8000,
            "proj_fd_ownership": 0.10, "vegas_game_total": 8.0,
            "vegas_team_total": 4.0,
            "game_key": f"{min(team,opp)}_vs_{max(team,opp)}", "proj_fd_mean": 20.0,
        })
    return pd.DataFrame(rows)


class TestGameEnvironment(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _make_optimizer_df()

    def test_compute_returns_environments_for_all_games(self) -> None:
        envs = compute_game_environments(self.df)
        game_keys = {e.game_key for e in envs}
        self.assertIn("AAA_vs_BBB", game_keys)
        self.assertIn("CCC_vs_DDD", game_keys)
        self.assertIn("EEE_vs_FFF", game_keys)

    def test_low_total_game_gets_avoid(self) -> None:
        envs = compute_game_environments(self.df)
        eee_env = [e for e in envs if e.game_key == "EEE_vs_FFF"][0]
        self.assertEqual(eee_env.environment_tier, "avoid")

    def test_high_total_low_ownership_has_higher_leverage(self) -> None:
        envs = compute_game_environments(self.df)
        aaa_env = [e for e in envs if e.game_key == "AAA_vs_BBB"][0]
        ccc_env = [e for e in envs if e.game_key == "CCC_vs_DDD"][0]
        # AAA_vs_BBB has low ownership → should have higher leverage than chalky CCC_vs_DDD
        self.assertGreater(aaa_env.game_leverage_score, ccc_env.game_leverage_score)

    def test_team_gpp_leverage_higher_for_underowned(self) -> None:
        result = merge_game_environment_columns(self.df)
        # AAA (low ownership) should have higher team_gpp_leverage than CCC (high ownership)
        aaa_lev = result[result["team_code"] == "AAA"]["team_gpp_leverage"].iloc[0]
        ccc_lev = result[result["team_code"] == "CCC"]["team_gpp_leverage"].iloc[0]
        self.assertGreater(aaa_lev, ccc_lev)

    def test_merge_adds_expected_columns(self) -> None:
        result = merge_game_environment_columns(self.df)
        self.assertIn("game_leverage_score", result.columns)
        self.assertIn("environment_tier", result.columns)
        self.assertIn("team_gpp_leverage", result.columns)


if __name__ == "__main__":
    unittest.main()
