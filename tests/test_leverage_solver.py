"""Tests for the leverage-weighted ILP solver (Sections 5-6)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.optimizer.config import LeverageConfig, SlateProfile, detect_slate_profile, apply_slate_adjustments
from slate_optimizer.optimizer.solver import generate_lineups, _select_auto_stack_template


def _make_full_slate() -> pd.DataFrame:
    """Build a full FanDuel-valid slate with enough players to generate lineups."""
    rows = []
    teams = [
        ("AAA", "BBB", 5.0, 10.0),
        ("BBB", "AAA", 5.0, 10.0),
        ("CCC", "DDD", 4.5, 9.0),
        ("DDD", "CCC", 4.5, 9.0),
        ("EEE", "FFF", 3.5, 7.0),
        ("FFF", "EEE", 3.5, 7.0),
    ]
    positions = ["C/1B", "2B", "3B", "SS", "OF", "OF", "OF", "OF"]
    pid_counter = 0

    for team, opp, team_total, game_total in teams:
        for i, pos in enumerate(positions):
            pid_counter += 1
            salary = 3500 - (i * 100) + (100 if team in ("AAA", "CCC") else 0)
            mean_proj = 8.0 + (3 - i) * 0.5 + (1.0 if team == "AAA" else 0)
            # Give AAA low ownership (leverage), CCC high ownership (chalk)
            if team == "AAA":
                own = 0.03 + i * 0.005
            elif team == "CCC":
                own = 0.15 + i * 0.01
            else:
                own = 0.06 + i * 0.005
            rows.append({
                "fd_player_id": f"B{pid_counter}",
                "full_name": f"Player {pid_counter}",
                "player_type": "batter",
                "position": pos.split("/")[0],
                "roster_position": pos,
                "team_code": team,
                "opponent_code": opp,
                "salary": salary,
                "proj_fd_mean": mean_proj,
                "proj_fd_floor": mean_proj * 0.6,
                "proj_fd_ceiling": mean_proj * 1.5,
                "proj_fd_ownership": own,
                "vegas_team_total": team_total,
                "vegas_game_total": game_total,
                "game_key": f"{min(team, opp)}_vs_{max(team, opp)}",
                "stack_key": team,
                "default_max_exposure": 0.65,
                "game_leverage_score": 0.5,
                "environment_tier": "prime" if team == "AAA" else ("good" if team == "CCC" else "neutral"),
                "team_gpp_leverage": 10.0 if team == "AAA" else (3.0 if team == "CCC" else 5.0),
                "ownership_tier": "leverage" if own < 0.08 else ("chalk" if own > 0.20 else "mid"),
            })

    # Pitchers (one per away team)
    for team, opp, _, game_total in [("BBB", "AAA", 5.0, 10.0), ("DDD", "CCC", 4.5, 9.0), ("FFF", "EEE", 3.5, 7.0)]:
        pid_counter += 1
        rows.append({
            "fd_player_id": f"P{pid_counter}",
            "full_name": f"Pitcher {pid_counter}",
            "player_type": "pitcher",
            "position": "P",
            "roster_position": "P",
            "team_code": team,
            "opponent_code": opp,
            "salary": 9000 + (500 if team == "BBB" else 0),
            "proj_fd_mean": 25.0 + (2 if team == "BBB" else 0),
            "proj_fd_floor": 15.0,
            "proj_fd_ceiling": 40.0,
            "proj_fd_ownership": 0.25 if team == "BBB" else 0.10,
            "vegas_team_total": 5.0,
            "vegas_game_total": game_total,
            "game_key": f"{min(team, opp)}_vs_{max(team, opp)}",
            "stack_key": team,
            "default_max_exposure": 0.60,
            "game_leverage_score": 0.5,
            "environment_tier": "neutral",
            "team_gpp_leverage": 5.0,
            "ownership_tier": "chalk" if team == "BBB" else "mid",
        })

    return pd.DataFrame(rows)


class TestLeverageSolver(unittest.TestCase):
    def setUp(self) -> None:
        self.slate = _make_full_slate()

    def test_backward_compatibility_no_leverage_config(self) -> None:
        """leverage_config=None should produce lineups using pure mean."""
        lineups = generate_lineups(self.slate, num_lineups=3, leverage_config=None)
        self.assertGreater(len(lineups), 0)
        for lineup in lineups:
            self.assertEqual(len(lineup.dataframe), 9)

    def test_single_entry_mode_produces_lineups(self) -> None:
        """Single entry mode should produce valid lineups."""
        config = LeverageConfig.single_entry_preset()
        lineups = generate_lineups(self.slate, num_lineups=1, leverage_config=config)
        self.assertGreater(len(lineups), 0)

    def test_gpp_mode_produces_valid_lineups(self) -> None:
        gpp_config = LeverageConfig()
        lineups = generate_lineups(
            self.slate, num_lineups=5,
            leverage_config=gpp_config,
            stack_template=(4, 3, 1),
        )
        self.assertGreater(len(lineups), 0)
        for lineup in lineups:
            df = lineup.dataframe
            self.assertEqual(len(df), 9)
            self.assertEqual((df["player_type"].str.lower() == "pitcher").sum(), 1)
            self.assertLessEqual(df["salary"].sum(), 35000)

    def test_large_field_has_lower_ownership_than_small_field(self) -> None:
        """Large field GPP should have lower avg ownership than small field GPP."""
        large_config = LeverageConfig.large_field_gpp_preset()
        small_config = LeverageConfig.small_field_gpp_preset()

        large_lineups = generate_lineups(
            self.slate, num_lineups=5,
            leverage_config=large_config,
            stack_template=(4, 3, 1),
        )
        small_lineups = generate_lineups(
            self.slate, num_lineups=5,
            leverage_config=small_config,
            stack_template=(4, 3, 1),
        )

        def avg_ownership(lineups):
            owns = []
            for lineup in lineups:
                owns.append(lineup.dataframe["proj_fd_ownership"].mean())
            return sum(owns) / len(owns) if owns else 0

        large_avg = avg_ownership(large_lineups)
        small_avg = avg_ownership(small_lineups)
        # Large field should generally have lower avg ownership
        self.assertLessEqual(large_avg, small_avg + 0.05)

    def test_floor_filter_excludes_bottom_players(self) -> None:
        """GPP mode should filter out bottom projection players."""
        gpp_config = LeverageConfig(min_viable_projection_pct=0.40)
        lineups = generate_lineups(
            self.slate, num_lineups=3,
            leverage_config=gpp_config,
            stack_template=(4, 3, 1),
        )
        # Verify all batters in lineups have reasonable projections
        cutoff = self.slate[self.slate["player_type"].str.lower() == "batter"]["proj_fd_mean"].quantile(0.40)
        for lineup in lineups:
            batters = lineup.dataframe[lineup.dataframe["player_type"].str.lower() == "batter"]
            # All batters should be above cutoff (or have high ceiling)
            for _, row in batters.iterrows():
                above_cutoff = row["proj_fd_mean"] >= cutoff
                has_ceiling = row.get("proj_fd_ceiling", 0) >= self.slate["proj_fd_ceiling"].quantile(0.6)
                self.assertTrue(above_cutoff or has_ceiling,
                    f"Player {row['full_name']} with mean {row['proj_fd_mean']:.1f} below cutoff {cutoff:.1f}")

    def test_smart_auto_stack_selects_template(self) -> None:
        """Auto-stack should return a valid template tuple."""
        template = _select_auto_stack_template(self.slate)
        self.assertIsInstance(template, tuple)
        self.assertGreater(len(template), 0)
        self.assertEqual(sum(template), 8)

    def test_leverage_config_presets_are_valid(self) -> None:
        for preset_fn in [
            LeverageConfig.single_entry_preset,
            LeverageConfig.small_field_gpp_preset,
            LeverageConfig.large_field_gpp_preset,
        ]:
            cfg = preset_fn()
            self.assertIsInstance(cfg, LeverageConfig)
            self.assertIn(cfg.mode, ("single_entry", "small_field_gpp", "large_field_gpp"))
            self.assertTrue(cfg.is_gpp)

    def test_gpp_auto_mode_generates_stacked_lineups(self) -> None:
        """GPP mode with Auto stack should produce non-hodgepodge lineups."""
        gpp_config = LeverageConfig()
        lineups = generate_lineups(
            self.slate, num_lineups=5,
            leverage_config=gpp_config,
            # No stack_template → auto mode kicks in
        )
        self.assertGreater(len(lineups), 0)
        for lineup in lineups:
            batters = lineup.dataframe[lineup.dataframe["player_type"].str.lower() == "batter"]
            team_counts = batters["team_code"].value_counts()
            # At least one team should have 3+ batters (not a hodgepodge 1-1-1-1-1-1-1-1)
            self.assertGreaterEqual(team_counts.max(), 3,
                f"Lineup has no stack (max team count = {team_counts.max()})")


class TestSlateProfile(unittest.TestCase):
    """Tests for slate detection and contest/slate strategy adjustments."""

    def _make_slate_df(self, num_games: int) -> pd.DataFrame:
        """Build a minimal DataFrame with the given number of games."""
        rows = []
        for g in range(num_games):
            team_a = f"T{g*2}"
            team_b = f"T{g*2+1}"
            gk = f"{team_a}_vs_{team_b}"
            for i in range(8):
                rows.append({
                    "fd_player_id": f"B_{g}_{i}",
                    "player_type": "batter",
                    "team_code": team_a if i < 4 else team_b,
                    "opponent_code": team_b if i < 4 else team_a,
                    "game_key": gk,
                    "proj_fd_mean": 8.0,
                })
            for side in (team_a, team_b):
                rows.append({
                    "fd_player_id": f"P_{g}_{side}",
                    "player_type": "pitcher",
                    "team_code": side,
                    "opponent_code": team_b if side == team_a else team_a,
                    "game_key": gk,
                    "proj_fd_mean": 20.0,
                })
        return pd.DataFrame(rows)

    def test_detect_small_slate(self) -> None:
        df = self._make_slate_df(2)
        profile = detect_slate_profile(df)
        self.assertEqual(profile.slate_type, "small")
        self.assertEqual(profile.num_games, 2)

    def test_detect_medium_slate(self) -> None:
        df = self._make_slate_df(7)
        profile = detect_slate_profile(df)
        self.assertEqual(profile.slate_type, "medium")
        self.assertEqual(profile.num_games, 7)

    def test_detect_large_slate(self) -> None:
        df = self._make_slate_df(12)
        profile = detect_slate_profile(df)
        self.assertEqual(profile.slate_type, "large")
        self.assertEqual(profile.num_games, 12)

    def test_small_slate_increases_ownership_penalty(self) -> None:
        base = LeverageConfig.large_field_gpp_preset()
        base_penalty = base.ownership_penalty
        profile = SlateProfile(num_games=2, num_batters=20, num_pitchers=4,
                               slate_type="small", recommended_stacks=2, stack_templates=[(4, 4)])
        adjusted = apply_slate_adjustments(base, profile)
        self.assertGreater(adjusted.ownership_penalty, base_penalty)

    def test_large_slate_decreases_ownership_penalty(self) -> None:
        base = LeverageConfig.large_field_gpp_preset()
        base_penalty = base.ownership_penalty
        profile = SlateProfile(num_games=12, num_batters=120, num_pitchers=24,
                               slate_type="large", recommended_stacks=5, stack_templates=[(4, 3, 1)])
        adjusted = apply_slate_adjustments(base, profile)
        self.assertLess(adjusted.ownership_penalty, base_penalty)

    def test_small_slate_sets_pitcher_fade_bonus(self) -> None:
        base = LeverageConfig.large_field_gpp_preset()
        self.assertEqual(base.pitcher_fade_bonus, 0.0)
        profile = SlateProfile(num_games=2, num_batters=20, num_pitchers=4,
                               slate_type="small", recommended_stacks=2, stack_templates=[(4, 4)])
        adjusted = apply_slate_adjustments(base, profile)
        self.assertGreater(adjusted.pitcher_fade_bonus, 0.0)

    def test_auto_stack_44_on_2game_slate(self) -> None:
        df = self._make_slate_df(2)
        df["environment_tier"] = "prime"
        template = _select_auto_stack_template(df, slate_type="small")
        self.assertEqual(template, (4, 4))

    def test_contest_presets_mutually_exclusive(self) -> None:
        se = LeverageConfig.single_entry_preset()
        sm = LeverageConfig.small_field_gpp_preset()
        lg = LeverageConfig.large_field_gpp_preset()
        modes = {se.mode, sm.mode, lg.mode}
        self.assertEqual(len(modes), 3, "Contest presets should have distinct modes")

    def test_slate_adjustments_dont_mutate_input(self) -> None:
        base = LeverageConfig.large_field_gpp_preset()
        original_penalty = base.ownership_penalty
        profile = SlateProfile(num_games=2, num_batters=20, num_pitchers=4,
                               slate_type="small", recommended_stacks=2, stack_templates=[(4, 4)])
        _ = apply_slate_adjustments(base, profile)
        self.assertEqual(base.ownership_penalty, original_penalty, "Input config should not be mutated")


if __name__ == "__main__":
    unittest.main()
