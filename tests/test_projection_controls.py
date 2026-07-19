from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.projection import (
    OwnershipModelConfig,
    blend_projection_sources,
    compute_baseline_projections,
    compute_ownership_series,
)


class TestProjectionControls(unittest.TestCase):
    def setUp(self) -> None:
        self.players = pd.DataFrame(
            [
                {
                    "fd_player_id": "H1",
                    "full_name": "Lefty Leadoff",
                    "player_type": "batter",
                    "position": "OF",
                    "salary": 3600,
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
                    "salary": 3200,
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
                    "salary": 8500,
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
                    "salary": 8200,
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

    def test_projection_blend_weights_players(self) -> None:
        base = compute_baseline_projections(self.players)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "source.csv"
            pd.DataFrame(
                [
                    {"fd_player_id": "H1", "proj_fd_mean": 20.0, "proj_fd_floor": 10.0, "proj_fd_ceiling": 30.0},
                    {"fd_player_id": "H2", "proj_fd_mean": 5.0, "proj_fd_floor": 2.0, "proj_fd_ceiling": 12.0},
                ]
            ).to_csv(path, index=False)

            blended, summary = blend_projection_sources(
                self.players,
                base,
                source_paths=[path],
                weights=[1.0],
                baseline_weight=1.0,
            )

        self.assertAlmostEqual(summary.baseline_share, 0.5, places=2)
        self.assertEqual(summary.source_count, 1)
        self.assertAlmostEqual(summary.sources[0].weight, 0.5, places=2)

        base_map = base.set_index("fd_player_id")
        blended_map = blended.set_index("fd_player_id")
        expected_h1 = 0.5 * base_map.loc["H1", "proj_fd_mean"] + 0.5 * 20.0
        self.assertAlmostEqual(blended_map.loc["H1", "proj_fd_mean"], expected_h1, places=5)

        expected_floor = 0.5 * base_map.loc["H1", "proj_fd_floor"] + 0.5 * 10.0
        self.assertAlmostEqual(blended_map.loc["H1", "proj_fd_floor"], expected_floor, places=5)

    def test_projection_blend_fallback_to_baseline(self) -> None:
        base = compute_baseline_projections(self.players)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mean_only.csv"
            pd.DataFrame(
                [
                    {"fd_player_id": "H1", "proj_fd_mean": 18.0},
                ]
            ).to_csv(path, index=False)

            blended, summary = blend_projection_sources(
                self.players,
                base,
                source_paths=[path],
                weights=[1.0],
                baseline_weight=0.0,
            )

        self.assertAlmostEqual(summary.baseline_share, 0.0, places=5)
        blended_map = blended.set_index("fd_player_id")
        # Player covered by the source should adopt the external value.
        self.assertAlmostEqual(blended_map.loc["H1", "proj_fd_mean"], 18.0, places=4)
        # Players missing from the external file should retain their baseline projection.
        self.assertAlmostEqual(
            blended_map.loc["H2", "proj_fd_mean"], base.set_index("fd_player_id").loc["H2", "proj_fd_mean"], places=5
        )

    def test_projection_blend_normalizes_disjoint_sources_per_player(self) -> None:
        base = compute_baseline_projections(self.players)
        with tempfile.TemporaryDirectory() as tmpdir:
            hitter_path = Path(tmpdir) / "hitters.csv"
            pitcher_path = Path(tmpdir) / "pitchers.csv"
            pd.DataFrame([{"fd_player_id": "H1", "proj_fd_mean": 20.0}]).to_csv(hitter_path, index=False)
            pd.DataFrame([{"fd_player_id": "P_BBB", "proj_fd_mean": 45.0}]).to_csv(pitcher_path, index=False)

            blended, _ = blend_projection_sources(
                self.players,
                base,
                source_paths=[hitter_path, pitcher_path],
                weights=[1.0, 1.0],
                baseline_weight=0.0,
            )

        blended_map = blended.set_index("fd_player_id")
        self.assertAlmostEqual(blended_map.loc["H1", "proj_fd_mean"], 20.0, places=5)
        self.assertAlmostEqual(blended_map.loc["P_BBB", "proj_fd_mean"], 45.0, places=5)
        self.assertAlmostEqual(
            blended_map.loc["H2", "proj_fd_mean"], base.set_index("fd_player_id").loc["H2", "proj_fd_mean"], places=5
        )

    def test_ballpark_optimizer_xlsx_blend_imports_sim_columns(self) -> None:
        base = compute_baseline_projections(self.players)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ballpark_optimizer.xlsx"
            source = pd.DataFrame(
                [
                    {
                        "Tm": "AAA",
                        "Pos": "OF",
                        "Players": "L. Leadoff",
                        "$": 3.6,
                        "Points": 20.0,
                        "Bust": 0.12,
                        "Median": 14.0,
                        "Upside": 36.0,
                        "Pts/$": 5.6,
                        "Med/$": 3.9,
                        "Ups/$": 10.0,
                        "PlayerType": "b",
                        "StackCount": 1,
                        "FullName": "Lefty Leadoff",
                    }
                ]
            )
            with pd.ExcelWriter(path) as writer:
                source.to_excel(writer, index=False, startrow=1)
                writer.sheets["Sheet1"].cell(row=1, column=1).value = "Ballpark DFS Optimizer | Ballpark Pal"

            blended, summary = blend_projection_sources(
                self.players,
                base,
                source_paths=[path],
                weights=[1.0],
                baseline_weight=1.0,
            )

        self.assertEqual(summary.source_count, 1)
        self.assertTrue(summary.sources[0].has_bust)
        self.assertTrue(summary.sources[0].has_median)
        self.assertTrue(summary.sources[0].has_upside)
        row = blended.set_index("fd_player_id").loc["H1"]
        expected_mean = 0.5 * base.set_index("fd_player_id").loc["H1", "proj_fd_mean"] + 0.5 * 20.0
        self.assertAlmostEqual(row["proj_fd_mean"], expected_mean, places=5)
        self.assertAlmostEqual(row["proj_fd_bust_rate"], 0.12, places=5)
        self.assertAlmostEqual(row["proj_fd_median"], 14.0, places=5)
        self.assertAlmostEqual(row["proj_fd_upside"], 36.0, places=5)

    def test_ballpark_dfs_xlsx_title_row_blend_imports_sim_columns(self) -> None:
        base = compute_baseline_projections(self.players)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ballpark_dfs.xlsx"
            source = pd.DataFrame(
                [
                    {
                        "Tm": "AAA",
                        "Pos": "OF",
                        "Players": "L. Leadoff",
                        "$": 3.6,
                        "Points": 20.0,
                        "Bust": 0.12,
                        "Median": 14.0,
                        "Upside": 36.0,
                        "Pts/$": 5.6,
                        "Med/$": 3.9,
                        "Ups/$": 10.0,
                        "PlayerType": "b",
                        "StackCount": 1,
                        "FullName": "Lefty Leadoff",
                    }
                ]
            )
            with pd.ExcelWriter(path) as writer:
                source.to_excel(writer, index=False, startrow=1)
                writer.sheets["Sheet1"].cell(row=1, column=1).value = "Ballpark DFS | Ballpark Pal"

            blended, summary = blend_projection_sources(
                self.players,
                base,
                source_paths=[path],
                weights=[1.0],
                baseline_weight=1.0,
            )

        self.assertEqual(summary.source_count, 1)
        self.assertTrue(summary.sources[0].has_bust)
        self.assertTrue(summary.sources[0].has_median)
        self.assertTrue(summary.sources[0].has_upside)
        row = blended.set_index("fd_player_id").loc["H1"]
        self.assertAlmostEqual(row["proj_fd_bust_rate"], 0.12, places=5)
        self.assertAlmostEqual(row["proj_fd_median"], 14.0, places=5)
        self.assertAlmostEqual(row["proj_fd_upside"], 36.0, places=5)

    def test_projection_blend_drops_unmatched_name_rows(self) -> None:
        base = compute_baseline_projections(self.players)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "source.csv"
            pd.DataFrame(
                [
                    {"full_name": "Lefty Leadoff", "proj_fd_mean": 19.0},
                    {"full_name": "Unknown Player", "proj_fd_mean": 99.0},
                ]
            ).to_csv(path, index=False)

            blended, summary = blend_projection_sources(
                self.players,
                base,
                source_paths=[path],
                weights=[1.0],
                baseline_weight=0.0,
            )

        blended_map = blended.set_index("fd_player_id")
        self.assertEqual(summary.sources[0].matched_players, 1)
        self.assertEqual(summary.sources[0].source_players, 2)
        self.assertIn("Unknown Player", summary.sources[0].unmatched_players)
        self.assertAlmostEqual(blended_map.loc["H1", "proj_fd_mean"], 19.0, places=5)
        self.assertNotIn("nan", set(blended["fd_player_id"].astype(str).str.lower()))

    def test_projection_blend_uses_suffix_alias_name_matching(self) -> None:
        players = self.players.copy()
        players.loc[players["fd_player_id"] == "H1", "full_name"] = "Jazz Chisholm Jr."
        base = compute_baseline_projections(players)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "source.csv"
            pd.DataFrame(
                [{"full_name": "Jazz Chisholm", "proj_fd_mean": 21.0, "Bust": 0.15, "Median": 12.0, "Upside": 32.0}]
            ).to_csv(path, index=False)

            blended, summary = blend_projection_sources(
                players,
                base,
                source_paths=[path],
                weights=[1.0],
                baseline_weight=0.0,
            )

        self.assertEqual(summary.sources[0].matched_players, 1)
        self.assertEqual(summary.sources[0].unmatched_players, [])
        self.assertAlmostEqual(blended.set_index("fd_player_id").loc["H1", "proj_fd_mean"], 21.0, places=5)

    def test_projection_blend_handles_duplicate_baseline_ids(self) -> None:
        base = compute_baseline_projections(self.players)
        duplicate = base[base["fd_player_id"] == "H1"].copy()
        duplicate["proj_fd_mean"] = duplicate["proj_fd_mean"] + 2.0
        duplicate_base = pd.concat([base, duplicate], ignore_index=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "source.csv"
            pd.DataFrame([{"fd_player_id": "H1", "proj_fd_mean": 20.0}]).to_csv(path, index=False)

            blended, _ = blend_projection_sources(
                self.players,
                duplicate_base,
                source_paths=[path],
                weights=[1.0],
                baseline_weight=1.0,
            )

        base_h1 = base.set_index("fd_player_id").loc["H1", "proj_fd_mean"]
        expected = 0.5 * (base_h1 + 1.0) + 0.5 * 20.0
        h1_values = blended.loc[blended["fd_player_id"] == "H1", "proj_fd_mean"].drop_duplicates()
        self.assertEqual(len(h1_values), 1)
        self.assertAlmostEqual(float(h1_values.iloc[0]), expected, places=5)

    def test_ownership_model_config_custom_weights(self) -> None:
        base = compute_baseline_projections(self.players)
        tight_range_cfg = OwnershipModelConfig(
            salary_weight=0.0,
            projection_weight=1.0,
            value_weight=0.0,
            team_weight=0.0,
            name_weight=0.0,
            position_weight=0.0,
            min_pct=0.1,
            max_pct=0.2,
        )
        ownership_result = compute_ownership_series(
            self.players,
            base,
            model_config=tight_range_cfg,
        )
        series = ownership_result.ownership.astype(float)
        self.assertTrue(((series >= 0.1) & (series <= 0.2)).all())

        salary_cfg = OwnershipModelConfig(
            salary_weight=1.0,
            projection_weight=0.0,
            value_weight=0.0,
            team_weight=0.0,
            name_weight=0.0,
            position_weight=0.0,
            min_pct=0.01,
            max_pct=0.5,
        )
        salary_result = compute_ownership_series(
            self.players,
            base,
            model_config=salary_cfg,
        )
        salary_series = salary_result.ownership.astype(float)
        self.assertGreater(salary_series.loc["H1"], salary_series.loc["H2"])

    def test_ownership_source_alias_matching_reports_coverage(self) -> None:
        players = self.players.copy()
        players.loc[players["fd_player_id"] == "H1", "full_name"] = "Jazz Chisholm Jr."
        base = compute_baseline_projections(players)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ownership.csv"
            pd.DataFrame(
                [
                    {"full_name": "Jazz Chisholm", "ownership": 18.0},
                    {"full_name": "Unknown Hitter", "ownership": 8.0},
                ]
            ).to_csv(path, index=False)

            result = compute_ownership_series(players, base, source_paths=[path])

        self.assertEqual(result.source_count, 1)
        self.assertEqual(result.sources[0].matched_players, 1)
        self.assertIn("Unknown Hitter", result.sources[0].unmatched_players)
        self.assertTrue(bool(result.external_coverage.loc["H1"]))
        self.assertAlmostEqual(float(result.ownership.loc["H1"]), 0.18, places=5)


if __name__ == "__main__":
    unittest.main()
