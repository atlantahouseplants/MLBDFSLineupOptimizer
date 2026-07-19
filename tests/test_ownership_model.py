from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.projection.ownership import compute_ownership_series
from slate_optimizer.projection.ownership_model import (
    HITTER_BUDGET,
    PITCHER_BUDGET,
    StructuralOwnershipParams,
    build_ownership_features,
    estimate_slate_games,
    estimate_structural_ownership,
    load_ownership_params,
    save_ownership_params,
)
from slate_optimizer.simulation.field_learning import compute_actual_ownership


def _synthetic_slate(n_teams: int = 8, hitters_per_team: int = 9) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    rows = []
    player_id = 0
    for team_idx in range(n_teams):
        team = f"T{team_idx}"
        opponent = f"T{(team_idx + 1) % n_teams}"
        rows.append(
            {
                "fd_player_id": f"p{player_id}",
                "full_name": f"Pitcher {player_id}",
                "position": "P",
                "player_type": "pitcher",
                "team_code": team,
                "salary": int(rng.integers(6000, 11500)),
                "fppg": float(rng.uniform(20, 40)),
                "bpp_strikeouts": float(rng.uniform(3, 9)),
                "bpp_win_pct": float(rng.uniform(0.3, 0.6)),
                "bpp_quality_start": float(rng.uniform(0.1, 0.5)),
                "bpp_runs": np.nan,
                "bpp_game_pk": str(800000 + team_idx // 2),
            }
        )
        player_id += 1
        team_runs = float(rng.uniform(3.0, 6.5))
        for order in range(1, hitters_per_team + 1):
            rows.append(
                {
                    "fd_player_id": f"p{player_id}",
                    "full_name": f"Hitter {player_id}",
                    "position": "OF",
                    "player_type": "batter",
                    "team_code": team,
                    "salary": int(rng.integers(2200, 4600)),
                    "fppg": float(rng.uniform(5, 14)),
                    "batting_order_position": order,
                    "is_confirmed_lineup": True,
                    "bpp_home_run_probability": float(rng.uniform(0.02, 0.25)),
                    "bpp_runs": team_runs,
                    "bpp_game_pk": str(800000 + team_idx // 2),
                }
            )
            player_id += 1
    players = pd.DataFrame(rows)
    projections = players[["fd_player_id", "salary", "player_type"]].copy()
    is_pitcher = projections["player_type"] == "pitcher"
    base = projections["salary"] / 1000.0 * 3.0
    noise = rng.uniform(0.9, 1.15, len(projections))
    projections["proj_fd_mean"] = base * noise * np.where(is_pitcher, 3.0, 1.0)
    return players, projections[["fd_player_id", "proj_fd_mean"]]


class TestStructuralOwnership(unittest.TestCase):
    def test_budget_sums(self) -> None:
        players, projections = _synthetic_slate()
        ownership = estimate_structural_ownership(players, projections)
        features = build_ownership_features(players, projections)
        pitcher_ids = set(features.loc[features["player_type"] == "pitcher", "fd_player_id"])
        pitcher_sum = ownership[ownership.index.isin(pitcher_ids)].sum()
        hitter_sum = ownership[~ownership.index.isin(pitcher_ids)].sum()
        self.assertAlmostEqual(pitcher_sum, PITCHER_BUDGET, delta=0.02)
        self.assertAlmostEqual(hitter_sum, HITTER_BUDGET, delta=0.05)

    def test_ownership_in_valid_range(self) -> None:
        players, projections = _synthetic_slate()
        ownership = estimate_structural_ownership(players, projections)
        self.assertTrue((ownership >= 0.0).all())
        self.assertTrue((ownership <= 1.0).all())
        self.assertEqual(len(ownership), len(players))

    def test_value_drives_ownership(self) -> None:
        players, projections = _synthetic_slate()
        # Force one hitter to be a massive value play, another to be a punt with no projection.
        proj = projections.set_index("fd_player_id")
        features = build_ownership_features(players, projections)
        hitters = features[features["player_type"] == "batter"]["fd_player_id"].tolist()
        stud, dud = hitters[0], hitters[1]
        proj.loc[stud, "proj_fd_mean"] = 25.0
        proj.loc[dud, "proj_fd_mean"] = 0.5
        players = players.copy()
        players.loc[players["fd_player_id"] == stud, "salary"] = 2500
        players.loc[players["fd_player_id"] == dud, "salary"] = 4500
        ownership = estimate_structural_ownership(players, proj.reset_index())
        self.assertGreater(ownership[stud], ownership[dud] * 3)

    def test_small_slate_concentrates_ownership(self) -> None:
        small_players, small_proj = _synthetic_slate(n_teams=4)
        large_players, large_proj = _synthetic_slate(n_teams=24)
        small_own = estimate_structural_ownership(small_players, small_proj)
        large_own = estimate_structural_ownership(large_players, large_proj)
        # Highest-owned hitter on a 2-game slate should exceed the top hitter on a 12-game slate.
        self.assertGreater(small_own.max(), large_own.max())

    def test_missing_optional_columns(self) -> None:
        players = pd.DataFrame(
            {
                "fd_player_id": ["a", "b", "c", "d"],
                "position": ["P", "OF", "SS", "1B"],
                "salary": [9000, 3000, 3500, 4000],
            }
        )
        projections = pd.DataFrame(
            {
                "fd_player_id": ["a", "b", "c", "d"],
                "proj_fd_mean": [35.0, 10.0, 11.0, 12.0],
            }
        )
        ownership = estimate_structural_ownership(players, projections)
        self.assertEqual(len(ownership), 4)
        self.assertTrue(np.isfinite(ownership).all())

    def test_estimate_slate_games(self) -> None:
        players, _ = _synthetic_slate(n_teams=8)
        self.assertEqual(estimate_slate_games(players), 4)

    def test_params_roundtrip(self, ) -> None:
        import tempfile

        params = StructuralOwnershipParams(
            hitter_weights={"value": 2.0},
            pitcher_weights={"proj_mean": 1.5},
            hitter_temperature=0.9,
            pitcher_temperature=0.7,
            slates_used=12,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ownership_model.json"
            save_ownership_params(params, path)
            loaded = load_ownership_params(path)
        self.assertEqual(loaded.hitter_weights, {"value": 2.0})
        self.assertEqual(loaded.pitcher_weights, {"proj_mean": 1.5})
        self.assertAlmostEqual(loaded.pitcher_temperature, 0.7)
        self.assertEqual(loaded.slates_used, 12)

    def test_compute_ownership_series_uses_structural_model(self) -> None:
        players, projections = _synthetic_slate()
        result = compute_ownership_series(players, projections, source_paths=[])
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].name, "structural_model")
        pitcher_ids = set(players.loc[players["player_type"] == "pitcher", "fd_player_id"])
        pitcher_sum = result.ownership[result.ownership.index.isin(pitcher_ids)].sum()
        self.assertAlmostEqual(pitcher_sum, PITCHER_BUDGET, delta=0.02)


class TestActualOwnership(unittest.TestCase):
    def test_compute_actual_ownership_from_contest_export(self) -> None:
        roster_cols = ["P", "C/1B", "2B", "3B", "SS", "OF", "OF.1", "OF.2", "UTIL"]
        base = [
            "Gerrit Cole", "Freddie Freeman", "Jose Altuve", "Jose Ramirez",
            "Bobby Witt", "Aaron Judge", "Juan Soto", "Kyle Tucker", "Yordan Alvarez",
        ]
        alt = [
            "Zack Wheeler", "Freddie Freeman", "Marcus Semien", "Austin Riley",
            "Bobby Witt", "Ronald Acuna", "Juan Soto", "Corbin Carroll", "Yordan Alvarez",
        ]
        contest_df = pd.DataFrame([base, base, alt, alt], columns=roster_cols)
        optimizer_df = pd.DataFrame(
            {
                "full_name": base + alt,
                "fd_player_id": [f"id{i}" for i in range(18)],
            }
        ).drop_duplicates("full_name")

        ownership = compute_actual_ownership(contest_df, optimizer_df)
        own_map = ownership.set_index("player_name")["actual_ownership_pct"].to_dict()
        self.assertAlmostEqual(own_map["Freddie Freeman"], 1.0)
        self.assertAlmostEqual(own_map["Gerrit Cole"], 0.5)
        self.assertAlmostEqual(own_map["Zack Wheeler"], 0.5)
        matched = ownership.dropna(subset=["fd_player_id"])
        self.assertEqual(len(matched), len(ownership))

    def test_empty_contest_returns_empty(self) -> None:
        result = compute_actual_ownership(pd.DataFrame())
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
