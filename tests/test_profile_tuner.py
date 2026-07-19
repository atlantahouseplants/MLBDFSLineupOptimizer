from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.analysis.profile_tuner import build_profile_tuning_report
from slate_optimizer.data.storage import SlateDatabase
from slate_optimizer.simulation.field_learning import match_lineups_to_contest_entries

PROFILES = {
    "medium": {
        "optimizer": {"ownership_penalty_weight": 10.0, "ceiling_weight": 35.0},
    },
}


def _config_json(profile: str) -> str:
    return json.dumps({"slate_profile": profile, "slate_games": 7, "optimizer": {}})


def _slate_results(date_idx: int, chalk_wins: bool, upside_wins: bool) -> pd.DataFrame:
    rng = np.random.default_rng(date_idx)
    n = 10
    ownership = np.linspace(0.8, 2.4, n)
    upside = np.linspace(90, 140, n)
    base = rng.normal(0, 2, n)
    actual = 100 + base
    actual = actual + (ownership * 20 if chalk_wins else ownership * -20)
    actual = actual + (upside * 0.5 if upside_wins else upside * -0.5)
    return pd.DataFrame(
        {
            "lineup_id": range(1, n + 1),
            "total_actual_points": actual,
            "rank": [None] * n,
            "payout": [None] * n,
            "roi": [0.1] * n,
            "strategy_config_json": [_config_json("medium")] * n,
            "total_ownership": ownership,
            "total_upside": upside,
            "total_salary": [34500] * n,
        }
    )


class TestProfileTuner(unittest.TestCase):
    def _build_db(self, n_slates: int, chalk_wins: bool, upside_wins: bool) -> Path:
        tmp = Path(tempfile.mkdtemp())
        db_path = tmp / "slates.db"
        db = SlateDatabase(db_path)
        for i in range(n_slates):
            date = f"202606{i + 10:02d}"
            db.insert_lineup_results(date, _slate_results(i, chalk_wins, upside_wins))
        db.close()
        return db_path

    def test_chalk_outperformance_lowers_penalty(self) -> None:
        db_path = self._build_db(8, chalk_wins=True, upside_wins=True)
        report = build_profile_tuning_report(db_path, "20260601", "20260630", PROFILES)
        self.assertFalse(report.recommendations.empty)
        own_rec = report.recommendations[report.recommendations["setting"] == "ownership_penalty_weight"]
        self.assertEqual(len(own_rec), 1)
        self.assertEqual(float(own_rec.iloc[0]["recommended"]), 8.0)
        ceil_rec = report.recommendations[report.recommendations["setting"] == "ceiling_weight"]
        self.assertEqual(float(ceil_rec.iloc[0]["recommended"]), 40.0)
        self.assertIn("medium", report.adjustments)

    def test_leverage_outperformance_raises_penalty(self) -> None:
        db_path = self._build_db(8, chalk_wins=False, upside_wins=False)
        report = build_profile_tuning_report(db_path, "20260601", "20260630", PROFILES)
        own_rec = report.recommendations[report.recommendations["setting"] == "ownership_penalty_weight"]
        self.assertEqual(float(own_rec.iloc[0]["recommended"]), 12.0)
        ceil_rec = report.recommendations[report.recommendations["setting"] == "ceiling_weight"]
        self.assertEqual(float(ceil_rec.iloc[0]["recommended"]), 30.0)

    def test_small_sample_gated(self) -> None:
        db_path = self._build_db(3, chalk_wins=True, upside_wins=True)
        report = build_profile_tuning_report(db_path, "20260601", "20260630", PROFILES)
        self.assertTrue(report.recommendations.empty)
        self.assertTrue(any("3/8 slates" in note for note in report.notes))
        self.assertFalse(report.per_profile.empty)

    def test_untagged_results_skipped(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        db_path = tmp / "slates.db"
        db = SlateDatabase(db_path)
        frame = _slate_results(1, True, True)
        frame["strategy_config_json"] = json.dumps({"optimizer": {}})  # no slate_profile key
        db.insert_lineup_results("20260610", frame)
        db.close()
        report = build_profile_tuning_report(db_path, "20260601", "20260630", PROFILES)
        self.assertTrue(report.recommendations.empty)
        self.assertTrue(report.notes)


class TestEntryMatching(unittest.TestCase):
    def test_match_rank_and_payout_by_signature(self) -> None:
        our_players = [f"Player {chr(65 + i)}" for i in range(9)]
        other_players = [f"Rival {chr(65 + i)}" for i in range(9)]
        contest_df = pd.DataFrame(
            {
                "Rank": [1, 50, 2000],
                "Winnings": ["$500.00", "$10", ""],
                "P": [our_players[0], other_players[0], our_players[0]],
                "C/1B": [our_players[1], other_players[1], our_players[1]],
                "2B": [our_players[2], other_players[2], our_players[2]],
                "3B": [our_players[3], other_players[3], our_players[3]],
                "SS": [our_players[4], other_players[4], our_players[4]],
                "OF": [our_players[5], other_players[5], our_players[5]],
                "OF.1": [our_players[6], other_players[6], our_players[6]],
                "OF.2": [our_players[7], other_players[7], our_players[7]],
                "UTIL": [our_players[8], other_players[8], our_players[8]],
            }
        )
        lineup_df = pd.DataFrame(
            {
                "lineup_id": [1] * 9 + [2] * 9,
                "full_name": our_players + [f"Nobody {i}" for i in range(9)],
            }
        )
        matches = match_lineups_to_contest_entries(contest_df, lineup_df)
        self.assertEqual(len(matches), 1)
        row = matches.iloc[0]
        self.assertEqual(int(row["lineup_id"]), 1)
        # Duplicate field copy at rank 2000: we keep the best rank / payout.
        self.assertEqual(int(row["rank"]), 1)
        self.assertAlmostEqual(float(row["payout"]), 500.0)

    def test_empty_inputs(self) -> None:
        self.assertTrue(match_lineups_to_contest_entries(pd.DataFrame(), pd.DataFrame()).empty)


if __name__ == "__main__":
    unittest.main()
