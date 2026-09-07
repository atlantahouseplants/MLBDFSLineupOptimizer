from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.simulation.sim_calibration import (
    HISTORY_COLUMNS,
    UnmatchedHeaderError,
    append_history,
    brier_score,
    calibration_by_decile,
    format_report,
    join_entered_lineups,
    lineup_signature,
    parse_player_ids_cell,
    resolve_entries_columns,
    score_calibration,
    spearman_correlation,
)


def _sim_row(lineup_id: int, players: list[str], cash: float, top1: float, win: float = 0.01) -> dict:
    return {
        "lineup_id": lineup_id,
        "player_ids": str(players),
        "cash_rate": cash,
        "top_1pct_rate": top1,
        "win_rate": win,
        "expected_roi": 1.0 + cash,
    }


def _upload_row(players: list[str]) -> dict:
    p, c, second, third, ss, of1, of2, of3, util = players
    return {
        "P": p,
        "C/1B": c,
        "2B": second,
        "3B": third,
        "SS": ss,
        "OF": of1,
        "OF.1": of2,
        "OF.2": of3,
        "UTIL": util,
    }


def _ids(seed: int) -> list[str]:
    return [f"133202-{seed + offset}" for offset in range(9)]


class TestLineupSignature(unittest.TestCase):
    def test_order_independent_and_id_extraction(self) -> None:
        a = lineup_signature(["133202-1", "133202-2", "Name (133202-3)"])
        b = lineup_signature(["Name (133202-3)", "133202-2", "133202-1"])
        self.assertEqual(a, b)
        self.assertEqual(a, frozenset({"133202-1", "133202-2", "133202-3"}))

    def test_parse_player_ids_list_literal(self) -> None:
        parsed = parse_player_ids_cell("['133202-1', '133202-2']")
        self.assertEqual(parsed, ["133202-1", "133202-2"])


class TestBrierAndDeciles(unittest.TestCase):
    def test_brier_known_value(self) -> None:
        # ((0.7-1)^2 + (0.3-0)^2) / 2 = 0.09
        self.assertAlmostEqual(brier_score([0.7, 0.3], [1, 0]), 0.09)
        self.assertAlmostEqual(brier_score([1.0, 0.0], [1, 0]), 0.0)

    def test_decile_table_splits_high_vs_low(self) -> None:
        predicted = [0.05, 0.06, 0.07, 0.08, 0.90, 0.91, 0.92, 0.93]
        actual = [0, 0, 0, 0, 1, 1, 1, 1]
        table = calibration_by_decile(predicted, actual, n_bins=2)
        self.assertGreaterEqual(len(table), 2)
        self.assertIn("predicted_mean", table.columns)
        self.assertIn("actual_rate", table.columns)
        self.assertAlmostEqual(float(table["actual_rate"].iloc[0]), 0.0)
        self.assertAlmostEqual(float(table["actual_rate"].iloc[-1]), 1.0)
        self.assertLess(float(table["predicted_mean"].iloc[0]), float(table["predicted_mean"].iloc[-1]))

    def test_spearman_perfect_inverse_ranks(self) -> None:
        predicted = [0.9, 0.6, 0.3, 0.1]
        percentile = [0.01, 0.10, 0.40, 0.90]
        rho = spearman_correlation(predicted, percentile)
        self.assertAlmostEqual(rho, -1.0)


class TestJoinLogic(unittest.TestCase):
    def setUp(self) -> None:
        self.ids_a = _ids(100)
        self.ids_b = _ids(200)
        self.ids_c = _ids(300)
        self.sim = pd.DataFrame(
            [
                _sim_row(11, self.ids_a, 0.80, 0.20),
                _sim_row(22, self.ids_b, 0.40, 0.05),
                _sim_row(33, self.ids_c, 0.10, 0.01),
            ]
        )
        self.upload = pd.DataFrame([_upload_row(self.ids_a), _upload_row(self.ids_b)])

    def test_join_by_player_set_signature(self) -> None:
        entries = pd.DataFrame(
            [
                {
                    **_upload_row(self.ids_a),
                    "Rank": 10,
                    "Points": 180.5,
                    "Winnings": "$20",
                },
                {
                    **_upload_row(self.ids_b),
                    "Rank": 400,
                    "Points": 110.0,
                    "Winnings": "$0",
                },
            ]
        )
        joined = join_entered_lineups(self.sim, self.upload, entries, field_size=1000)
        self.assertEqual(len(joined), 2)
        by_id = joined.set_index("lineup_id")
        self.assertEqual(int(by_id.loc[11, "rank"]), 10)
        self.assertEqual(float(by_id.loc[11, "cashed"]), 1.0)
        self.assertEqual(float(by_id.loc[22, "cashed"]), 0.0)
        self.assertAlmostEqual(float(by_id.loc[11, "finish_percentile"]), 0.01)
        self.assertEqual(float(by_id.loc[11, "top1"]), 1.0)

    def test_join_by_lineup_id(self) -> None:
        entries = pd.DataFrame(
            {
                "lineup_id": [11, 22],
                "Place": [50, 800],
                "Score": [160.0, 90.0],
            }
        )
        joined = join_entered_lineups(self.sim, self.upload, entries, field_size=500)
        self.assertEqual(set(joined["lineup_id"]), {11, 22})
        self.assertEqual(int(joined.loc[joined["lineup_id"] == 22, "rank"].iloc[0]), 800)

    def test_name_plus_id_cells_match_upload_ids(self) -> None:
        named = [f"Player {i} ({pid})" for i, pid in enumerate(self.ids_a)]
        entries = pd.DataFrame(
            [
                {**_upload_row(named), "Rank": 1, "Points": 200.0, "Winnings": 100},
            ]
        )
        upload = pd.DataFrame([_upload_row(self.ids_a)])
        joined = join_entered_lineups(self.sim, upload, entries, field_size=100)
        self.assertEqual(len(joined), 1)
        self.assertEqual(int(joined.iloc[0]["lineup_id"]), 11)


class TestHeaderFailure(unittest.TestCase):
    def test_unmatched_headers_print_actual_columns(self) -> None:
        entries = pd.DataFrame({"Foo": [1], "Bar": [2], "P": ["133202-1"]})
        with self.assertRaises(UnmatchedHeaderError) as ctx:
            resolve_entries_columns(entries)
        message = str(ctx.exception)
        self.assertIn("Foo", message)
        self.assertIn("Bar", message)
        self.assertIn("Unable to match FanDuel entries column", message)

    def test_explicit_missing_col_fails_loudly(self) -> None:
        entries = pd.DataFrame({"Rank": [1], "Points": [12]})
        with self.assertRaises(UnmatchedHeaderError) as ctx:
            resolve_entries_columns(entries, rank_col="Place")
        self.assertIn("Place", str(ctx.exception))
        self.assertIn("Rank", str(ctx.exception))


class TestHistoryAndScore(unittest.TestCase):
    def test_history_append_and_full_score(self) -> None:
        ids_a = _ids(100)
        ids_b = _ids(200)
        sim = pd.DataFrame([_sim_row(1, ids_a, 0.7, 0.2), _sim_row(2, ids_b, 0.2, 0.01)])
        upload = pd.DataFrame([_upload_row(ids_a), _upload_row(ids_b)])
        entries = pd.DataFrame(
            [
                {**_upload_row(ids_a), "Rank": 5, "Points": 175.0, "Winnings": 40},
                {**_upload_row(ids_b), "Rank": 900, "Points": 88.0, "Winnings": 0},
            ]
        )
        with TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "sim_calibration_history.csv"
            result = score_calibration(
                sim,
                upload,
                entries,
                tag="2026-04-01",
                field_size=1000,
                history_path=history_path,
                notes="unit-test",
            )
            self.assertEqual(result.n_lineups, 2)
            self.assertTrue(history_path.exists())
            history = pd.read_csv(history_path)
            self.assertEqual(list(history.columns), HISTORY_COLUMNS)
            self.assertEqual(history.iloc[0]["tag"], "2026-04-01")
            self.assertEqual(int(history.iloc[0]["n_lineups"]), 2)
            self.assertEqual(history.iloc[0]["notes"], "unit-test")
            self.assertFalse(result.deciles.empty)
            report = format_report(result)
            self.assertIn("Brier (cash)", report)
            self.assertIn("2026-04-01", report)
            self.assertIn("running mean", report)

            second = score_calibration(
                sim,
                upload,
                entries,
                tag="2026-04-02",
                field_size=1000,
                history_path=history_path,
            )
            self.assertEqual(len(second.history), 2)

            extra = append_history(
                history_path,
                {
                    "tag": "manual",
                    "n_lineups": 1,
                    "brier_cash": 0.1,
                    "brier_top1": 0.2,
                    "spearman": -0.5,
                    "notes": "",
                },
            )
            self.assertEqual(len(extra), 3)

    def test_score_via_csv_roundtrip(self) -> None:
        ids_a = _ids(400)
        sim = pd.DataFrame([_sim_row(7, ids_a, 0.55, 0.08)])
        upload = pd.DataFrame([_upload_row(ids_a)])
        entries = pd.DataFrame([{**_upload_row(ids_a), "Rank": 20, "Points": 140.0}])
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sim_path = root / "tag_sim_results.csv"
            upload_path = root / "tag_simulated_upload.csv"
            entries_path = root / "entries.csv"
            sim.to_csv(sim_path, index=False)
            # Write a true FanDuel upload with duplicate OF headers.
            of_header = "P,C/1B,2B,3B,SS,OF,OF,OF,UTIL\n"
            of_row = ",".join(ids_a) + "\n"
            upload_path.write_text(of_header + of_row, encoding="utf-8")
            entries.to_csv(entries_path, index=False)
            result = score_calibration(
                sim_path,
                upload_path,
                entries_path,
                field_size=200,
                append=False,
            )
            self.assertEqual(result.n_matched, 1)
            self.assertEqual(result.tag, "tag")


if __name__ == "__main__":
    unittest.main()
