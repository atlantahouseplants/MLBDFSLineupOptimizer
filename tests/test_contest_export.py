from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.optimizer.export import (
    FANDUEL_UPLOAD_COLUMNS,
    assign_lineups_to_contests,
    lineups_to_fanduel_template,
)
from slate_optimizer.optimizer.solver import LineupResult


def _make_lineup(tag: int) -> LineupResult:
    data = [
        {"fd_player_id": f"P{tag}", "position": "P", "player_type": "pitcher"},
        {"fd_player_id": f"C{tag}", "position": "C/1B", "player_type": "batter"},
        {"fd_player_id": f"2B{tag}", "position": "2B", "player_type": "batter"},
        {"fd_player_id": f"3B{tag}", "position": "3B", "player_type": "batter"},
        {"fd_player_id": f"SS{tag}", "position": "SS", "player_type": "batter"},
        {"fd_player_id": f"OFa{tag}", "position": "OF", "player_type": "batter"},
        {"fd_player_id": f"OFb{tag}", "position": "OF", "player_type": "batter"},
        {"fd_player_id": f"OFc{tag}", "position": "OF", "player_type": "batter"},
        {"fd_player_id": f"U{tag}", "position": "1B", "player_type": "batter"},
    ]
    return LineupResult(dataframe=pd.DataFrame(data), total_salary=34000, total_projection=100.0)


def _template(contest_sizes: dict[str, int]) -> pd.DataFrame:
    rows = []
    entry_id = 1000
    for cid, size in contest_sizes.items():
        for _ in range(size):
            rows.append(
                {
                    "entry_id": str(entry_id),
                    "contest_id": cid,
                    "contest_name": f"Contest {cid}",
                    "entry_fee": "5",
                }
            )
            entry_id += 1
    return pd.DataFrame(rows)


def _lineup_signature(row: pd.Series) -> frozenset:
    return frozenset(row.iloc[4:13].tolist())


class TestContestAssignment(unittest.TestCase):
    def test_even_split_two_contests(self) -> None:
        assignment = assign_lineups_to_contests(6, _template({"A": 3, "B": 3}))
        self.assertEqual(assignment["A"], [0, 2, 4])
        self.assertEqual(assignment["B"], [1, 3, 5])

    def test_unequal_contests(self) -> None:
        assignment = assign_lineups_to_contests(5, _template({"A": 3, "B": 2}))
        flat = assignment["A"] + assignment["B"]
        self.assertCountEqual(flat, [0, 1, 2, 3, 4])
        self.assertEqual(len(set(assignment["A"])), 3)
        self.assertEqual(len(set(assignment["B"])), 2)

    def test_shortage_reuses_across_but_not_within_contest(self) -> None:
        # 4 lineups, two contests of 3 entries: cross-contest reuse fills all 6.
        assignment = assign_lineups_to_contests(4, _template({"A": 3, "B": 3}))
        for cid in ("A", "B"):
            filled = [idx for idx in assignment[cid] if idx is not None]
            self.assertEqual(len(filled), 3)
            self.assertEqual(len(set(filled)), 3, f"duplicate lineup within contest {cid}")

    def test_hard_shortage_leaves_entries_unfilled(self) -> None:
        # 2 lineups cannot fill a 4-entry contest without duplicating.
        assignment = assign_lineups_to_contests(2, _template({"A": 4}))
        filled = [idx for idx in assignment["A"] if idx is not None]
        self.assertEqual(len(filled), 2)
        self.assertEqual(assignment["A"].count(None), 2)


class TestContestAwareTemplate(unittest.TestCase):
    def test_single_contest_unchanged(self) -> None:
        lineups = [_make_lineup(i) for i in range(3)]
        template = _template({"A": 3})
        result = lineups_to_fanduel_template(lineups, template)
        self.assertEqual(len(result), 3)
        self.assertEqual(
            list(result.columns),
            ["entry_id", "contest_id", "contest_name", "entry_fee"] + FANDUEL_UPLOAD_COLUMNS,
        )
        # All three OF columns must survive (duplicate labels).
        self.assertEqual(sum(1 for col in result.columns if col == "OF"), 3)
        self.assertTrue(result.iloc[:, 4:13].notna().all().all())

    def test_two_contests_no_duplicates_within_contest(self) -> None:
        lineups = [_make_lineup(i) for i in range(4)]
        template = _template({"A": 3, "B": 3})
        result = lineups_to_fanduel_template(lineups, template)
        self.assertEqual(len(result), 6)
        for cid, group in result.groupby("contest_id"):
            signatures = group.apply(_lineup_signature, axis=1).tolist()
            self.assertEqual(len(signatures), len(set(signatures)), f"duplicate in contest {cid}")

    def test_old_round_robin_bug_regression(self) -> None:
        # The old implementation repeated lineups round-robin: 2 lineups into a
        # 4-entry contest produced two duplicate pairs inside one contest.
        lineups = [_make_lineup(i) for i in range(2)]
        template = _template({"A": 4})
        result = lineups_to_fanduel_template(lineups, template)
        self.assertEqual(len(result), 2)  # unfilled entries omitted, never duplicated
        signatures = result.apply(_lineup_signature, axis=1).tolist()
        self.assertEqual(len(signatures), len(set(signatures)))

    def test_no_template_returns_plain_upload(self) -> None:
        lineups = [_make_lineup(i) for i in range(2)]
        result = lineups_to_fanduel_template(lineups, None)
        self.assertEqual(list(result.columns), FANDUEL_UPLOAD_COLUMNS)
        self.assertEqual(len(result), 2)

    def test_entries_stay_mapped_to_their_contest(self) -> None:
        lineups = [_make_lineup(i) for i in range(10)]
        template = _template({"A": 4, "B": 6})
        result = lineups_to_fanduel_template(lineups, template)
        merged = result.merge(template, on="entry_id", suffixes=("", "_orig"))
        self.assertTrue((merged["contest_id"] == merged["contest_id_orig"]).all())


if __name__ == "__main__":
    unittest.main()
