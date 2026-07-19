from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.simulation.field_learning import learn_field_duplication_profile


class TestFieldLearning(unittest.TestCase):
    def test_learns_duplicate_profile_from_roster_columns(self) -> None:
        contest = pd.DataFrame(
            [
                {"P": "Pitcher A", "C/1B": "A One", "2B": "A Two", "3B": "A Three", "SS": "A Four", "OF": "B One", "OF2": "B Two", "OF3": "B Three", "UTIL": "C One"},
                {"P": "Pitcher A", "C/1B": "A One", "2B": "A Two", "3B": "A Three", "SS": "A Four", "OF": "B One", "OF2": "B Two", "OF3": "B Three", "UTIL": "C One"},
                {"P": "Pitcher B", "C/1B": "D One", "2B": "D Two", "3B": "D Three", "SS": "D Four", "OF": "E One", "OF2": "E Two", "OF3": "E Three", "UTIL": "F One"},
            ]
        )
        optimizer = pd.DataFrame(
            [
                {"full_name": name, "salary": 4000, "team_code": name.split()[0], "player_type": "batter"}
                for name in ["A One", "A Two", "A Three", "A Four", "B One", "B Two", "B Three", "C One", "D One", "D Two", "D Three", "D Four", "E One", "E Two", "E Three", "F One"]
            ]
            + [
                {"full_name": "Pitcher A", "salary": 9000, "team_code": "P", "player_type": "pitcher"},
                {"full_name": "Pitcher B", "salary": 9000, "team_code": "P", "player_type": "pitcher"},
            ]
        )
        profile = learn_field_duplication_profile(contest, optimizer_df=optimizer)

        self.assertEqual(profile.samples, 3)
        self.assertEqual(profile.unique_lineups, 2)
        self.assertGreater(profile.duplicate_lineup_pct, 0.0)
        self.assertGreater(profile.avg_salary_used, 0.0)


if __name__ == "__main__":
    unittest.main()
