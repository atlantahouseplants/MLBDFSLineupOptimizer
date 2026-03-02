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
    lineups_to_fanduel_upload,
)
from slate_optimizer.optimizer.solver import LineupResult


class TestFanduelExport(unittest.TestCase):
    def test_lineups_to_upload_dataframe(self) -> None:
        data = [
            {"fd_player_id": "P1", "position": "P", "player_type": "pitcher"},
            {"fd_player_id": "C1", "position": "C/1B", "player_type": "batter"},
            {"fd_player_id": "2B1", "position": "2B", "player_type": "batter"},
            {"fd_player_id": "3B1", "position": "3B", "player_type": "batter"},
            {"fd_player_id": "SS1", "position": "SS", "player_type": "batter"},
            {"fd_player_id": "OF1", "position": "OF", "player_type": "batter"},
            {"fd_player_id": "OF2", "position": "OF/1B", "player_type": "batter"},
            {"fd_player_id": "OF3", "position": "OF/2B", "player_type": "batter"},
            {"fd_player_id": "UTIL1", "position": "1B/OF", "player_type": "batter"},
        ]
        lineup_df = pd.DataFrame(data)
        lineup = LineupResult(dataframe=lineup_df, total_salary=35000, total_projection=100.0)
        upload_df = lineups_to_fanduel_upload([lineup])

        self.assertEqual(list(upload_df.columns), FANDUEL_UPLOAD_COLUMNS)
        self.assertEqual(len(upload_df), 1)
        row = upload_df.iloc[0]
        self.assertEqual(row["P"], "P1")
        self.assertEqual(row["C/1B"], "C1")
        self.assertEqual(row["2B"], "2B1")
        self.assertEqual(row["3B"], "3B1")
        self.assertEqual(row["SS"], "SS1")
        of_values = row.iloc[5:8].tolist()
        self.assertCountEqual(of_values, ["OF1", "OF2", "OF3"])
        self.assertEqual(row["UTIL"], "UTIL1")


if __name__ == "__main__":
    unittest.main()
