from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.optimizer.export import lineups_to_fanduel_upload
from slate_optimizer.optimizer.solver import generate_lineups


POSITIONS = ["C/1B", "2B", "3B", "SS", "OF", "OF", "OF", "OF"]


def _solver_dataset() -> pd.DataFrame:
    rows = [
        {
            "fd_player_id": "P_AAA",
            "full_name": "Pitcher AAA",
            "player_type": "pitcher",
            "position": "P",
            "team_code": "AAA",
            "opponent_code": "BBB",
            "proj_fd_mean": 35.0,
            "salary": 8500,
            "default_max_exposure": 1.0,
        },
        {
            "fd_player_id": "P_CCC",
            "full_name": "Pitcher CCC",
            "player_type": "pitcher",
            "position": "P",
            "team_code": "CCC",
            "opponent_code": "DDD",
            "proj_fd_mean": 33.0,
            "salary": 8300,
            "default_max_exposure": 1.0,
        },
    ]
    for team_idx, team in enumerate(("AAA", "CCC", "DDD")):
        for idx, pos in enumerate(POSITIONS):
            rows.append(
                {
                    "fd_player_id": f"{team}_{idx}",
                    "full_name": f"{team} Hitter {idx}",
                    "player_type": "batter",
                    "position": pos,
                    "team_code": team,
                    "opponent_code": "BBB" if team == "AAA" else "AAA",
                    "proj_fd_mean": 14.0 - team_idx - idx * 0.1,
                    "salary": 2500 + idx * 50,
                    "default_max_exposure": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _position_trap_dataset() -> pd.DataFrame:
    rows = [
        {
            "fd_player_id": "P_1",
            "full_name": "Pitcher",
            "player_type": "pitcher",
            "position": "P",
            "roster_position": "P",
            "team_code": "AAA",
            "opponent_code": "BBB",
            "proj_fd_mean": 30.0,
            "salary": 8000,
            "default_max_exposure": 1.0,
        },
        {
            "fd_player_id": "MULTI_23",
            "full_name": "Only Middle Infielder",
            "player_type": "batter",
            "position": "2B/3B",
            "roster_position": "2B/3B/UTIL",
            "team_code": "AAA",
            "opponent_code": "BBB",
            "proj_fd_mean": 20.0,
            "salary": 3000,
            "default_max_exposure": 1.0,
        },
        {
            "fd_player_id": "C1",
            "full_name": "C One",
            "player_type": "batter",
            "position": "C",
            "roster_position": "C/1B/UTIL",
            "team_code": "CCC",
            "opponent_code": "DDD",
            "proj_fd_mean": 19.0,
            "salary": 3000,
            "default_max_exposure": 1.0,
        },
        {
            "fd_player_id": "C2",
            "full_name": "C Two",
            "player_type": "batter",
            "position": "C",
            "roster_position": "C/1B/UTIL",
            "team_code": "DDD",
            "opponent_code": "CCC",
            "proj_fd_mean": 18.0,
            "salary": 3000,
            "default_max_exposure": 1.0,
        },
        {
            "fd_player_id": "SS1",
            "full_name": "Shortstop",
            "player_type": "batter",
            "position": "SS",
            "roster_position": "SS/UTIL",
            "team_code": "EEE",
            "opponent_code": "FFF",
            "proj_fd_mean": 17.0,
            "salary": 3000,
            "default_max_exposure": 1.0,
        },
    ]
    for idx, proj in enumerate((16.0, 15.0, 14.0, 13.0), start=1):
        rows.append(
            {
                "fd_player_id": f"OF{idx}",
                "full_name": f"Outfielder {idx}",
                "player_type": "batter",
                "position": "OF",
                "roster_position": "OF/UTIL",
                "team_code": "GGG" if idx <= 2 else "HHH",
                "opponent_code": "HHH" if idx <= 2 else "GGG",
                "proj_fd_mean": proj,
                "salary": 3000,
                "default_max_exposure": 1.0,
            }
        )
    rows.extend(
        [
            {
                "fd_player_id": "2B_EXTRA",
                "full_name": "Second Base Fallback",
                "player_type": "batter",
                "position": "2B",
                "roster_position": "2B/UTIL",
                "team_code": "III",
                "opponent_code": "JJJ",
                "proj_fd_mean": 9.0,
                "salary": 3000,
                "default_max_exposure": 1.0,
            },
            {
                "fd_player_id": "3B_EXTRA",
                "full_name": "Third Base Fallback",
                "player_type": "batter",
                "position": "3B",
                "roster_position": "3B/UTIL",
                "team_code": "JJJ",
                "opponent_code": "III",
                "proj_fd_mean": 8.0,
                "salary": 3000,
                "default_max_exposure": 1.0,
            },
        ]
    )
    return pd.DataFrame(rows)


class TestSolverConstraints(unittest.TestCase):
    def test_min_stack_size_enforced_without_template(self) -> None:
        lineups = generate_lineups(
            _solver_dataset(),
            num_lineups=3,
            min_stack_size=4,
            randomness=0.0,
        )
        self.assertEqual(len(lineups), 3)
        for lineup in lineups:
            hitters = lineup.dataframe[lineup.dataframe["player_type"] != "pitcher"]
            max_team_count = int(hitters["team_code"].value_counts().max())
            self.assertGreaterEqual(max_team_count, 4)

    def test_zero_exposure_hard_fades_player(self) -> None:
        dataset = _solver_dataset()
        dataset.loc[dataset["fd_player_id"] == "AAA_0", "default_max_exposure"] = 0.0
        lineups = generate_lineups(
            dataset,
            num_lineups=2,
            min_stack_size=4,
            randomness=0.0,
        )
        selected_ids = set(
            pd.concat([lineup.dataframe for lineup in lineups], ignore_index=True)["fd_player_id"].astype(str)
        )
        self.assertNotIn("AAA_0", selected_ids)

    def test_locked_player_ids_are_enforced(self) -> None:
        lineups = generate_lineups(
            _solver_dataset(),
            num_lineups=1,
            min_stack_size=4,
            randomness=0.0,
            locked_player_ids=["CCC_7"],
        )
        self.assertEqual(len(lineups), 1)
        selected_ids = set(lineups[0].dataframe["fd_player_id"].astype(str))
        self.assertIn("CCC_7", selected_ids)

    def test_lineups_are_fanduel_slot_exportable(self) -> None:
        lineups = generate_lineups(
            _position_trap_dataset(),
            num_lineups=1,
            salary_cap=35000,
            randomness=0.0,
        )

        self.assertEqual(len(lineups), 1)
        upload_df = lineups_to_fanduel_upload(lineups, strict=True)
        self.assertEqual(len(upload_df), 1)


if __name__ == "__main__":
    unittest.main()
