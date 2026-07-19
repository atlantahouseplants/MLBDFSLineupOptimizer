from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.simulation.duplication import estimate_lineup_duplication


class TestDuplication(unittest.TestCase):
    def test_chalk_stack_scores_as_more_duplicated(self) -> None:
        chalk = self._lineup([28, 24, 22, 20, 18, 16, 14, 12, 35], salary=34900, stack_team="AAA")
        contrarian = self._lineup([8, 7, 6, 5, 4, 4, 3, 3, 18], salary=33800, stack_team="BBB")

        chalk_est = estimate_lineup_duplication(chalk, field_size=50000)
        contrarian_est = estimate_lineup_duplication(contrarian, field_size=50000)

        self.assertGreater(chalk_est.duplication_score, contrarian_est.duplication_score)
        self.assertGreater(chalk_est.estimated_dupes, contrarian_est.estimated_dupes)
        self.assertLess(chalk_est.uniqueness_score, contrarian_est.uniqueness_score)

    def _lineup(self, owns, salary: int, stack_team: str) -> pd.DataFrame:
        rows = []
        per_player_salary = salary // len(owns)
        for idx, own in enumerate(owns):
            rows.append(
                {
                    "fd_player_id": f"P{idx}",
                    "proj_fd_ownership": own,
                    "salary": per_player_salary,
                    "team_code": stack_team if idx < 5 else "ZZZ",
                    "player_type": "pitcher" if idx == 8 else "batter",
                }
            )
        return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
