from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.analysis.candidate_pool_quality import build_candidate_pool_quality_report


class TestCandidatePoolQuality(unittest.TestCase):
    def test_flags_shallow_duplicate_candidate_pool(self) -> None:
        rows = []
        for lineup_id in [1, 2]:
            for idx in range(9):
                rows.append(
                    {
                        "lineup_id": lineup_id,
                        "fd_player_id": f"P{idx}",
                        "full_name": f"Player {idx}",
                        "team_code": "AAA" if idx < 4 else "BBB",
                        "player_type": "pitcher" if idx == 8 else "batter",
                        "proj_fd_mean": 10,
                        "proj_fd_ownership": 0.10,
                        "player_leverage_score": 0.01,
                        "salary": 3000,
                    }
                )

        report = build_candidate_pool_quality_report(pd.DataFrame(rows), target_final_lineups=3)
        statuses = dict(zip(report.checks["check"], report.checks["status"]))

        self.assertEqual(statuses["Pool depth"], "Warn")
        self.assertEqual(statuses["Duplicate candidates"], "Fail")
        self.assertEqual(report.summary["duplicate_lineups"], 1)


if __name__ == "__main__":
    unittest.main()
