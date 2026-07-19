from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.analysis.run_audit import build_run_audit_report


class TestRunAudit(unittest.TestCase):
    def test_flags_missing_bust_when_projection_source_was_blended(self) -> None:
        optimizer_df = pd.DataFrame(
            [
                {
                    "fd_player_id": "H1",
                    "full_name": "Hitter One",
                    "proj_fd_mean": 10.0,
                    "proj_fd_bust_rate": 0.0,
                    "proj_fd_median": 0.0,
                    "proj_fd_upside": 0.0,
                    "proj_fd_ownership": 0.12,
                }
            ]
        )

        report = build_run_audit_report(
            optimizer_df,
            projection_source_count=1,
            ownership_source_count=0,
        )

        bust_row = report.checks[report.checks["check"] == "Bust input"].iloc[0]
        leverage_row = report.checks[report.checks["check"] == "Leverage confidence"].iloc[0]
        self.assertEqual(bust_row["status"], "Fail")
        self.assertEqual(leverage_row["status"], "Warn")
        self.assertEqual(report.summary["bust_coverage_pct"], 0.0)
        self.assertEqual(report.summary["ownership_mode"], "Fallback")

    def test_accepts_populated_boom_bust_and_catches_lineup_duplicates(self) -> None:
        optimizer_df = pd.DataFrame(
            [
                {
                    "fd_player_id": f"H{i}",
                    "full_name": f"Hitter {i}",
                    "proj_fd_mean": 8.0 + i,
                    "proj_fd_bust_rate": 0.20,
                    "proj_fd_median": 7.5 + i,
                    "proj_fd_upside": 18.0 + i,
                    "proj_fd_ownership": 0.08,
                }
                for i in range(1, 10)
            ]
        )
        lineup_rows = []
        for lineup_id in [1, 2]:
            for i in range(1, 10):
                is_pitcher = i == 9
                lineup_rows.append(
                    {
                        "lineup_id": lineup_id,
                        "fd_player_id": f"H{i}",
                        "full_name": f"Hitter {i}",
                        "team_code": "PIT" if is_pitcher else ("AAA" if i <= 4 else "BBB"),
                        "player_type": "pitcher" if is_pitcher else "batter",
                        "salary": 3000,
                        "proj_fd_mean": 8.0 + i,
                        "proj_fd_ownership": 0.08,
                        "player_leverage_score": 0.05,
                    }
                )

        stack_plan = pd.DataFrame(
            [
                {
                    "team_code": "AAA",
                    "target_exposure": 0.35,
                    "recommended_max_exposure": 0.50,
                    "stack_ownership": 0.45,
                    "sim_upside_score": 0.90,
                    "stack_leverage_score": 0.40,
                    "stack_score": 0.80,
                    "p_runs_ge_6": 0.30,
                    "top5_upside": 120.0,
                },
                {
                    "team_code": "BBB",
                    "target_exposure": 0.20,
                    "recommended_max_exposure": 0.35,
                    "stack_ownership": 0.25,
                    "sim_upside_score": 0.60,
                    "stack_leverage_score": 0.10,
                    "stack_score": 0.50,
                    "p_runs_ge_6": 0.22,
                    "top5_upside": 100.0,
                },
            ]
        )

        report = build_run_audit_report(
            optimizer_df,
            lineup_df=pd.DataFrame(lineup_rows),
            expected_lineups=2,
            stack_plan=stack_plan,
            projection_source_count=1,
            ownership_source_count=1,
        )

        bust_row = report.checks[report.checks["check"] == "Bust input"].iloc[0]
        dup_row = report.checks[report.checks["check"] == "Duplicate lineups"].iloc[0]
        leverage_row = report.checks[report.checks["check"] == "Leverage confidence"].iloc[0]
        self.assertEqual(bust_row["status"], "Good")
        self.assertEqual(dup_row["status"], "Fail")
        self.assertEqual(leverage_row["status"], "Good")
        self.assertEqual(report.summary["duplicate_lineups"], 2)
        self.assertIn("positive_leverage_stack_exposure", report.summary)
        self.assertFalse(report.stack_alignment.empty)
        self.assertEqual(report.stack_mix.iloc[0]["stack_template"], "4-4")


if __name__ == "__main__":
    unittest.main()
