from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.optimizer.solver import LineupResult
from slate_optimizer.simulation.contest_simulator import ContestSimResult, LineupSimResult
from slate_optimizer.simulation.ownership_scenarios import apply_ownership_scenario_metrics


class TestOwnershipScenarios(unittest.TestCase):
    def test_adds_worst_case_fields_to_results(self) -> None:
        lineup_df = pd.DataFrame(
            [
                {"fd_player_id": f"P{i}", "full_name": f"P{i}", "proj_fd_ownership": 20 + i, "salary": 3500, "team_code": "AAA" if i < 5 else "BBB", "player_type": "batter"}
                for i in range(8)
            ]
            + [{"fd_player_id": "SP", "full_name": "SP", "proj_fd_ownership": 35, "salary": 9000, "team_code": "CCC", "player_type": "pitcher"}]
        )
        lineup = LineupResult(lineup_df, total_salary=35000, total_projection=100.0)
        result = _sim_result()
        contest = ContestSimResult([result], num_simulations=10, num_field_lineups=5, num_candidates=1, entry_fee=20)

        details = apply_ownership_scenario_metrics(contest, [lineup], lineup_df, field_size=50000)

        self.assertFalse(details.empty)
        self.assertGreaterEqual(result.worst_case_duplication_score, result.duplication_score)
        self.assertTrue(0.0 <= result.ownership_scenario_score <= 1.0)


def _sim_result() -> LineupSimResult:
    return LineupSimResult(
        lineup_id=0,
        player_ids=[f"P{i}" for i in range(8)] + ["SP"],
        mean_score=100,
        median_score=100,
        std_score=10,
        p10_score=80,
        p25_score=90,
        p75_score=110,
        p90_score=120,
        p99_score=140,
        max_score=160,
        win_rate=0.01,
        top_1pct_rate=0.05,
        top_10pct_rate=0.20,
        cash_rate=0.45,
        expected_roi=0.1,
        total_ownership=2.0,
        leverage_score=1.0,
        field_duplication_rate=0.1,
        stack_teams=["AAA"],
        duplication_score=0.1,
    )


if __name__ == "__main__":
    unittest.main()
