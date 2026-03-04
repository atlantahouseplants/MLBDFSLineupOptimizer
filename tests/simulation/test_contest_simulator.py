from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import unittest

import pandas as pd

from slate_optimizer.optimizer import LineupResult
from slate_optimizer.simulation import (
    FieldQualityMix,
    SimulationConfig,
    build_correlation_matrix,
    fit_player_distributions,
    select_portfolio,
    simulate_contest,
    simulate_field,
    simulate_slate,
)


class TestContestSimulator(unittest.TestCase):
    def _dataset(self) -> pd.DataFrame:
        rows = []
        for team in ("AAA", "BBB"):
            # pitcher
            rows.append(
                {
                    "fd_player_id": f"P_{team}",
                    "full_name": f"Pitcher {team}",
                    "player_type": "pitcher",
                    "position": "P",
                    "team_code": team,
                    "opponent_code": "BBB" if team == "AAA" else "AAA",
                    "stack_key": team,
                    "stack_priority": "high" if team == "AAA" else "mid",
                    "game_key": "AAA@BBB",
                    "proj_fd_mean": 34,
                    "proj_fd_floor": 20,
                    "proj_fd_ceiling": 52,
                    "salary": 9200,
                    "proj_fd_ownership": 20,
                    "player_leverage_score": 1.0,
                }
            )
            idx = 0
            for pos in ['C/1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF', 'OF', 'UTIL', '2B', '1B', 'OF']:
                rows.append(
                    {
                        "fd_player_id": f"{team}_H{idx}",
                        "full_name": f"{team} Hitter {idx}",
                        "player_type": "batter",
                        "position": pos,
                        "team_code": team,
                        "opponent_code": "BBB" if team == "AAA" else "AAA",
                        "stack_key": team,
                        "stack_priority": "high" if team == "AAA" else "mid",
                        "game_key": "AAA@BBB",
                        "proj_fd_mean": 10 + (idx % 4),
                        "proj_fd_floor": 6 + (idx % 2),
                        "proj_fd_ceiling": 16 + (idx % 3),
                        "salary": 2700 + (idx * 60),
                        "proj_fd_ownership": 8 + idx,
                        "player_leverage_score": 0.5,
                    }
                )
                idx += 1
        return pd.DataFrame(rows)

    def _candidate_lineups(self, dataset: pd.DataFrame) -> list[LineupResult]:
        hitters = dataset[dataset["player_type"] == "batter"].reset_index(drop=True)
        pitchers = dataset[dataset["player_type"] == "pitcher"].reset_index(drop=True)
        lineups: list[LineupResult] = []
        for idx in range(4):
            pitcher = pitchers.iloc[idx % len(pitchers)]
            offset = max(1, len(hitters) - 8)
            start = (idx * 2) % offset
            hitter_block = hitters.iloc[start : start + 8]
            frame = pd.concat([pitcher.to_frame().T, hitter_block], ignore_index=True)
            total_salary = int(frame["salary"].sum())
            total_projection = float(frame["proj_fd_mean"].sum())
            lineups.append(
                LineupResult(
                    dataframe=frame,
                    total_salary=total_salary,
                    total_projection=total_projection,
                )
            )
        return lineups

    def test_contest_results_shape(self) -> None:
        dataset = self._dataset()
        lineups = self._candidate_lineups(dataset)
        config = SimulationConfig(num_simulations=300, num_field_lineups=100)
        dists = fit_player_distributions(dataset)
        corr = build_correlation_matrix(dataset)
        slate = simulate_slate(dists, corr, num_simulations=config.num_simulations, seed=7)
        field = simulate_field(
            dataset,
            num_opponent_lineups=config.num_field_lineups,
            salary_cap=35000,
            seed=7,
            position_constraints=True,
            quality_mix=FieldQualityMix(),
        )
        contest = simulate_contest(
            lineups,
            slate,
            field,
            entry_fee=config.entry_fee,
            payout_structure=config.payout_structure,
        )
        df = contest.to_dataframe()
        self.assertFalse(df.empty)
        required = {"lineup_id", "mean_score", "win_rate", "top_1pct_rate", "expected_roi"}
        self.assertTrue(required.issubset(df.columns))

        portfolio = select_portfolio(contest, num_lineups=2)
        self.assertGreater(portfolio.num_selected, 0)
        self.assertTrue(0.0 <= portfolio.portfolio_win_rate <= 1.0)


if __name__ == "__main__":
    unittest.main()
