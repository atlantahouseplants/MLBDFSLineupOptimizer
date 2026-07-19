from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.analysis.late_news import build_late_news_report


class TestLateNews(unittest.TestCase):
    def test_flags_hitter_not_in_posted_lineup(self) -> None:
        now = pd.Timestamp("2026-04-24T20:00:00Z")
        lineups = pd.DataFrame(
            [
                {
                    "lineup_id": 1,
                    "full_name": "Bench Hitter",
                    "team_code": "AAA",
                    "player_type": "batter",
                    "batting_order_position": 5,
                    "is_confirmed_lineup": True,
                    "game_start_time": "2026-04-24T23:00:00Z",
                }
            ]
        )
        fresh = pd.DataFrame(
            [
                {"team": "AAA", "order_position": 1, "player_name": "Actual Starter"},
                {"team": "AAA", "order_position": 2, "player_name": "Another Starter"},
            ]
        )

        report = build_late_news_report(lineups, fresh_orders=fresh, now=now)

        self.assertEqual(report.affected_lineup_ids, [1])
        self.assertIn("Not in posted lineup", set(report.issues["category"]))

    def test_flags_order_change(self) -> None:
        now = pd.Timestamp("2026-04-24T20:00:00Z")
        lineups = pd.DataFrame(
            [
                {
                    "lineup_id": 3,
                    "full_name": "Actual Starter",
                    "team_code": "AAA",
                    "player_type": "batter",
                    "batting_order_position": 5,
                    "is_confirmed_lineup": True,
                    "game_start_time": "2026-04-24T23:00:00Z",
                }
            ]
        )
        fresh = pd.DataFrame([{"team": "AAA", "order_position": 1, "player_name": "Actual Starter"}])

        report = build_late_news_report(lineups, fresh_orders=fresh, now=now)

        self.assertIn("Order changed", set(report.issues["category"]))


if __name__ == "__main__":
    unittest.main()
