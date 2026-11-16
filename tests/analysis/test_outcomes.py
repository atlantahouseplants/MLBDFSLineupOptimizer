"""Actual-outcome ingestion from MLB box scores (no network in tests)."""
from __future__ import annotations

import pandas as pd
import pytest

from slate_optimizer.analysis.outcomes import (
    fanduel_hitter_points,
    fanduel_pitcher_points,
    fetch_actuals,
    final_game_pks,
    is_quality_start,
    load_or_fetch_actuals,
    parse_boxscore,
)


def _player(pid, name, order=None, batting=None, pitching=None):
    payload = {"person": {"id": pid, "fullName": name}, "stats": {"batting": batting or {}, "pitching": pitching or {}}}
    if order is not None:
        payload["battingOrder"] = order
    return payload


def _boxscore():
    return {
        "teams": {
            "away": {
                "team": {"abbreviation": "STL"},
                "teamStats": {"batting": {"runs": 3}},
                "players": {
                    "ID1": _player(
                        1,
                        "Lead Off",
                        "100",
                        batting={
                            "plateAppearances": 5,
                            "hits": 3,
                            "doubles": 1,
                            "triples": 0,
                            "homeRuns": 1,
                            "rbi": 2,
                            "runs": 2,
                            "baseOnBalls": 1,
                            "stolenBases": 1,
                            "hitByPitch": 0,
                        },
                    ),
                    "ID2": _player(2, "Pinch Hitter", "901", batting={"plateAppearances": 1, "hits": 0}),
                    "ID3": _player(
                        3,
                        "Away Starter",
                        pitching={"gamesStarted": 1, "outs": 18, "earnedRuns": 3, "strikeOuts": 7, "wins": 1},
                    ),
                },
            },
            "home": {
                "team": {"abbreviation": "TOR"},
                "teamStats": {"batting": {"runs": 1}},
                "players": {
                    "ID4": _player(
                        4,
                        "Home Reliever",
                        pitching={"gamesStarted": 0, "outs": 4, "earnedRuns": 3, "strikeOuts": 0, "wins": 0},
                    ),
                },
            },
        }
    }


def test_hitter_scoring_matches_fanduel_rules():
    # single 3 + double 6 + HR 12 + 2 RBI 7 + 2 R 6.4 + BB 3 + SB 6 = 43.4
    batting = {"hits": 3, "doubles": 1, "homeRuns": 1, "rbi": 2, "runs": 2, "baseOnBalls": 1, "stolenBases": 1}
    assert fanduel_hitter_points(batting) == pytest.approx(43.4)
    assert fanduel_hitter_points({}) == 0.0


def test_pitcher_scoring_and_quality_start():
    line = {"gamesStarted": 1, "outs": 18, "earnedRuns": 3, "strikeOuts": 7, "wins": 1}
    assert is_quality_start(line)
    # W 6 + QS 4 + 18 outs + 7 K * 3 - 3 ER * 3 = 40
    assert fanduel_pitcher_points(line) == pytest.approx(40.0)
    assert not is_quality_start({**line, "outs": 17})
    assert not is_quality_start({**line, "earnedRuns": 4})
    assert not is_quality_start({**line, "gamesStarted": 0})


def test_parse_boxscore_flags_starters_and_roles():
    frame = parse_boxscore(_boxscore(), game_pk=99, date="2026-08-01")
    by_name = frame.set_index("full_name")
    assert by_name.loc["Lead Off", "started"] and by_name.loc["Lead Off", "batting_order"] == 1
    assert by_name.loc["Lead Off", "fd_points"] == pytest.approx(43.4)
    assert by_name.loc["Lead Off", "team_runs"] == 3
    assert not by_name.loc["Pinch Hitter", "started"]
    assert by_name.loc["Pinch Hitter", "batting_order"] == 9
    assert by_name.loc["Away Starter", "role"] == "pitcher"
    assert by_name.loc["Away Starter", "quality_start"] == 1
    assert by_name.loc["Home Reliever", "fd_points"] == pytest.approx(4 - 9)
    assert by_name.loc["Home Reliever", "opponent"] == "STL"


def test_final_game_pks_skips_unfinished_and_postponed():
    schedule = {
        "dates": [
            {
                "games": [
                    {"gamePk": 3, "status": {"abstractGameState": "Final", "detailedState": "Final"}},
                    {"gamePk": 1, "status": {"abstractGameState": "Final", "detailedState": "Completed Early: Rain"}},
                    {"gamePk": 2, "status": {"abstractGameState": "Final", "detailedState": "Postponed"}},
                    {"gamePk": 4, "status": {"abstractGameState": "Live", "detailedState": "In Progress"}},
                ]
            }
        ]
    }
    assert final_game_pks(schedule) == [1, 3]


def test_fetch_and_cache_round_trip(tmp_path):
    calls = []

    def fake_fetch(url):
        calls.append(url)
        if "schedule" in url:
            return {"dates": [{"games": [{"gamePk": 99, "status": {"abstractGameState": "Final", "detailedState": "Final"}}]}]}
        return _boxscore()

    frame = fetch_actuals("2026-08-01", fetch_json=fake_fetch)
    assert len(frame) == 4 and set(frame["game_pk"]) == {99}

    cached = load_or_fetch_actuals("2026-08-01", actuals_dir=tmp_path, fetch_json=fake_fetch)
    assert (tmp_path / "actuals_2026-08-01.csv").exists()
    before = len(calls)
    again = load_or_fetch_actuals("2026-08-01", actuals_dir=tmp_path, fetch_json=fake_fetch)
    assert len(calls) == before, "second read must come from the cache, not the network"
    pd.testing.assert_series_equal(
        cached["fd_points"].reset_index(drop=True), again["fd_points"].reset_index(drop=True), check_dtype=False
    )
