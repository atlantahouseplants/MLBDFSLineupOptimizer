"""Pre-flight checks: each test is a failure that really happened in 2026."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from slate_optimizer.ingestion.preflight import (
    PreflightError,
    fanduel_filename_date,
    fanduel_filename_sport,
    normalize_team_code,
    run_preflight,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GAMES = [("NYM", "PIT"), ("ATH", "BOS"), ("LAA", "MIA"), ("TOR", "PHI")]


def _fd_players(games=GAMES) -> pd.DataFrame:
    rows = []
    counter = 0
    for away, home in games:
        for team, opp in ((away, home), (home, away)):
            for position, salary in (("P", 8000), ("C", 2500), ("1B", 2800), ("2B", 2700), ("3B", 2900), ("SS", 3000), ("OF", 2600), ("OF", 3100), ("OF", 2400)):
                counter += 1
                rows.append(
                    {
                        "fd_player_id": f"133100-{counter}",
                        "position": position,
                        "roster_position": "C/1B/UTIL" if position in {"C", "1B"} else ("P" if position == "P" else f"{position}/UTIL"),
                        "team": team,
                        "opponent": opp,
                        "game": f"{away}@{home}",
                        "salary": salary,
                    }
                )
    return pd.DataFrame(rows)


def _bpp_games(games=GAMES, date="2026-08-07") -> pd.DataFrame:
    return pd.DataFrame(
        {"game_date": date, "away_team": [away for away, _ in games], "home_team": [home + " " for _, home in games]}
    )


def test_filename_parsing_handles_fanduels_odd_formats():
    assert fanduel_filename_date("FanDuel-MLB-2026 EDT-08 EDT-07 EDT-133100-players-list.csv") == "2026-08-07"
    assert fanduel_filename_date("FanDuel-MLB-2026-03-26-127566-entries-upload-template.csv") == "2026-03-26"
    assert fanduel_filename_date("FanDuel-NFL-2026 EST-11 EST-15 EST-136490-players-list.csv") == "2026-11-15"
    assert fanduel_filename_date("my_players.csv") is None
    assert fanduel_filename_sport("FanDuel-NFL-2026 EST-11 EST-15 EST-136490-players-list.csv") == "NFL"
    assert fanduel_filename_sport("FanDuel-MLB-2026-03-26-127566-entries-upload-template.csv") == "MLB"
    assert normalize_team_code(" tb ") == "TB" and normalize_team_code("CHW") == "CWS"


def test_matching_slate_passes():
    report = run_preflight(
        _fd_players(), "FanDuel-MLB-2026 EDT-08 EDT-07 EDT-133100-players-list.csv", _bpp_games()
    )
    assert report.ok and not report.issues
    assert report.fanduel_games == 4 and report.matched_games == 4
    assert "Pre-flight OK" in report.format()
    report.raise_if_blocking()


def test_stale_fanduel_file_is_blocked():
    """About ten 2026 cron runs used a players list from a different day."""
    report = run_preflight(
        _fd_players(), "FanDuel-MLB-2026 EDT-07 EDT-04 EDT-132208-players-list.csv", _bpp_games(date="2026-07-20")
    )
    assert not report.ok
    assert any(issue.code == "date_mismatch" for issue in report.errors)
    with pytest.raises(PreflightError, match="2026-07-04"):
        report.raise_if_blocking()


def test_different_slate_is_blocked_even_without_a_dated_filename():
    other = [("SEA", "TEX"), ("HOU", "SD"), ("LAD", "ARI"), ("DET", "SF")]
    report = run_preflight(_fd_players(), "uploaded.csv", _bpp_games(games=other))
    assert [issue.code for issue in report.errors] == ["slate_mismatch"]
    assert "0 of the 4 games" in report.errors[0].message


def test_partial_slate_is_a_warning_not_a_block():
    """FanDuel's later slate is a subset of the day; one missing game should only warn."""
    report = run_preflight(_fd_players(), None, _bpp_games(games=GAMES[:3]))
    assert report.ok
    assert [issue.code for issue in report.warnings] == ["games_without_projections"]
    extra_bpp = _bpp_games(games=GAMES + [("SEA", "TEX")])
    assert not run_preflight(_fd_players(), None, extra_bpp).issues, "BallparkPal having more games is normal"


def test_wrong_sport_is_blocked_with_a_readable_reason():
    """The Oct 15 2026 morning lost to an NFL template surfacing as a pandas index error."""
    nfl = pd.DataFrame(
        {
            "fd_player_id": [f"n{i}" for i in range(8)],
            "position": ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "D"],
            "roster_position": ["QB", "RB/FLEX", "RB/FLEX", "WR/FLEX", "WR/FLEX", "WR/FLEX", "TE/FLEX", "DEF"],
            "team": ["KC", "KC", "ATL", "ATL", "BUF", "NYJ", "BUF", "NYJ"],
            "opponent": ["ATL", "ATL", "KC", "KC", "NYJ", "BUF", "NYJ", "BUF"],
            "salary": [9000, 8000, 7000, 7500, 6000, 6500, 5000, 4000],
        }
    )
    report = run_preflight(nfl, "FanDuel-NFL-2026 EST-11 EST-15 EST-136490-players-list.csv", _bpp_games())
    codes = [issue.code for issue in report.errors]
    assert codes.count("wrong_sport") == 2
    assert "NFL" in report.errors[0].message


def test_duplicate_players_and_empty_files_are_blocked():
    doubled = pd.concat([_fd_players(), _fd_players()], ignore_index=True)
    assert any(issue.code == "duplicate_players" for issue in run_preflight(doubled, None, _bpp_games()).errors)
    assert [issue.code for issue in run_preflight(pd.DataFrame(), None, _bpp_games()).errors] == ["empty_file"]


def test_missing_position_and_impossible_salaries_are_blocked():
    no_shortstops = _fd_players()
    no_shortstops = no_shortstops[no_shortstops["position"] != "SS"]
    assert any(issue.code == "missing_position" and "SS" in issue.message for issue in run_preflight(no_shortstops, None, _bpp_games()).errors)
    pricey = _fd_players().assign(salary=lambda df: df["salary"] * 3)
    assert any(issue.code == "salary_infeasible" for issue in run_preflight(pricey, None, _bpp_games()).errors)


def test_in_play_and_repeated_vegas_lines_are_flagged():
    vegas = pd.DataFrame(
        {"game": ["NYM@PIT", "ATH@BOS", "ATH@BOS", "CWS@TB"], "total": [8.5, 9.0, 8.0, 1.5], "home_ml": [-130, -260, -215, -700], "away_ml": [120, 235, 180, 550]}
    )
    report = run_preflight(_fd_players(), None, _bpp_games(), vegas_games=vegas)
    assert report.ok, "bad odds are a warning: the projection code ignores them"
    codes = {issue.code for issue in report.warnings}
    assert codes == {"in_play_lines", "repeated_lines"}
    assert "CWS@TB" in next(issue.message for issue in report.warnings if issue.code == "in_play_lines")


def _load_pipeline_module():
    path = REPO_ROOT / "scripts" / "run_daily_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_daily_pipeline_for_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pipeline_accepts_the_stack_rotation_the_dashboard_sends():
    """The dashboard's Run Analysis button passes --stack-rotation; the CLI used to reject it."""
    pipeline = _load_pipeline_module()
    args = pipeline.build_parser().parse_args(
        ["--bpp-source", ".", "--fanduel-csv", "x.csv", "--stack-rotation", "4,3|5,3|4,4|3,3,2|4,2,2|3,2,2,1"]
    )
    rotation = pipeline.parse_stack_rotation(args.stack_rotation)
    assert rotation == [(4, 3), (4, 3), (4, 4), (3, 3, 2), (4, 2, 2), (3, 2, 2, 1)], "5 is capped at FanDuel's 4 per team"
    assert pipeline.parse_stack_rotation(None) is None and pipeline.parse_stack_rotation("  ") is None
    assert args.skip_preflight is False
    with pytest.raises(SystemExit):
        pipeline.parse_stack_rotation("4,x")
    with pytest.raises(SystemExit):
        pipeline.parse_stack_rotation("4,4,4")
