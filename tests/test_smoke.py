from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.optimizer.solver import generate_lineups
from slate_optimizer.projection import compute_baseline_projections, data_quality_report


@pytest.fixture
def synthetic_players() -> pd.DataFrame:
    rows = [
        {
            "fd_player_id": "B1",
            "full_name": "Extra Outfielder",
            "player_type": "batter",
            "position": "OF",
            "roster_position": "OF",
            "team": "BBB",
            "team_code": "BBB",
            "opponent": "AAA",
            "opponent_code": "AAA",
            "salary": 3400,
            "bpp_points_fd": 10.5,
            "fppg": 10.0,
            "batting_order_position": 1,
            "is_confirmed_lineup": True,
            "batter_hand": "L",
            "recent_last7_fppg": 10.0,
            "recent_last14_fppg": 9.5,
            "recent_season_fppg": 10.0,
            "vegas_team_total": 4.1,
            "bpp_runs": 4.2,
            "bpp_win_percent": 0.46,
            "bpp_runs_allowed": 4.4,
            "bpp_runs_first_inning_pct": 0.45,
        },
        {
            "fd_player_id": "B2",
            "full_name": "Extra Corner Bat",
            "player_type": "batter",
            "position": "1B",
            "roster_position": "C/1B",
            "team": "BBB",
            "team_code": "BBB",
            "opponent": "AAA",
            "opponent_code": "AAA",
            "salary": 3150,
            "bpp_points_fd": None,
            "fppg": 9.2,
            "batting_order_position": 2,
            "is_confirmed_lineup": True,
            "batter_hand": "R",
            "recent_last7_fppg": 9.0,
            "recent_last14_fppg": 8.7,
            "recent_season_fppg": 9.2,
            "vegas_team_total": 4.1,
            "bpp_runs": 4.2,
            "bpp_win_percent": 0.46,
            "bpp_runs_allowed": 4.4,
            "bpp_runs_first_inning_pct": 0.45,
        },
        {
            "fd_player_id": "B3",
            "full_name": "Core First Base",
            "player_type": "batter",
            "position": "1B",
            "roster_position": "C/1B",
            "team": "DDD",
            "team_code": "DDD",
            "opponent": "CCC",
            "opponent_code": "CCC",
            "salary": 3300,
            "bpp_points_fd": 11.8,
            "fppg": 11.0,
            "batting_order_position": 1,
            "is_confirmed_lineup": True,
            "batter_hand": "L",
            "recent_last7_fppg": 11.0,
            "recent_last14_fppg": 10.6,
            "recent_season_fppg": 11.0,
            "vegas_team_total": 4.8,
            "bpp_runs": 4.9,
            "bpp_win_percent": 0.51,
            "bpp_runs_allowed": 4.0,
            "bpp_runs_first_inning_pct": 0.49,
        },
        {
            "fd_player_id": "B4",
            "full_name": "Core Second Base",
            "player_type": "batter",
            "position": "2B",
            "roster_position": "2B",
            "team": "DDD",
            "team_code": "DDD",
            "opponent": "CCC",
            "opponent_code": "CCC",
            "salary": 3200,
            "bpp_points_fd": 10.9,
            "fppg": 10.5,
            "batting_order_position": 2,
            "is_confirmed_lineup": True,
            "batter_hand": "R",
            "recent_last7_fppg": 10.2,
            "recent_last14_fppg": 10.0,
            "recent_season_fppg": 10.3,
            "vegas_team_total": 4.8,
            "bpp_runs": 4.9,
            "bpp_win_percent": 0.51,
            "bpp_runs_allowed": 4.0,
            "bpp_runs_first_inning_pct": 0.49,
        },
        {
            "fd_player_id": "B5",
            "full_name": "Core Third Base",
            "player_type": "batter",
            "position": "3B",
            "roster_position": "3B",
            "team": "DDD",
            "team_code": "DDD",
            "opponent": "CCC",
            "opponent_code": "CCC",
            "salary": 3100,
            "bpp_points_fd": 10.1,
            "fppg": 9.8,
            "batting_order_position": 3,
            "is_confirmed_lineup": True,
            "batter_hand": "S",
            "recent_last7_fppg": 9.8,
            "recent_last14_fppg": 9.6,
            "recent_season_fppg": 9.7,
            "vegas_team_total": 4.8,
            "bpp_runs": 4.9,
            "bpp_win_percent": 0.51,
            "bpp_runs_allowed": 4.0,
            "bpp_runs_first_inning_pct": 0.49,
        },
        {
            "fd_player_id": "B6",
            "full_name": "Core Outfielder",
            "player_type": "batter",
            "position": "OF",
            "roster_position": "OF",
            "team": "DDD",
            "team_code": "DDD",
            "opponent": "CCC",
            "opponent_code": "CCC",
            "salary": 3400,
            "bpp_points_fd": 11.2,
            "fppg": 10.8,
            "batting_order_position": 4,
            "is_confirmed_lineup": True,
            "batter_hand": "L",
            "recent_last7_fppg": 10.7,
            "recent_last14_fppg": 10.4,
            "recent_season_fppg": 10.8,
            "vegas_team_total": 4.8,
            "bpp_runs": 4.9,
            "bpp_win_percent": 0.51,
            "bpp_runs_allowed": 4.0,
            "bpp_runs_first_inning_pct": 0.49,
        },
        {
            "fd_player_id": "B7",
            "full_name": "Core Shortstop",
            "player_type": "batter",
            "position": "SS",
            "roster_position": "SS",
            "team": "EEE",
            "team_code": "EEE",
            "opponent": "FFF",
            "opponent_code": "FFF",
            "salary": 3000,
            "bpp_points_fd": 9.9,
            "fppg": 9.4,
            "batting_order_position": 1,
            "is_confirmed_lineup": True,
            "batter_hand": "R",
            "recent_last7_fppg": 9.5,
            "recent_last14_fppg": 9.1,
            "recent_season_fppg": 9.3,
            "vegas_team_total": 4.6,
            "bpp_runs": 4.6,
            "bpp_win_percent": 0.49,
            "bpp_runs_allowed": 4.1,
            "bpp_runs_first_inning_pct": 0.47,
        },
        {
            "fd_player_id": "B8",
            "full_name": "Speed Outfielder",
            "player_type": "batter",
            "position": "OF",
            "roster_position": "OF",
            "team": "EEE",
            "team_code": "EEE",
            "opponent": "FFF",
            "opponent_code": "FFF",
            "salary": 3200,
            "bpp_points_fd": 10.4,
            "fppg": 10.0,
            "batting_order_position": 2,
            "is_confirmed_lineup": True,
            "batter_hand": "L",
            "recent_last7_fppg": 10.0,
            "recent_last14_fppg": 9.8,
            "recent_season_fppg": 9.9,
            "vegas_team_total": 4.6,
            "bpp_runs": 4.6,
            "bpp_win_percent": 0.49,
            "bpp_runs_allowed": 4.1,
            "bpp_runs_first_inning_pct": 0.47,
        },
        {
            "fd_player_id": "B9",
            "full_name": "Power Outfielder",
            "player_type": "batter",
            "position": "OF",
            "roster_position": "OF",
            "team": "EEE",
            "team_code": "EEE",
            "opponent": "FFF",
            "opponent_code": "FFF",
            "salary": 3300,
            "bpp_points_fd": 11.0,
            "fppg": 10.7,
            "batting_order_position": 3,
            "is_confirmed_lineup": True,
            "batter_hand": "R",
            "recent_last7_fppg": 10.5,
            "recent_last14_fppg": 10.2,
            "recent_season_fppg": 10.4,
            "vegas_team_total": 4.6,
            "bpp_runs": 4.6,
            "bpp_win_percent": 0.49,
            "bpp_runs_allowed": 4.1,
            "bpp_runs_first_inning_pct": 0.47,
        },
        {
            "fd_player_id": "B10",
            "full_name": "Cleanup Outfielder",
            "player_type": "batter",
            "position": "OF",
            "roster_position": "OF",
            "team": "EEE",
            "team_code": "EEE",
            "opponent": "FFF",
            "opponent_code": "FFF",
            "salary": 3500,
            "bpp_points_fd": 11.6,
            "fppg": 11.0,
            "batting_order_position": 4,
            "is_confirmed_lineup": True,
            "batter_hand": "S",
            "recent_last7_fppg": 11.1,
            "recent_last14_fppg": 10.8,
            "recent_season_fppg": 11.0,
            "vegas_team_total": 4.6,
            "bpp_runs": 4.6,
            "bpp_win_percent": 0.49,
            "bpp_runs_allowed": 4.1,
            "bpp_runs_first_inning_pct": 0.47,
        },
        {
            "fd_player_id": "P1",
            "full_name": "Preferred Pitcher",
            "player_type": "pitcher",
            "position": "P",
            "roster_position": "P",
            "team": "AAA",
            "team_code": "AAA",
            "opponent": "BBB",
            "opponent_code": "BBB",
            "salary": 7600,
            "bpp_points_fd": 34.0,
            "fppg": 32.0,
            "is_confirmed_lineup": True,
            "pitcher_hand": "R",
            "vegas_team_total": 3.8,
            "bpp_runs": 3.9,
            "bpp_win_percent": 0.61,
            "bpp_runs_allowed": 3.4,
            "bpp_runs_first_inning_pct": 0.38,
            "recent_last7_fppg": 31.0,
            "recent_last14_fppg": 30.5,
            "recent_season_fppg": 30.0,
        },
        {
            "fd_player_id": "P2",
            "full_name": "Secondary Pitcher",
            "player_type": "pitcher",
            "position": "P",
            "roster_position": "P",
            "team": "CCC",
            "team_code": "CCC",
            "opponent": "DDD",
            "opponent_code": "DDD",
            "salary": 7200,
            "bpp_points_fd": 28.0,
            "fppg": 27.0,
            "is_confirmed_lineup": True,
            "pitcher_hand": "L",
            "vegas_team_total": 3.9,
            "bpp_runs": 4.1,
            "bpp_win_percent": 0.54,
            "bpp_runs_allowed": 3.9,
            "bpp_runs_first_inning_pct": 0.36,
            "recent_last7_fppg": 27.5,
            "recent_last14_fppg": 27.0,
            "recent_season_fppg": 26.8,
        },
    ]
    return pd.DataFrame(rows)


def test_compute_baseline_projections_positive(synthetic_players: pd.DataFrame) -> None:
    projections = compute_baseline_projections(synthetic_players)
    assert (projections["proj_fd_mean"] > 0).all()


def test_compute_baseline_projections_assigns_ownership(synthetic_players: pd.DataFrame) -> None:
    projections = compute_baseline_projections(synthetic_players)
    assert (projections["proj_fd_ownership"] > 0).all()


def test_ilp_solver_returns_valid_lineup(synthetic_players: pd.DataFrame) -> None:
    projections = compute_baseline_projections(synthetic_players)
    optimizer_df = synthetic_players.merge(
        projections[["fd_player_id", "proj_fd_mean", "proj_fd_floor", "proj_fd_ceiling", "proj_fd_ownership"]],
        on="fd_player_id",
        how="left",
    )

    lineups = generate_lineups(optimizer_df, num_lineups=1)

    assert len(lineups) == 1
    lineup = lineups[0].dataframe
    assert len(lineup) == 9
    assert lineup["salary"].sum() <= 35000
    assert (lineup["player_type"].str.lower() == "pitcher").sum() == 1


def test_data_quality_report_keys(synthetic_players: pd.DataFrame) -> None:
    report = data_quality_report(synthetic_players)
    expected_keys = {
        "n_players_with_bpp_projection",
        "n_players_using_fppg_fallback",
        "n_players_with_confirmed_batting_order",
        "n_players_with_vegas_data",
        "n_players_with_recent_stats",
    }
    assert isinstance(report, dict)
    assert set(report) == expected_keys


# ──────────────────────────────────────────────────────────────────────
# Auto-fetch module tests
# ──────────────────────────────────────────────────────────────────────

def test_mlb_api_fetch_structure(monkeypatch) -> None:
    """Mock MLB Stats API response and verify fetch_mlb_lineups returns correct columns."""
    import importlib
    import unittest.mock as mock

    from slate_optimizer.ingestion import mlb_api

    fake_response = {
        "totalItems": 1,
        "dates": [
            {
                "games": [
                    {
                        "teams": {
                            "home": {"team": {"abbreviation": "NYY"}, "probablePitcher": {"fullName": "Gerrit Cole", "pitchHand": {"code": "R"}}},
                            "away": {"team": {"abbreviation": "BOS"}, "probablePitcher": {"fullName": "Chris Sale", "pitchHand": {"code": "L"}}},
                        },
                        "lineups": {
                            "homePlayers": [
                                {"fullName": "Aaron Judge"},
                                {"fullName": "Juan Soto"},
                                {"fullName": "Giancarlo Stanton"},
                                {"fullName": "Anthony Volpe"},
                                {"fullName": "Austin Wells"},
                                {"fullName": "Jazz Chisholm"},
                                {"fullName": "Paul Goldschmidt"},
                                {"fullName": "Oswaldo Cabrera"},
                                {"fullName": "Ben Rice"},
                            ],
                            "awayPlayers": [
                                {"fullName": "Jarren Duran"},
                                {"fullName": "Rafael Devers"},
                                {"fullName": "Triston Casas"},
                                {"fullName": "Masataka Yoshida"},
                                {"fullName": "Tyler O'Neill"},
                                {"fullName": "David Hamilton"},
                                {"fullName": "Connor Wong"},
                                {"fullName": "Romy Gonzalez"},
                                {"fullName": "Ceddanne Rafaela"},
                            ],
                        },
                    }
                ]
            }
        ],
    }

    mock_resp = mock.MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = fake_response

    with mock.patch.object(mlb_api._requests, "get", return_value=mock_resp):
        batting_df, pitchers_df = mlb_api.fetch_mlb_lineups(date_str="2026-04-15")

    assert set(batting_df.columns) >= {"team_code", "order_position", "player_name", "confirmed"}
    assert len(batting_df) == 18  # 9 home + 9 away
    assert set(batting_df["team_code"].unique()) == {"NYY", "BOS"}
    assert batting_df["order_position"].max() == 9

    assert set(pitchers_df.columns) >= {"team_code", "player_name", "pitcher_hand"}
    assert len(pitchers_df) == 2
    cole = pitchers_df[pitchers_df["player_name"] == "Gerrit Cole"].iloc[0]
    assert cole["pitcher_hand"] == "R"
    sale = pitchers_df[pitchers_df["player_name"] == "Chris Sale"].iloc[0]
    assert sale["pitcher_hand"] == "L"


def test_odds_api_no_key(monkeypatch) -> None:
    """fetch_vegas_lines returns None when no API key is set."""
    import os
    from slate_optimizer.ingestion.odds_api import fetch_vegas_lines

    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    result = fetch_vegas_lines(api_key=None)
    assert result is None
