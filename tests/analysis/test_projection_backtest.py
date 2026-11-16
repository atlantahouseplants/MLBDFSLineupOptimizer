"""Projection backtest harness on a tiny synthetic season."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from slate_optimizer.analysis.projection_backtest import (
    build_slate_frame,
    default_variants,
    game_start_minutes,
    join_actuals,
    normalize_name,
    normalize_team,
    paired_bootstrap,
    run_backtest,
    score_projection,
)

TEAMS = [("AAA", "BBB"), ("CCC", "DDD")]


def _write_slate(live_dir, actuals_dir, date, seed, snapshot_evening=True):
    rng = np.random.default_rng(seed)
    batters, pitchers, games, teams, orders, vegas, actual_rows = [], [], [], [], [], [], []
    pid = seed * 1000
    for game_idx, (away, home) in enumerate(TEAMS):
        game_pk = seed * 10 + game_idx
        runs = {away: 4.0 + game_idx * 0.5, home: 5.0 - game_idx * 0.5}
        games.append({"GamePk": game_pk, "GameDate": date, "AwayTeam": away, "HomeTeam": home, "RunsAway": runs[away], "RunsHome": runs[home], "AwayWinPct": 0.45, "HomeWinPct": 0.55})
        vegas.append({"game": f"{away}@{home}", "total": 9.0, "home_ml": -130, "away_ml": 115})
        for team, opp, side in ((away, home, "A"), (home, away, "H")):
            teams.append({"GamePk": game_pk, "GameDate": date, "Side": side, "Team": team, "Opponent": opp, "Runs": runs[team], "WinPercent": 0.5})
            for slot in range(1, 10):
                pid += 1
                points = 7.0 + (10 - slot) * 0.5
                batters.append(
                    {
                        "GamePk": game_pk, "GameDate": date, "GameTime": "7:05", "PlayerId": pid,
                        "FullName": f"{team} Bat{slot}", "LastName": f"Bat{slot}", "BatterStand": "R", "Side": side,
                        "Team": team, "Opponent": opp, "BattingPosition": slot, "PlateAppearances": 4.2,
                        "Runs": 0.5, "PointsFD": points, "HomeRunProbability": 0.1, "HitProbability": 0.6,
                        "StolenBaseProbability": 0.05,
                    }
                )
                orders.append({"team": team, "order_position": slot, "player_name": f"{team} Bat{slot}"})
                actual_rows.append(
                    {
                        "date": date, "game_pk": game_pk, "mlb_id": pid, "full_name": f"{team} Bat{slot}", "team": team,
                        "opponent": opp, "role": "batter", "started": True, "batting_order": slot,
                        "plate_appearances": 4, "fd_points": float(max(0.0, points + rng.normal(0, 6))),
                        "team_runs": int(round(runs[team])),
                    }
                )
            pid += 1
            pitchers.append(
                {
                    "GamePk": game_pk, "GameDate": date, "GameTime": "7:05", "PlayerId": pid, "FullName": f"{team} Arm",
                    "LastName": "Arm", "PitcherHand": "R", "Side": side, "Team": team, "Opponent": opp, "Innings": 5.5,
                    "WinPct": 0.3, "QualityStart": 0.4, "PointsFD": 28.0, "RunsAllowed": 2.8, "Strikeouts": 5.5,
                }
            )
            actual_rows.append(
                {
                    "date": date, "game_pk": game_pk, "mlb_id": pid, "full_name": f"{team} Arm", "team": team,
                    "opponent": opp, "role": "pitcher", "started": True, "batting_order": 0, "plate_appearances": 0,
                    "fd_points": float(28.0 + rng.normal(0, 12)), "team_runs": int(round(runs[team])),
                }
            )
    pd.DataFrame(batters).to_csv(live_dir / f"bpp_batters_{date}.csv", index=False)
    pd.DataFrame(pitchers).to_csv(live_dir / f"bpp_pitchers_{date}.csv", index=False)
    pd.DataFrame(games).to_csv(live_dir / f"bpp_games_{date}.csv", index=False)
    pd.DataFrame(teams).to_csv(live_dir / f"bpp_teams_{date}.csv", index=False)
    pd.DataFrame(orders).to_csv(live_dir / f"batting_orders_{date}.csv", index=False)
    pd.DataFrame(vegas).to_csv(live_dir / f"vegas_lines_{date}.csv", index=False)
    pd.DataFrame(actual_rows).to_csv(actuals_dir / f"actuals_{date}.csv", index=False)


@pytest.fixture()
def season(tmp_path):
    live_dir, actuals_dir = tmp_path / "live", tmp_path / "actuals"
    live_dir.mkdir()
    actuals_dir.mkdir()
    dates = [f"2026-07-{day:02d}" for day in range(1, 7)]
    for idx, date in enumerate(dates, start=1):
        _write_slate(live_dir, actuals_dir, date, seed=idx)
    return live_dir, actuals_dir, dates


def test_name_team_and_time_helpers():
    assert normalize_team(" TB ") == "TB" and normalize_team("CHW") == "CWS" and normalize_team("AZ") == "ARI"
    assert normalize_name("José Ramírez Jr.") == normalize_name("Jose Ramirez")
    assert game_start_minutes("7:05") == 19 * 60 + 5
    assert game_start_minutes("12:35") == 12 * 60 + 35
    assert game_start_minutes("1:10") == 13 * 60 + 10
    assert np.isnan(game_start_minutes("TBD"))


def test_build_slate_frame_mirrors_production_columns(season):
    live_dir, _, dates = season
    frame = build_slate_frame(dates[0], live_dir, snapshot_minutes=18 * 60 + 5)
    batters = frame[frame["player_type"] == "batter"]
    pitchers = frame[frame["player_type"] == "pitcher"]
    assert len(batters) == 36 and len(pitchers) == 4
    assert batters["bpp_runs"].eq(0.5).all(), "batter rows keep the individual-runs quirk"
    assert pitchers["bpp_runs"].ge(4.0).all()
    assert batters["bpp_team_runs"].ge(4.0).all()
    assert batters["is_confirmed_lineup"].all() and batters["batting_order_position"].between(1, 9).all()
    assert frame["pregame_snapshot"].all(), "7:05 first pitch is after an 18:05 snapshot"
    home = batters[batters["team_code"] == "BBB"].iloc[0]
    assert home["vegas_team_total"] < home["vegas_team_total_legacy"], "corrected totals shrink the favourite"
    late = build_slate_frame(dates[0], live_dir, snapshot_minutes=21 * 60)
    assert not late["pregame_snapshot"].any()


def test_join_actuals_uses_exact_ids(season):
    live_dir, actuals_dir, dates = season
    frame = build_slate_frame(dates[0], live_dir, snapshot_minutes=1000)
    actuals = pd.read_csv(actuals_dir / f"actuals_{dates[0]}.csv")
    joined = join_actuals(frame, actuals)
    assert joined["played"].all() and joined["actual_started"].all()
    assert len(joined) == len(frame)
    dropped = join_actuals(frame, actuals.iloc[5:])
    assert int((~dropped["played"]).sum()) == 5


def test_score_projection_and_bootstrap_detect_a_better_projection():
    rng = np.random.default_rng(0)
    truth = rng.normal(10, 3, size=4000)
    actual = pd.Series(truth + rng.normal(0, 6, size=4000))
    good = pd.Series(truth)
    bad = pd.Series(10 + (truth - 10) * 2.5 + rng.normal(0, 3, size=4000))  # over-spread and noisy
    dates = pd.Series(np.repeat([f"d{i}" for i in range(20)], 200))
    good_score, bad_score = score_projection(good, actual, dates), score_projection(bad, actual, dates)
    assert good_score["rmse"] < bad_score["rmse"]
    assert good_score["slope"] == pytest.approx(1.0, abs=0.1)
    assert bad_score["slope"] < 0.6, "an over-spread projection shows up as a slope well under 1"
    boot = paired_bootstrap(bad, good, actual, dates, reps=300)
    assert boot["d_rmse_lo"] > 0, "interval must exclude zero: the bad projection is reliably worse"
    assert boot["p_better_rmse"] < 0.05


def test_run_backtest_end_to_end(season):
    live_dir, actuals_dir, dates = season
    result = run_backtest(live_dir, actuals_dir, bootstrap_reps=100)
    assert result.dates == dates
    names = {variant.name for variant in default_variants()}
    assert {"raw_bpp", "legacy_dashboard", "legacy_cron", "calibrated_config"} <= names
    summary = result.summary
    hitters = summary[(summary["role"] == "batter") & (summary["subset"] == "all")].set_index("variant")
    assert hitters.loc["raw_bpp", "n"] == 36 * len(dates)
    # Synthetic truth IS the BallparkPal number, so stacking multipliers on top can only hurt.
    assert hitters.loc["raw_bpp", "rmse"] <= hitters.loc["legacy_dashboard", "rmse"]
    assert hitters.loc["only_batting_order", "slope"] < hitters.loc["raw_bpp", "slope"]
    assert not result.player_games.empty and "proj_raw_bpp" in result.player_games.columns
