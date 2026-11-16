"""Outcome-fitted simulator: distributions, correlations, and contest ranking."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from slate_optimizer.analysis.sim_structure import (
    calibrate,
    empirical_correlations,
    fit_hitter_model,
    fit_pitcher_model,
    simulated_correlations,
    starters,
)
from slate_optimizer.simulation import (
    SimulationCalibration,
    build_correlation_matrix,
    fit_player_distributions,
    load_simulation_calibration,
    save_simulation_calibration,
    simulate_slate,
)
from slate_optimizer.simulation.contest_simulator import _percentiles_vs_field


def _players() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"fd_player_id": "bat", "full_name": "Bat", "player_type": "batter", "proj_fd_mean": 9.0, "salary": 3000},
            {"fd_player_id": "star", "full_name": "Star", "player_type": "batter", "proj_fd_mean": 13.0, "salary": 4200},
            {"fd_player_id": "bench", "full_name": "Bench", "player_type": "batter", "proj_fd_mean": 1.0, "salary": 2000},
            {"fd_player_id": "none", "full_name": "None", "player_type": "batter", "proj_fd_mean": 0.0, "salary": 2000},
            {"fd_player_id": "arm", "full_name": "Arm", "player_type": "pitcher", "proj_fd_mean": 30.0, "salary": 9500},
        ]
    )


def test_calibrated_hitter_can_score_zero_and_keeps_its_mean():
    calibration = SimulationCalibration()
    dists = fit_player_distributions(_players(), calibration=calibration)
    rng = np.random.default_rng(1)
    bat = dists["bat"].sample(200_000, rng)
    assert dists["bat"].dist_type == "hurdle_gamma"
    assert bat.mean() == pytest.approx(9.0, abs=0.1)
    assert dists["bat"].mean() == pytest.approx(9.0, abs=1e-6)
    zero_rate = float((bat == 0).mean())
    assert zero_rate == pytest.approx(calibration.hitter_zero_probability(9.0), abs=0.01)
    assert 0.15 < zero_rate < 0.35, "about a quarter of real starts score nothing"
    assert bat[bat > 0].min() >= calibration.hitter_positive_shift - 1e-9, "smallest real score is a walk or single"
    assert bat.std() > 8.0, "real hitter SD is ~9; the hand-set model gave ~4.5"

    star = dists["star"].sample(200_000, rng)
    assert (star == 0).mean() < zero_rate, "better hitters get blanked less often"
    assert star.mean() == pytest.approx(13.0, abs=0.15)


def test_calibrated_edge_cases_do_not_break():
    dists = fit_player_distributions(_players(), calibration=SimulationCalibration())
    rng = np.random.default_rng(2)
    bench = dists["bench"].sample(100_000, rng)
    assert bench.mean() == pytest.approx(1.0, abs=0.05)
    assert (bench == 0).mean() > 0.6
    nobody = dists["none"].sample(10_000, rng)
    assert np.isfinite(nobody).all() and nobody.mean() < 0.1
    grid = np.array([0.001, 0.2, 0.5, 0.9, 0.999])
    for dist in dists.values():
        values = dist.ppf(grid)
        assert np.all(np.diff(values) >= -1e-9), "quantile function must be monotone"
        assert np.isfinite(values).all()


def test_calibrated_pitcher_is_wide_bounded_and_centred():
    calibration = SimulationCalibration()
    arm = fit_player_distributions(_players(), calibration=calibration)["arm"]
    draws = arm.sample(200_000, np.random.default_rng(3))
    assert draws.mean() == pytest.approx(30.0, abs=0.15)
    assert draws.std() > 11.0, "real pitcher SD is ~14; the hand-set model gave ~4.3"
    assert draws.min() >= calibration.pitcher_lower_bound - 1e-6
    assert (draws < 0).mean() > 0.0, "pitchers do post negative scores"
    assert draws.max() <= 30.0 + calibration.pitcher_upper_sd * calibration.pitcher_sd(30.0) + 1e-6


def test_legacy_distributions_are_unchanged_without_calibration():
    players = _players().assign(proj_fd_floor=lambda d: d.proj_fd_mean * 0.8, proj_fd_ceiling=lambda d: d.proj_fd_mean * 1.3)
    legacy = fit_player_distributions(players)
    assert legacy["bat"].dist_type == "lognormal" and legacy["arm"].dist_type == "truncated_normal"
    assert legacy["arm"].lower_bound == 0.0
    assert legacy["bat"].sample(5000, np.random.default_rng(4)).min() > 0.0


def test_calibration_file_round_trip_and_fallback(tmp_path):
    path = tmp_path / "simulation_calibration.json"
    assert load_simulation_calibration(path) is None, "no file means the hand-set simulator"
    chosen = SimulationCalibration(teammate_base=0.14, copula_nu=30, source="test")
    save_simulation_calibration(chosen, path)
    assert load_simulation_calibration(path) == chosen
    path.write_text("{broken", encoding="utf-8")
    assert load_simulation_calibration(path) is None
    path.write_text(json.dumps({"teammate_base": 0.2, "unknown": 1}), encoding="utf-8")
    assert load_simulation_calibration(path).teammate_base == 0.2


def test_calibrated_correlation_config_drops_unsupported_bonuses():
    config = SimulationCalibration(teammate_base=0.14, pitcher_vs_opposing=-0.33).correlation_config()
    assert config.teammate_adjacent_bonus == 0.0 and config.stack_priority_bonus == 0.0
    assert config.vegas_scale_floor == config.vegas_scale_cap == 1.0
    assert config.cross_game == 0.0


def test_simulated_correlations_track_the_settings():
    low = simulated_correlations(SimulationCalibration(teammate_base=0.05, copula_nu=30), num_simulations=12_000)
    high = simulated_correlations(SimulationCalibration(teammate_base=0.30, copula_nu=30), num_simulations=12_000)
    assert high["teammates"] > low["teammates"] + 0.1
    assert low["pitcher_vs_opposing"] < -0.15
    assert abs(low["cross_game_hitters"]) < 0.03, "different games must be close to independent"
    heavy_tail = simulated_correlations(SimulationCalibration(teammate_base=0.05, copula_nu=4), num_simulations=12_000)
    assert heavy_tail["cross_game_hitters"] > low["cross_game_hitters"], "a low nu ties the whole slate together"


def _synthetic_player_games(seed: int = 0, slates: int = 24, games_per_slate: int = 20) -> pd.DataFrame:
    """Outcomes drawn from a known model so the fit can be checked against truth.

    One two-team game is simulated many times; every simulation becomes its own
    game on some slate.
    """
    truth = SimulationCalibration(
        hitter_zero_scale=6.0,
        hitter_positive_cv=0.75,
        pitcher_sd_intercept=13.0,
        pitcher_sd_slope=0.0,
        teammate_base=0.20,
        same_game_opponent=0.0,
        pitcher_vs_opposing=-0.35,
        pitcher_own_hitters=0.05,
        copula_nu=30,
    )
    players = []
    for team, opp in (("AAA", "BBB"), ("BBB", "AAA")):
        for slot in range(1, 10):
            players.append(
                {
                    "fd_player_id": f"{team}-{slot}", "full_name": "x", "player_type": "batter",
                    "team_code": team, "opponent_code": opp, "stack_key": team, "game_key": "g",
                    "batting_order_position": slot, "stack_priority": "mid", "vegas_game_total": 0.0,
                    "proj_fd_mean": 7.0 + (10 - slot) * 0.5, "salary": 3000,
                }
            )
        players.append(
            {
                "fd_player_id": f"{team}-p", "full_name": "x", "player_type": "pitcher",
                "team_code": team, "opponent_code": opp, "stack_key": team, "game_key": "g",
                "batting_order_position": 0, "stack_priority": "mid", "vegas_game_total": 0.0,
                "proj_fd_mean": 26.0, "salary": 8500,
            }
        )
    frame = pd.DataFrame(players)
    total_games = slates * games_per_slate
    sim = simulate_slate(
        fit_player_distributions(frame, calibration=truth),
        build_correlation_matrix(frame, truth.correlation_config()),
        num_simulations=total_games,
        seed=seed,
        use_antithetic=False,
    )
    rows = []
    for game in range(total_games):
        played = frame.copy()
        played["actual_fd_points"] = sim.scores[game]
        played["date"] = f"2026-06-{game // games_per_slate + 1:02d}"
        played["game_pk"] = game
        rows.append(played)
    out = pd.concat(rows, ignore_index=True)
    out["actual_started"] = True
    out["actual_batting_order"] = out["batting_order_position"]
    out["proj_calibrated_config"] = out["proj_fd_mean"]
    return out


def test_fit_recovers_a_known_model():
    player_games = _synthetic_player_games()
    frame = starters(player_games)
    hitters = fit_hitter_model(frame[frame["player_type"] == "batter"])
    assert hitters["hitter_zero_scale"] == pytest.approx(6.0, abs=1.0)
    assert hitters["hitter_positive_cv"] == pytest.approx(0.75, abs=0.08)
    pitchers = fit_pitcher_model(frame[frame["player_type"] == "pitcher"])
    assert pitchers["observed_sd"] == pytest.approx(12.7, abs=1.0)
    measured = empirical_correlations(frame, bootstrap_reps=30).set_index("pair_class")["correlation"]
    assert measured["teammates"] > 0.08
    assert measured["pitcher_vs_opposing"] < -0.15
    assert abs(measured["opposing_hitters"]) < 0.08

    report = calibrate(player_games, validate=False, num_simulations=12_000)
    fit = report.calibration
    assert fit.teammate_base == pytest.approx(0.20, abs=0.08)
    assert fit.pitcher_vs_opposing == pytest.approx(-0.35, abs=0.10)
    assert fit.slates_used == 24 and fit.hitter_starts == 24 * 20 * 18
    for label, target in measured.items():
        if label in report.simulated:
            assert report.simulated[label] == pytest.approx(target, abs=0.04), label


def test_calibrate_refuses_tiny_samples():
    tiny = _synthetic_player_games(slates=1, games_per_slate=3)
    with pytest.raises(ValueError, match="Not enough outcomes"):
        calibrate(tiny, validate=False)


def test_candidates_are_ranked_against_the_field_only():
    field = np.array([[10.0, 20.0, 30.0, 40.0], [10.0, 20.0, 30.0, 40.0]])
    candidates = np.array([[45.0, 44.0, 25.0], [5.0, 20.0, 41.0]])
    pct = _percentiles_vs_field(candidates, field)
    # Sim 0: two candidates beat the whole field - BOTH are 100th percentile,
    # they do not knock each other down.
    assert pct[0, 0] == pytest.approx(100.0) and pct[0, 1] == pytest.approx(100.0)
    assert pct[0, 2] == pytest.approx(3 / 5 * 100)
    assert pct[1, 0] == pytest.approx(1 / 5 * 100), "beats nobody"
    assert pct[1, 1] == pytest.approx(2 / 5 * 100), "a tie with a field lineup is not a win over it"
    assert pct[1, 2] == pytest.approx(100.0)


def test_shipped_calibration_matches_the_measured_ballpark():
    shipped = load_simulation_calibration()
    assert shipped is not None, "config/simulation_calibration.json should ship with the branch"
    assert shipped.hitter_starts > 3000 and shipped.slates_used >= 20
    assert 0.05 <= shipped.teammate_base <= 0.25
    assert shipped.pitcher_vs_opposing < -0.2
    assert abs(shipped.same_game_opponent) < 0.08
