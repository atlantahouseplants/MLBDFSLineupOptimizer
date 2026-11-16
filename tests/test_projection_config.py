"""Projection switches, the corrected Vegas maths and the bugs the backtest found."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from slate_optimizer.ingestion.vegas import VegasLoader, implied_team_totals
from slate_optimizer.optimizer.dataset import build_optimizer_dataset
from slate_optimizer.projection.baseline import (
    ProjectionConfig,
    compute_baseline_projections,
    load_projection_config,
    save_projection_config,
    team_run_expectation,
)


def _slate(with_vegas: bool = True) -> pd.DataFrame:
    rows = []
    # Two teams: AAA is the home favourite, BBB the road underdog.
    for team, opp, side, hand in (("AAA", "BBB", "H", "L"), ("BBB", "AAA", "A", "R")):
        for slot in range(1, 10):
            rows.append(
                {
                    "fd_player_id": f"{team}-b{slot}",
                    "full_name": f"{team} Bat {slot}",
                    "team": team,
                    "team_code": team,
                    "opponent": opp,
                    "opponent_code": opp,
                    "player_type": "batter",
                    "position": "OF",
                    "salary": 3000,
                    "bpp_points_fd": 10.0,
                    "bpp_runs": 0.5,  # the batter's OWN runs, as in production
                    "bpp_side": side,
                    "bpp_runs_home": 5.0,
                    "bpp_runs_away": 4.0,
                    "batting_order_position": slot,
                    "is_confirmed_lineup": True,
                    "batter_hand": "R",
                    "pitcher_hand": "",
                }
            )
        rows.append(
            {
                "fd_player_id": f"{team}-p",
                "full_name": f"{team} Pitcher",
                "team": team,
                "team_code": team,
                "opponent": opp,
                "opponent_code": opp,
                "player_type": "pitcher",
                "position": "P",
                "salary": 9000,
                "bpp_points_fd": 30.0,
                "bpp_runs": 5.0 if side == "H" else 4.0,
                "bpp_side": side,
                "bpp_runs_home": 5.0,
                "bpp_runs_away": 4.0,
                "bpp_win_percent": 0.5,
                "bpp_runs_allowed": 3.0,
                "batter_hand": "",
                "pitcher_hand": hand,
            }
        )
    frame = pd.DataFrame(rows)
    if with_vegas:
        frame["vegas_game_total"] = 9.0
        frame["vegas_team_total"] = frame["team_code"].map({"AAA": 6.05, "BBB": 2.95})
        frame["vegas_implied_win_prob"] = frame["team_code"].map({"AAA": 0.7, "BBB": 0.3})
    return frame


def _proj(frame, config=None):
    out = compute_baseline_projections(frame, config=config)
    return out.set_index("fd_player_id")["proj_fd_mean"]


def test_default_config_is_the_legacy_projection():
    frame = _slate()
    pd.testing.assert_series_equal(_proj(frame), _proj(frame, ProjectionConfig()))


def test_raw_config_leaves_ballparkpal_points_untouched():
    proj = _proj(_slate(), ProjectionConfig.raw_bpp())
    assert proj[[pid for pid in proj.index if "-b" in pid]].eq(10.0).all()
    assert proj[["AAA-p", "BBB-p"]].eq(30.0).all()


def test_each_switch_isolates_one_adjustment():
    frame = _slate()
    raw = ProjectionConfig.raw_bpp().to_dict()
    order_only = _proj(frame, ProjectionConfig.from_dict({**raw, "apply_batting_order": True}))
    assert order_only["AAA-b1"] == pytest.approx(11.2)
    assert order_only["AAA-b9"] == pytest.approx(9.2)
    platoon_only = _proj(frame, ProjectionConfig.from_dict({**raw, "apply_platoon": True}))
    # AAA right-handed bats face BBB's right-hander (same hand); BBB bats face a lefty.
    assert platoon_only["AAA-b1"] == pytest.approx(9.5)
    assert platoon_only["BBB-b1"] == pytest.approx(10.6)


def test_market_ratio_moves_hitters_by_disagreement_not_level():
    frame = _slate()
    config = ProjectionConfig.from_dict(
        {**ProjectionConfig.raw_bpp().to_dict(), "team_total_mode": "market_ratio", "market_blend_alpha": 0.5, "market_ratio_clip": 0.5}
    )
    proj = _proj(frame, config)
    assert proj["AAA-b1"] == pytest.approx(10.0 * (6.05 / 5.0) ** 0.5)
    assert proj["BBB-b1"] == pytest.approx(10.0 * (2.95 / 4.0) ** 0.5)
    assert proj["AAA-p"] == pytest.approx(30.0), "pitchers are not touched by the hitter team-total blend"

    agree = frame.copy()
    agree["vegas_team_total"] = agree["team_code"].map({"AAA": 5.0, "BBB": 4.0})
    assert _proj(agree, config)[[pid for pid in proj.index if "-b" in pid]].eq(10.0).all()


def test_market_ratio_ignores_in_play_lines():
    frame = _slate()
    frame["vegas_game_total"] = 1.5  # a live line quoted after first pitch
    frame["vegas_team_total"] = frame["team_code"].map({"AAA": 1.9, "BBB": 1.6})
    config = ProjectionConfig.from_dict({**ProjectionConfig.raw_bpp().to_dict(), "team_total_mode": "market_ratio"})
    assert _proj(frame, config)["AAA-b1"] == pytest.approx(10.0)


def test_missing_vegas_no_longer_cuts_pitchers_by_twenty_percent():
    """Regression: a run with no odds file used to read the 0.0 placeholder as a 0% win chance."""
    no_vegas = _slate(with_vegas=False)
    only_pitchers = ProjectionConfig.from_dict({**ProjectionConfig.raw_bpp().to_dict(), "apply_pitcher_adjustments": True})
    proj = _proj(no_vegas, only_pitchers)
    # bpp_win_percent 0.5 and 3.0 runs allowed -> multiplier 1.0, additive 0.0.
    assert proj["AAA-p"] == pytest.approx(30.0)
    zero_prob = no_vegas.assign(vegas_implied_win_prob=0.0)
    assert _proj(zero_prob, only_pitchers)["AAA-p"] == pytest.approx(30.0)


def test_implied_team_totals_use_run_ratio_not_win_probability():
    team, opponent = implied_team_totals(8.5, 0.72)
    assert team + opponent == pytest.approx(8.5)
    assert team == pytest.approx(5.32, abs=0.03)
    assert team < 8.5 * 0.72, "the old total x win-prob formula over-credited favourites"
    even_team, even_opp = implied_team_totals(9.0, 0.5)
    assert even_team == pytest.approx(4.5) and even_opp == pytest.approx(4.5)
    series_team, _ = implied_team_totals(pd.Series([8.5, 9.0]), pd.Series([0.72, 0.5]))
    assert series_team.tolist() == pytest.approx([team, 4.5])


def test_vegas_loader_emits_corrected_team_totals(tmp_path):
    path = tmp_path / "vegas.csv"
    path.write_text("game,total,home_ml,away_ml\nBBB@AAA,8.5,-260,235\n", encoding="utf-8")
    totals = VegasLoader(path).load().team_totals.set_index("team_code")
    assert totals.loc["AAA", "vegas_team_total"] + totals.loc["BBB", "vegas_team_total"] == pytest.approx(8.5)
    assert 5.0 < totals.loc["AAA", "vegas_team_total"] < 5.6
    assert 2.9 < totals.loc["BBB", "vegas_team_total"] < 3.5


def test_team_run_expectation_resolves_team_totals_for_batters():
    runs = team_run_expectation(_slate())
    frame = _slate()
    assert runs[frame["team_code"] == "AAA"].eq(5.0).all()
    assert runs[frame["team_code"] == "BBB"].eq(4.0).all()
    # Without the Games columns it must not mistake a batter's own 0.5 for a team total.
    stripped = frame.drop(columns=["bpp_side", "bpp_runs_home", "bpp_runs_away"])
    fallback = team_run_expectation(stripped)
    assert fallback[stripped["team_code"] == "AAA"].eq(5.0).all(), "inherits the pitcher row's team total"


def test_optimizer_dataset_ranks_team_leverage_on_team_runs():
    frame = _slate(with_vegas=False)
    # Make the WORSE offence's first hitter look great individually.
    frame.loc[frame["fd_player_id"] == "BBB-b1", "bpp_runs"] = 0.9
    projections = compute_baseline_projections(frame, config=ProjectionConfig.raw_bpp())
    projections["proj_fd_ownership"] = 0.1
    dataset = build_optimizer_dataset(frame, projections)
    team_runs = dataset.groupby("team_code")["bpp_team_runs"].max()
    assert team_runs["AAA"] == pytest.approx(5.0) and team_runs["BBB"] == pytest.approx(4.0)
    leverage = dataset[dataset["player_type"] == "batter"].groupby("team_code")["team_leverage_score"].first()
    assert leverage["AAA"] > leverage["BBB"], "equal ownership: the better run environment must carry more leverage"


def test_projection_config_round_trip_and_safe_fallback(tmp_path):
    path = tmp_path / "projection_model.json"
    assert load_projection_config(path) == ProjectionConfig(), "missing file means legacy behaviour"
    chosen = ProjectionConfig.from_dict({**ProjectionConfig.raw_bpp().to_dict(), "team_total_mode": "market_ratio", "source": "test"})
    save_projection_config(chosen, path)
    assert load_projection_config(path) == chosen
    path.write_text("{not json", encoding="utf-8")
    assert load_projection_config(path) == ProjectionConfig()
    path.write_text(json.dumps({"team_total_mode": "nonsense"}), encoding="utf-8")
    assert load_projection_config(path) == ProjectionConfig()
    path.write_text(json.dumps({"apply_platoon": False, "future_setting": 1}), encoding="utf-8")
    assert load_projection_config(path).apply_platoon is False, "unknown keys are ignored, known ones applied"


def test_shipped_config_is_loadable_and_backtest_backed():
    config = load_projection_config()
    assert config.team_total_mode in {"market_ratio", "off", "legacy"}
    # The shipped file must be the calibrated one, not an accidental legacy fallback.
    assert config.source != "legacy-defaults"
    assert not config.apply_batting_order and not config.apply_platoon
