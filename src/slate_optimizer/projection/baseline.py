"""Baseline projection helpers built on top of BallparkPal sims."""
from __future__ import annotations

import pandas as pd

BATTING_ORDER_MULTIPLIERS = {
    1: 1.12,
    2: 1.08,
    3: 1.10,
    4: 1.06,
    5: 1.02,
    6: 1.00,
    7: 0.96,
    8: 0.94,
    9: 0.92,
}
BENCH_MULTIPLIER = 0.85

OPPOSITE_HAND_BOOST = 1.06
SAME_HAND_PENALTY = 0.95
SWITCH_HITTER_BOOST = 1.03
RECENT_WINDOW_WEIGHTS = (0.5, 0.5)
DEFAULT_RECENCY_BLEND = (0.7, 0.3)
RECENCY_MIN_FACTOR = 0.8
RECENCY_MAX_FACTOR = 1.25

PROJECTION_COLUMNS = [
    "fd_player_id",
    "full_name",
    "team",
    "opponent",
    "position",
    "player_type",
    "salary",
    "proj_fd_mean",
    "proj_fd_floor",
    "proj_fd_ceiling",
    "proj_fd_ownership",
    "base_projection",
    "value_score",
    "batting_order_position",
    "order_factor",
    "is_confirmed_lineup",
    "batter_hand",
    "pitcher_hand",
    "platoon_factor",
    "recent_fppg",
    "season_fppg",
    "recency_factor",
    "vegas_multiplier",
    "floor_multiplier",
    "ceiling_multiplier",
    "vegas_game_total",
    "vegas_team_total",
    "vegas_opponent_total",
    "vegas_moneyline",
    "vegas_implied_win_prob",
    "park_adjustment",
    "pitch_adjustment",
    "weather_score",
]


def _team_run_multiplier(df: pd.DataFrame) -> pd.Series:
    if "bpp_runs" not in df.columns:
        return pd.Series(1.0, index=df.index)
    runs = pd.to_numeric(df["bpp_runs"], errors="coerce")
    temp = pd.DataFrame({"team": df["team"], "runs": runs}).dropna()
    if temp.empty:
        return pd.Series(1.0, index=df.index)
    team_means = temp.groupby("team")["runs"].mean()
    baseline = team_means.mean()
    if pd.isna(baseline) or baseline == 0:
        return pd.Series(1.0, index=df.index)
    multipliers = (team_means / baseline).clip(lower=0.75, upper=1.25)
    return df["team"].map(multipliers).fillna(1.0)


def _batting_order_adjustment(df: pd.DataFrame) -> pd.Series:
    positions = pd.to_numeric(df.get("batting_order_position"), errors="coerce")
    confirmed = df.get("is_confirmed_lineup")
    hitters = df.get("player_type", "").astype(str).str.lower() == "batter"
    if confirmed is None or not bool(pd.Series(confirmed).astype(bool).any()) or positions.notna().sum() == 0:
        return pd.Series(1.0, index=df.index)

    confirmed_bool = pd.Series(confirmed).astype(bool)
    factors = pd.Series(BENCH_MULTIPLIER, index=df.index, dtype=float)
    mapped = positions.map(BATTING_ORDER_MULTIPLIERS)
    factors.loc[confirmed_bool] = mapped.loc[confirmed_bool].fillna(BENCH_MULTIPLIER)
    result = pd.Series(1.0, index=df.index, dtype=float)
    result.loc[hitters] = factors.loc[hitters]
    return result.fillna(1.0)


def _opponent_pitcher_map(df: pd.DataFrame) -> pd.Series:
    pitchers = df[df.get("player_type", "").astype(str).str.lower() == "pitcher"].copy()
    if pitchers.empty:
        return pd.Series(dtype=str)
    pitchers["pitcher_hand"] = pitchers.get("pitcher_hand", "").astype(str).str.upper()
    pitchers = pitchers[pitchers["pitcher_hand"].isin(["L", "R"])]
    pitcher_map = (
        pitchers.dropna(subset=["team_code"])
        .drop_duplicates("team_code")
        .set_index("team_code")["pitcher_hand"]
    )
    pitcher_map.index = pitcher_map.index.astype(str)
    return pitcher_map


def _platoon_adjustment(
    df: pd.DataFrame,
    opposite_boost: float = OPPOSITE_HAND_BOOST,
    same_penalty: float = SAME_HAND_PENALTY,
    switch_boost: float = SWITCH_HITTER_BOOST,
) -> pd.Series:
    bats = df.get("batter_hand", "").astype(str).str.upper()
    opponent_codes = df.get("opponent_code", "").astype(str)
    hitters = df.get("player_type", "").astype(str).str.lower() == "batter"
    pitcher_map = _opponent_pitcher_map(df)
    opponent_hand = opponent_codes.map(pitcher_map).fillna("")

    factors = pd.Series(1.0, index=df.index, dtype=float)
    switch_mask = hitters & (bats == "S")
    factors.loc[switch_mask] = switch_boost

    opposite_mask = hitters & (
        ((bats == "L") & (opponent_hand == "R"))
        | ((bats == "R") & (opponent_hand == "L"))
    )
    same_mask = hitters & (
        ((bats == "L") & (opponent_hand == "L"))
        | ((bats == "R") & (opponent_hand == "R"))
    )
    factors.loc[opposite_mask] = opposite_boost
    factors.loc[same_mask] = same_penalty
    return factors.fillna(1.0)


def _recency_adjustment(
    df: pd.DataFrame,
    recency_blend: tuple[float, float] | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    last7 = pd.to_numeric(df.get("recent_last7_fppg"), errors="coerce")
    last14 = pd.to_numeric(df.get("recent_last14_fppg"), errors="coerce")
    season = pd.to_numeric(df.get("recent_season_fppg"), errors="coerce")

    if recency_blend is None:
        recency_blend = DEFAULT_RECENCY_BLEND
    season_weight, recent_weight = recency_blend
    total = season_weight + recent_weight
    if total <= 0:
        season_ratio = 1.0
        recent_ratio = 0.0
    else:
        season_ratio = season_weight / total
        recent_ratio = recent_weight / total

    w7, w14 = RECENT_WINDOW_WEIGHTS
    window_total = (w7 + w14) if (w7 + w14) != 0 else 1.0
    recency_score = ((last7 * w7) + (last14 * w14)) / window_total
    recency_score = recency_score.fillna(last7).fillna(last14)
    recency_score = recency_score.fillna(season)

    season_safe = season.where(season > 0)
    recency_ratio = recency_score / season_safe
    recency_ratio = recency_ratio.replace([pd.NA, pd.NaT], 1.0)
    recency_ratio = recency_ratio.fillna(1.0)
    recency_ratio = recency_ratio.replace([float("inf"), float("-inf")], 1.0)

    recency_factor = season_ratio * 1.0 + recent_ratio * recency_ratio
    recency_factor = recency_factor.clip(RECENCY_MIN_FACTOR, RECENCY_MAX_FACTOR)
    recency_factor = recency_factor.fillna(1.0)

    recent_score = recency_score.fillna(0.0)
    season_score = season.fillna(0.0)
    return recency_factor, recent_score, season_score


def _vegas_team_multiplier(df: pd.DataFrame) -> pd.Series:
    vegas_totals = pd.to_numeric(df.get("vegas_team_total"), errors="coerce")
    valid_mask = vegas_totals.notna() & (vegas_totals > 0)
    fallback = _team_run_multiplier(df)
    if not valid_mask.any():
        return fallback
    baseline = vegas_totals.loc[valid_mask].mean()
    if pd.isna(baseline) or baseline <= 0:
        return fallback
    vegas_multiplier = (vegas_totals / baseline).clip(lower=0.7, upper=1.3)
    result = fallback.fillna(1.0)
    result.loc[valid_mask] = vegas_multiplier.loc[valid_mask].fillna(result.loc[valid_mask])
    return result.fillna(1.0)


def _pitcher_win_multiplier(df: pd.DataFrame) -> pd.Series:
    if "bpp_win_percent" not in df.columns:
        return pd.Series(1.0, index=df.index)
    win_pct = pd.to_numeric(df["bpp_win_percent"], errors="coerce")
    adjustment = ((win_pct - 0.5) * 0.6).clip(-0.25, 0.25)
    multiplier = (1 + adjustment).fillna(1.0)
    vegas_prob = pd.to_numeric(df.get("vegas_implied_win_prob"), errors="coerce")
    if vegas_prob.notna().any():
        vegas_adj = ((vegas_prob - 0.5) * 0.8).clip(-0.2, 0.2)
        multiplier = multiplier * (1 + vegas_adj.fillna(0.0))
    return multiplier.fillna(1.0)


def _park_adjustment(df: pd.DataFrame) -> pd.Series:
    if "bpp_runs" not in df.columns:
        return pd.Series(0.0, index=df.index)
    runs = pd.to_numeric(df["bpp_runs"], errors="coerce")
    delta = runs - runs.mean()
    return delta.clip(-2, 2).fillna(0.0)


def _pitch_adjustment(df: pd.DataFrame) -> pd.Series:
    win_pct = pd.to_numeric(df.get("bpp_win_percent"), errors="coerce").fillna(0.5)
    runs_allowed = pd.to_numeric(df.get("bpp_runs_allowed"), errors="coerce")
    fills = runs_allowed.mean() if runs_allowed.notna().any() else 4.0
    runs_allowed = runs_allowed.fillna(fills)
    return ((0.5 - runs_allowed / 6) * 2.5 + (win_pct - 0.5) * 5).fillna(0.0)


def _weather_score(df: pd.DataFrame) -> pd.Series:
    weather = pd.to_numeric(df.get("bpp_runs_first_inning_pct"), errors="coerce")
    if weather.isna().all():
        weather = pd.to_numeric(df.get("bpp_runs_first5away"), errors="coerce")
    if weather.isna().all():
        return pd.Series(0.0, index=df.index)
    weather = (weather - weather.mean()).fillna(0.0)
    return weather.clip(-0.5, 0.5)


def _normalize_ownership(series: pd.Series, index: pd.Index) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if values.notna().any() and values.max(skipna=True) > 1.0:
        values = values / 100.0
    return values.reindex(index).clip(lower=0.0, upper=1.0)


def _scaled_percentile(values: pd.Series, min_own: float, max_own: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series((min_own + max_own) / 2.0, index=values.index, dtype=float)
    valid = numeric.notna()
    if not valid.any():
        return result

    valid_values = numeric.loc[valid]
    if len(valid_values) == 1:
        result.loc[valid] = max_own
        return result

    ranks = valid_values.rank(method="average", pct=False)
    percentiles = (ranks - 1.0) / (len(valid_values) - 1.0)
    result.loc[valid] = min_own + percentiles * (max_own - min_own)
    return result.clip(lower=min_own, upper=max_own)


def _estimate_ownership(df: pd.DataFrame) -> pd.Series:
    ownership = pd.Series(0.0, index=df.index, dtype=float)

    if "bpp_ownership" in df.columns:
        bpp_ownership = _normalize_ownership(df["bpp_ownership"], df.index)
        ownership.loc[bpp_ownership.notna()] = bpp_ownership.loc[bpp_ownership.notna()]

    player_type = df.get("player_type", "").astype(str).str.lower()
    batter_mask = player_type == "batter"
    pitcher_mask = player_type == "pitcher"
    missing_mask = ownership <= 0

    batter_missing = batter_mask & missing_mask
    if batter_missing.any():
        batter_estimates = _scaled_percentile(
            df.loc[batter_missing, "value_score"],
            min_own=0.05,
            max_own=0.35,
        )
        ownership.loc[batter_missing] = batter_estimates

    pitcher_missing = pitcher_mask & missing_mask
    if pitcher_missing.any():
        pitcher_estimates = _scaled_percentile(
            df.loc[pitcher_missing, "proj_fd_mean"],
            min_own=0.10,
            max_own=0.30,
        )
        ownership.loc[pitcher_missing] = pitcher_estimates

    fallback_missing = ownership <= 0
    if fallback_missing.any():
        ownership.loc[fallback_missing] = 0.05

    return ownership.clip(lower=0.0, upper=1.0)


def data_quality_report(players_df: pd.DataFrame) -> dict[str, int]:
    df = players_df.copy()

    def _numeric_column(column: str) -> pd.Series:
        values = df.get(column)
        if isinstance(values, pd.Series):
            return pd.to_numeric(values, errors="coerce")
        return pd.Series(pd.NA, index=df.index, dtype="Float64")

    bpp_projection = _numeric_column("bpp_points_fd")
    fppg = _numeric_column("fppg")
    batting_order = _numeric_column("batting_order_position")
    confirmed_lineup = pd.Series(df.get("is_confirmed_lineup", False), index=df.index).astype(bool)
    vegas_team_total = _numeric_column("vegas_team_total")
    recent_last7 = _numeric_column("recent_last7_fppg")

    report = {
        "n_players_with_bpp_projection": int(bpp_projection.notna().sum()),
        "n_players_using_fppg_fallback": int(bpp_projection.isna().mul(fppg.notna()).sum()),
        "n_players_with_confirmed_batting_order": int(batting_order.notna().mul(confirmed_lineup).sum()),
        "n_players_with_vegas_data": int((vegas_team_total.fillna(0.0) > 0).sum()),
        "n_players_with_recent_stats": int(recent_last7.notna().sum()),
    }
    return report



def compute_baseline_projections(
    players_df: pd.DataFrame,
    recency_blend: tuple[float, float] | None = None,
    platoon_opposite_boost: float | None = None,
    platoon_same_penalty: float | None = None,
    platoon_switch_boost: float | None = None,
) -> pd.DataFrame:
    """Generate naive mean/floor/ceiling projections from available columns."""
    df = players_df.copy()
    fd_points = pd.to_numeric(df.get("bpp_points_fd"), errors="coerce")
    fallback = pd.to_numeric(df.get("fppg"), errors="coerce")
    df["proj_fd_mean"] = fd_points.fillna(fallback).fillna(0.0)
    df["base_projection"] = df["proj_fd_mean"]

    if "player_type" not in df.columns:
        df["player_type"] = ""

    if "is_confirmed_lineup" not in df.columns:
        df["is_confirmed_lineup"] = False
    if "batting_order_position" not in df.columns:
        df["batting_order_position"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    if "batter_hand" not in df.columns:
        df["batter_hand"] = ""
    if "pitcher_hand" not in df.columns:
        df["pitcher_hand"] = ""

    for col in (
        "vegas_game_total",
        "vegas_team_total",
        "vegas_opponent_total",
        "vegas_moneyline",
        "vegas_implied_win_prob",
        "order_factor",
        "platoon_factor",
        "recent_last7_fppg",
        "recent_last14_fppg",
        "recent_season_fppg",
    ):
        if col not in df.columns:
            df[col] = 0.0

    park_adj = _park_adjustment(df)
    weather_score = _weather_score(df)
    hitters_mask = df["player_type"].str.lower() == "batter"
    df.loc[hitters_mask, "proj_fd_mean"] += park_adj.loc[hitters_mask] * 0.5 + weather_score.loc[hitters_mask]

    order_factor = _batting_order_adjustment(df)
    df["order_factor"] = order_factor
    df.loc[hitters_mask, "proj_fd_mean"] *= order_factor.loc[hitters_mask]

    opp_boost = (
        float(platoon_opposite_boost)
        if platoon_opposite_boost is not None
        else OPPOSITE_HAND_BOOST
    )
    same_penalty = (
        float(platoon_same_penalty)
        if platoon_same_penalty is not None
        else SAME_HAND_PENALTY
    )
    switch_boost = (
        float(platoon_switch_boost)
        if platoon_switch_boost is not None
        else SWITCH_HITTER_BOOST
    )

    platoon_factor = _platoon_adjustment(
        df,
        opposite_boost=opp_boost,
        same_penalty=same_penalty,
        switch_boost=switch_boost,
    )
    df["platoon_factor"] = platoon_factor
    df.loc[hitters_mask, "proj_fd_mean"] *= platoon_factor.loc[hitters_mask]

    recency_factor, recent_score, season_score = _recency_adjustment(
        df, recency_blend
    )
    df["recency_factor"] = recency_factor
    df["recent_fppg"] = recent_score
    df["season_fppg"] = season_score
    df["proj_fd_mean"] *= recency_factor

    pitch_adj = _pitch_adjustment(df)
    pitchers_mask = df["player_type"].str.lower() == "pitcher"
    win_multiplier = _pitcher_win_multiplier(df)
    df.loc[pitchers_mask, "proj_fd_mean"] = (
        df.loc[pitchers_mask, "proj_fd_mean"] * win_multiplier.loc[pitchers_mask]
        + pitch_adj.loc[pitchers_mask]
    )

    df["park_adjustment"] = park_adj.fillna(0.0)
    df["pitch_adjustment"] = pitch_adj.fillna(0.0)
    df["weather_score"] = weather_score.fillna(0.0)

    team_multiplier = _vegas_team_multiplier(df)
    df.loc[hitters_mask, "proj_fd_mean"] *= team_multiplier.loc[hitters_mask]
    df["vegas_multiplier"] = team_multiplier
    salary_raw = df.get("salary")
    if not isinstance(salary_raw, pd.Series):
        salary_raw = pd.Series(salary_raw, index=df.index)
    salary_series = pd.to_numeric(salary_raw, errors="coerce")
    salary_series = salary_series.replace(0, pd.NA)
    df["value_score"] = (df["proj_fd_mean"] / salary_series).fillna(0.0) * 1000.0
    df["proj_fd_ownership"] = _estimate_ownership(df)

    floor_multiplier = pd.Series(1.0, index=df.index, dtype=float)
    ceiling_multiplier = pd.Series(1.0, index=df.index, dtype=float)

    hitter_floor = (
        0.55
        + 0.15 * recency_factor.clip(0.7, 1.3)
        + 0.1 * order_factor.clip(0.8, 1.2)
    ).clip(0.5, 0.95)
    pitcher_floor = (
        0.65
        + 0.1 * win_multiplier.clip(0.8, 1.2)
    ).clip(0.55, 0.95)
    floor_multiplier.loc[hitters_mask] = hitter_floor.loc[hitters_mask]
    floor_multiplier.loc[pitchers_mask] = pitcher_floor.loc[pitchers_mask]

    hitter_ceiling = (
        1.2
        + 0.1 * (team_multiplier - 1.0)
        + 0.05 * (platoon_factor - 1.0)
        + 0.05 * recency_factor
    ).clip(1.05, 1.8)
    pitcher_ceiling = (
        1.25
        + 0.1 * win_multiplier
    ).clip(1.1, 1.8)
    ceiling_multiplier.loc[hitters_mask] = hitter_ceiling.loc[hitters_mask]
    ceiling_multiplier.loc[pitchers_mask] = pitcher_ceiling.loc[pitchers_mask]

    df["floor_multiplier"] = floor_multiplier.fillna(0.8)
    df["ceiling_multiplier"] = ceiling_multiplier.fillna(1.2)
    df["proj_fd_floor"] = df["proj_fd_mean"] * df["floor_multiplier"]
    df["proj_fd_ceiling"] = df["proj_fd_mean"] * df["ceiling_multiplier"]

    for col in ("team", "opponent", "position", "full_name"):
        if col not in df.columns:
            df[col] = ""

    missing_cols = [col for col in PROJECTION_COLUMNS if col not in df.columns]
    for col in missing_cols:
        if col in {"park_adjustment", "pitch_adjustment", "weather_score", "proj_fd_ownership"}:
            df[col] = 0.0
        elif col in {"platoon_factor", "order_factor", "recency_factor"}:
            df[col] = 1.0
        elif col in {"batting_order_position"}:
            df[col] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        elif col in {"is_confirmed_lineup"}:
            df[col] = False
        elif col in {"batter_hand", "pitcher_hand"}:
            df[col] = ""
        elif col in {"recent_fppg", "season_fppg"}:
            df[col] = 0.0
        else:
            df[col] = ""

    output = df[PROJECTION_COLUMNS].copy()
    output["salary"] = pd.to_numeric(output["salary"], errors="coerce").fillna(0).astype(int)
    return output


__all__ = [
    "compute_baseline_projections",
    "data_quality_report",
    "PROJECTION_COLUMNS",
]
