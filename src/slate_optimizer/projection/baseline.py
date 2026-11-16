"""Baseline projection helpers built on top of BallparkPal sims."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_PROJECTION_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "projection_model.json"

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

TEAM_TOTAL_MODES = ("legacy", "market_ratio", "off")


@dataclass
class ProjectionConfig:
    """Switches and strengths for every adjustment layered on BallparkPal points.

    The dataclass defaults reproduce the 2026-season behaviour exactly, so
    callers that pass nothing get the legacy projection.  The production
    pipelines call :func:`load_projection_config`, which reads
    ``config/projection_model.json`` - the settings chosen by
    ``scripts/backtest_projections.py`` against real box scores.

    ``team_total_mode``:
      * ``legacy``       - hitters x (Vegas team total / slate average), clipped.
                           Double counts the run environment BallparkPal has
                           already simulated.
      * ``market_ratio`` - hitters x (Vegas team total / BallparkPal team runs)
                           ** ``market_blend_alpha``.  Only the *disagreement*
                           between the market and BallparkPal moves a projection.
      * ``off``          - no team-total adjustment.
    """

    apply_park_weather: bool = True
    apply_batting_order: bool = True
    apply_platoon: bool = True
    apply_recency: bool = True
    apply_pitcher_adjustments: bool = True
    team_total_mode: str = "legacy"
    market_blend_alpha: float = 0.5
    market_ratio_clip: float = 0.20
    source: str = "legacy-defaults"

    def __post_init__(self) -> None:
        if self.team_total_mode not in TEAM_TOTAL_MODES:
            raise ValueError(
                f"team_total_mode must be one of {TEAM_TOTAL_MODES}, got {self.team_total_mode!r}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "ProjectionConfig":
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in known})

    @classmethod
    def raw_bpp(cls) -> "ProjectionConfig":
        """BallparkPal points untouched - the reference every variant must beat."""
        return cls(
            apply_park_weather=False,
            apply_batting_order=False,
            apply_platoon=False,
            apply_recency=False,
            apply_pitcher_adjustments=False,
            team_total_mode="off",
            source="raw-bpp",
        )


def load_projection_config(path: Path | str | None = None) -> ProjectionConfig:
    """Read the calibrated projection settings, falling back to legacy defaults."""
    target = Path(path) if path else DEFAULT_PROJECTION_CONFIG_PATH
    if not target.exists():
        return ProjectionConfig()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return ProjectionConfig()
    if not isinstance(payload, dict):
        return ProjectionConfig()
    try:
        return ProjectionConfig.from_dict(payload)
    except (TypeError, ValueError):
        return ProjectionConfig()


def save_projection_config(config: ProjectionConfig, path: Path | str | None = None) -> None:
    target = Path(path) if path else DEFAULT_PROJECTION_CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")


def team_run_expectation(df: pd.DataFrame) -> pd.Series:
    """BallparkPal's simulated runs for each row's own team.

    ``bpp_runs`` cannot be used for this: after the slate merge it holds a
    batter's *individual* projected runs scored (~0.5) on batter rows and the
    team total (~4.5) on pitcher rows.  The Games sheet columns are
    unambiguous, so prefer them and only fall back to ``bpp_runs`` when it
    is clearly a team-level number.
    """
    result = pd.Series(np.nan, index=df.index, dtype=float)
    if {"bpp_side", "bpp_runs_home", "bpp_runs_away"}.issubset(df.columns):
        side = df["bpp_side"].astype(str).str.upper().str[:1]
        home = pd.to_numeric(df["bpp_runs_home"], errors="coerce")
        away = pd.to_numeric(df["bpp_runs_away"], errors="coerce")
        result = home.where(side == "H", away.where(side == "A"))
    if "bpp_team_runs" in df.columns:
        result = result.fillna(pd.to_numeric(df["bpp_team_runs"], errors="coerce"))
    if "bpp_runs" in df.columns:
        runs = pd.to_numeric(df["bpp_runs"], errors="coerce")
        result = result.fillna(runs.where(runs >= 1.5))
    team_key = "team_code" if "team_code" in df.columns else ("team" if "team" in df.columns else None)
    if team_key is not None and result.notna().any():
        # Bench bats and unmatched rows inherit their team's number.
        team_values = result.groupby(df[team_key].astype(str)).transform("max")
        result = result.fillna(team_values)
    return result


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


def _market_ratio_multiplier(df: pd.DataFrame, alpha: float, clip: float) -> pd.Series:
    """Move hitters by the market's disagreement with BallparkPal, not its level."""
    vegas_totals = df.get("vegas_team_total")
    if not isinstance(vegas_totals, pd.Series):
        return pd.Series(1.0, index=df.index)
    vegas_totals = pd.to_numeric(vegas_totals, errors="coerce")
    bpp_totals = team_run_expectation(df)
    valid = vegas_totals.notna() & (vegas_totals > 1.5) & bpp_totals.notna() & (bpp_totals > 1.5)
    game_totals = df.get("vegas_game_total")
    if isinstance(game_totals, pd.Series):
        # Pre-game MLB totals live between about 6 and 13.  Anything outside
        # 5.5-15 is an in-play or broken line (the odds feed keeps quoting
        # games after first pitch) and must not touch a projection.
        game_totals = pd.to_numeric(game_totals, errors="coerce")
        valid &= game_totals.isna() | game_totals.between(5.5, 15.0)
    ratio = (vegas_totals / bpp_totals).where(valid, 1.0)
    clip = abs(float(clip))
    multiplier = ratio.pow(float(alpha)).clip(lower=1.0 - clip, upper=1.0 + clip)
    return multiplier.fillna(1.0)


def _pitcher_win_multiplier(df: pd.DataFrame) -> pd.Series:
    if "bpp_win_percent" not in df.columns:
        return pd.Series(1.0, index=df.index)
    win_pct = pd.to_numeric(df["bpp_win_percent"], errors="coerce")
    adjustment = ((win_pct - 0.5) * 0.6).clip(-0.25, 0.25)
    multiplier = (1 + adjustment).fillna(1.0)
    vegas_prob = df.get("vegas_implied_win_prob")
    if not isinstance(vegas_prob, pd.Series):
        return multiplier.fillna(1.0)
    vegas_prob = pd.to_numeric(vegas_prob, errors="coerce")
    # A missing Vegas file arrives here as 0.0, not NaN.  Reading that as a
    # 0% win chance cut every pitcher's projection by 20% on runs with no
    # odds - so only genuine probabilities are allowed to move a projection.
    vegas_prob = vegas_prob.where((vegas_prob > 0.0) & (vegas_prob < 1.0))
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
    # df.get() returns None for a missing column and pd.to_numeric(None)
    # yields a SCALAR nan (numpy.float64) — .isna() then blows up. The
    # dashboard path uploads the BPP Excel (column present); the cron's
    # auto-fetched data lacks it. Normalize missing columns to a Series.
    def _col(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(float("nan"), index=df.index, dtype=float)

    weather = _col("bpp_runs_first_inning_pct")
    if weather.isna().all():
        weather = _col("bpp_runs_first5away")
    if weather.isna().all():
        return pd.Series(0.0, index=df.index)
    weather = (weather - weather.mean()).fillna(0.0)
    return weather.clip(-0.5, 0.5)


def _probability_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    else:
        values = pd.Series(0.0, index=df.index, dtype=float)
    if values.max() > 1.5:
        values = values / 100.0
    return values.clip(lower=0.0, upper=1.0)



def compute_baseline_projections(
    players_df: pd.DataFrame,
    recency_blend: tuple[float, float] | None = None,
    platoon_opposite_boost: float | None = None,
    platoon_same_penalty: float | None = None,
    platoon_switch_boost: float | None = None,
    config: Optional[ProjectionConfig] = None,
) -> pd.DataFrame:
    """Generate mean/floor/ceiling projections from available columns.

    ``config`` selects which adjustments run; ``None`` means the legacy
    all-multipliers behaviour (see :class:`ProjectionConfig`).
    """
    config = config or ProjectionConfig()
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

    hitters_mask = df["player_type"].str.lower() == "batter"
    if config.apply_park_weather:
        park_adj = _park_adjustment(df)
        weather_score = _weather_score(df)
    else:
        park_adj = pd.Series(0.0, index=df.index)
        weather_score = pd.Series(0.0, index=df.index)
    df.loc[hitters_mask, "proj_fd_mean"] += park_adj.loc[hitters_mask] * 0.5 + weather_score.loc[hitters_mask]

    if config.apply_batting_order:
        order_factor = _batting_order_adjustment(df)
    else:
        order_factor = pd.Series(1.0, index=df.index, dtype=float)
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

    if config.apply_platoon:
        platoon_factor = _platoon_adjustment(
            df,
            opposite_boost=opp_boost,
            same_penalty=same_penalty,
            switch_boost=switch_boost,
        )
    else:
        platoon_factor = pd.Series(1.0, index=df.index, dtype=float)
    df["platoon_factor"] = platoon_factor
    df.loc[hitters_mask, "proj_fd_mean"] *= platoon_factor.loc[hitters_mask]

    recency_factor, recent_score, season_score = _recency_adjustment(
        df, recency_blend
    )
    if not config.apply_recency:
        recency_factor = pd.Series(1.0, index=df.index, dtype=float)
    df["recency_factor"] = recency_factor
    df["recent_fppg"] = recent_score
    df["season_fppg"] = season_score
    df["proj_fd_mean"] *= recency_factor

    pitchers_mask = df["player_type"].str.lower() == "pitcher"
    if config.apply_pitcher_adjustments:
        pitch_adj = _pitch_adjustment(df)
        win_multiplier = _pitcher_win_multiplier(df)
    else:
        pitch_adj = pd.Series(0.0, index=df.index)
        win_multiplier = pd.Series(1.0, index=df.index, dtype=float)
    df.loc[pitchers_mask, "proj_fd_mean"] = (
        df.loc[pitchers_mask, "proj_fd_mean"] * win_multiplier.loc[pitchers_mask]
        + pitch_adj.loc[pitchers_mask]
    )

    df["park_adjustment"] = park_adj.fillna(0.0)
    df["pitch_adjustment"] = pitch_adj.fillna(0.0)
    df["weather_score"] = weather_score.fillna(0.0)

    if config.team_total_mode == "legacy":
        team_multiplier = _vegas_team_multiplier(df)
    elif config.team_total_mode == "market_ratio":
        team_multiplier = _market_ratio_multiplier(
            df, config.market_blend_alpha, config.market_ratio_clip
        )
    else:
        team_multiplier = pd.Series(1.0, index=df.index, dtype=float)
    df.loc[hitters_mask, "proj_fd_mean"] *= team_multiplier.loc[hitters_mask]
    df["vegas_multiplier"] = team_multiplier
    salary_raw = df.get("salary")
    if not isinstance(salary_raw, pd.Series):
        salary_raw = pd.Series(salary_raw, index=df.index)
    salary_series = pd.to_numeric(salary_raw, errors="coerce")
    salary_series = salary_series.replace(0, pd.NA)
    df["value_score"] = (df["proj_fd_mean"] / salary_series).fillna(0.0) * 1000.0

    floor_multiplier = pd.Series(1.0, index=df.index, dtype=float)
    ceiling_multiplier = pd.Series(1.0, index=df.index, dtype=float)

    hitter_floor = (
        0.55
        + 0.15 * recency_factor.clip(0.7, 1.3)
        + 0.1 * order_factor.clip(0.8, 1.2)
    ).clip(0.5, 0.95)
    hit_probability = _probability_column(df, "bpp_hit_probability")
    hitter_floor = (hitter_floor + 0.05 * hit_probability).clip(0.5, 0.98)
    pitcher_floor = (
        0.65
        + 0.1 * win_multiplier.clip(0.8, 1.2)
    ).clip(0.55, 0.95)
    quality_start_probability = _probability_column(df, "bpp_quality_start")
    pitcher_floor = (pitcher_floor + 0.08 * quality_start_probability).clip(0.55, 0.98)
    floor_multiplier.loc[hitters_mask] = hitter_floor.loc[hitters_mask]
    floor_multiplier.loc[pitchers_mask] = pitcher_floor.loc[pitchers_mask]

    home_run_probability = _probability_column(df, "bpp_home_run_probability")
    stolen_base_probability = _probability_column(df, "bpp_stolen_base_probability")
    hitter_ceiling = (
        1.2
        + 0.1 * (team_multiplier - 1.0)
        + 0.05 * (platoon_factor - 1.0)
        + 0.05 * recency_factor
        + 0.30 * home_run_probability
        + 0.10 * stolen_base_probability
        + 0.04 * hit_probability
    ).clip(1.05, 2.0)
    pitcher_ceiling = (
        1.25
        + 0.1 * win_multiplier
        + 0.15 * quality_start_probability
    ).clip(1.1, 1.9)
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


__all__ = [
    "DEFAULT_PROJECTION_CONFIG_PATH",
    "PROJECTION_COLUMNS",
    "ProjectionConfig",
    "compute_baseline_projections",
    "data_quality_report",
    "load_projection_config",
    "save_projection_config",
    "team_run_expectation",
]
