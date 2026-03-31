"""Builder routines for optimizer-ready datasets."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import re
from datetime import timezone, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from slate_optimizer.projection.game_environment import merge_game_environment_columns

_TIME_PATTERN = re.compile(r"(\d{1,2}:\d{2}\s*[AP]M)", re.IGNORECASE)
_DATE_PATTERN = re.compile(r"(\d{1,2}/\d{1,2})")
try:
    _EASTERN_TZ = ZoneInfo("US/Eastern")
except Exception:
    # Fallback for environments missing tzdata (e.g. Streamlit Cloud)
    _EASTERN_TZ = timezone(timedelta(hours=-4))

OPTIMIZER_COLUMNS = [
    "fd_player_id",
    "full_name",
    "position",
    "roster_position",
    "player_type",
    "team",
    "team_code",
    "opponent",
    "opponent_code",
    "game_pk",
    "stack_key",
    "game_key",
    "salary",
    "proj_fd_mean",
    "proj_fd_floor",
    "proj_fd_ceiling",
    "proj_fd_ownership",
    "batting_order_position",
    "order_factor",
    "is_confirmed_lineup",
    "batter_hand",
    "pitcher_hand",
    "platoon_factor",
    "recent_fppg",
    "season_fppg",
    "recency_factor",
    "game_start_time",
    "vegas_game_total",
    "vegas_team_total",
    "vegas_opponent_total",
    "vegas_moneyline",
    "vegas_implied_win_prob",
    "stack_priority",
    "default_max_exposure",
    "player_leverage_score",
    "team_leverage_score",
    "ownership_tier",
    "game_leverage_score",
    "environment_tier",
    "team_gpp_leverage",
    "bpp_runs",
    "bpp_win_percent",
]


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _classify_ownership_tier(ownership: pd.Series) -> pd.Series:
    """Classify players into ownership tiers: chalk / mid / leverage / deep."""
    tiers = pd.Series("mid", index=ownership.index)
    tiers = tiers.mask(ownership > 0.20, "chalk")
    tiers = tiers.mask((ownership >= 0.03) & (ownership <= 0.08), "leverage")
    tiers = tiers.mask(ownership < 0.03, "deep")
    return tiers


def _add_leverage_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "proj_fd_ownership" not in df.columns:
        df["player_leverage_score"] = 0.0
        df["team_leverage_score"] = 0.0
        df["ownership_tier"] = "mid"
        return df
    df["proj_fd_ownership"] = pd.to_numeric(df["proj_fd_ownership"], errors="coerce").fillna(0.0)
    df["player_leverage_score"] = df["proj_fd_mean"].rank(pct=True) - df["proj_fd_ownership"].rank(pct=True)
    df["ownership_tier"] = _classify_ownership_tier(df["proj_fd_ownership"])
    team_ownership = df.groupby("team_code")["proj_fd_ownership"].mean().rename("team_avg_ownership")
    df = df.merge(team_ownership, on="team_code", how="left")
    run_rank = pd.to_numeric(df.get("bpp_runs"), errors="coerce").rank(pct=True)
    own_rank = df["team_avg_ownership"].rank(pct=True)
    df["team_leverage_score"] = run_rank.fillna(0.0) - own_rank.fillna(0.0)
    df.drop(columns=["team_avg_ownership"], inplace=True)
    return df

def _assign_stack_priority(df: pd.DataFrame) -> pd.Series:
    runs = _safe_numeric(df.get("bpp_runs"))
    if runs.notna().sum() < 3:
        return pd.Series(["mid"] * len(df), index=df.index)
    q_low, q_high = runs.quantile([0.33, 0.66])
    priority = pd.Series("mid", index=df.index)
    priority = priority.mask(runs <= q_low, "low")
    priority = priority.mask(runs >= q_high, "high")
    return priority.fillna("mid")


def _default_max_exposure(df: pd.DataFrame) -> pd.Series:
    return df["player_type"].str.lower().map({"pitcher": 0.4}).fillna(0.65)


def _build_game_key(row: pd.Series) -> str:
    teams = sorted([row.get("team_code", ""), row.get("opponent_code", "")])
    return "_vs_".join(teams)


def _parse_fd_game_string(value: str, reference_year: int) -> Optional[pd.Timestamp]:
    if not isinstance(value, str):
        return None
    time_match = _TIME_PATTERN.search(value)
    if not time_match:
        return None
    time_str = time_match.group(1).upper().replace(" ", "")
    try:
        time_obj = datetime.strptime(time_str, "%I:%M%p").time()
    except ValueError:
        return None
    date_match = _DATE_PATTERN.search(value)
    if date_match:
        month, day = map(int, date_match.group(1).split("/"))
        date_obj = datetime(reference_year, month, day).date()
    else:
        date_obj = datetime.now(_EASTERN_TZ).date()
    return pd.Timestamp.combine(date_obj, time_obj)


def _derive_game_start_times(df: pd.DataFrame) -> pd.Series:
    current_year = datetime.now(_EASTERN_TZ).year

    def parse_row(row: pd.Series) -> pd.Timestamp:
        ts = pd.NaT
        date_value = row.get("bpp_game_date")
        time_value = row.get("bpp_game_time")
        if pd.notna(date_value) and pd.notna(time_value):
            ts = pd.to_datetime(f"{date_value} {time_value}", errors="coerce")
        if (pd.isna(ts) or ts is None) and pd.notna(row.get("game")):
            parsed = _parse_fd_game_string(row.get("game"), current_year)
            if parsed is not None:
                ts = pd.Timestamp(parsed)
        if pd.isna(ts) or ts is None:
            return pd.NaT
        if ts.tzinfo is None:
            try:
                ts = ts.tz_localize(_EASTERN_TZ)
            except (TypeError, ValueError):
                ts = pd.Timestamp(ts, tz=_EASTERN_TZ)
        return ts.tz_convert("UTC")

    return df.apply(parse_row, axis=1)


def build_optimizer_dataset(
    players_df: pd.DataFrame,
    projections_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = players_df.merge(
        projections_df,
        on="fd_player_id",
        how="left",
        suffixes=("", "_proj"),
    )

    for col in ("team_code", "opponent_code"):
        if col not in merged.columns:
            merged[col] = merged.get(col.replace("_code", ""), "").astype(str).str.upper()

    merged["stack_key"] = merged["team_code"].fillna("")
    merged["game_key"] = merged.apply(_build_game_key, axis=1)
    merged["stack_priority"] = _assign_stack_priority(merged)
    merged["default_max_exposure"] = _default_max_exposure(merged)

    merged["salary"] = _safe_numeric(merged.get("salary")).fillna(0).astype(int)

    if "batting_order_position" not in merged.columns:
        merged["batting_order_position"] = pd.Series(pd.NA, index=merged.index, dtype="Int64")
    if "order_factor" not in merged.columns:
        merged["order_factor"] = 1.0
    if "platoon_factor" not in merged.columns:
        merged["platoon_factor"] = 1.0
    if "is_confirmed_lineup" not in merged.columns:
        merged["is_confirmed_lineup"] = False
    if "batter_hand" not in merged.columns:
        merged["batter_hand"] = ""
    if "pitcher_hand" not in merged.columns:
        merged["pitcher_hand"] = ""
    if "recent_fppg" not in merged.columns:
        merged["recent_fppg"] = 0.0
    if "season_fppg" not in merged.columns:
        merged["season_fppg"] = 0.0
    if "recency_factor" not in merged.columns:
        merged["recency_factor"] = 1.0
    merged["game_start_time"] = _derive_game_start_times(merged)

    missing = [col for col in OPTIMIZER_COLUMNS if col not in merged.columns]
    _string_defaults = {"team", "opponent", "roster_position"}
    for col in missing:
        merged[col] = "" if col.startswith("bpp") or col in _string_defaults else 0

    dataset = merged[OPTIMIZER_COLUMNS].copy()
    numeric_cols = [
        "proj_fd_mean",
        "proj_fd_floor",
        "proj_fd_ceiling",
        "proj_fd_ownership",
        "order_factor",
        "platoon_factor",
        "recent_fppg",
        "season_fppg",
        "recency_factor",
        "game_start_time",
        "bpp_runs",
        "bpp_win_percent",
        "vegas_game_total",
        "vegas_team_total",
        "vegas_opponent_total",
        "vegas_moneyline",
        "vegas_implied_win_prob",
    ]
    for col in numeric_cols:
        dataset[col] = _safe_numeric(dataset[col]).fillna(0.0)

    dataset["is_confirmed_lineup"] = dataset["is_confirmed_lineup"].astype(bool)
    dataset["batter_hand"] = dataset["batter_hand"].astype(str)
    dataset["pitcher_hand"] = dataset["pitcher_hand"].astype(str)

    dataset = _add_leverage_columns(dataset)
    dataset = merge_game_environment_columns(dataset)
    dataset.sort_values(by=["player_type", "team_code", "proj_fd_mean"], ascending=[True, True, False], inplace=True)
    return dataset.reset_index(drop=True)


__all__ = ["build_optimizer_dataset", "OPTIMIZER_COLUMNS"]
