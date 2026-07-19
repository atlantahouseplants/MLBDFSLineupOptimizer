"""Builder routines for optimizer-ready datasets."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import re
from zoneinfo import ZoneInfo

import pandas as pd

_TIME_PATTERN = re.compile(r"(\d{1,2}:\d{2}\s*[AP]M)", re.IGNORECASE)
_DATE_PATTERN = re.compile(r"(\d{1,2}/\d{1,2})")
_EASTERN_TZ = ZoneInfo("US/Eastern")

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
    "proj_fd_median",
    "proj_fd_bust_rate",
    "proj_fd_upside",
    "proj_fd_pts_per_salary",
    "proj_fd_median_per_salary",
    "proj_fd_upside_per_salary",
    "proj_fd_ownership",
    "ownership_source_covered",
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
    "bpp_runs",
    "bpp_win_percent",
    "bpp_home_runs",
    "bpp_runs_first_inning_pct",
    "bpp_runs_first5away",
    "bpp_runs_first5home",
    "bpp_stack_count",
]

BPP_STACK_SIM_COLUMNS = (
    [f"bpp_runs{i}" for i in range(16)]
    + [f"bpp_home_runs{i}" for i in range(6)]
    + [f"bpp_runs_inning{i}" for i in range(1, 10)]
    + ["bpp_runs_inning_extra"]
)

OPTIMIZER_COLUMNS.extend(BPP_STACK_SIM_COLUMNS)


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _add_leverage_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "proj_fd_ownership" not in df.columns:
        df["player_leverage_score"] = 0.0
        df["team_leverage_score"] = 0.0
        return df
    df["proj_fd_ownership"] = pd.to_numeric(df["proj_fd_ownership"], errors="coerce").fillna(0.0)
    df["player_leverage_score"] = df["proj_fd_mean"].rank(pct=True) - df["proj_fd_ownership"].rank(pct=True)
    player_type = df.get("player_type", "").astype(str).str.lower()
    batters = df[player_type != "pitcher"].copy()
    team_ownership = batters.groupby("team_code")["proj_fd_ownership"].mean().rename("team_avg_ownership")
    df = df.merge(team_ownership, on="team_code", how="left")
    team_runs = (
        batters.assign(_bpp_runs=pd.to_numeric(batters.get("bpp_runs"), errors="coerce"))
        .dropna(subset=["_bpp_runs"])
        .drop_duplicates("team_code")
        .set_index("team_code")["_bpp_runs"]
    )
    run_rank = df["team_code"].map(team_runs.rank(pct=True))
    own_rank = df["team_code"].map(team_ownership.rank(pct=True))
    df["team_leverage_score"] = run_rank.fillna(0.0) - own_rank.fillna(0.0)
    df.drop(columns=["team_avg_ownership"], inplace=True)
    return df

def _assign_stack_priority(df: pd.DataFrame) -> pd.Series:
    runs = _safe_numeric(df["bpp_runs"]) if "bpp_runs" in df.columns else pd.Series(pd.NA, index=df.index)
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
    ).copy()

    for col in ("team_code", "opponent_code"):
        if col not in merged.columns:
            merged[col] = merged.get(col.replace("_code", ""), "").astype(str).str.upper()

    derived = pd.DataFrame(index=merged.index)
    derived["stack_key"] = merged["team_code"].fillna("")
    derived["game_key"] = merged.apply(_build_game_key, axis=1)
    derived["stack_priority"] = _assign_stack_priority(merged)
    derived["default_max_exposure"] = _default_max_exposure(merged)
    derived["salary"] = _safe_numeric(merged.get("salary")).fillna(0).astype(int)
    derived["game_start_time"] = _derive_game_start_times(merged)
    for col, default in {
        "batting_order_position": pd.Series(pd.NA, index=merged.index, dtype="Int64"),
        "order_factor": 1.0,
        "platoon_factor": 1.0,
        "is_confirmed_lineup": False,
        "batter_hand": "",
        "pitcher_hand": "",
        "recent_fppg": 0.0,
        "season_fppg": 0.0,
        "recency_factor": 1.0,
        "ownership_source_covered": False,
    }.items():
        if col not in merged.columns:
            derived[col] = default
    derived = derived[[col for col in derived.columns if col not in merged.columns]]
    if not derived.empty:
        merged = pd.concat([merged, derived], axis=1).copy()
    merged["salary"] = _safe_numeric(merged.get("salary")).fillna(0).astype(int)

    missing = [col for col in OPTIMIZER_COLUMNS if col not in merged.columns]
    _string_defaults = {"team", "opponent", "roster_position"}
    if missing:
        defaults = {
            col: ("" if col.startswith("bpp") or col in _string_defaults else False if col == "ownership_source_covered" else 0)
            for col in missing
        }
        merged = pd.concat([merged, pd.DataFrame(defaults, index=merged.index)], axis=1).copy()

    dataset = merged[OPTIMIZER_COLUMNS].copy()
    numeric_cols = [
        "proj_fd_mean",
        "proj_fd_floor",
        "proj_fd_ceiling",
        "proj_fd_median",
        "proj_fd_bust_rate",
        "proj_fd_upside",
        "proj_fd_pts_per_salary",
        "proj_fd_median_per_salary",
        "proj_fd_upside_per_salary",
        "proj_fd_ownership",
        "order_factor",
        "platoon_factor",
        "recent_fppg",
        "season_fppg",
        "recency_factor",
        "bpp_runs",
        "bpp_win_percent",
        "bpp_home_runs",
        "bpp_runs_first_inning_pct",
        "bpp_runs_first5away",
        "bpp_runs_first5home",
        "bpp_stack_count",
        "vegas_game_total",
        "vegas_team_total",
        "vegas_opponent_total",
        "vegas_moneyline",
        "vegas_implied_win_prob",
    ]
    numeric_cols.extend(BPP_STACK_SIM_COLUMNS)
    for col in numeric_cols:
        dataset[col] = _safe_numeric(dataset[col]).fillna(0.0)
    dataset["proj_fd_median"] = dataset["proj_fd_median"].where(
        dataset["proj_fd_median"] > 0,
        dataset["proj_fd_mean"],
    )
    dataset["proj_fd_upside"] = dataset["proj_fd_upside"].where(
        dataset["proj_fd_upside"] > 0,
        dataset["proj_fd_ceiling"],
    )
    dataset["proj_fd_bust_rate"] = dataset["proj_fd_bust_rate"].clip(lower=0.0, upper=1.0)
    dataset["game_start_time"] = pd.to_datetime(
        dataset["game_start_time"], errors="coerce", utc=True
    )
    dataset["game_start_time"] = dataset["game_start_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("")

    dataset["is_confirmed_lineup"] = dataset["is_confirmed_lineup"].astype(bool)
    dataset["ownership_source_covered"] = dataset["ownership_source_covered"].fillna(False).astype(bool)
    dataset["batter_hand"] = dataset["batter_hand"].astype(str)
    dataset["pitcher_hand"] = dataset["pitcher_hand"].astype(str)

    dataset = _add_leverage_columns(dataset)
    dataset.sort_values(by=["player_type", "team_code", "proj_fd_mean"], ascending=[True, True, False], inplace=True)
    return dataset.reset_index(drop=True)


__all__ = ["build_optimizer_dataset", "OPTIMIZER_COLUMNS"]
