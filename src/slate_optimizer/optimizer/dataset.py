"""Builder routines for optimizer-ready datasets."""
from __future__ import annotations

from typing import List

import pandas as pd

OPTIMIZER_COLUMNS = [
    "fd_player_id",
    "full_name",
    "position",
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
    "stack_priority",
    "default_max_exposure",
    "player_leverage_score",
    "team_leverage_score",
    "bpp_runs",
    "bpp_win_percent",
]


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _add_leverage_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "proj_fd_ownership" not in df.columns:
        df["player_leverage_score"] = 0.0
        df["team_leverage_score"] = 0.0
        return df
    df["proj_fd_ownership"] = pd.to_numeric(df["proj_fd_ownership"], errors="coerce").fillna(0.0)
    df["player_leverage_score"] = df["proj_fd_mean"].rank(pct=True) - df["proj_fd_ownership"].rank(pct=True)
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

    missing = [col for col in OPTIMIZER_COLUMNS if col not in merged.columns]
    for col in missing:
        merged[col] = "" if col.startswith("bpp") or col in {"team", "opponent"} else 0

    dataset = merged[OPTIMIZER_COLUMNS].copy()
    dataset["proj_fd_mean"] = _safe_numeric(dataset["proj_fd_mean"]).fillna(0.0)
    dataset["proj_fd_floor"] = _safe_numeric(dataset["proj_fd_floor"]).fillna(0.0)
    dataset["proj_fd_ceiling"] = _safe_numeric(dataset["proj_fd_ceiling"]).fillna(0.0)
    dataset["proj_fd_ownership"] = _safe_numeric(dataset["proj_fd_ownership"]).fillna(0.0)
    dataset["bpp_runs"] = _safe_numeric(dataset["bpp_runs"]).fillna(0.0)
    dataset["bpp_win_percent"] = _safe_numeric(dataset["bpp_win_percent"]).fillna(0.0)

    dataset = _add_leverage_columns(dataset)
    dataset.sort_values(by=["player_type", "team_code", "proj_fd_mean"], ascending=[True, True, False], inplace=True)
    return dataset.reset_index(drop=True)


__all__ = ["build_optimizer_dataset", "OPTIMIZER_COLUMNS"]
