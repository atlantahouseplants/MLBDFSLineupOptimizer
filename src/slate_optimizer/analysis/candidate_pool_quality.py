"""Candidate lineup-pool quality gates before final portfolio selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class CandidatePoolQualityReport:
    summary: dict[str, float]
    checks: pd.DataFrame
    player_exposure: pd.DataFrame
    stack_exposure: pd.DataFrame
    stack_template_mix: pd.DataFrame


def build_candidate_pool_quality_report(
    lineup_df: Optional[pd.DataFrame],
    target_final_lineups: int = 300,
    min_pool_multiple: float = 1.6,
    max_player_exposure: float = 0.55,
    max_pitcher_exposure: float = 0.65,
    max_stack_exposure: float = 0.40,
) -> CandidatePoolQualityReport:
    if lineup_df is None or lineup_df.empty or "lineup_id" not in lineup_df.columns:
        return CandidatePoolQualityReport({}, _checks([_row("Fail", "Pool exists", "No candidate lineups available.", "0", "Run Step 3 first.")]), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    df = lineup_df.copy()
    for col in ["fd_player_id", "full_name", "team_code", "player_type"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)
    for col in ["proj_fd_mean", "proj_fd_ownership", "player_leverage_score", "salary"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if df["proj_fd_ownership"].max() > 1.5:
        df["proj_fd_ownership"] = df["proj_fd_ownership"] / 100.0

    lineup_count = int(df["lineup_id"].nunique())
    unique_keys = _lineup_keys(df)
    unique_lineups = int(unique_keys.nunique())
    duplicate_lineups = int(lineup_count - unique_lineups)
    player_exposure = _player_exposure(df, lineup_count)
    stack_exposure = _stack_exposure(df, lineup_count)
    stack_template_mix = _stack_template_mix(df, lineup_count)
    max_player = float(player_exposure["exposure_pct"].max()) if not player_exposure.empty else 0.0
    max_pitcher = float(player_exposure.loc[player_exposure["is_pitcher"], "exposure_pct"].max()) if not player_exposure.empty and player_exposure["is_pitcher"].any() else 0.0
    max_stack = float(stack_exposure["exposure_pct"].max()) if not stack_exposure.empty else 0.0
    unique_players = int(df["fd_player_id"].nunique())
    avg_projection = float(df.groupby("lineup_id")["proj_fd_mean"].sum().mean())
    avg_leverage = float(df.groupby("lineup_id")["player_leverage_score"].mean().mean())

    checks = []
    required_pool = int(round(target_final_lineups * min_pool_multiple))
    if lineup_count >= required_pool:
        checks.append(_row("Good", "Pool depth", "Candidate pool is deep enough for final selection.", f"{lineup_count}/{required_pool}", ""))
    else:
        checks.append(_row("Warn", "Pool depth", "Candidate pool may be too shallow for a clean 300-lineup final portfolio.", f"{lineup_count}/{required_pool}", "Generate more candidates or loosen Step 3 constraints."))
    if duplicate_lineups:
        checks.append(_row("Fail", "Duplicate candidates", "Candidate pool contains duplicate player sets.", str(duplicate_lineups), "Increase randomness or tighten uniqueness."))
    else:
        checks.append(_row("Good", "Duplicate candidates", "Candidate player sets are unique.", str(unique_lineups), ""))
    if max_player > max_player_exposure:
        checks.append(_row("Warn", "Player concentration", "One hitter/batter exposure is too high in the candidate pool.", f"{max_player:.0%}", "Lower player caps or increase randomness before simulation."))
    else:
        checks.append(_row("Good", "Player concentration", "No non-pitcher dominates the candidate pool.", f"{max_player:.0%}", ""))
    if max_pitcher > max_pitcher_exposure:
        checks.append(_row("Warn", "Pitcher concentration", "One pitcher exposure is too high in the candidate pool.", f"{max_pitcher:.0%}", "Lower pitcher caps or widen pitcher pool."))
    else:
        checks.append(_row("Good", "Pitcher concentration", "Pitcher exposure has enough room for final selection.", f"{max_pitcher:.0%}", ""))
    if max_stack > max_stack_exposure:
        checks.append(_row("Warn", "Stack concentration", "One team stack is carrying too much of the candidate pool.", f"{max_stack:.0%}", "Lower team stack caps or broaden stack templates."))
    else:
        checks.append(_row("Good", "Stack concentration", "Team stack exposure is not overly concentrated.", f"{max_stack:.0%}", ""))

    summary = {
        "lineups": lineup_count,
        "required_pool": required_pool,
        "unique_lineups": unique_lineups,
        "duplicate_lineups": duplicate_lineups,
        "unique_players": unique_players,
        "max_player_exposure": max_player,
        "max_pitcher_exposure": max_pitcher,
        "max_stack_exposure": max_stack,
        "avg_projection": avg_projection,
        "avg_leverage": avg_leverage,
    }
    return CandidatePoolQualityReport(summary, _checks(checks), player_exposure, stack_exposure, stack_template_mix)


def _lineup_keys(df: pd.DataFrame) -> pd.Series:
    return df.groupby("lineup_id")["fd_player_id"].apply(lambda values: "|".join(sorted(values.astype(str))))


def _player_exposure(df: pd.DataFrame, lineup_count: int) -> pd.DataFrame:
    grouped = (
        df.groupby(["fd_player_id", "full_name", "team_code", "player_type"])
        .agg(
            lineups=("lineup_id", "nunique"),
            projection=("proj_fd_mean", "mean"),
            ownership=("proj_fd_ownership", "mean"),
            leverage=("player_leverage_score", "mean"),
        )
        .reset_index()
    )
    grouped["exposure_pct"] = grouped["lineups"] / max(1, lineup_count)
    grouped["is_pitcher"] = grouped["player_type"].str.lower() == "pitcher"
    return grouped.sort_values("exposure_pct", ascending=False)


def _stack_exposure(df: pd.DataFrame, lineup_count: int) -> pd.DataFrame:
    hitters = df[df["player_type"].str.lower() != "pitcher"]
    if hitters.empty:
        return pd.DataFrame()
    stacks = hitters.groupby(["lineup_id", "team_code"]).size().reset_index(name="hitters")
    stacks = stacks[stacks["hitters"] >= 3]
    if stacks.empty:
        return pd.DataFrame()
    grouped = stacks.groupby("team_code").agg(lineups=("lineup_id", "nunique"), avg_stack_size=("hitters", "mean")).reset_index()
    grouped["exposure_pct"] = grouped["lineups"] / max(1, lineup_count)
    return grouped.sort_values("exposure_pct", ascending=False)


def _stack_template_mix(df: pd.DataFrame, lineup_count: int) -> pd.DataFrame:
    hitters = df[df["player_type"].str.lower() != "pitcher"]
    if hitters.empty:
        return pd.DataFrame()
    team_counts = hitters.groupby(["lineup_id", "team_code"]).size().reset_index(name="hitters")
    rows = []
    for lineup_id, group in team_counts.groupby("lineup_id"):
        counts = sorted([int(v) for v in group["hitters"].tolist() if int(v) >= 2], reverse=True)
        rows.append({"lineup_id": lineup_id, "template": "-".join(str(v) for v in counts) if counts else "No stacks"})
    mix = pd.DataFrame(rows).groupby("template")["lineup_id"].nunique().reset_index(name="lineups")
    mix["exposure_pct"] = mix["lineups"] / max(1, lineup_count)
    return mix.sort_values("lineups", ascending=False)


def _row(status: str, check: str, detail: str, value: str, action: str) -> dict[str, str]:
    return {"status": status, "check": check, "detail": detail, "value": value, "action": action}


def _checks(rows: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["status", "check", "detail", "value", "action"])


__all__ = ["CandidatePoolQualityReport", "build_candidate_pool_quality_report"]
