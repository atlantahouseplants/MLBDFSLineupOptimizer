"""Game environment scoring for GPP leverage analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

__all__ = [
    "GameEnvironment",
    "compute_game_environments",
    "merge_game_environment_columns",
]


@dataclass
class GameEnvironment:
    game_key: str
    home_team: str
    away_team: str
    vegas_game_total: float
    home_implied_total: float
    away_implied_total: float
    home_team_ownership: float
    away_team_ownership: float
    game_leverage_score: float
    environment_tier: str  # "prime" / "good" / "neutral" / "avoid"


def compute_game_environments(optimizer_df: pd.DataFrame) -> List[GameEnvironment]:
    """Score every game on the slate for GPP attractiveness.

    Returns one ``GameEnvironment`` per distinct game.
    """
    df = optimizer_df.copy()
    df["vegas_game_total"] = pd.to_numeric(df.get("vegas_game_total"), errors="coerce").fillna(0.0)
    df["vegas_team_total"] = pd.to_numeric(df.get("vegas_team_total"), errors="coerce").fillna(0.0)
    df["proj_fd_ownership"] = pd.to_numeric(df.get("proj_fd_ownership"), errors="coerce").fillna(0.0)

    is_batter = df["player_type"].astype(str).str.lower() == "batter"

    # Build per-team aggregates (batters only)
    batter_df = df[is_batter].copy()
    team_ownership = batter_df.groupby("team_code")["proj_fd_ownership"].sum()
    team_implied = batter_df.groupby("team_code")["vegas_team_total"].first()

    # Build per-game rows
    game_rows: Dict[str, dict] = {}
    for _, row in df.drop_duplicates("game_key").iterrows():
        gk = str(row.get("game_key", ""))
        if not gk:
            continue
        tc = str(row.get("team_code", ""))
        oc = str(row.get("opponent_code", ""))
        # Determine home/away — use alphabetical sort of game_key teams
        teams_sorted = sorted([tc, oc])
        home, away = teams_sorted[0], teams_sorted[1]
        game_rows[gk] = {
            "game_key": gk,
            "home_team": home,
            "away_team": away,
            "vegas_game_total": float(row.get("vegas_game_total", 0)),
            "home_implied_total": float(team_implied.get(home, 0)),
            "away_implied_total": float(team_implied.get(away, 0)),
            "home_team_ownership": float(team_ownership.get(home, 0)),
            "away_team_ownership": float(team_ownership.get(away, 0)),
        }

    if not game_rows:
        return []

    games_df = pd.DataFrame(game_rows.values())

    # Percentile ranks for leverage score
    games_df["vgt_rank"] = games_df["vegas_game_total"].rank(pct=True)
    # Team ownership rank — average of both sides
    games_df["avg_team_own"] = (games_df["home_team_ownership"] + games_df["away_team_ownership"]) / 2.0
    games_df["own_rank"] = games_df["avg_team_own"].rank(pct=True)

    # game_leverage_score = high total rank * (1 - ownership rank)
    games_df["game_leverage_score"] = games_df["vgt_rank"] * (1.0 - games_df["own_rank"])

    # Assign tiers
    leverage_75 = games_df["game_leverage_score"].quantile(0.75)
    leverage_50 = games_df["game_leverage_score"].quantile(0.50)

    def _tier(row: pd.Series) -> str:
        vgt = row["vegas_game_total"]
        score = row["game_leverage_score"]
        if vgt < 6.5:
            return "avoid"
        if score >= leverage_75 and vgt >= 8.5:
            return "prime"
        if score >= leverage_50 and vgt >= 7.5:
            return "good"
        if vgt >= 6.5:
            return "neutral"
        return "avoid"

    games_df["environment_tier"] = games_df.apply(_tier, axis=1)

    results: List[GameEnvironment] = []
    for _, row in games_df.iterrows():
        results.append(GameEnvironment(
            game_key=row["game_key"],
            home_team=row["home_team"],
            away_team=row["away_team"],
            vegas_game_total=row["vegas_game_total"],
            home_implied_total=row["home_implied_total"],
            away_implied_total=row["away_implied_total"],
            home_team_ownership=row["home_team_ownership"],
            away_team_ownership=row["away_team_ownership"],
            game_leverage_score=row["game_leverage_score"],
            environment_tier=row["environment_tier"],
        ))
    return results


def _team_gpp_leverage(optimizer_df: pd.DataFrame) -> pd.Series:
    """Compute per-team GPP leverage: implied_total / aggregate_ownership."""
    df = optimizer_df.copy()
    is_batter = df["player_type"].astype(str).str.lower() == "batter"
    batter_df = df[is_batter]

    team_own = batter_df.groupby("team_code")["proj_fd_ownership"].sum()
    team_implied = pd.to_numeric(
        batter_df.groupby("team_code")["vegas_team_total"].first(), errors="coerce"
    ).fillna(0.0)

    leverage = team_implied / team_own.clip(lower=0.01)
    return leverage.rename("team_gpp_leverage")


def merge_game_environment_columns(optimizer_df: pd.DataFrame) -> pd.DataFrame:
    """Add game_leverage_score, environment_tier, and team_gpp_leverage columns."""
    df = optimizer_df.copy()
    envs = compute_game_environments(df)

    # Build lookup dicts keyed by game_key
    env_scores: Dict[str, float] = {}
    env_tiers: Dict[str, str] = {}
    for env in envs:
        env_scores[env.game_key] = env.game_leverage_score
        env_tiers[env.game_key] = env.environment_tier

    df["game_leverage_score"] = df["game_key"].map(env_scores).fillna(0.0)
    df["environment_tier"] = df["game_key"].map(env_tiers).fillna("neutral")

    # Per-team leverage
    team_leverage = _team_gpp_leverage(df)
    df["team_gpp_leverage"] = df["team_code"].map(team_leverage).fillna(0.0)

    return df
