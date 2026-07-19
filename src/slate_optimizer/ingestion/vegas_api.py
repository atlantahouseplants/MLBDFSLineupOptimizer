"""Fetch MLB Vegas odds from The Odds API and format for the optimizer pipeline.

Requires an API key from https://the-odds-api.com (free tier: 500 requests/month).
Set the environment variable ODDS_API_KEY or pass it directly.

Usage:
    from slate_optimizer.ingestion.vegas_api import fetch_vegas_lines
    vegas = fetch_vegas_lines()  # returns VegasLines object
"""
from __future__ import annotations

import os
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd

from .vegas import VegasLines, _moneyline_to_prob

_BASE_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"

# Standard 3-letter codes used by FanDuel
_TEAM_MAP = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


def _team_code(full_name: str) -> str:
    """Convert full team name to 2-3 letter code."""
    return _TEAM_MAP.get(full_name, full_name.upper()[:3])


def fetch_vegas_lines(
    api_key: Optional[str] = None,
    save_csv: Optional[Path] = None,
) -> VegasLines:
    """Fetch today's MLB odds from The Odds API and return a VegasLines object.

    Parameters
    ----------
    api_key : str, optional
        The Odds API key. Falls back to ODDS_API_KEY environment variable.
    save_csv : Path, optional
        If provided, also saves the raw data as a CSV for debugging/caching.

    Returns
    -------
    VegasLines
        Same object type as VegasLoader.load(), drop-in compatible.
    """
    import urllib.request
    import json

    key = api_key or os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise ValueError(
            "No API key provided. Set the ODDS_API_KEY environment variable "
            "or pass api_key= to fetch_vegas_lines(). "
            "Get a free key at https://the-odds-api.com"
        )

    url = (
        f"{_BASE_URL}"
        f"?apiKey={key}"
        f"&regions=us"
        f"&markets=totals,h2h"
        f"&oddsFormat=american"
        f"&dateFormat=iso"
    )

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())

    if not data:
        raise ValueError("The Odds API returned no games. The slate may not be posted yet.")

    rows = []
    for event in data:
        away_team = _team_code(event.get("away_team", ""))
        home_team = _team_code(event.get("home_team", ""))
        if not away_team or not home_team:
            continue

        game_total = None
        home_ml = None
        away_ml = None

        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] == "totals" and game_total is None:
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == "Over":
                            game_total = outcome.get("point")
                            break
                elif market["key"] == "h2h":
                    for outcome in market.get("outcomes", []):
                        team_code = _team_code(outcome["name"])
                        if team_code == home_team and home_ml is None:
                            home_ml = outcome.get("price")
                        elif team_code == away_team and away_ml is None:
                            away_ml = outcome.get("price")
            # Use the first bookmaker that has both markets
            if game_total is not None and home_ml is not None and away_ml is not None:
                break

        if game_total is None:
            continue  # Skip games without totals

        rows.append({
            "game": f"{away_team}@{home_team}",
            "total": game_total,
            "home_ml": home_ml or 0,
            "away_ml": away_ml or 0,
        })

    if not rows:
        raise ValueError("No valid game lines found in API response.")

    df = pd.DataFrame(rows)

    if save_csv:
        df.to_csv(save_csv, index=False)

    # Build VegasLines using the same logic as VegasLoader
    df["total"] = pd.to_numeric(df["total"], errors="coerce").clip(lower=0)
    df["home_ml"] = pd.to_numeric(df["home_ml"], errors="coerce")
    df["away_ml"] = pd.to_numeric(df["away_ml"], errors="coerce")

    away_teams = []
    home_teams = []
    for game_str in df["game"]:
        parts = game_str.split("@")
        away_teams.append(parts[0].strip())
        home_teams.append(parts[1].strip() if len(parts) > 1 else "")
    df["away_team"] = away_teams
    df["home_team"] = home_teams

    df["away_prob_raw"] = df["away_ml"].map(_moneyline_to_prob)
    df["home_prob_raw"] = df["home_ml"].map(_moneyline_to_prob)

    prob_sum = df[["away_prob_raw", "home_prob_raw"]].fillna(0).sum(axis=1)
    valid = prob_sum > 0
    df["away_prob"] = 0.5
    df["home_prob"] = 0.5
    df.loc[valid, "away_prob"] = df.loc[valid, "away_prob_raw"].fillna(0) / prob_sum[valid]
    df.loc[valid, "home_prob"] = df.loc[valid, "home_prob_raw"].fillna(0) / prob_sum[valid]

    df["away_implied_total"] = (df["total"] * df["away_prob"]).fillna(0.0)
    df["home_implied_total"] = (df["total"] * df["home_prob"]).fillna(0.0)
    df["game_code"] = df["away_team"] + "@" + df["home_team"]

    team_rows = []
    for _, row in df.iterrows():
        away = row["away_team"]
        home = row["home_team"]
        if not away or not home:
            continue
        for team, opp, total, opp_total, ml, prob in [
            (away, home, row["away_implied_total"], row["home_implied_total"], row["away_ml"], row["away_prob"]),
            (home, away, row["home_implied_total"], row["away_implied_total"], row["home_ml"], row["home_prob"]),
        ]:
            team_rows.append({
                "team_code": team,
                "opponent_code": opp,
                "vegas_game_code": row["game_code"],
                "vegas_game_total": row["total"],
                "vegas_team_total": total,
                "vegas_opponent_total": opp_total,
                "vegas_moneyline": ml,
                "vegas_implied_win_prob": prob,
            })

    return VegasLines(games=df, team_totals=pd.DataFrame(team_rows))


__all__ = ["fetch_vegas_lines"]
