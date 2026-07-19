"""Fetch today's MLB starting lineups from the MLB Stats API.

The MLB Stats API is public and requires no API key.

Usage:
    from slate_optimizer.ingestion.mlb_lineups_api import fetch_batting_orders
    orders_df = fetch_batting_orders()  # returns DataFrame with team, order_position, player_name
"""
from __future__ import annotations

import json
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

_MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
_MLB_GAME_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"

# MLB API full team names → standard FanDuel codes
_MLB_NAME_MAP: Dict[str, str] = {
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
    "Athletics": "OAK",
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

# Also handle abbreviations that may come from other API endpoints
_MLB_ABBREV_MAP: Dict[str, str] = {
    "AZ": "ARI", "ARI": "ARI",
    "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CWS": "CWS", "CHA": "CWS",
    "CIN": "CIN", "CLE": "CLE", "COL": "COL",
    "DET": "DET", "HOU": "HOU",
    "KC": "KC", "KCA": "KC",
    "LAA": "LAA", "ANA": "LAA",
    "LAD": "LAD", "LA": "LAD",
    "MIA": "MIA", "MIL": "MIL", "MIN": "MIN",
    "NYM": "NYM", "NYY": "NYY", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT",
    "SD": "SD", "SDP": "SD",
    "SF": "SF", "SFG": "SF",
    "SEA": "SEA", "STL": "STL",
    "TB": "TB", "TBR": "TB",
    "TEX": "TEX", "TOR": "TOR",
    "WSH": "WSH", "WAS": "WSH",
}


def _normalize_team(name_or_abbrev: str) -> str:
    """Resolve a full team name or abbreviation to standard FanDuel code."""
    # Try full name first
    code = _MLB_NAME_MAP.get(name_or_abbrev)
    if code:
        return code
    # Then abbreviation
    return _MLB_ABBREV_MAP.get(name_or_abbrev.upper(), name_or_abbrev.upper())


def _api_get(url: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_batting_orders(
    game_date: Optional[date] = None,
    save_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch today's starting lineups from the MLB Stats API.

    Parameters
    ----------
    game_date : date, optional
        Which day to fetch. Defaults to today (Eastern time).
    save_csv : Path, optional
        If provided, saves the result as a CSV for debugging/caching.

    Returns
    -------
    pd.DataFrame
        Columns: team, order_position, player_name
        - order_position 0 = starting pitcher
        - order_position 1-9 = batting order
        Compatible with BattingOrderLoader.load() output format.
    """
    if game_date is None:
        game_date = datetime.now(ZoneInfo("US/Eastern")).date()

    date_str = game_date.strftime("%Y-%m-%d")
    schedule_url = f"{_MLB_SCHEDULE_URL}?sportId=1&date={date_str}&hydrate=probablePitcher,lineups"
    schedule_data = _api_get(schedule_url)

    rows: List[Dict] = []
    games_with_lineups = 0
    games_without_lineups = 0

    for game_date_entry in schedule_data.get("dates", []):
        for game in game_date_entry.get("games", []):
            game_pk = game.get("gamePk")
            status = game.get("status", {}).get("abstractGameState", "")

            teams_data = game.get("teams", {})
            away_info = teams_data.get("away", {})
            home_info = teams_data.get("home", {})

            away_team_name = away_info.get("team", {}).get("name", "")
            home_team_name = home_info.get("team", {}).get("name", "")
            away_code = _normalize_team(away_team_name)
            home_code = _normalize_team(home_team_name)

            # Try to get probable pitchers from schedule data
            away_pitcher = away_info.get("probablePitcher", {})
            home_pitcher = home_info.get("probablePitcher", {})

            if away_pitcher.get("fullName"):
                rows.append({
                    "team": away_code,
                    "order_position": 0,
                    "player_name": away_pitcher["fullName"],
                })
            if home_pitcher.get("fullName"):
                rows.append({
                    "team": home_code,
                    "order_position": 0,
                    "player_name": home_pitcher["fullName"],
                })

            # Try to get lineups from the schedule hydration first
            lineups = game.get("lineups", {})
            away_lineup = lineups.get("awayPlayers", [])
            home_lineup = lineups.get("homePlayers", [])

            if away_lineup or home_lineup:
                games_with_lineups += 1
                for order_pos, player in enumerate(away_lineup, start=1):
                    name = player.get("fullName", "")
                    if name:
                        rows.append({
                            "team": away_code,
                            "order_position": order_pos,
                            "player_name": name,
                        })
                for order_pos, player in enumerate(home_lineup, start=1):
                    name = player.get("fullName", "")
                    if name:
                        rows.append({
                            "team": home_code,
                            "order_position": order_pos,
                            "player_name": name,
                        })
            else:
                games_without_lineups += 1
                # If lineups aren't in schedule, try the live feed
                if game_pk and status in ("Live", "Preview"):
                    try:
                        _fetch_lineups_from_live_feed(game_pk, away_code, home_code, rows)
                        games_with_lineups += 1
                        games_without_lineups -= 1
                    except Exception:
                        pass  # lineups just not posted yet

    if not rows:
        raise ValueError(
            f"No lineup data found for {date_str}. "
            f"Lineups may not be posted yet (they typically come out 2-4 hours before first pitch)."
        )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["team", "order_position", "player_name"], keep="first")

    if save_csv:
        df.to_csv(save_csv, index=False)

    return df


def _fetch_lineups_from_live_feed(
    game_pk: int,
    away_code: str,
    home_code: str,
    rows: List[Dict],
) -> None:
    """Fallback: pull lineups from the game's live feed."""
    url = _MLB_GAME_URL.format(game_pk=game_pk)
    data = _api_get(url)

    boxscore = data.get("liveData", {}).get("boxscore", {})

    for side, team_code in [("away", away_code), ("home", home_code)]:
        team_box = boxscore.get("teams", {}).get(side, {})
        batting_order = team_box.get("battingOrder", [])
        players = team_box.get("players", {})

        for order_pos, player_id in enumerate(batting_order, start=1):
            player_key = f"ID{player_id}"
            player_data = players.get(player_key, {})
            name = player_data.get("person", {}).get("fullName", "")
            if name:
                rows.append({
                    "team": team_code,
                    "order_position": order_pos,
                    "player_name": name,
                })


def lineup_fetch_status(game_date: Optional[date] = None) -> Dict[str, int]:
    """Quick check: how many games have lineups posted today."""
    if game_date is None:
        game_date = datetime.now(ZoneInfo("US/Eastern")).date()

    date_str = game_date.strftime("%Y-%m-%d")
    url = f"{_MLB_SCHEDULE_URL}?sportId=1&date={date_str}&hydrate=probablePitcher,lineups"
    data = _api_get(url)

    total = 0
    with_lineups = 0
    with_pitchers = 0

    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            total += 1
            teams = game.get("teams", {})
            if teams.get("away", {}).get("probablePitcher") or teams.get("home", {}).get("probablePitcher"):
                with_pitchers += 1
            lineups = game.get("lineups", {})
            if lineups.get("awayPlayers") or lineups.get("homePlayers"):
                with_lineups += 1

    return {
        "total_games": total,
        "games_with_lineups": with_lineups,
        "games_with_pitchers": with_pitchers,
    }


__all__ = ["fetch_batting_orders", "lineup_fetch_status"]
