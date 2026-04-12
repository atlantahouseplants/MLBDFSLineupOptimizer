"""Vegas lines auto-fetch via The Odds API (free tier, key-gated).

Sign up free at https://the-odds-api.com — 500 requests/month free tier.
Set ODDS_API_KEY in your environment or .env file.

Returns the same VegasLines object as VegasLoader so it's a drop-in replacement.
"""
from __future__ import annotations

import os
import warnings
from typing import Optional

import pandas as pd

from .vegas import VegasLines, _moneyline_to_prob

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

_ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
_PREFERRED_BOOKS = ("fanduel", "draftkings", "betmgm", "caesars")


def _find_bookmaker(bookmakers: list[dict]) -> Optional[dict]:
    """Pick preferred bookmaker from the list, fall back to first available."""
    book_map = {b.get("key", "").lower(): b for b in bookmakers}
    for name in _PREFERRED_BOOKS:
        if name in book_map:
            return book_map[name]
    return bookmakers[0] if bookmakers else None


def _extract_market(bookmaker: dict, market_key: str) -> Optional[dict]:
    for market in bookmaker.get("markets", []):
        if market.get("key") == market_key:
            return market
    return None


def _get_total(market: dict) -> Optional[float]:
    """Extract Over/Under total from a totals market."""
    for outcome in market.get("outcomes", []):
        if outcome.get("name", "").lower() == "over":
            return float(outcome.get("point", 0)) or None
    return None


def _get_moneyline(market: dict, team_name: str) -> Optional[float]:
    """Extract moneyline for a specific team from h2h market."""
    for outcome in market.get("outcomes", []):
        if outcome.get("name", "") == team_name:
            try:
                return float(outcome.get("price", 0))
            except (TypeError, ValueError):
                pass
    return None


def fetch_vegas_lines(
    api_key: Optional[str] = None,
    date_str: Optional[str] = None,
) -> Optional[VegasLines]:
    """Fetch MLB Vegas lines from The Odds API.

    Args:
        api_key: Odds API key. Falls back to ODDS_API_KEY env var.
        date_str: Not used by the API (returns upcoming games). Included for
                  interface consistency with manual loaders.

    Returns:
        VegasLines object (same as VegasLoader.load()) or None if no key/error.
    """
    if not _REQUESTS_AVAILABLE:
        warnings.warn("requests package not installed.", stacklevel=2)
        return None

    key = api_key or os.getenv("ODDS_API_KEY")
    if not key:
        return None  # Caller should fall back to manual CSV

    params = {
        "apiKey": key,
        "regions": "us",
        "markets": "totals,h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    try:
        resp = requests.get(_ODDS_API_URL, params=params, timeout=15)
        remaining = resp.headers.get("x-requests-remaining", "?")
        used = resp.headers.get("x-requests-used", "?")
        print(f"  [Odds API] credits used: {used}, remaining: {remaining}")
        resp.raise_for_status()
        games_data = resp.json()
    except Exception as exc:
        warnings.warn(f"Odds API request failed: {exc}", stacklevel=2)
        return None

    if not games_data:
        return VegasLines(games=pd.DataFrame(), team_totals=pd.DataFrame())

    games_rows: list[dict] = []
    team_rows: list[dict] = []

    for game in games_data:
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        bookmakers = game.get("bookmakers", [])

        if not home_team or not away_team or not bookmakers:
            continue

        book = _find_bookmaker(bookmakers)
        if not book:
            continue

        totals_market = _extract_market(book, "totals")
        h2h_market = _extract_market(book, "h2h")

        game_total = _get_total(totals_market) if totals_market else None
        home_ml = _get_moneyline(h2h_market, home_team) if h2h_market else None
        away_ml = _get_moneyline(h2h_market, away_team) if h2h_market else None

        if game_total is None:
            continue

        # Use team name as code for now — caller can alias as needed
        # Try to extract abbreviation from team name (last word, uppercase)
        def _abbrev(name: str) -> str:
            # Common MLB full name → abbrev mapping
            _MAP = {
                "New York Yankees": "NYY", "New York Mets": "NYM",
                "Boston Red Sox": "BOS", "Chicago Cubs": "CHC",
                "Chicago White Sox": "CWS", "Los Angeles Dodgers": "LAD",
                "Los Angeles Angels": "LAA", "Houston Astros": "HOU",
                "Atlanta Braves": "ATL", "Philadelphia Phillies": "PHI",
                "San Francisco Giants": "SF", "San Diego Padres": "SD",
                "Seattle Mariners": "SEA", "Texas Rangers": "TEX",
                "Toronto Blue Jays": "TOR", "Tampa Bay Rays": "TB",
                "Minnesota Twins": "MIN", "Detroit Tigers": "DET",
                "Cleveland Guardians": "CLE", "Kansas City Royals": "KC",
                "St. Louis Cardinals": "STL", "Milwaukee Brewers": "MIL",
                "Cincinnati Reds": "CIN", "Pittsburgh Pirates": "PIT",
                "Colorado Rockies": "COL", "Arizona Diamondbacks": "ARI",
                "Miami Marlins": "MIA", "Baltimore Orioles": "BAL",
                "Washington Nationals": "WSH", "Oakland Athletics": "OAK",
            }
            return _MAP.get(name, name.upper().replace(" ", "_")[:3])

        home_code = _abbrev(home_team)
        away_code = _abbrev(away_team)
        game_code = f"{away_code}@{home_code}"

        # Implied team totals from moneylines
        away_prob_raw = _moneyline_to_prob(away_ml) if away_ml is not None else float("nan")
        home_prob_raw = _moneyline_to_prob(home_ml) if home_ml is not None else float("nan")
        prob_sum = (away_prob_raw if not pd.isna(away_prob_raw) else 0) + \
                   (home_prob_raw if not pd.isna(home_prob_raw) else 0)

        if prob_sum > 0:
            away_prob = away_prob_raw / prob_sum if not pd.isna(away_prob_raw) else 0.5
            home_prob = home_prob_raw / prob_sum if not pd.isna(home_prob_raw) else 0.5
        else:
            away_prob = home_prob = 0.5

        away_total = game_total * away_prob
        home_total = game_total * home_prob

        games_rows.append({
            "game": f"{away_code}@{home_code}",
            "total": game_total,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "away_team": away_code,
            "home_team": home_code,
            "game_code": game_code,
            "away_implied_total": away_total,
            "home_implied_total": home_total,
            "away_prob": away_prob,
            "home_prob": home_prob,
        })

        for code, opp, total, opp_total, ml, prob in [
            (away_code, home_code, away_total, home_total, away_ml, away_prob),
            (home_code, away_code, home_total, away_total, home_ml, home_prob),
        ]:
            team_rows.append({
                "team_code": code,
                "opponent_code": opp,
                "vegas_game_code": game_code,
                "vegas_game_total": game_total,
                "vegas_team_total": total,
                "vegas_opponent_total": opp_total,
                "vegas_moneyline": ml,
                "vegas_implied_win_prob": prob,
            })

    games_df = pd.DataFrame(games_rows)
    team_totals_df = pd.DataFrame(team_rows)
    return VegasLines(games=games_df, team_totals=team_totals_df)


__all__ = ["fetch_vegas_lines"]
