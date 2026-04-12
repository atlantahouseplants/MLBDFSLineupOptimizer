"""Auto-fetch batting orders and probable pitchers from the free MLB Stats API.

No API key required. Uses: https://statsapi.mlb.com/api/v1/schedule
"""
from __future__ import annotations

import warnings
from datetime import date
from typing import Optional, Tuple

import pandas as pd

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _requests = None  # type: ignore[assignment]
    _REQUESTS_AVAILABLE = False

_MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
# 'team' hydrate gives us team.abbreviation; pitchHand requires a separate people call
_HYDRATE = "lineups,probablePitcher,team"

# MLB API sometimes uses different abbreviations than FanDuel
_MLB_ABBREV_OVERRIDES: dict[str, str] = {
    "KCR": "KC",
    "SDP": "SD",
    "SFG": "SF",
    "TBR": "TB",
    "CHW": "CWS",
}


def _normalize_team(abbrev: str) -> str:
    abbrev = str(abbrev).upper().strip()
    return _MLB_ABBREV_OVERRIDES.get(abbrev, abbrev)


def _empty_batting_orders() -> pd.DataFrame:
    return pd.DataFrame(columns=["team_code", "order_position", "player_name", "confirmed"])


def _empty_pitchers() -> pd.DataFrame:
    return pd.DataFrame(columns=["team_code", "player_name", "pitcher_hand"])


def _fetch_pitcher_hand(player_id: int) -> str:
    """Look up a pitcher's throwing hand from the MLB people API. Returns '' on error."""
    if not _REQUESTS_AVAILABLE:
        return ""
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
        resp = _requests.get(url, params={"hydrate": "currentTeam"}, timeout=8)
        resp.raise_for_status()
        people = resp.json().get("people", [])
        if people:
            hand = people[0].get("pitchHand", {})
            if isinstance(hand, dict):
                return str(hand.get("code", "")).upper()[:1]
    except Exception:
        pass
    return ""


def fetch_mlb_lineups(
    date_str: Optional[str] = None,
    confirmed_only: bool = True,
    fetch_pitcher_hands: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch batting orders and probable pitchers from the MLB Stats API.

    Args:
        date_str: Date in YYYY-MM-DD format. Defaults to today.
        confirmed_only: If True, only include teams with confirmed lineups.
        fetch_pitcher_hands: If True, make extra API calls to get pitcher handedness.
                             Set False to skip for speed (returns empty pitcher_hand).

    Returns:
        Tuple of (batting_orders_df, pitchers_df).
        - batting_orders_df columns: team_code, order_position, player_name, confirmed
        - pitchers_df columns: team_code, player_name, pitcher_hand
        Returns empty DataFrames on network error or no games found.
    """
    if not _REQUESTS_AVAILABLE:
        warnings.warn(
            "requests package not installed. Run: pip install requests.",
            stacklevel=2,
        )
        return _empty_batting_orders(), _empty_pitchers()

    if date_str is None:
        date_str = date.today().strftime("%Y-%m-%d")

    params = {"sportId": 1, "date": date_str, "hydrate": _HYDRATE}

    try:
        resp = _requests.get(_MLB_SCHEDULE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        warnings.warn(f"MLB Stats API request failed: {exc}", stacklevel=2)
        return _empty_batting_orders(), _empty_pitchers()

    batting_rows: list[dict] = []
    pitcher_rows: list[dict] = []

    if not data.get("dates"):
        return _empty_batting_orders(), _empty_pitchers()

    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            teams = game.get("teams", {})

            for side in ("home", "away"):
                team_info = teams.get(side, {})
                abbrev = team_info.get("team", {}).get("abbreviation", "")
                team_code = _normalize_team(abbrev)

                # Probable pitchers
                probable = team_info.get("probablePitcher")
                if probable and team_code:
                    full_name = probable.get("fullName", "")
                    player_id = probable.get("id")
                    # pitchHand may be embedded (some hydrations include it)
                    pitch_hand_raw = probable.get("pitchHand", "")
                    if isinstance(pitch_hand_raw, dict):
                        pitcher_hand = str(pitch_hand_raw.get("code", "")).upper()[:1]
                    else:
                        pitcher_hand = str(pitch_hand_raw).upper()[:1]
                    if pitcher_hand not in ("L", "R"):
                        pitcher_hand = ""
                    # Fall back to separate people API call if needed
                    if not pitcher_hand and fetch_pitcher_hands and player_id:
                        pitcher_hand = _fetch_pitcher_hand(player_id)
                    if full_name:
                        pitcher_rows.append({
                            "team_code": team_code,
                            "player_name": full_name,
                            "pitcher_hand": pitcher_hand,
                            "player_id": player_id,
                        })

            # Confirmed batting lineups
            lineups = game.get("lineups", {})
            if not lineups:
                continue
            for side, side_key in [("homePlayers", "home"), ("awayPlayers", "away")]:
                players_list = lineups.get(side, [])
                if not players_list:
                    continue
                abbrev = teams.get(side_key, {}).get("team", {}).get("abbreviation", "")
                team_code = _normalize_team(abbrev)
                for order_pos, player in enumerate(players_list, start=1):
                    full_name = player.get("fullName", "")
                    if full_name and team_code:
                        batting_rows.append({
                            "team_code": team_code,
                            "order_position": order_pos,
                            "player_name": full_name,
                            "confirmed": True,
                        })

    batting_df = pd.DataFrame(batting_rows) if batting_rows else _empty_batting_orders()
    pitchers_df = pd.DataFrame(pitcher_rows) if pitcher_rows else _empty_pitchers()

    if not batting_df.empty:
        batting_df["order_position"] = pd.to_numeric(
            batting_df["order_position"], errors="coerce"
        ).astype("Int64")
        batting_df["confirmed"] = batting_df["confirmed"].astype(bool)
        batting_df = batting_df.drop_duplicates(
            subset=["team_code", "player_name"], keep="first"
        )
        if confirmed_only:
            batting_df = batting_df[batting_df["confirmed"]]

    if not pitchers_df.empty:
        if "player_id" in pitchers_df.columns:
            pitchers_df = pitchers_df.drop(columns=["player_id"])
        pitchers_df = pitchers_df.drop_duplicates(subset=["team_code"], keep="first")

    return batting_df.reset_index(drop=True), pitchers_df.reset_index(drop=True)


__all__ = ["fetch_mlb_lineups"]
