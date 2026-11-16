"""Actual FanDuel outcomes from MLB box scores.

The optimizer spent the whole 2026 season without recording a single actual
score (``actual_scores`` in slates.db is empty), which is why none of the
projection multipliers, simulation distributions or correlations were ever
checked against reality.  This module closes that loop with data that needs
no manual download: the free MLB Stats API box score for every final game.

BallparkPal rows carry MLB ``GamePk`` and ``PlayerId``, so outcomes join to
projections on exact ids - no name matching involved.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import pandas as pd

MLB_STATS_API = "https://statsapi.mlb.com/api/v1"
DEFAULT_ACTUALS_DIR = Path(__file__).resolve().parents[3] / "data" / "actuals"

# FanDuel MLB classic scoring.
FD_HITTER_SCORING: Dict[str, float] = {
    "singles": 3.0,
    "doubles": 6.0,
    "triples": 9.0,
    "home_runs": 12.0,
    "rbi": 3.5,
    "runs": 3.2,
    "walks": 3.0,
    "stolen_bases": 6.0,
    "hit_by_pitch": 3.0,
}
FD_PITCHER_SCORING: Dict[str, float] = {
    "win": 6.0,
    "quality_start": 4.0,
    "earned_runs": -3.0,
    "strikeouts": 3.0,
    "outs": 1.0,  # 3 points per inning pitched
}

ACTUALS_COLUMNS = [
    "date",
    "game_pk",
    "mlb_id",
    "full_name",
    "team",
    "opponent",
    "role",
    "started",
    "batting_order",
    "plate_appearances",
    "fd_points",
    "singles",
    "doubles",
    "triples",
    "home_runs",
    "rbi",
    "runs",
    "walks",
    "stolen_bases",
    "hit_by_pitch",
    "outs",
    "earned_runs",
    "strikeouts",
    "win",
    "quality_start",
    "team_runs",
]

JsonFetcher = Callable[[str], Mapping[str, Any]]


def _http_get_json(url: str, retries: int = 3, timeout: float = 30.0) -> Mapping[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "mlb-dfs-backtest"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"MLB Stats API request failed after {retries} attempts: {url}") from last_error


def _int(stats: Mapping[str, Any], key: str) -> int:
    try:
        return int(stats.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def fanduel_hitter_points(batting: Mapping[str, Any]) -> float:
    """FanDuel points from an MLB Stats API batting line."""
    hits = _int(batting, "hits")
    doubles = _int(batting, "doubles")
    triples = _int(batting, "triples")
    home_runs = _int(batting, "homeRuns")
    singles = max(0, hits - doubles - triples - home_runs)
    return (
        FD_HITTER_SCORING["singles"] * singles
        + FD_HITTER_SCORING["doubles"] * doubles
        + FD_HITTER_SCORING["triples"] * triples
        + FD_HITTER_SCORING["home_runs"] * home_runs
        + FD_HITTER_SCORING["rbi"] * _int(batting, "rbi")
        + FD_HITTER_SCORING["runs"] * _int(batting, "runs")
        + FD_HITTER_SCORING["walks"] * _int(batting, "baseOnBalls")
        + FD_HITTER_SCORING["stolen_bases"] * _int(batting, "stolenBases")
        + FD_HITTER_SCORING["hit_by_pitch"] * _int(batting, "hitByPitch")
    )


def is_quality_start(pitching: Mapping[str, Any]) -> bool:
    return (
        _int(pitching, "gamesStarted") >= 1
        and _int(pitching, "outs") >= 18
        and _int(pitching, "earnedRuns") <= 3
    )


def fanduel_pitcher_points(pitching: Mapping[str, Any]) -> float:
    """FanDuel points from an MLB Stats API pitching line."""
    return (
        FD_PITCHER_SCORING["win"] * _int(pitching, "wins")
        + FD_PITCHER_SCORING["quality_start"] * (1 if is_quality_start(pitching) else 0)
        + FD_PITCHER_SCORING["earned_runs"] * _int(pitching, "earnedRuns")
        + FD_PITCHER_SCORING["strikeouts"] * _int(pitching, "strikeOuts")
        + FD_PITCHER_SCORING["outs"] * _int(pitching, "outs")
    )


def parse_boxscore(boxscore: Mapping[str, Any], game_pk: int, date: str) -> pd.DataFrame:
    """Flatten one box score into per-player outcome rows.

    A two-way player (bats and pitches in the same game) yields two rows, one
    per role, mirroring how FanDuel lists such players.
    """
    rows: List[Dict[str, Any]] = []
    teams = boxscore.get("teams") or {}
    abbreviations = {
        side: str(((teams.get(side) or {}).get("team") or {}).get("abbreviation") or "")
        for side in ("away", "home")
    }
    team_runs = {
        side: _int((((teams.get(side) or {}).get("teamStats") or {}).get("batting") or {}), "runs")
        for side in ("away", "home")
    }
    for side, other in (("away", "home"), ("home", "away")):
        players = ((teams.get(side) or {}).get("players") or {})
        for payload in players.values():
            person = payload.get("person") or {}
            stats = payload.get("stats") or {}
            base = {
                "date": date,
                "game_pk": int(game_pk),
                "mlb_id": int(person.get("id") or 0),
                "full_name": str(person.get("fullName") or ""),
                "team": abbreviations[side],
                "opponent": abbreviations[other],
                "team_runs": team_runs[side],
            }
            batting = stats.get("batting") or {}
            order_raw = str(payload.get("battingOrder") or "")
            if batting and order_raw:
                # "300" = starter in slot 3; "301" = first substitute into slot 3.
                slot = int(order_raw[0]) if order_raw[:1].isdigit() else 0
                started = order_raw.endswith("00")
                hits = _int(batting, "hits")
                doubles = _int(batting, "doubles")
                triples = _int(batting, "triples")
                home_runs = _int(batting, "homeRuns")
                rows.append(
                    {
                        **base,
                        "role": "batter",
                        "started": bool(started),
                        "batting_order": slot,
                        "plate_appearances": _int(batting, "plateAppearances"),
                        "fd_points": round(fanduel_hitter_points(batting), 2),
                        "singles": max(0, hits - doubles - triples - home_runs),
                        "doubles": doubles,
                        "triples": triples,
                        "home_runs": home_runs,
                        "rbi": _int(batting, "rbi"),
                        "runs": _int(batting, "runs"),
                        "walks": _int(batting, "baseOnBalls"),
                        "stolen_bases": _int(batting, "stolenBases"),
                        "hit_by_pitch": _int(batting, "hitByPitch"),
                    }
                )
            pitching = stats.get("pitching") or {}
            if pitching and (_int(pitching, "outs") > 0 or _int(pitching, "battersFaced") > 0 or _int(pitching, "gamesStarted") > 0):
                rows.append(
                    {
                        **base,
                        "role": "pitcher",
                        "started": _int(pitching, "gamesStarted") >= 1,
                        "batting_order": 0,
                        "plate_appearances": 0,
                        "fd_points": round(fanduel_pitcher_points(pitching), 2),
                        "outs": _int(pitching, "outs"),
                        "earned_runs": _int(pitching, "earnedRuns"),
                        "strikeouts": _int(pitching, "strikeOuts"),
                        "win": _int(pitching, "wins"),
                        "quality_start": int(is_quality_start(pitching)),
                    }
                )
    frame = pd.DataFrame(rows)
    for column in ACTUALS_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0
    return frame[ACTUALS_COLUMNS].fillna(0)


def final_game_pks(schedule: Mapping[str, Any]) -> List[int]:
    """Game ids for completed games in a schedule payload (skips postponed)."""
    pks: List[int] = []
    for day in schedule.get("dates") or []:
        for game in day.get("games") or []:
            status = game.get("status") or {}
            if str(status.get("abstractGameState") or "") != "Final":
                continue
            detailed = str(status.get("detailedState") or "")
            if detailed and not (detailed.startswith("Final") or detailed.startswith("Completed")):
                continue  # Postponed / Cancelled / Suspended
            pks.append(int(game["gamePk"]))
    return sorted(set(pks))


def fetch_actuals(date: str, fetch_json: Optional[JsonFetcher] = None) -> pd.DataFrame:
    """Fetch FanDuel outcomes for every final MLB game on ``date`` (YYYY-MM-DD)."""
    fetch_json = fetch_json or _http_get_json
    schedule = fetch_json(f"{MLB_STATS_API}/schedule?sportId=1&date={date}")
    frames = []
    for game_pk in final_game_pks(schedule):
        boxscore = fetch_json(f"{MLB_STATS_API}/game/{game_pk}/boxscore")
        frames.append(parse_boxscore(boxscore, game_pk, date))
    if not frames:
        return pd.DataFrame(columns=ACTUALS_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def actuals_path(date: str, actuals_dir: Path | str | None = None) -> Path:
    return Path(actuals_dir or DEFAULT_ACTUALS_DIR) / f"actuals_{date}.csv"


def load_or_fetch_actuals(
    date: str,
    actuals_dir: Path | str | None = None,
    refresh: bool = False,
    fetch_json: Optional[JsonFetcher] = None,
) -> pd.DataFrame:
    """Read the cached outcomes for ``date``, fetching and caching if missing."""
    path = actuals_path(date, actuals_dir)
    if path.exists() and not refresh:
        return pd.read_csv(path)
    frame = fetch_actuals(date, fetch_json=fetch_json)
    if not frame.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return frame


def load_actuals_range(dates: Iterable[str], actuals_dir: Path | str | None = None) -> pd.DataFrame:
    """Concatenate cached outcomes for several dates (missing dates are skipped)."""
    frames = []
    for date in dates:
        path = actuals_path(date, actuals_dir)
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame(columns=ACTUALS_COLUMNS)
    return pd.concat(frames, ignore_index=True)


__all__ = [
    "ACTUALS_COLUMNS",
    "DEFAULT_ACTUALS_DIR",
    "FD_HITTER_SCORING",
    "FD_PITCHER_SCORING",
    "actuals_path",
    "fanduel_hitter_points",
    "fanduel_pitcher_points",
    "fetch_actuals",
    "final_game_pks",
    "is_quality_start",
    "load_actuals_range",
    "load_or_fetch_actuals",
    "parse_boxscore",
]
