"""Fetch BallparkPal data via the official BallparkPal JSON API (v1).

This replaces the fragile PHPSESSID cookie scrape (``bpp_api.py``) with the
supported API: https://www.ballparkpal.com/api/v1 — key auth via the
``X-API-Key`` header. Request a key from the API Access page on the site.

What the official API provides
------------------------------
- ``/games?date=``                → game ids, teams, start times
- ``/projections/averages``       → per-game sim averages: batters (incl.
                                    battingPosition + fantasyPointsFD), pitchers
                                    (incl. win/QS probability + fantasyPointsFD),
                                    team runs
- ``/projections/probabilities``  → market probabilities per game (moneyline,
                                    batter/pitcher props) — used for win pcts and
                                    HR/Hit/SB probabilities when present
- ``/parkfactors?date=``          → game-level park factors (weather baked in)
- ``/parkfactors/hitters``        → per-hitter park factors (stadium vs weather)
- ``/teams``                      → teamId → abbreviation reference

What it does NOT provide (and where that data comes from now)
-------------------------------------------------------------
- Batter/pitcher handedness (``BatterStand``/``PitcherHand``) → filled from the
  free MLB Stats API bulk player endpoint (one request, whole league).
- Bust%/Median/Upside DFS table → not in the API. The pipeline already handles
  its absence: ``baseline.py`` derives floor/ceiling from multipliers and the
  ownership model falls back to percentile estimates.
- Run distributions / inning splits / win margins → unused by the core
  pipeline; omitted from the generated frames.

Output contract
---------------
``BppApiBundle`` builds the same four DataFrames (same column names) as the
legacy Export Center Excel files, and ``to_excels()`` writes the same
``BallparkPal_{Batters,Pitchers,Games,Teams}_{date}.xlsx`` files — so
``BallparkPalLoader``, ``slate_builder``, ``baseline`` and the dashboard all
work unchanged. ``to_csvs()`` additionally writes the new API-only tables
(``bpp_probabilities_``, ``bpp_hitter_park_factors_``).

Rate limits: default allowance is 60 requests/min and 15k/month. A full daily
fetch is ~35 requests (1 teams + 1 games + 2×N games + 2 park factors + 1 MLB
bulk), so even three fetches/day stays well under 4k requests/month.
"""
from __future__ import annotations

import math
import os
import re
import time
import warnings
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import requests as _requests

    _REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover - requests is a hard dep in practice
    _requests = None  # type: ignore[assignment]
    _REQUESTS_AVAILABLE = False

_BASE_URL = "https://www.ballparkpal.com/api/v1"
_MLB_PLAYERS_URL = "https://statsapi.mlb.com/api/v1/sports/1/players"

# BPP teamId→abbreviation comes from /teams. MLB Stats API uses a few different
# abbreviations than FanDuel/BPP — normalize to the FD-style codes the pipeline
# already uses (mirrors slate_builder._TEAM_ALIASES).
_TEAM_ALIASES = {
    "CHW": "CWS",
    "WAS": "WSH",
    "WSN": "WSH",
    "KCR": "KC",
    "SDP": "SD",
    "SFG": "SF",
    "TBR": "TB",
    "ANA": "LAA",
}

_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Typical MLB stolen-base success rate, used to approximate StolenBaseAttempts
# (the official API exposes only successes). Nothing downstream consumes
# attempts; this keeps the legacy column populated and plausible.
_SB_SUCCESS_RATE = 0.80

# Market-key patterns for the probabilities endpoint. Exact keys aren't
# documented, so match defensively and fall back to Poisson estimates.
_MONEYLINE_RE = re.compile(r"moneyline|money_line|\bml\b|winner", re.IGNORECASE)
_HR_MARKET_RE = re.compile(r"home.?run|\bhr\b", re.IGNORECASE)
_HIT_MARKET_RE = re.compile(r"\bhits?\b|to_get.*hit|record.*hit", re.IGNORECASE)
_SB_MARKET_RE = re.compile(r"stolen.?base|\bsb\b|steal", re.IGNORECASE)


# ── Errors ────────────────────────────────────────────────────────────────────


class BppApiError(Exception):
    """Base error for official BallparkPal API failures."""


class BppAuthError(BppApiError):
    """401 (bad/missing key) or 403 (valid key, access blocked)."""


class BppRateLimitError(BppApiError):
    """429 after retries exhausted."""


class BppDateRangeError(BppApiError):
    """Requested date is outside the API's today-and-future window."""


# ── HTTP helpers ──────────────────────────────────────────────────────────────


def _make_session(api_key: str) -> "_requests.Session":
    session = _requests.Session()
    session.headers.update(
        {
            "X-API-Key": api_key,
            "Accept": "application/json",
            "User-Agent": "MLBDFSLineupOptimizer/1.0 (+ballparkpal api client)",
        }
    )
    return session


def _error_message(payload: object, default: str) -> str:
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            code = err.get("code", "")
            msg = err.get("message", "")
            return f"{code}: {msg}".strip(": ") or default
    return default


def _get(
    session: "_requests.Session",
    path: str,
    params: Optional[dict] = None,
    max_retries: int = 3,
    backoff: float = 5.0,
) -> dict:
    """GET an API endpoint with retry/backoff. Returns parsed JSON."""
    url = f"{_BASE_URL}{path}"
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=20)
        except Exception as exc:  # connection errors are retried
            last_exc = exc
            if attempt < max_retries:
                time.sleep(backoff * (attempt + 1))
                continue
            raise BppApiError(f"Request to {path} failed: {exc}") from exc

        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError as exc:
                raise BppApiError(f"Non-JSON response from {path}") from exc

        payload = None
        try:
            payload = resp.json()
        except ValueError:
            pass
        message = _error_message(payload, resp.text[:200])

        if resp.status_code == 401:
            raise BppAuthError(f"BallparkPal API key missing or invalid (401). {message}")
        if resp.status_code == 403:
            raise BppAuthError(
                f"BallparkPal API access blocked (403) — check subscription/API access. {message}"
            )
        if resp.status_code == 400 and "date_out_of_range" in message:
            raise BppDateRangeError(f"BallparkPal API: {message}")
        if resp.status_code == 429:
            if attempt < max_retries:
                time.sleep(backoff * (2**attempt))
                continue
            raise BppRateLimitError(
                f"BallparkPal API rate limited (429) after {max_retries + 1} attempts. {message}"
            )
        # Other 4xx/5xx: retry transient 5xx, fail fast otherwise
        if resp.status_code >= 500 and attempt < max_retries:
            time.sleep(backoff * (attempt + 1))
            continue
        raise BppApiError(f"BallparkPal API error {resp.status_code} on {path}: {message}")

    raise BppApiError(f"Request to {path} failed: {last_exc}")


def _unwrap_data(payload: dict) -> list:
    """Return the ``data`` payload as a flat list of item dicts.

    The API wraps results as {meta, data}. ``data`` may be a list, a dict of
    lists (e.g. {batters: [...], pitchers: [...]}), or a single object.
    """
    data = payload.get("data", payload)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Single object (e.g. /games/{id}) → wrap; dict of lists → flatten.
        if data and all(isinstance(v, list) for v in data.values()):
            out: list = []
            for v in data.values():
                out.extend(v)
            return out
        return [data] if data else []
    return []


def _as_fraction(value: object) -> float:
    """Normalize a probability to 0–1 (API may serve 0–100 percentages)."""
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    if v > 1.5:  # clearly a percentage
        v = v / 100.0
    return max(0.0, min(1.0, v))


def _poisson_prob(mean: object) -> float:
    """P(at least one event) for a Poisson rate — fallback probability."""
    try:
        mu = float(mean)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    if mu <= 0:
        return 0.0
    return 1.0 - math.exp(-mu)


def _last_name(full_name: str) -> str:
    parts = str(full_name).replace(".", "").split()
    if not parts:
        return ""
    last = parts[-1]
    if last.lower() in _NAME_SUFFIXES and len(parts) > 1:
        last = parts[-2]
    return last


def _normalize_team(abbrev: object) -> str:
    code = str(abbrev or "").upper().strip()
    return _TEAM_ALIASES.get(code, code)


# ── Endpoint fetchers ─────────────────────────────────────────────────────────


def fetch_teams(session: "_requests.Session") -> dict[int, str]:
    """Return {teamId: abbreviation} from /teams."""
    payload = _get(session, "/teams")
    mapping: dict[int, str] = {}
    for row in _unwrap_data(payload):
        try:
            mapping[int(row["teamId"])] = _normalize_team(row.get("abv"))
        except (KeyError, TypeError, ValueError):
            continue
    return mapping


def fetch_games(session: "_requests.Session", date_str: str) -> list[dict]:
    payload = _get(session, "/games", params={"date": date_str})
    games = [g for g in _unwrap_data(payload) if isinstance(g, dict) and "gameId" in g]
    games.sort(key=lambda g: (str(g.get("gameTime", "")), int(g["gameId"])))
    return games


def fetch_averages(session: "_requests.Session", game_id: int) -> dict:
    """Return {'batters': [...], 'pitchers': [...], 'teams': [...]} for a game."""
    payload = _get(session, "/projections/averages", params={"gameId": game_id})
    data = payload.get("data", payload)
    if isinstance(data, dict):
        return {
            "batters": [b for b in data.get("batters", []) if isinstance(b, dict)],
            "pitchers": [p for p in data.get("pitchers", []) if isinstance(p, dict)],
            "teams": [t for t in data.get("teams", []) if isinstance(t, dict)],
        }
    # Unexpected shape — treat as empty
    return {"batters": [], "pitchers": [], "teams": []}


def fetch_probabilities(session: "_requests.Session", game_id: int) -> list[dict]:
    payload = _get(session, "/projections/probabilities", params={"gameId": game_id})
    return [i for i in _unwrap_data(payload) if isinstance(i, dict)]


def fetch_park_factors(session: "_requests.Session", date_str: str) -> list[dict]:
    payload = _get(session, "/parkfactors", params={"date": date_str})
    return [i for i in _unwrap_data(payload) if isinstance(i, dict)]


def fetch_hitter_park_factors(session: "_requests.Session", date_str: str) -> list[dict]:
    payload = _get(session, "/parkfactors/hitters", params={"date": date_str})
    return [i for i in _unwrap_data(payload) if isinstance(i, dict)]


def fetch_mlb_handedness(season: Optional[int] = None) -> pd.DataFrame:
    """Bulk-fetch bat/throw handedness for every MLB player (free, no key).

    Returns DataFrame[player_name, mlb_team, bats, throws]. One HTTP request.
    """
    if not _REQUESTS_AVAILABLE:
        return pd.DataFrame(columns=["player_name", "mlb_team", "bats", "throws"])
    if season is None:
        season = date.today().year
    try:
        resp = _requests.get(
            _MLB_PLAYERS_URL,
            params={"season": season, "hydrate": "currentTeam"},
            timeout=30,
        )
        resp.raise_for_status()
        people = resp.json().get("people", [])
    except Exception as exc:
        warnings.warn(f"MLB handedness fetch failed: {exc}", stacklevel=2)
        return pd.DataFrame(columns=["player_name", "mlb_team", "bats", "throws"])

    rows = []
    for p in people:
        name = p.get("fullName")
        if not name:
            continue
        team = (p.get("currentTeam") or {}).get("abbreviation", "")
        bats = (p.get("batSide") or {}).get("code", "")
        throws = (p.get("pitchHand") or {}).get("code", "")
        rows.append(
            {
                "player_name": name,
                "mlb_team": _normalize_team(team),
                "bats": str(bats).upper()[:1],
                "throws": str(throws).upper()[:1],
            }
        )
    return pd.DataFrame(rows, columns=["player_name", "mlb_team", "bats", "throws"])


# ── Handedness lookup ─────────────────────────────────────────────────────────


def _canon(name: object) -> str:
    """Light name canonicalization for joining to MLB handedness."""
    s = re.sub(r"[^a-z ]", "", str(name or "").lower())
    parts = [p for p in s.split() if p not in _NAME_SUFFIXES]
    return " ".join(parts)


class _HandednessLookup:
    def __init__(self, mlb_df: pd.DataFrame):
        self._by_name: dict[str, dict] = {}
        self._by_name_team: dict[tuple[str, str], dict] = {}
        if mlb_df is None or mlb_df.empty:
            return
        for row in mlb_df.to_dict("records"):
            key = _canon(row.get("player_name"))
            if not key:
                continue
            entry = {
                "bats": str(row.get("bats", "") or ""),
                "throws": str(row.get("throws", "") or ""),
                "team": str(row.get("mlb_team", "") or ""),
            }
            # First wins for bare-name collisions; name+team is exact.
            self._by_name.setdefault(key, entry)
            if entry["team"]:
                self._by_name_team[(key, entry["team"])] = entry

    def bats(self, name: object, team: object = "") -> str:
        entry = self._lookup(name, team)
        return entry.get("bats", "") if entry else ""

    def throws(self, name: object, team: object = "") -> str:
        entry = self._lookup(name, team)
        return entry.get("throws", "") if entry else ""

    def _lookup(self, name: object, team: object) -> Optional[dict]:
        key = _canon(name)
        if not key:
            return None
        t = _normalize_team(team)
        if t and (key, t) in self._by_name_team:
            return self._by_name_team[(key, t)]
        return self._by_name.get(key)


# ── Probabilities parsing ─────────────────────────────────────────────────────


def _subject_key(item: dict) -> tuple[str, Optional[int]]:
    subj = item.get("subject") or {}
    stype = str(subj.get("type", "")).lower()
    try:
        sid = int(subj.get("id"))
    except (TypeError, ValueError):
        sid = None
    return stype, sid


def _market_text(item: dict) -> str:
    return " ".join(
        str(item.get(k, "")) for k in ("marketType", "marketKey", "displayName")
    )


def _is_over05(item: dict) -> bool:
    """True for over-0.5 style props (or line-less markets)."""
    line = item.get("line")
    side = str(item.get("side") or "").lower()
    # An explicit non-over side (e.g. "under") is not a "to record" probability.
    if side and side not in {"over", "yes"}:
        return False
    if line is None:
        return True
    try:
        return float(line) <= 0.5
    except (TypeError, ValueError):
        return True


def parse_win_probabilities(items: list[dict], team_map: dict[int, str]) -> dict[int, float]:
    """Extract {teamId: win probability} from moneyline items."""
    out: dict[int, float] = {}
    for item in items:
        if not _MONEYLINE_RE.search(_market_text(item)):
            continue
        stype, sid = _subject_key(item)
        tid = item.get("teamId")
        try:
            tid = int(tid) if tid is not None else (sid if stype == "team" else None)
        except (TypeError, ValueError):
            continue
        if tid is None or tid not in team_map:
            continue
        prob = _as_fraction(item.get("probability"))
        if not math.isnan(prob):
            out[tid] = prob
    return out


def parse_batter_prop_probabilities(items: list[dict]) -> dict[int, dict[str, float]]:
    """Extract per-batter {playerId: {hr, hit, sb}} probabilities from prop markets."""
    out: dict[int, dict[str, float]] = {}
    for item in items:
        stype, sid = _subject_key(item)
        if stype != "batter" or sid is None or not _is_over05(item):
            continue
        text = _market_text(item)
        if _SB_MARKET_RE.search(text):
            kind = "sb"
        elif _HR_MARKET_RE.search(text):
            kind = "hr"
        elif _HIT_MARKET_RE.search(text):
            kind = "hit"
        else:
            continue
        prob = _as_fraction(item.get("probability"))
        if math.isnan(prob):
            continue
        out.setdefault(sid, {})[kind] = prob
    return out


# ── Frame builders (legacy Export Center column names) ────────────────────────


def _game_time_str(value: object) -> str:
    """Keep BPP's 'H:MM' style if present; pass through whatever the API sends."""
    return str(value or "").strip()


def build_batters_frame(
    games: list[dict],
    averages: dict[int, dict],
    team_map: dict[int, str],
    handedness: _HandednessLookup,
    batter_props: dict[int, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for game in games:
        gid = int(game["gameId"])
        away_id, home_id = game.get("teamAwayId"), game.get("teamHomeId")
        for b in averages.get(gid, {}).get("batters", []):
            tid = b.get("teamId")
            try:
                tid = int(tid)
            except (TypeError, ValueError):
                continue
            team = _normalize_team(b.get("team") or team_map.get(tid, ""))
            if tid == home_id:
                side, opp_id = "H", away_id
            else:
                side, opp_id = "A", home_id
            opponent = team_map.get(opp_id, "")
            name = str(b.get("playerName", "")).strip()
            pid = b.get("playerId")
            try:
                pid = int(pid)
            except (TypeError, ValueError):
                pid = None
            props = batter_props.get(pid, {}) if pid is not None else {}
            sb_successes = b.get("stolenBaseSuccesses")
            try:
                sb_successes_f = float(sb_successes) if sb_successes is not None else 0.0
            except (TypeError, ValueError):
                sb_successes_f = 0.0
            rows.append(
                {
                    "GamePk": gid,
                    "GameDate": str(game.get("gameDate", "")),
                    "GameTime": _game_time_str(game.get("gameTime")),
                    "PlayerId": pid,
                    "FullName": name,
                    "LastName": _last_name(name),
                    "BatterStand": handedness.bats(name, team),
                    "Side": side,
                    "Team": team,
                    "Opponent": opponent,
                    "BattingPosition": b.get("battingPosition"),
                    "PlateAppearances": b.get("plateAppearances"),
                    "AtBats": b.get("atBats"),
                    "Hits": b.get("hits"),
                    "Bases": b.get("totalBases"),
                    "Strikeouts": b.get("strikeouts"),
                    "Walks": b.get("walks"),
                    "Singles": b.get("singles"),
                    "Doubles": b.get("doubles"),
                    "Triples": b.get("triples"),
                    "HomeRuns": b.get("homeRuns"),
                    "RBIs": b.get("rbis"),
                    "Runs": b.get("runs"),
                    "StolenBaseAttempts": round(sb_successes_f / _SB_SUCCESS_RATE, 3),
                    "StolenBaseSuccesses": sb_successes,
                    "PointsDK": b.get("fantasyPointsDK"),
                    "PointsFD": b.get("fantasyPointsFD"),
                    # Prefer market probabilities; fall back to Poisson estimate
                    # from the sim average (monotone and close for rare events).
                    "HomeRunProbability": props.get("hr", _poisson_prob(b.get("homeRuns"))),
                    "HitProbability": props.get("hit", _poisson_prob(b.get("hits"))),
                    "StolenBaseProbability": props.get(
                        "sb", _poisson_prob(b.get("stolenBaseSuccesses"))
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_pitchers_frame(
    games: list[dict],
    averages: dict[int, dict],
    team_map: dict[int, str],
    handedness: _HandednessLookup,
) -> pd.DataFrame:
    rows = []
    for game in games:
        gid = int(game["gameId"])
        away_id, home_id = game.get("teamAwayId"), game.get("teamHomeId")
        for p in averages.get(gid, {}).get("pitchers", []):
            if p.get("isStarter") is False:
                continue  # match Export Center behavior: starters only
            tid = p.get("teamId")
            try:
                tid = int(tid)
            except (TypeError, ValueError):
                continue
            team = _normalize_team(p.get("team") or team_map.get(tid, ""))
            if tid == home_id:
                side, opp_id = "H", away_id
            else:
                side, opp_id = "A", home_id
            opponent = team_map.get(opp_id, "")
            name = str(p.get("playerName", "")).strip()
            pid = p.get("playerId")
            try:
                pid = int(pid)
            except (TypeError, ValueError):
                pid = None
            win = _as_fraction(p.get("winProbability"))
            loss = _as_fraction(p.get("lossProbability"))
            nd = (
                max(0.0, 1.0 - win - loss)
                if not (math.isnan(win) or math.isnan(loss))
                else None
            )
            rows.append(
                {
                    "GamePk": gid,
                    "GameDate": str(game.get("gameDate", "")),
                    "GameTime": _game_time_str(game.get("gameTime")),
                    "PlayerId": pid,
                    "FullName": name,
                    "LastName": _last_name(name),
                    "PitcherHand": handedness.throws(name, team),
                    "Side": side,
                    "Team": team,
                    "Opponent": opponent,
                    "BattersFaced": p.get("battersFaced"),
                    "Innings": p.get("innings"),
                    "WinPct": win,
                    "LossPct": loss,
                    "NdPct": nd,
                    "QualityStart": _as_fraction(p.get("qualityStartProbability")),
                    "PointsDK": p.get("fantasyPointsDK"),
                    "PointsFD": p.get("fantasyPointsFD"),
                    "RunsAllowed": p.get("runsAllowed"),
                    "HitsAllowed": p.get("hitsAllowed"),
                    "Strikeouts": p.get("strikeouts"),
                    "Walks": p.get("walks"),
                }
            )
    return pd.DataFrame(rows)


def build_games_frame(
    games: list[dict],
    averages: dict[int, dict],
    team_map: dict[int, str],
    win_probs: dict[int, dict[int, float]],
) -> pd.DataFrame:
    rows = []
    for game in games:
        gid = int(game["gameId"])
        away_id, home_id = game.get("teamAwayId"), game.get("teamHomeId")
        team_runs = {}
        for t in averages.get(gid, {}).get("teams", []):
            try:
                team_runs[int(t["teamId"])] = t.get("runs")
            except (KeyError, TypeError, ValueError):
                continue
        probs = win_probs.get(gid, {})
        rows.append(
            {
                "GamePk": gid,
                "GameDate": str(game.get("gameDate", "")),
                "AwayTeam": team_map.get(away_id, ""),
                "HomeTeam": team_map.get(home_id, ""),
                "RunsAway": team_runs.get(away_id),
                "RunsHome": team_runs.get(home_id),
                "AwayWinPct": probs.get(away_id),
                "HomeWinPct": probs.get(home_id),
            }
        )
    return pd.DataFrame(rows)


def build_teams_frame(
    games: list[dict],
    averages: dict[int, dict],
    team_map: dict[int, str],
    win_probs: dict[int, dict[int, float]],
    batters_df: pd.DataFrame,
) -> pd.DataFrame:
    # Team batting counting stats = sum of that game's starting batters.
    sums: dict[tuple[int, str], dict[str, float]] = {}
    stat_cols = ["HomeRuns", "Triples", "Doubles", "Singles", "Walks", "Strikeouts"]
    if not batters_df.empty:
        grouped = batters_df.groupby(["GamePk", "Team"])
        for col in stat_cols:
            for (gid, team), val in grouped[col].sum().items():
                sums.setdefault((gid, team), {})[col] = float(val)

    rows = []
    for game in games:
        gid = int(game["gameId"])
        away_id, home_id = game.get("teamAwayId"), game.get("teamHomeId")
        team_runs = {}
        for t in averages.get(gid, {}).get("teams", []):
            try:
                team_runs[int(t["teamId"])] = t.get("runs")
            except (KeyError, TypeError, ValueError):
                continue
        probs = win_probs.get(gid, {})
        for tid, opp_id, side in ((away_id, home_id, "A"), (home_id, away_id, "H")):
            team = team_map.get(tid, "")
            counting = sums.get((gid, team), {})
            rows.append(
                {
                    "GamePk": gid,
                    "GameDate": str(game.get("gameDate", "")),
                    "Side": side,
                    "Team": team,
                    "Opponent": team_map.get(opp_id, ""),
                    "Runs": team_runs.get(tid),
                    "WinPercent": probs.get(tid),
                    "HomeRuns": counting.get("HomeRuns"),
                    "Triples": counting.get("Triples"),
                    "Doubles": counting.get("Doubles"),
                    "Singles": counting.get("Singles"),
                    "Walks": counting.get("Walks"),
                    "Strikeouts": counting.get("Strikeouts"),
                }
            )
    return pd.DataFrame(rows)


def build_park_factors_frame(rows: list[dict]) -> pd.DataFrame:
    """Game-level park factors. Keeps legacy column names where they map.

    Legacy scrape had a 'stadium' name and 100-centered indices; the API serves
    per-game percents (100 = neutral) with weather baked in, plus absolute
    amounts. 'stadium' is left blank (not exposed by the API).
    """
    out = []
    for r in rows:
        home = _normalize_team(r.get("teamHome"))
        out.append(
            {
                "team": home,
                "stadium": "",
                "park_runs": r.get("runsPercent"),
                "park_hits": r.get("singlesPercent"),
                "park_hr": r.get("homeRunsPercent"),
                "park_xbh": r.get("doublesTriplesPercent"),
                "park_1b": r.get("singlesPercent"),
                # New API-only fields
                "game_id": r.get("gameId"),
                "game_time": r.get("gameTime"),
                "team_away": _normalize_team(r.get("teamAway")),
                "team_home": home,
                "runs_amount": r.get("runsAmount"),
                "home_runs_amount": r.get("homeRunsAmount"),
                "doubles_triples_amount": r.get("doublesTriplesAmount"),
                "singles_amount": r.get("singlesAmount"),
            }
        )
    return pd.DataFrame(out)


def build_hitter_park_frame(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        out.append(
            {
                "game_id": r.get("gameId"),
                "game_time": r.get("gameTime"),
                "team_away": _normalize_team(r.get("teamAway")),
                "team_home": _normalize_team(r.get("teamHome")),
                "player_id": r.get("playerId"),
                "player_name": r.get("playerName"),
                "team": _normalize_team(r.get("team")),
                "home_runs": r.get("homeRuns"),
                "doubles_triples": r.get("doublesTriples"),
                "singles": r.get("singles"),
                "home_runs_stadium": r.get("homeRunsStadium"),
                "doubles_triples_stadium": r.get("doublesTriplesStadium"),
                "singles_stadium": r.get("singlesStadium"),
                "home_runs_weather": r.get("homeRunsWeather"),
                "doubles_triples_weather": r.get("doublesTriplesWeather"),
                "singles_weather": r.get("singlesWeather"),
            }
        )
    return pd.DataFrame(out)


def build_probabilities_frame(items_by_game: dict[int, list[dict]]) -> pd.DataFrame:
    """Raw probabilities dump for future leverage/ownership research."""
    rows = []
    for gid, items in items_by_game.items():
        for item in items:
            stype, sid = _subject_key(item)
            rows.append(
                {
                    "game_id": gid,
                    "market_type": item.get("marketType"),
                    "market_key": item.get("marketKey"),
                    "display_name": item.get("displayName"),
                    "line": item.get("line"),
                    "side": item.get("side"),
                    "odds": item.get("odds"),
                    "probability": item.get("probability"),
                    "average": item.get("average"),
                    "subject_type": stype or None,
                    "subject_id": sid,
                    "team_id": item.get("teamId"),
                }
            )
    return pd.DataFrame(rows)


# ── Bundle ────────────────────────────────────────────────────────────────────


class BppApiBundle:
    """Container for a full official-API fetch.

    Exposes the same interface as bpp_api.BallparkPalBundle (batters, pitchers,
    games, teams, park_factors, batting_orders(), handedness(), to_csvs(),
    to_excels(), summary()) so fetch_live_data.py and the downstream pipeline
    work unchanged. dfs_projections is always empty (not served by the API).
    """

    def __init__(
        self,
        batters: pd.DataFrame,
        pitchers: pd.DataFrame,
        games: pd.DataFrame,
        teams: pd.DataFrame,
        park_factors: pd.DataFrame,
        hitter_park_factors: pd.DataFrame,
        probabilities: pd.DataFrame,
        handedness_df: pd.DataFrame,
        fetch_date: str,
    ):
        self.batters = batters
        self.pitchers = pitchers
        self.games = games
        self.teams = teams
        self.dfs_projections = pd.DataFrame()  # not served by the official API
        self.park_factors = park_factors
        self.hitter_park_factors = hitter_park_factors
        self.probabilities = probabilities
        self._handedness_df = handedness_df
        self.fetch_date = fetch_date
        self.source = "official_api"

    def summary(self) -> dict:
        return {
            "date": self.fetch_date,
            "batters": len(self.batters),
            "pitchers": len(self.pitchers),
            "games": len(self.games),
            "teams": len(self.teams),
            "dfs_projections": 0,
            "park_factors": len(self.park_factors),
            "hitter_park_factors": len(self.hitter_park_factors),
            "probabilities": len(self.probabilities),
            "handedness": len(self._handedness_df),
        }

    def batting_orders(self) -> pd.DataFrame:
        """team, order_position, player_name — BattingOrderLoader format."""
        if self.batters.empty:
            return pd.DataFrame(columns=["team", "order_position", "player_name"])
        df = self.batters[["Team", "BattingPosition", "FullName"]].copy()
        df = df.rename(
            columns={
                "Team": "team",
                "BattingPosition": "order_position",
                "FullName": "player_name",
            }
        )
        df["order_position"] = pd.to_numeric(df["order_position"], errors="coerce")
        return df.dropna(subset=["order_position"]).reset_index(drop=True)

    def handedness(self) -> pd.DataFrame:
        """player_name, team, bats, throws — HandednessLoader format."""
        rows = []
        if not self.batters.empty:
            b = self.batters[["FullName", "Team", "BatterStand"]].copy()
            b["throws"] = ""
            b = b.rename(
                columns={"FullName": "player_name", "Team": "team", "BatterStand": "bats"}
            )
            rows.append(b)
        if not self.pitchers.empty:
            p = self.pitchers[["FullName", "Team", "PitcherHand"]].copy()
            p["bats"] = ""
            p = p.rename(
                columns={"FullName": "player_name", "Team": "team", "PitcherHand": "throws"}
            )
            rows.append(p)
        if not rows:
            return pd.DataFrame(columns=["player_name", "team", "bats", "throws"])
        return pd.concat(rows, ignore_index=True)

    def to_csvs(self, output_dir: str) -> dict[str, str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        d = self.fetch_date
        for name, df in [
            ("batters", self.batters),
            ("pitchers", self.pitchers),
            ("games", self.games),
            ("teams", self.teams),
            ("park_factors", self.park_factors),
            ("hitter_park_factors", self.hitter_park_factors),
            ("probabilities", self.probabilities),
        ]:
            if df is not None and not df.empty:
                path = out / f"bpp_{name}_{d}.csv"
                df.to_csv(path, index=False)
                paths[name] = str(path)

        bo = self.batting_orders()
        if not bo.empty:
            p = out / f"batting_orders_{d}.csv"
            bo.to_csv(p, index=False)
            paths["batting_orders"] = str(p)

        hand = self.handedness()
        if not hand.empty:
            p = out / f"handedness_{d}.csv"
            hand.to_csv(p, index=False)
            paths["handedness"] = str(p)
        return paths

    def to_excels(self, output_dir: str) -> dict[str, str]:
        """Write BallparkPal-format xlsx files for the existing ingestion pipeline."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        d = self.fetch_date
        name_map = {
            "batters": f"BallparkPal_Batters_{d}.xlsx",
            "pitchers": f"BallparkPal_Pitchers_{d}.xlsx",
            "games": f"BallparkPal_Games_{d}.xlsx",
            "teams": f"BallparkPal_Teams_{d}.xlsx",
        }
        for key, fname in name_map.items():
            df = getattr(self, key)
            if df is not None and not df.empty:
                path = out / fname
                df.to_excel(path, index=False)
                paths[key] = str(path)
        return paths


# ── Main entry point ──────────────────────────────────────────────────────────


def fetch_bpp_api_data(
    api_key: Optional[str] = None,
    date_str: Optional[str] = None,
    sleep_seconds: float = 1.05,
    include_probabilities: bool = True,
    include_park_factors: bool = True,
    include_handedness: bool = True,
) -> Optional[BppApiBundle]:
    """Fetch today's full BallparkPal dataset via the official API.

    Args:
        api_key: BallparkPal API key. Falls back to BPP_API_KEY env var.
        date_str: Date in YYYY-MM-DD format. Defaults to today.
        sleep_seconds: Pause between calls (60 req/min cap → default ~57/min).
        include_probabilities: Also fetch per-game market probabilities
            (moneyline win pcts + batter prop probabilities). Recommended.
        include_park_factors: Fetch game-level + hitter park factors.
        include_handedness: Bulk-fetch MLB handedness (free, no key) to fill
            BatterStand/PitcherHand.

    Returns:
        BppApiBundle, or None on auth failure / date out of range.
    """
    if not _REQUESTS_AVAILABLE:
        warnings.warn("requests package not installed.", stacklevel=2)
        return None

    key = api_key or os.getenv("BPP_API_KEY")
    if not key:
        warnings.warn(
            "BPP_API_KEY not set. Request API access from your BallparkPal "
            "API Access page, then set BPP_API_KEY=<key> in your .env file.",
            stacklevel=2,
        )
        return None

    if date_str is None:
        date_str = date.today().strftime("%Y-%m-%d")

    session = _make_session(key)

    try:
        team_map = fetch_teams(session)
        if not team_map:
            warnings.warn("BallparkPal /teams returned no teams.", stacklevel=2)
            return None
        time.sleep(sleep_seconds)

        games = fetch_games(session, date_str)
        if not games:
            warnings.warn(f"BallparkPal: no games found for {date_str}.", stacklevel=2)
            return None
    except (BppAuthError, BppDateRangeError) as exc:
        warnings.warn(str(exc), stacklevel=2)
        return None

    averages: dict[int, dict] = {}
    prob_items: dict[int, list[dict]] = {}
    for game in games:
        gid = int(game["gameId"])
        try:
            time.sleep(sleep_seconds)
            averages[gid] = fetch_averages(session, gid)
            if include_probabilities:
                time.sleep(sleep_seconds)
                prob_items[gid] = fetch_probabilities(session, gid)
        except BppApiError as exc:
            warnings.warn(f"Game {gid}: {exc}", stacklevel=2)

    park_rows: list[dict] = []
    hitter_park_rows: list[dict] = []
    if include_park_factors:
        try:
            time.sleep(sleep_seconds)
            park_rows = fetch_park_factors(session, date_str)
            time.sleep(sleep_seconds)
            hitter_park_rows = fetch_hitter_park_factors(session, date_str)
        except BppApiError as exc:
            warnings.warn(f"Park factors: {exc}", stacklevel=2)

    mlb_hand = pd.DataFrame(columns=["player_name", "mlb_team", "bats", "throws"])
    if include_handedness:
        season = int(date_str[:4]) if re.match(r"^\d{4}-", date_str) else None
        mlb_hand = fetch_mlb_handedness(season)
    handedness = _HandednessLookup(mlb_hand)

    # Win probabilities per game per teamId
    win_probs: dict[int, dict[int, float]] = {
        gid: parse_win_probabilities(items, team_map) for gid, items in prob_items.items()
    }
    # Batter prop probabilities across the slate (playerId → hr/hit/sb)
    batter_props: dict[int, dict[str, float]] = {}
    for items in prob_items.values():
        for pid, props in parse_batter_prop_probabilities(items).items():
            batter_props.setdefault(pid, {}).update(props)

    batters_df = build_batters_frame(games, averages, team_map, handedness, batter_props)
    pitchers_df = build_pitchers_frame(games, averages, team_map, handedness)
    games_df = build_games_frame(games, averages, team_map, win_probs)
    teams_df = build_teams_frame(games, averages, team_map, win_probs, batters_df)
    park_df = build_park_factors_frame(park_rows)
    hitter_park_df = build_hitter_park_frame(hitter_park_rows)
    probs_df = build_probabilities_frame(prob_items)

    return BppApiBundle(
        batters=batters_df,
        pitchers=pitchers_df,
        games=games_df,
        teams=teams_df,
        park_factors=park_df,
        hitter_park_factors=hitter_park_df,
        probabilities=probs_df,
        handedness_df=mlb_hand,
        fetch_date=date_str,
    )


__all__ = [
    "fetch_bpp_api_data",
    "BppApiBundle",
    "BppApiError",
    "BppAuthError",
    "BppRateLimitError",
    "BppDateRangeError",
]
