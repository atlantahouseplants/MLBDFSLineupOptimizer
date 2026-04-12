"""Auto-fetch BallparkPal simulation data using a subscriber session cookie.

BallparkPal has no public API — this uses the Export Center endpoints
that subscribers can access. Requires a valid PHPSESSID session cookie.

How to get/refresh your session cookie:
  1. Log in to ballparkpal.com in your browser
  2. Open DevTools (F12) → Application → Cookies → ballparkpal.com
  3. Copy the PHPSESSID value
  4. Set it in your .env file: BPP_SESSION=<value>

Session cookies expire after ~7 days. Refresh when you see auth errors.
"""
from __future__ import annotations

import io
import os
import warnings
from datetime import date
from typing import Optional

import pandas as pd

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _requests = None  # type: ignore[assignment]
    _REQUESTS_AVAILABLE = False

_BASE_URL = "https://www.ballparkpal.com"
_EXPORT_ENDPOINTS = {
    "batters":  "/ExportBatters.php",
    "pitchers": "/ExportPitchers.php",
    "games":    "/ExportGames.php",
    "teams":    "/ExportTeams.php",
}
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    )
}


def _make_session(phpsessid: str) -> "_requests.Session":
    session = _requests.Session()
    session.headers.update(_HEADERS)
    session.cookies.set("PHPSESSID", phpsessid, domain="www.ballparkpal.com")
    return session


def _check_auth(session: "_requests.Session") -> bool:
    """Return True if the session cookie is still valid."""
    try:
        r = session.get(f"{_BASE_URL}/Export-Center.php", timeout=10)
        return "logout" in r.text.lower() or "sign out" in r.text.lower()
    except Exception:
        return False


def _fetch_excel(session: "_requests.Session", endpoint: str, date_str: str) -> Optional[pd.DataFrame]:
    """POST to an Export Center endpoint and return parsed DataFrame."""
    url = _BASE_URL + endpoint
    try:
        r = session.post(url, data={"date": date_str}, timeout=20)
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "")
        if "spreadsheet" in ct or "excel" in ct or "octet" in ct:
            return pd.read_excel(io.BytesIO(r.content))
        else:
            warnings.warn(
                f"Unexpected Content-Type from {endpoint}: {ct}. "
                "Session may have expired — refresh BPP_SESSION cookie.",
                stacklevel=3,
            )
            return None
    except Exception as exc:
        warnings.warn(f"Failed to fetch {endpoint}: {exc}", stacklevel=3)
        return None


class BallparkPalBundle:
    """Container for a full BallparkPal data fetch."""

    def __init__(
        self,
        batters: pd.DataFrame,
        pitchers: pd.DataFrame,
        games: pd.DataFrame,
        teams: pd.DataFrame,
        fetch_date: str,
    ):
        self.batters = batters
        self.pitchers = pitchers
        self.games = games
        self.teams = teams
        self.fetch_date = fetch_date

    def summary(self) -> dict:
        return {
            "date": self.fetch_date,
            "batters": len(self.batters),
            "pitchers": len(self.pitchers),
            "games": len(self.games),
            "teams": len(self.teams),
        }

    def to_csvs(self, output_dir: str) -> dict[str, str]:
        """Save each DataFrame as a CSV and return a dict of {name: path}."""
        import os
        from pathlib import Path
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {}
        d = self.fetch_date
        for name, df in [
            ("batters", self.batters),
            ("pitchers", self.pitchers),
            ("games", self.games),
            ("teams", self.teams),
        ]:
            path = out / f"bpp_{name}_{d}.csv"
            df.to_csv(path, index=False)
            paths[name] = str(path)
        return paths

    def to_excels(self, output_dir: str) -> dict[str, str]:
        """Save each DataFrame as an Excel file (BallparkPal format) for ingestion."""
        from pathlib import Path
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {}
        d = self.fetch_date
        name_map = {
            "batters":  f"BallparkPal_Batters_{d}.xlsx",
            "pitchers": f"BallparkPal_Pitchers_{d}.xlsx",
            "games":    f"BallparkPal_Games_{d}.xlsx",
            "teams":    f"BallparkPal_Teams_{d}.xlsx",
        }
        for key, fname in name_map.items():
            df = getattr(self, key)
            path = out / fname
            df.to_excel(path, index=False)
            paths[key] = str(path)
        return paths


def fetch_bpp_data(
    phpsessid: Optional[str] = None,
    date_str: Optional[str] = None,
) -> Optional[BallparkPalBundle]:
    """Fetch today's BallparkPal simulation data.

    Args:
        phpsessid: Session cookie. Falls back to BPP_SESSION env var.
        date_str: Date in YYYY-MM-DD format. Defaults to today.

    Returns:
        BallparkPalBundle with all four DataFrames, or None on auth failure.
    """
    if not _REQUESTS_AVAILABLE:
        warnings.warn("requests package not installed.", stacklevel=2)
        return None

    cookie = phpsessid or os.getenv("BPP_SESSION")
    if not cookie:
        warnings.warn(
            "BPP_SESSION not set. Log in to ballparkpal.com, copy your PHPSESSID "
            "cookie, and set BPP_SESSION=<value> in your .env file.",
            stacklevel=2,
        )
        return None

    if date_str is None:
        date_str = date.today().strftime("%Y-%m-%d")

    session = _make_session(cookie)

    if not _check_auth(session):
        warnings.warn(
            "BallparkPal session expired or invalid. "
            "Log in to ballparkpal.com and update BPP_SESSION in your .env file.",
            stacklevel=2,
        )
        return None

    results = {}
    for name, endpoint in _EXPORT_ENDPOINTS.items():
        df = _fetch_excel(session, endpoint, date_str)
        if df is None:
            return None
        results[name] = df

    return BallparkPalBundle(
        batters=results["batters"],
        pitchers=results["pitchers"],
        games=results["games"],
        teams=results["teams"],
        fetch_date=date_str,
    )


__all__ = ["fetch_bpp_data", "BallparkPalBundle"]
