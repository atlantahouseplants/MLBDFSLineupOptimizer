"""Auto-fetch BallparkPal simulation data using a subscriber session cookie.

Fetches:
  - Export Center: Batters, Pitchers, Games, Teams (simulation data)
  - Daily Fantasy Projections: Bust%, Median, Upside per player
  - Park Factors: per-stadium park factor index

BattingPosition, BatterStand, and PitcherHand are all included in the
Batters/Pitchers exports — so no separate batting orders or handedness
CSVs are needed when using this module.

How to get/refresh your session cookie:
  1. Log in to ballparkpal.com in your browser
  2. Open DevTools (F12) → Application → Cookies → ballparkpal.com
  3. Copy the PHPSESSID value
  4. Set BPP_SESSION=<value> in your .env file

Session cookies expire after ~7 days. Refresh when you see auth errors.
"""
from __future__ import annotations

import io
import os
import re
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_session(phpsessid: str) -> "_requests.Session":
    session = _requests.Session()
    session.headers.update(_HEADERS)
    session.cookies.set("PHPSESSID", phpsessid, domain="www.ballparkpal.com")
    return session


def _check_auth(session: "_requests.Session") -> bool:
    try:
        r = session.get(f"{_BASE_URL}/Export-Center.php", timeout=10)
        return "logout" in r.text.lower()
    except Exception:
        return False


def _fetch_excel(
    session: "_requests.Session", endpoint: str, date_str: str
) -> Optional[pd.DataFrame]:
    url = _BASE_URL + endpoint
    try:
        r = session.post(url, data={"date": date_str}, timeout=20)
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "")
        if "spreadsheet" in ct or "excel" in ct or "octet" in ct:
            return pd.read_excel(io.BytesIO(r.content))
        warnings.warn(
            f"Unexpected Content-Type from {endpoint}: {ct}. "
            "Session may have expired — refresh BPP_SESSION.",
            stacklevel=3,
        )
        return None
    except Exception as exc:
        warnings.warn(f"Failed to fetch {endpoint}: {exc}", stacklevel=3)
        return None


def _scrape_table(html: str) -> tuple[list[str], list[list[str]]]:
    """Parse the first HTML table's headers + body rows."""
    thead = re.findall(r"<thead[^>]*>(.*?)</thead>", html, re.DOTALL | re.IGNORECASE)
    headers: list[str] = []
    for th_block in thead:
        ths = re.findall(r"<th[^>]*>(.*?)</th>", th_block, re.DOTALL | re.IGNORECASE)
        headers = [re.sub(r"<[^>]+>", "", t).strip() for t in ths]

    tbody = re.findall(r"<tbody[^>]*>(.*?)</tbody>", html, re.DOTALL | re.IGNORECASE)
    rows: list[list[str]] = []
    if tbody:
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tbody[0], re.DOTALL | re.IGNORECASE):
            cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL | re.IGNORECASE)
            cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
            if any(cells):
                rows.append(cells)
    return headers, rows


def _scrape_dfs_projections(session: "_requests.Session") -> pd.DataFrame:
    """Scrape Bust%/Median/Upside from Daily-Fantasy-Projections.php.

    Eliminates the need to manually download the BPP DFS Optimizer Excel.
    Columns returned: team, full_name, position, salary, dfs_avg, bust_pct, upside
    """
    try:
        r = session.get(f"{_BASE_URL}/Daily-Fantasy-Projections.php", timeout=15)
        _, rows = _scrape_table(r.text)
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Column order: Team, Player, Pos, ?, Opp, Salary(K), Avg, Bust, Upside, Pts/$, ...
        col_map = {
            0: "team", 1: "player_display", 2: "position",
            4: "opponent", 5: "salary_k", 6: "dfs_avg",
            7: "bust_pct", 8: "upside", 9: "pts_per_k",
        }
        df = df.rename(columns={i: col_map.get(i, f"col_{i}") for i in df.columns})

        # Extract clean full name (page concatenates full + abbreviated name)
        if "player_display" in df.columns:
            # Full name ends where the abbreviated version starts (e.g. "Byron BuxtonB. Buxton")
            df["full_name"] = df["player_display"].str.extract(
                r"^([A-Z][a-z']+(?:\s+[A-Z][a-z'.]+)+)"
            )[0].str.strip()
            df["full_name"] = df["full_name"].fillna(df["player_display"])

        for col in ("dfs_avg", "upside", "pts_per_k"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "bust_pct" in df.columns:
            df["bust_pct"] = (
                pd.to_numeric(
                    df["bust_pct"].astype(str).str.replace("%", "", regex=False),
                    errors="coerce",
                )
                / 100.0
            )

        if "salary_k" in df.columns:
            df["salary"] = pd.to_numeric(df["salary_k"], errors="coerce") * 1000

        keep = [c for c in ["team", "full_name", "position", "opponent", "salary",
                             "dfs_avg", "bust_pct", "upside", "pts_per_k"] if c in df.columns]
        return df[keep].dropna(subset=["full_name"])

    except Exception as exc:
        warnings.warn(f"Failed to scrape DFS projections: {exc}", stacklevel=3)
        return pd.DataFrame()


def _scrape_park_factors(session: "_requests.Session") -> pd.DataFrame:
    """Scrape general park factors from Park-Factors-General.php.

    Returns: team, stadium, park_runs, park_hr, park_hits (index, 100=neutral)
    """
    try:
        r = session.get(f"{_BASE_URL}/Park-Factors-General.php", timeout=15)
        _, rows = _scrape_table(r.text)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # Cols: Team, Stadium, Runs, Hits, HR, XBH, 1B, K, BB, ...
        col_map = {0: "team", 1: "stadium", 2: "park_runs", 3: "park_hits",
                   4: "park_hr", 5: "park_xbh", 6: "park_1b", 7: "park_k", 8: "park_bb"}
        df = df.rename(columns={i: col_map.get(i, f"col_{i}") for i in df.columns})
        for col in ["park_runs", "park_hits", "park_hr", "park_xbh", "park_1b"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        keep = [c for c in col_map.values() if c in df.columns]
        return df[keep].dropna(subset=["team"])
    except Exception as exc:
        warnings.warn(f"Failed to scrape park factors: {exc}", stacklevel=3)
        return pd.DataFrame()


# ── Bundle ────────────────────────────────────────────────────────────────────

class BallparkPalBundle:
    """Container for a full BallparkPal data fetch."""

    def __init__(
        self,
        batters: pd.DataFrame,
        pitchers: pd.DataFrame,
        games: pd.DataFrame,
        teams: pd.DataFrame,
        dfs_projections: pd.DataFrame,
        park_factors: pd.DataFrame,
        fetch_date: str,
    ):
        self.batters = batters
        self.pitchers = pitchers
        self.games = games
        self.teams = teams
        self.dfs_projections = dfs_projections
        self.park_factors = park_factors
        self.fetch_date = fetch_date

    def summary(self) -> dict:
        return {
            "date": self.fetch_date,
            "batters": len(self.batters),
            "pitchers": len(self.pitchers),
            "games": len(self.games),
            "teams": len(self.teams),
            "dfs_projections": len(self.dfs_projections),
            "park_factors": len(self.park_factors),
        }

    def batting_orders(self) -> pd.DataFrame:
        """Return batting orders in the format BattingOrderLoader expects.

        Eliminates the need for a separate batting orders CSV.
        Columns: team, order_position, player_name
        """
        if self.batters.empty:
            return pd.DataFrame(columns=["team", "order_position", "player_name"])
        df = self.batters[["Team", "BattingPosition", "FullName"]].copy()
        df = df.rename(columns={
            "Team": "team",
            "BattingPosition": "order_position",
            "FullName": "player_name",
        })
        df["order_position"] = pd.to_numeric(df["order_position"], errors="coerce")
        return df.dropna(subset=["order_position"]).reset_index(drop=True)

    def handedness(self) -> pd.DataFrame:
        """Return player handedness in HandednessLoader format.

        Eliminates the need for a separate handedness CSV.
        Columns: player_name, team, bats, throws
        """
        rows = []
        if not self.batters.empty and "BatterStand" in self.batters.columns:
            b = self.batters[["FullName", "Team", "BatterStand"]].copy()
            b["throws"] = ""
            b = b.rename(columns={"FullName": "player_name", "Team": "team", "BatterStand": "bats"})
            rows.append(b)

        if not self.pitchers.empty and "PitcherHand" in self.pitchers.columns:
            p = self.pitchers[["FullName", "Team", "PitcherHand"]].copy()
            p["bats"] = ""
            p = p.rename(columns={"FullName": "player_name", "Team": "team", "PitcherHand": "throws"})
            rows.append(p)

        if not rows:
            return pd.DataFrame(columns=["player_name", "team", "bats", "throws"])
        return pd.concat(rows, ignore_index=True)

    def to_csvs(self, output_dir: str) -> dict[str, str]:
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
            ("dfs_projections", self.dfs_projections),
            ("park_factors", self.park_factors),
        ]:
            if not df.empty:
                path = out / f"bpp_{name}_{d}.csv"
                df.to_csv(path, index=False)
                paths[name] = str(path)

        # Also save derived convenience files
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
        """Save Excel files in BallparkPal format for existing ingestion pipeline."""
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
            if not df.empty:
                path = out / fname
                df.to_excel(path, index=False)
                paths[key] = str(path)
        return paths


# ── Main entry point ──────────────────────────────────────────────────────────

def fetch_bpp_data(
    phpsessid: Optional[str] = None,
    date_str: Optional[str] = None,
) -> Optional[BallparkPalBundle]:
    """Fetch today's full BallparkPal dataset.

    Includes: simulation data (4 exports) + DFS projections + park factors.
    The bundle also exposes .batting_orders() and .handedness() derived tables.

    Args:
        phpsessid: Session cookie. Falls back to BPP_SESSION env var.
        date_str: Date in YYYY-MM-DD format. Defaults to today.

    Returns:
        BallparkPalBundle or None on auth failure.
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
            "BallparkPal session expired. Log in and update BPP_SESSION in .env.",
            stacklevel=2,
        )
        return None

    # Fetch the 4 Export Center Excel files
    results = {}
    for name, endpoint in _EXPORT_ENDPOINTS.items():
        df = _fetch_excel(session, endpoint, date_str)
        if df is None:
            return None
        results[name] = df

    # Scrape DFS projections (Bust/Median/Upside)
    dfs_proj = _scrape_dfs_projections(session)

    # Scrape park factors
    park_factors = _scrape_park_factors(session)

    return BallparkPalBundle(
        batters=results["batters"],
        pitchers=results["pitchers"],
        games=results["games"],
        teams=results["teams"],
        dfs_projections=dfs_proj,
        park_factors=park_factors,
        fetch_date=date_str,
    )


__all__ = ["fetch_bpp_data", "BallparkPalBundle"]
