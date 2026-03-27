"""Loader utilities for batting order CSVs and paste-based lineup data."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .aliases import CanonicalMap, apply_aliases
from .text_utils import canonicalize_series

# Full team name → standard abbreviation (includes short names used by FantasyLabs)
_TEAM_NAME_TO_CODE: Dict[str, str] = {
    "arizona diamondbacks": "ARI", "diamondbacks": "ARI", "d-backs": "ARI",
    "atlanta braves": "ATL", "braves": "ATL",
    "baltimore orioles": "BAL", "orioles": "BAL",
    "boston red sox": "BOS", "red sox": "BOS",
    "chicago cubs": "CHC", "cubs": "CHC",
    "chicago white sox": "CWS", "white sox": "CWS",
    "cincinnati reds": "CIN", "reds": "CIN",
    "cleveland guardians": "CLE", "guardians": "CLE",
    "colorado rockies": "COL", "rockies": "COL",
    "detroit tigers": "DET", "tigers": "DET",
    "houston astros": "HOU", "astros": "HOU",
    "kansas city royals": "KC", "royals": "KC",
    "los angeles angels": "LAA", "angels": "LAA",
    "los angeles dodgers": "LAD", "dodgers": "LAD",
    "miami marlins": "MIA", "marlins": "MIA",
    "milwaukee brewers": "MIL", "brewers": "MIL",
    "minnesota twins": "MIN", "twins": "MIN",
    "new york mets": "NYM", "mets": "NYM",
    "new york yankees": "NYY", "yankees": "NYY",
    "oakland athletics": "OAK", "athletics": "OAK",
    "philadelphia phillies": "PHI", "phillies": "PHI",
    "pittsburgh pirates": "PIT", "pirates": "PIT",
    "san diego padres": "SD", "padres": "SD",
    "san francisco giants": "SF", "giants": "SF",
    "seattle mariners": "SEA", "mariners": "SEA",
    "st. louis cardinals": "STL", "st louis cardinals": "STL", "cardinals": "STL",
    "tampa bay rays": "TB", "rays": "TB",
    "texas rangers": "TEX", "rangers": "TEX",
    "toronto blue jays": "TOR", "blue jays": "TOR",
    "washington nationals": "WSH", "nationals": "WSH",
}


@dataclass
class BattingOrderTable:
    entries: pd.DataFrame
    source_path: Path

    def summary(self) -> dict[str, int]:
        df = self.entries
        return {
            "total": len(df),
            "teams": df["team_code"].nunique(),
        }


class BattingOrderLoader:
    """Loads batting order CSVs and normalizes player/team names."""

    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path).expanduser().resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Batting orders CSV not found: {self.csv_path}")

    def load(self, alias_map: Optional[CanonicalMap] = None) -> BattingOrderTable:
        df = pd.read_csv(self.csv_path)
        required = {"team", "order_position", "player_name"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Batting orders CSV missing required columns: {sorted(missing)}"
            )

        entries = df.copy()
        entries["team_code"] = (
            entries["team"].astype(str).str.upper().str.strip()
        )
        entries["batting_order_position"] = (
            pd.to_numeric(entries["order_position"], errors="coerce")
            .round()
            .astype("Int64")
        )
        entries["player_name"] = entries["player_name"].astype(str).str.strip()
        canonical = canonicalize_series(entries["player_name"])
        if alias_map:
            canonical = apply_aliases(canonical, alias_map)
        entries["canonical_name"] = canonical
        entries = entries.dropna(subset=["batting_order_position", "canonical_name"])  # type: ignore[arg-type]
        entries = entries[entries["batting_order_position"].between(0, 9, inclusive="both")]
        entries = entries.drop_duplicates(["team_code", "canonical_name"], keep="first")
        keep_cols = [
            "team_code",
            "canonical_name",
            "batting_order_position",
        ]
        return BattingOrderTable(entries=entries[keep_cols], source_path=self.csv_path)


def parse_lineup_paste(text: str) -> pd.DataFrame:
    """Parse pasted lineup data from FantasyLabs or similar sources.

    Handles both single-line and multi-line pitcher formats:
        Single-line:  Paul Skenes (R) $10.8K
        Multi-line:   Paul Skenes (R)
                      $10.8K

    Batter lines:
        1 - Nick Kurtz (L) 1B $4.0K
        * 1 - Nick Kurtz (L) 1B $4.0K

    Returns a DataFrame with columns: team, order_position, player_name
    Pitchers are included with order_position=0.
    """
    lines = text.strip().splitlines()
    rows: List[Dict[str, object]] = []

    # Pattern: "Team A (odds) @ Team B (odds)" or "Team A @ Team B"
    matchup_re = re.compile(
        r"^(.+?)\s*\([^)]*\)\s*[@vV][sS]?\.?\s*(.+?)\s*\([^)]*\)\s*$"
    )
    # Pattern: "* 1 - Player Name (R) POS $X.XK" with optional status flags
    player_re = re.compile(
        r"^\*?\s*(\d)\s*[-–—]\s*(.+?)\s*\(\s*([RLBS]?)\s*\)\s*"
        r"([\w/]+)\s*\$[\d.]+K?"
    )
    # Pattern: "Player Name (R) $X.XK" — pitcher on single line
    pitcher_single_re = re.compile(
        r"^([A-Z][a-zA-Z'.]+(?:\s+[A-Za-z'.]+)+)\s*\(([RLBS]?)\)\s*\$[\d.]+K?\s*$"
    )
    # Pattern: "Player Name (R)" — pitcher name only (salary on next line)
    pitcher_name_re = re.compile(
        r"^([A-Z][a-zA-Z'.]+(?:\s+[A-Za-z'.]+)+)\s*\(([RLBS]?)\)\s*$"
    )
    # Pattern: "$X.XK" — standalone salary line
    salary_re = re.compile(r"^\$[\d.]+K?\s*$")

    current_away: Optional[str] = None
    current_home: Optional[str] = None
    lineup_count = 0  # 0=before first, 1=away, 2=home
    pitcher_count = 0  # tracks pitchers seen for this matchup
    pending_pitcher_name: Optional[str] = None  # for multi-line pitcher format

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # If we have a pending pitcher name, check if this line is the salary
        if pending_pitcher_name is not None:
            if salary_re.match(line):
                pitcher_count += 1
                team = current_away if pitcher_count == 1 else current_home
                if team:
                    rows.append({
                        "team": team,
                        "order_position": 0,
                        "player_name": pending_pitcher_name,
                    })
                pending_pitcher_name = None
                continue
            # Not a salary line — the pending name wasn't a pitcher, discard it
            pending_pitcher_name = None

        # Check for matchup line
        m = matchup_re.match(line)
        if m:
            away_name = m.group(1).strip().lower()
            home_name = m.group(2).strip().lower()
            current_away = _TEAM_NAME_TO_CODE.get(away_name, away_name.upper())
            current_home = _TEAM_NAME_TO_CODE.get(home_name, home_name.upper())
            lineup_count = 0
            pitcher_count = 0
            continue

        # Check for "Projected Lineup" or "Confirmed Lineup"
        if re.match(r"^(projected|confirmed)\s+lineup", line, re.IGNORECASE):
            lineup_count += 1
            continue

        # Check for pitcher line (before lineups start)
        if lineup_count == 0 and (current_away or current_home):
            # Try single-line pitcher format first
            pm_pitcher = pitcher_single_re.match(line)
            if pm_pitcher:
                pitcher_name = pm_pitcher.group(1).strip()
                pitcher_count += 1
                team = current_away if pitcher_count == 1 else current_home
                if team:
                    rows.append({
                        "team": team,
                        "order_position": 0,
                        "player_name": pitcher_name,
                    })
                continue
            # Try multi-line: name only (salary on next line)
            pm_name = pitcher_name_re.match(line)
            if pm_name:
                pending_pitcher_name = pm_name.group(1).strip()
                continue

        # Check for player line
        pm = player_re.match(line)
        if pm and (current_away or current_home):
            order_pos = int(pm.group(1))
            player_name = pm.group(2).strip()
            # Remove trailing status indicators like "Q", "DTD", "O", "IL"
            player_name = re.sub(r"\s+[A-Z]{1,3}$", "", player_name)
            team = current_away if lineup_count <= 1 else current_home
            if team:
                rows.append({
                    "team": team,
                    "order_position": order_pos,
                    "player_name": player_name,
                })

    return pd.DataFrame(rows, columns=["team", "order_position", "player_name"])


__all__ = ["BattingOrderLoader", "BattingOrderTable", "parse_lineup_paste"]

