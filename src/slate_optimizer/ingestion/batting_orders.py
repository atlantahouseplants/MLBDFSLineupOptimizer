"""Loader utilities for batting order CSVs and paste-based lineup data."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .aliases import CanonicalMap, apply_aliases
from .text_utils import canonicalize_series

# Full team name → standard abbreviation
_TEAM_NAME_TO_CODE: Dict[str, str] = {
    "arizona diamondbacks": "ARI", "atlanta braves": "ATL",
    "baltimore orioles": "BAL", "boston red sox": "BOS",
    "chicago cubs": "CHC", "chicago white sox": "CWS",
    "cincinnati reds": "CIN", "cleveland guardians": "CLE",
    "colorado rockies": "COL", "detroit tigers": "DET",
    "houston astros": "HOU", "kansas city royals": "KC",
    "los angeles angels": "LAA", "los angeles dodgers": "LAD",
    "miami marlins": "MIA", "milwaukee brewers": "MIL",
    "minnesota twins": "MIN", "new york mets": "NYM",
    "new york yankees": "NYY", "oakland athletics": "OAK",
    "philadelphia phillies": "PHI", "pittsburgh pirates": "PIT",
    "san diego padres": "SD", "san francisco giants": "SF",
    "seattle mariners": "SEA", "st. louis cardinals": "STL",
    "st louis cardinals": "STL", "tampa bay rays": "TB",
    "texas rangers": "TEX", "toronto blue jays": "TOR",
    "washington nationals": "WSH",
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

    Expects a format like:
        Pittsburgh Pirates (-100) @ New York Mets (-120)
        ...
        Paul Skenes (R) $10.8K
        Freddy Peralta (R) $9.6K
        Projected Lineup
        * 1 - Oneil Cruz (L) SS/OF $3.0K
        ...
        Projected Lineup
        * 1 - Francisco Lindor (B) SS $4.0K
        ...

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
    # Pattern: "Player Name (R) $X.XK" — pitcher line (no order number, no position)
    pitcher_re = re.compile(
        r"^([A-Z][a-zA-Z'.]+(?:\s+[A-Za-z'.]+)+)\s*\(([RLBS]?)\)\s*\$[\d.]+K?\s*$"
    )

    current_away: Optional[str] = None
    current_home: Optional[str] = None
    lineup_count = 0  # 0=before first, 1=away, 2=home
    pitcher_count = 0  # tracks pitchers seen for this matchup

    for line in lines:
        line = line.strip()
        if not line:
            continue

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
            pm_pitcher = pitcher_re.match(line)
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

