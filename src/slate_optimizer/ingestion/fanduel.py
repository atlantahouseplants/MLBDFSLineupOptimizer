"""Loader utilities for FanDuel player list CSV exports."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

_COLUMN_MAP: Dict[str, str] = {
    "Id": "fd_player_id",
    "Position": "position",
    "First Name": "first_name",
    "Nickname": "nickname",
    "Last Name": "last_name",
    "FPPG": "fppg",
    "Played": "played",
    "Salary": "salary",
    "Game": "game",
    "Team": "team",
    "Opponent": "opponent",
    "Injury Indicator": "injury_indicator",
    "Injury Details": "injury_details",
    "Tier": "tier",
    "Probable Pitcher": "probable_pitcher",
    "Batting Order": "batting_order",
    "Roster Position": "roster_position",
}

_UPLOAD_TEMPLATE_PLAYER_HEADERS = [
    "Player ID + Player Name", "Id", "Position", "First Name", "Nickname",
    "Last Name", "FPPG", "Played", "Salary", "Game", "Team", "Opponent",
    "Injury Indicator", "Injury Details", "Tier", "Probable Pitcher",
    "Batting Order", "Roster Position",
]


def _is_upload_template(csv_path: Path) -> bool:
    """Detect if the CSV is a FanDuel entries upload template."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        return len(header) > 0 and header[0].strip().lower() == "entry_id"


def _parse_upload_template(csv_path: Path) -> pd.DataFrame:
    """Extract player roster data from FanDuel upload template format."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Find the row containing the player column headers
    header_row_idx = None
    player_col_start = None
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            if cell.strip() == "Player ID + Player Name":
                header_row_idx = i
                player_col_start = j
                break
        if header_row_idx is not None:
            break

    if header_row_idx is None or player_col_start is None:
        raise ValueError("Could not find player data in FanDuel upload template")

    headers = rows[header_row_idx][player_col_start:]
    player_data: List[List[str]] = []
    for row in rows[header_row_idx + 1:]:
        if len(row) > player_col_start and row[player_col_start].strip():
            player_data.append(row[player_col_start:player_col_start + len(headers)])

    df = pd.DataFrame(player_data, columns=headers)
    return df


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns=_COLUMN_MAP)
    # Normalize any columns we do not have explicit mappings for
    for column in renamed.columns:
        if " " in column:
            normalized = column.strip().lower().replace(" ", "_")
            if normalized not in renamed.columns:
                renamed = renamed.rename(columns={column: normalized})
    return renamed


@dataclass
class FanduelPlayerList:
    """Container for the normalized FanDuel CSV."""

    players: pd.DataFrame
    source_path: Path

    def summary(self) -> Dict[str, int]:
        df = self.players
        return {
            "total": len(df),
            "pitchers": int((df["position"].str.upper() == "P").sum()),
            "hitters": int((df["position"].str.upper() != "P").sum()),
        }


class FanduelCSVLoader:
    """Loads FanDuel player list CSV exports."""

    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path).expanduser().resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"FanDuel CSV not found: {self.csv_path}")

    def load(self) -> FanduelPlayerList:
        if _is_upload_template(self.csv_path):
            df = _parse_upload_template(self.csv_path)
        else:
            df = pd.read_csv(self.csv_path)
        df = _rename_columns(df)
        df["position"] = df["position"].str.upper().fillna("")
        df["team"] = df["team"].str.upper().fillna("")
        df["opponent"] = df["opponent"].str.upper().fillna("")
        df["salary"] = (
            pd.to_numeric(df["salary"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        df["fppg"] = pd.to_numeric(df.get("fppg"), errors="coerce")
        df["full_name"] = (
            df.get("first_name", "").fillna("")
            + " "
            + df.get("last_name", "").fillna("")
        ).str.strip()
        df["full_name"] = df["full_name"].replace("^$", pd.NA, regex=True)
        return FanduelPlayerList(players=df, source_path=self.csv_path)
