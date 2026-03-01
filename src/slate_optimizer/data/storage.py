"""Lightweight SQLite persistence for slate datasets."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

_CREATE_SLATES = """
CREATE TABLE IF NOT EXISTS slates (
    slate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag TEXT NOT NULL,
    fanduel_csv TEXT NOT NULL,
    ballparkpal_dir TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

_CREATE_PLAYERS = """
CREATE TABLE IF NOT EXISTS slate_players (
    slate_id INTEGER NOT NULL,
    fd_player_id TEXT NOT NULL,
    position TEXT,
    team TEXT,
    salary INTEGER,
    player_type TEXT,
    payload JSON,
    FOREIGN KEY (slate_id) REFERENCES slates(slate_id)
);
"""


@dataclass
class SlateRecord:
    slate_id: int
    tag: str
    fanduel_csv: str
    ballparkpal_dir: str
    created_at: datetime


class SlateDatabase:
    """Simple SQLite-backed store for slate snapshots."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path).expanduser().resolve()
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(_CREATE_SLATES)
        cur.execute(_CREATE_PLAYERS)
        self.conn.commit()

    def insert_slate(self, tag: str, fanduel_csv: Path, ballparkpal_dir: Path) -> SlateRecord:
        created_at = datetime.utcnow().isoformat()
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO slates(tag, fanduel_csv, ballparkpal_dir, created_at) VALUES (?, ?, ?, ?)",
            (tag, str(fanduel_csv), str(ballparkpal_dir), created_at),
        )
        slate_id = cur.lastrowid
        self.conn.commit()
        return self._row_to_record(
            {
                "slate_id": slate_id,
                "tag": tag,
                "fanduel_csv": str(fanduel_csv),
                "ballparkpal_dir": str(ballparkpal_dir),
                "created_at": created_at,
            }
        )

    def _row_to_record(self, row) -> SlateRecord:
        return SlateRecord(
            slate_id=int(row["slate_id"]),
            tag=row["tag"],
            fanduel_csv=row["fanduel_csv"],
            ballparkpal_dir=row["ballparkpal_dir"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def get_latest_slate(self) -> Optional[SlateRecord]:
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT * FROM slates ORDER BY slate_id DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def get_slate(self, slate_id: int) -> Optional[SlateRecord]:
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT * FROM slates WHERE slate_id = ?",
            (slate_id,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def write_players(self, slate_id: int, players_df: pd.DataFrame) -> None:
        records = []
        for _, row in players_df.iterrows():
            payload = row.to_dict()
            records.append(
                (
                    slate_id,
                    row.get("fd_player_id"),
                    row.get("position"),
                    row.get("team"),
                    int(row.get("salary", 0) or 0),
                    row.get("player_type"),
                    json.dumps(payload),
                )
            )
        cur = self.conn.cursor()
        cur.executemany(
            "INSERT INTO slate_players(slate_id, fd_player_id, position, team, salary, player_type, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
            records,
        )
        self.conn.commit()

    def fetch_players(self, slate_id: int) -> pd.DataFrame:
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT payload FROM slate_players WHERE slate_id = ?",
            (slate_id,),
        ).fetchall()
        if not rows:
            return pd.DataFrame()
        data = [json.loads(r["payload"]) for r in rows]
        return pd.DataFrame(data)

    def close(self) -> None:
        self.conn.close()
