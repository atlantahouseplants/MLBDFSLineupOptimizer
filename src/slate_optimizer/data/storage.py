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

_CREATE_SLATE_RESULTS = """
CREATE TABLE IF NOT EXISTS slate_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slate_tag TEXT NOT NULL,
    date TEXT NOT NULL,
    contest_type TEXT,
    entry_fee REAL,
    num_entries INTEGER,
    winning_score REAL,
    cash_line REAL,
    created_at TEXT NOT NULL
);
"""

_CREATE_ACTUAL_SCORES = """
CREATE TABLE IF NOT EXISTS actual_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    fd_player_id TEXT,
    player_name TEXT,
    actual_fd_points REAL,
    actual_ownership_pct REAL,
    created_at TEXT NOT NULL
);
"""

_CREATE_LINEUP_RESULTS = """
CREATE TABLE IF NOT EXISTS lineup_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    lineup_id INTEGER NOT NULL,
    total_actual_points REAL,
    rank INTEGER,
    payout REAL,
    roi REAL,
    strategy_config_json TEXT,
    created_at TEXT NOT NULL
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
        cur.execute(_CREATE_SLATE_RESULTS)
        cur.execute(_CREATE_ACTUAL_SCORES)
        cur.execute(_CREATE_LINEUP_RESULTS)
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

    def get_slate_by_tag(self, tag: str) -> Optional[SlateRecord]:
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT * FROM slates WHERE tag = ? ORDER BY slate_id DESC LIMIT 1",
            (tag,),
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

    def fetch_players_by_tag(self, tag: str) -> pd.DataFrame:
        record = self.get_slate_by_tag(tag)
        if record is None:
            return pd.DataFrame()
        return self.fetch_players(record.slate_id)

    def insert_slate_result(
        self,
        slate_tag: str,
        date: str,
        contest_type: Optional[str],
        entry_fee: Optional[float],
        num_entries: Optional[int],
        winning_score: Optional[float],
        cash_line: Optional[float],
    ) -> int:
        created_at = datetime.utcnow().isoformat()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO slate_results(
                slate_tag, date, contest_type, entry_fee, num_entries, winning_score, cash_line, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                slate_tag,
                date,
                contest_type,
                entry_fee,
                num_entries,
                winning_score,
                cash_line,
                created_at,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def insert_actual_scores(self, date: str, scores: pd.DataFrame) -> None:
        created_at = datetime.utcnow().isoformat()
        records = []
        for _, row in scores.iterrows():
            records.append(
                (
                    date,
                    row.get("fd_player_id"),
                    row.get("player_name"),
                    row.get("actual_fd_points"),
                    row.get("actual_ownership_pct"),
                    created_at,
                )
            )
        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT INTO actual_scores(date, fd_player_id, player_name, actual_fd_points, actual_ownership_pct, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        self.conn.commit()

    def insert_lineup_results(self, date: str, results: pd.DataFrame) -> None:
        created_at = datetime.utcnow().isoformat()
        records = []
        for _, row in results.iterrows():
            records.append(
                (
                    date,
                    int(row.get("lineup_id")),
                    row.get("total_actual_points"),
                    row.get("rank"),
                    row.get("payout"),
                    row.get("roi"),
                    row.get("strategy_config_json"),
                    created_at,
                )
            )
        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT INTO lineup_results(date, lineup_id, total_actual_points, rank, payout, roi, strategy_config_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        self.conn.commit()

    def fetch_actual_scores(self, date: str) -> pd.DataFrame:
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT * FROM actual_scores WHERE date = ?",
            (date,),
        ).fetchall()
        return pd.DataFrame(rows, columns=[col[0] for col in cur.description]) if rows else pd.DataFrame()

    def fetch_lineup_results(self, date: str) -> pd.DataFrame:
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT * FROM lineup_results WHERE date = ?",
            (date,),
        ).fetchall()
        return pd.DataFrame(rows, columns=[col[0] for col in cur.description]) if rows else pd.DataFrame()

    def fetch_slate_results_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT * FROM slate_results WHERE date BETWEEN ? AND ? ORDER BY date",
            (start_date, end_date),
        ).fetchall()
        return pd.DataFrame(rows, columns=[col[0] for col in cur.description]) if rows else pd.DataFrame()

    def close(self) -> None:
        self.conn.close()

