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


_CREATE_SIMULATION_ACCURACY = """
CREATE TABLE IF NOT EXISTS simulation_accuracy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    num_players INTEGER,
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


_CREATE_OWNERSHIP_HISTORY = """
CREATE TABLE IF NOT EXISTS ownership_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    fd_player_id TEXT NOT NULL,
    player_name TEXT,
    player_type TEXT,
    predicted_own REAL,
    actual_own REAL,
    features JSON,
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
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
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
        cur.execute(_CREATE_SIMULATION_ACCURACY)
        cur.execute(_CREATE_OWNERSHIP_HISTORY)
        self.conn.commit()
        self._ensure_column("lineup_results", "total_ownership", "REAL")
        self._ensure_column("lineup_results", "total_upside", "REAL")
        self._ensure_column("lineup_results", "total_salary", "REAL")

    def _ensure_column(self, table: str, column: str, ddl_type: str) -> None:
        cur = self.conn.cursor()
        existing = {row["name"] for row in cur.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")
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

        def _num(value):
            try:
                out = float(value)
            except (TypeError, ValueError):
                return None
            return out if out == out else None  # drop NaN

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
                    _num(row.get("total_ownership")),
                    _num(row.get("total_upside")),
                    _num(row.get("total_salary")),
                    created_at,
                )
            )
        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT INTO lineup_results(date, lineup_id, total_actual_points, rank, payout, roi, strategy_config_json, total_ownership, total_upside, total_salary, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        self.conn.commit()

    def insert_simulation_accuracy(self, date: str, metrics: dict[str, float], num_players: int) -> None:
        """Persist simulation calibration metrics."""
        if not metrics:
            return
        created_at = datetime.utcnow().isoformat()
        rows = []
        for name, value in metrics.items():
            if value is None:
                continue
            rows.append((date, name, float(value), num_players, created_at))
        if not rows:
            return
        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT INTO simulation_accuracy(date, metric_name, metric_value, num_players, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def insert_ownership_history(self, date: str, rows_df: pd.DataFrame) -> None:
        """Store the per-player ownership training snapshot for a slate.

        Expects columns fd_player_id, player_name, player_type, predicted_own,
        actual_own, features (JSON string). Replaces any prior rows for the
        same date so reprocessing a slate does not duplicate training data.
        """
        if rows_df is None or rows_df.empty:
            return
        created_at = datetime.utcnow().isoformat()
        records = []
        for _, row in rows_df.iterrows():
            records.append(
                (
                    date,
                    str(row.get("fd_player_id") or ""),
                    row.get("player_name"),
                    row.get("player_type"),
                    row.get("predicted_own"),
                    row.get("actual_own"),
                    row.get("features"),
                    created_at,
                )
            )
        cur = self.conn.cursor()
        cur.execute("DELETE FROM ownership_history WHERE date = ?", (date,))
        cur.executemany(
            """
            INSERT INTO ownership_history(date, fd_player_id, player_name, player_type, predicted_own, actual_own, features, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        self.conn.commit()

    def fetch_ownership_history(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        cur = self.conn.cursor()
        query = "SELECT * FROM ownership_history"
        params: list[str] = []
        if start_date and end_date:
            query += " WHERE date BETWEEN ? AND ?"
            params = [start_date, end_date]
        query += " ORDER BY date"
        rows = cur.execute(query, params).fetchall()
        return pd.DataFrame(rows, columns=[col[0] for col in cur.description]) if rows else pd.DataFrame()

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

    def fetch_lineup_results_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT * FROM lineup_results WHERE date BETWEEN ? AND ? ORDER BY date",
            (start_date, end_date),
        ).fetchall()
        return pd.DataFrame(rows, columns=[col[0] for col in cur.description]) if rows else pd.DataFrame()

    def fetch_slate_results_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT * FROM slate_results WHERE date BETWEEN ? AND ? ORDER BY date",
            (start_date, end_date),
        ).fetchall()
        return pd.DataFrame(rows, columns=[col[0] for col in cur.description]) if rows else pd.DataFrame()

    def fetch_simulation_accuracy(self, date: str) -> pd.DataFrame:
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT * FROM simulation_accuracy WHERE date = ? ORDER BY created_at",
            (date,),
        ).fetchall()
        return pd.DataFrame(rows, columns=[col[0] for col in cur.description]) if rows else pd.DataFrame()

    def close(self) -> None:
        self.conn.close()
