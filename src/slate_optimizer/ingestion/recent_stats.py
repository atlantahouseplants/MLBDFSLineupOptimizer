"""Loader utilities for recent performance stats."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .aliases import CanonicalMap, apply_aliases
from .text_utils import canonicalize_series


@dataclass
class RecentStatsTable:
    entries: pd.DataFrame
    source_path: Path

    def summary(self) -> dict[str, int]:
        df = self.entries
        return {"total": len(df), "teams": df["team_code"].nunique()}


class RecentStatsLoader:
    """Loads recent performance blends for hitters/pitchers."""

    REQUIRED_COLUMNS = {
        "player_name",
        "team",
        "position",
        "last_7_fppg",
        "last_14_fppg",
        "season_fppg",
    }

    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path).expanduser().resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Recent stats CSV not found: {self.csv_path}")

    def load(self, alias_map: Optional[CanonicalMap] = None) -> RecentStatsTable:
        df = pd.read_csv(self.csv_path)
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Recent stats CSV missing required columns: {sorted(missing)}"
            )

        entries = df.copy()
        entries["team_code"] = entries["team"].astype(str).str.upper().str.strip()
        entries["player_name"] = entries["player_name"].astype(str).str.strip()
        canonical = canonicalize_series(entries["player_name"])
        if alias_map:
            canonical = apply_aliases(canonical, alias_map)
        entries["canonical_name"] = canonical

        for column in ("last_7_fppg", "last_14_fppg", "season_fppg"):
            entries[column] = pd.to_numeric(entries[column], errors="coerce")

        keep_cols = [
            "team_code",
            "canonical_name",
            "last_7_fppg",
            "last_14_fppg",
            "season_fppg",
        ]
        trimmed = entries[keep_cols].drop_duplicates(
            subset=["team_code", "canonical_name"], keep="first"
        )
        renamed = trimmed.rename(
            columns={
                "last_7_fppg": "recent_last7_fppg",
                "last_14_fppg": "recent_last14_fppg",
                "season_fppg": "recent_season_fppg",
            }
        )
        return RecentStatsTable(entries=renamed, source_path=self.csv_path)


__all__ = ["RecentStatsLoader", "RecentStatsTable"]

