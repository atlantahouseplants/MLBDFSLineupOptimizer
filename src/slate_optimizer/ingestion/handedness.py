"""Loader utilities for batter/pitcher handedness reference data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .aliases import CanonicalMap, apply_aliases
from .text_utils import canonicalize_series


@dataclass
class HandednessTable:
    entries: pd.DataFrame
    source_path: Path

    def summary(self) -> dict[str, int]:
        df = self.entries
        return {"total": len(df), "teams": df["team_code"].nunique()}


class HandednessLoader:
    """Loads player handedness reference data."""

    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path).expanduser().resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Handedness CSV not found: {self.csv_path}")

    def load(self, alias_map: Optional[CanonicalMap] = None) -> HandednessTable:
        df = pd.read_csv(self.csv_path)
        required = {"player_name", "team"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Handedness CSV missing required columns: {sorted(missing)}"
            )

        entries = df.copy()
        entries["team_code"] = entries["team"].astype(str).str.upper().str.strip()
        entries["player_name"] = entries["player_name"].astype(str).str.strip()
        canonical = canonicalize_series(entries["player_name"])
        if alias_map:
            canonical = apply_aliases(canonical, alias_map)
        entries["canonical_name"] = canonical

        def _normalize_hand(series_name: str) -> pd.Series:
            values = entries.get(series_name)
            if values is None:
                return pd.Series(["" for _ in range(len(entries))])
            return values.astype(str).str.upper().str.strip().str[:1]

        entries["batter_hand"] = _normalize_hand("bats").where(
            lambda s: s.isin(["L", "R", "S"]), ""
        )
        entries["pitcher_hand"] = _normalize_hand("throws").where(
            lambda s: s.isin(["L", "R"]), ""
        )

        keep_cols = [
            "team_code",
            "canonical_name",
            "batter_hand",
            "pitcher_hand",
        ]
        trimmed = entries[keep_cols].drop_duplicates(
            subset=["team_code", "canonical_name"], keep="first"
        )
        return HandednessTable(entries=trimmed, source_path=self.csv_path)


__all__ = ["HandednessLoader", "HandednessTable"]

