"""Loader utilities for batting order CSVs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .aliases import CanonicalMap, apply_aliases
from .text_utils import canonicalize_series


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
        entries = entries[entries["batting_order_position"].between(1, 9, inclusive="both")]
        entries = entries.drop_duplicates(["team_code", "canonical_name"], keep="first")
        keep_cols = [
            "team_code",
            "canonical_name",
            "batting_order_position",
        ]
        return BattingOrderTable(entries=entries[keep_cols], source_path=self.csv_path)


__all__ = ["BattingOrderLoader", "BattingOrderTable"]

