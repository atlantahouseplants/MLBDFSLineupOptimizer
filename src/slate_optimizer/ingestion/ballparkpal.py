"""Utilities for loading BallparkPal Excel outputs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import pandas as pd

_BPP_FILE_STEMS = {
    "batters": "BallparkPal_Batters",
    "pitchers": "BallparkPal_Pitchers",
    "games": "BallparkPal_Games",
    "teams": "BallparkPal_Teams",
}


@dataclass(frozen=True)
class BallparkPalPaths:
    """Concrete file paths for a single BallparkPal export bundle."""

    batters: Path
    pitchers: Path
    games: Path
    teams: Path


def _normalize_column(name: str) -> str:
    """Convert BallparkPal column names to snake_case."""
    cleaned = name.strip().replace("%", "pct").replace("#", "num")
    buffer = []
    for char in cleaned:
        if char in " /()-":
            buffer.append("_")
        elif char == "\n":
            buffer.append("_")
        elif char.isupper() and buffer and buffer[-1].isalnum() and buffer[-1].islower():
            buffer.append("_" + char.lower())
        else:
            buffer.append(char.lower())
    normalized = "".join(buffer)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    standardized = df.copy()
    standardized.columns = [_normalize_column(col) for col in standardized.columns]
    return standardized


@dataclass
class BallparkPalBundle:
    """In-memory representation of the BallparkPal worksheets."""

    batters: pd.DataFrame
    pitchers: pd.DataFrame
    games: pd.DataFrame
    teams: pd.DataFrame

    def frames(self) -> Dict[str, pd.DataFrame]:
        return {
            "batters": self.batters,
            "pitchers": self.pitchers,
            "games": self.games,
            "teams": self.teams,
        }

    def summary(self) -> Dict[str, int]:
        return {name: len(frame) for name, frame in self.frames().items()}


class BallparkPalLoader:
    """Loads and normalizes BallparkPal Excel exports."""

    def __init__(self, source_dir: Path):
        self.source_dir = Path(source_dir).expanduser().resolve()
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory {self.source_dir} does not exist")

    def _latest_file(self, stem: str) -> Path:
        candidates = sorted(
            self.source_dir.glob(f"{stem}*.xlsx"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No Excel files matching '{stem}*.xlsx' found in {self.source_dir}"
            )
        return candidates[0]

    def resolve_paths(self, overrides: Optional[Mapping[str, Path]] = None) -> BallparkPalPaths:
        overrides = overrides or {}
        resolved: Dict[str, Path] = {}
        for key, stem in _BPP_FILE_STEMS.items():
            if key in overrides and overrides[key] is not None:
                resolved[key] = Path(overrides[key]).expanduser().resolve()
            else:
                resolved[key] = self._latest_file(stem)
        return BallparkPalPaths(**resolved)

    def load_bundle(self, overrides: Optional[Mapping[str, Path]] = None) -> BallparkPalBundle:
        paths = self.resolve_paths(overrides)
        batters = _standardize_dataframe(pd.read_excel(paths.batters))
        pitchers = _standardize_dataframe(pd.read_excel(paths.pitchers))
        games = _standardize_dataframe(pd.read_excel(paths.games))
        teams = _standardize_dataframe(pd.read_excel(paths.teams))
        return BallparkPalBundle(batters=batters, pitchers=pitchers, games=games, teams=teams)
