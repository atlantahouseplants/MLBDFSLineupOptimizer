"""Vegas line ingestion utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

_REQUIRED_COLUMNS = {"game", "total", "home_ml", "away_ml"}
_COLUMN_NORMALIZE = {
    "game": "game",
    "matchup": "game",
    "total": "total",
    "over_under": "total",
    "home_ml": "home_ml",
    "home_moneyline": "home_ml",
    "away_ml": "away_ml",
    "away_moneyline": "away_ml",
}


def _normalize_column(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _moneyline_to_prob(value) -> float:
    try:
        line = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if line < 0:
        return (-line) / ((-line) + 100)
    if line > 0:
        return 100 / (line + 100)
    return float("nan")


def _parse_game(value: str) -> Tuple[str, str]:
    if not isinstance(value, str) or "@" not in value:
        return "", ""
    away, home = value.split("@", 1)
    return away.strip().upper(), home.strip().upper()


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for column in df.columns:
        normalized = _normalize_column(str(column))
        if normalized in _COLUMN_NORMALIZE:
            rename_map[column] = _COLUMN_NORMALIZE[normalized]
    standardized = df.rename(columns=rename_map)
    missing = _REQUIRED_COLUMNS - set(standardized.columns)
    if missing:
        raise ValueError(f"Vegas CSV missing required columns: {sorted(missing)}")
    return standardized


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


@dataclass
class VegasLines:
    """Container for raw game lines plus per-team implied totals."""

    games: pd.DataFrame
    team_totals: pd.DataFrame

    def summary(self) -> Dict[str, int]:
        return {
            "games": len(self.games),
            "teams": len(self.team_totals),
        }


class VegasLoader:
    """Loads a CSV of Vegas lines and derives implied team totals."""

    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path).expanduser().resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Vegas CSV not found: {self.csv_path}")

    def load(self) -> VegasLines:
        df = pd.read_csv(self.csv_path)
        df = _standardize_columns(df)
        df["game"] = df["game"].astype(str).str.strip()
        df["total"] = _clean_numeric(df["total"]).clip(lower=0)
        df["home_ml"] = _clean_numeric(df["home_ml"])
        df["away_ml"] = _clean_numeric(df["away_ml"])

        away_teams: list[str] = []
        home_teams: list[str] = []
        for value in df["game"]:
            away, home = _parse_game(value)
            away_teams.append(away)
            home_teams.append(home)
        df["away_team"] = away_teams
        df["home_team"] = home_teams

        df["away_prob_raw"] = df["away_ml"].map(_moneyline_to_prob)
        df["home_prob_raw"] = df["home_ml"].map(_moneyline_to_prob)

        df["away_prob"] = 0.0
        df["home_prob"] = 0.0
        prob_sum = df[["away_prob_raw", "home_prob_raw"]].fillna(0).sum(axis=1)
        valid_mask = prob_sum > 0
        df.loc[valid_mask, "away_prob"] = (
            df.loc[valid_mask, "away_prob_raw"].fillna(0) / prob_sum.loc[valid_mask]
        )
        df.loc[valid_mask, "home_prob"] = (
            df.loc[valid_mask, "home_prob_raw"].fillna(0) / prob_sum.loc[valid_mask]
        )
        df.loc[~valid_mask, ["away_prob", "home_prob"]] = 0.5

        df["away_implied_total"] = (df["total"] * df["away_prob"]).fillna(0.0)
        df["home_implied_total"] = (df["total"] * df["home_prob"]).fillna(0.0)
        df["game_code"] = df["away_team"].str.cat(df["home_team"], sep="@")

        team_rows = []
        for _, row in df.iterrows():
            away_team = row.get("away_team", "")
            home_team = row.get("home_team", "")
            if not away_team or not home_team:
                continue
            game_total = row.get("total")
            away_total = row.get("away_implied_total")
            home_total = row.get("home_implied_total")
            away_prob = row.get("away_prob")
            home_prob = row.get("home_prob")
            team_rows.append(
                {
                    "team_code": away_team,
                    "opponent_code": home_team,
                    "vegas_game_code": row.get("game_code"),
                    "vegas_game_total": game_total,
                    "vegas_team_total": away_total,
                    "vegas_opponent_total": home_total,
                    "vegas_moneyline": row.get("away_ml"),
                    "vegas_implied_win_prob": away_prob,
                }
            )
            team_rows.append(
                {
                    "team_code": home_team,
                    "opponent_code": away_team,
                    "vegas_game_code": row.get("game_code"),
                    "vegas_game_total": game_total,
                    "vegas_team_total": home_total,
                    "vegas_opponent_total": away_total,
                    "vegas_moneyline": row.get("home_ml"),
                    "vegas_implied_win_prob": home_prob,
                }
            )

        team_totals = pd.DataFrame(team_rows)
        return VegasLines(games=df, team_totals=team_totals)


__all__ = ["VegasLoader", "VegasLines"]

