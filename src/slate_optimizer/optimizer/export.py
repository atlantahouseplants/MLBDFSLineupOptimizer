"""FanDuel lineup export helpers bound to optimizer outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from .solver import LineupResult

FANDUEL_UPLOAD_COLUMNS = ["P", "C/1B", "2B", "3B", "SS", "OF", "OF", "OF", "UTIL"]


def _parse_positions(value: str) -> List[str]:
    if not isinstance(value, str):
        return []
    tokens = value.upper().replace("-", "/").split("/")
    return [token.strip() for token in tokens if token.strip()]


def _pop_candidate(rows: List[Dict], tokens: Sequence[str]) -> Dict:
    desired = {token.upper() for token in tokens if token}
    if not desired:
        raise ValueError("No tokens supplied for slot selection")
    best_idx = None
    best_len = None
    for idx, row in enumerate(rows):
        positions = set(_parse_positions(row.get("position")))
        if positions and positions.intersection(desired):
            length = len(positions) or 1
            if best_idx is None or length < best_len:
                best_idx = idx
                best_len = length
    if best_idx is None:
        raise ValueError(f"Unable to fill slot with tokens={tuple(desired)}")
    return rows.pop(best_idx)


def _pop_outfielder(rows: List[Dict]) -> Dict:
    return _pop_candidate(rows, ["OF"])


def _pop_util(rows: List[Dict]) -> Dict:
    hitters = [
        (idx, row)
        for idx, row in enumerate(rows)
        if str(row.get("player_type", "")).lower() != "pitcher"
    ]
    if not hitters:
        raise ValueError("Unable to assign UTIL slot (no hitters remaining)")
    idx, _ = hitters[0]
    return rows.pop(idx)


def _lineup_to_row(lineup_df: pd.DataFrame) -> List[str]:
    required_cols = {"fd_player_id", "position", "player_type"}
    missing = required_cols.difference(lineup_df.columns)
    if missing:
        raise ValueError(f"Lineup dataframe missing required columns: {sorted(missing)}")

    rows = lineup_df.to_dict("records")
    for row in rows:
        row["fd_player_id"] = str(row.get("fd_player_id", "")).strip()
        row["position"] = str(row.get("position", ""))
        row["player_type"] = str(row.get("player_type", ""))

    ordered: List[str] = []
    ordered.append(_pop_candidate(rows, ["P"])["fd_player_id"])
    ordered.append(_pop_candidate(rows, ["C", "1B"])["fd_player_id"])
    ordered.append(_pop_candidate(rows, ["2B"])["fd_player_id"])
    ordered.append(_pop_candidate(rows, ["3B"])["fd_player_id"])
    ordered.append(_pop_candidate(rows, ["SS"])["fd_player_id"])
    ordered.append(_pop_outfielder(rows)["fd_player_id"])
    ordered.append(_pop_outfielder(rows)["fd_player_id"])
    ordered.append(_pop_outfielder(rows)["fd_player_id"])
    util_player = _pop_util(rows)
    if str(util_player.get("player_type", "")).lower() == "pitcher":
        raise ValueError("Pitcher cannot occupy UTIL slot")
    ordered.append(util_player["fd_player_id"])
    return ordered


def lineups_to_fanduel_upload(lineups: Sequence[LineupResult]) -> pd.DataFrame:
    if not lineups:
        return pd.DataFrame(columns=FANDUEL_UPLOAD_COLUMNS)
    rows = [_lineup_to_row(lineup.dataframe) for lineup in lineups]
    upload_df = pd.DataFrame(rows, columns=FANDUEL_UPLOAD_COLUMNS)
    return upload_df


def write_fanduel_upload(lineups: Sequence[LineupResult], output_path: Path | str) -> pd.DataFrame:
    upload_df = lineups_to_fanduel_upload(lineups)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    upload_df.to_csv(output, index=False)
    return upload_df


__all__ = [
    "FANDUEL_UPLOAD_COLUMNS",
    "lineups_to_fanduel_upload",
    "write_fanduel_upload",
]
