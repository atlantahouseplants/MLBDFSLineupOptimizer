"""FanDuel lineup export helpers."""
from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

SLOT_ORDER = ["P", "C1", "SS", "3B", "2B", "OF1", "OF2", "OF3", "UTIL"]


def _parse_positions(value: str) -> List[str]:
    if not isinstance(value, str):
        return []
    tokens = value.upper().replace("-", "/").split("/")
    return [tok.strip() for tok in tokens if tok.strip()]


def _pop_candidate(rows: List[Dict], token: str) -> Dict:
    candidates = [
        (idx, row)
        for idx, row in enumerate(rows)
        if token.upper() in set(_parse_positions(row.get("position")))
    ]
    if not candidates:
        raise ValueError(f"No candidate available for slot {token}")
    idx, row = min(candidates, key=lambda item: len(_parse_positions(item[1].get("position"))) or 1)
    return rows.pop(idx)


def _pop_with_fallback(rows: List[Dict], tokens: Iterable[str]) -> Dict:
    for token in tokens:
        try:
            return _pop_candidate(rows, token)
        except ValueError:
            continue
    raise ValueError("Unable to fill slot with provided tokens")


def lineup_to_fanduel_row(lineup_df: pd.DataFrame) -> Dict[str, str]:
    rows = lineup_df.to_dict("records")
    result: Dict[str, str] = {}

    result["P"] = _pop_with_fallback(rows, ["P"])["full_name"]
    result["C1"] = _pop_with_fallback(rows, ["C", "C1", "1B"])["full_name"]
    result["SS"] = _pop_with_fallback(rows, ["SS"])["full_name"]
    result["3B"] = _pop_with_fallback(rows, ["3B"])["full_name"]
    result["2B"] = _pop_with_fallback(rows, ["2B"])["full_name"]
    for slot in ["OF1", "OF2", "OF3"]:
        result[slot] = _pop_with_fallback(rows, ["OF"])["full_name"]
    result["UTIL"] = rows.pop(0)["full_name"]
    return result


def build_upload_dataframe(lineups_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lineup_id, group in lineups_df.groupby("lineup_id"):
        lineup_row = lineup_to_fanduel_row(group)
        lineup_row["lineup_id"] = lineup_id
        rows.append(lineup_row)
    upload_df = pd.DataFrame(rows)
    upload_df = upload_df[["lineup_id"] + SLOT_ORDER]
    return upload_df.sort_values("lineup_id").reset_index(drop=True)


__all__ = ["build_upload_dataframe", "lineup_to_fanduel_row", "SLOT_ORDER"]
