"""FanDuel lineup export helpers bound to optimizer outputs."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .solver import LineupResult

FANDUEL_UPLOAD_COLUMNS = ["P", "C/1B", "2B", "3B", "SS", "OF", "OF", "OF", "UTIL"]
FANDUEL_TEMPLATE_META_COLUMNS = ["entry_id", "contest_id", "contest_name", "entry_fee"]


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
        raw_id = row.get("fd_player_id", "")
        try:
            row["fd_player_id"] = str(int(float(raw_id)))
        except (ValueError, TypeError):
            row["fd_player_id"] = str(raw_id).strip()
        # Prefer roster_position (full FanDuel eligibility) over position
        roster_pos = str(row.get("roster_position", "")).strip()
        raw_pos = str(row.get("position", "")).strip()
        row["position"] = roster_pos if roster_pos and roster_pos.upper() not in ("", "NAN", "NONE") else raw_pos
        row["player_type"] = str(row.get("player_type", ""))

    # Assign slots using backtracking to avoid greedy ordering issues
    slots = [
        ("P", ["P"]),
        ("C/1B", ["C", "1B"]),
        ("2B", ["2B"]),
        ("3B", ["3B"]),
        ("SS", ["SS"]),
        ("OF1", ["OF"]),
        ("OF2", ["OF"]),
        ("OF3", ["OF"]),
    ]

    def _fits(row: Dict, tokens: List[str]) -> bool:
        positions = set(_parse_positions(row.get("position")))
        return bool(positions.intersection(t.upper() for t in tokens))

    def _backtrack(remaining: List[Dict], slot_idx: int, assigned: List[Dict]) -> bool:
        if slot_idx == len(slots):
            return True
        _, tokens = slots[slot_idx]
        for i, row in enumerate(remaining):
            if _fits(row, tokens):
                rest = remaining[:i] + remaining[i + 1:]
                assigned.append(row)
                if _backtrack(rest, slot_idx + 1, assigned):
                    return True
                assigned.pop()
        return False

    assigned: List[Dict] = []
    if not _backtrack(rows, 0, assigned):
        raise ValueError("Unable to assign all players to valid FanDuel slots")

    # The UTIL slot gets whoever is left
    assigned_ids = {r["fd_player_id"] for r in assigned}
    util_candidates = [r for r in rows if r["fd_player_id"] not in assigned_ids]
    if not util_candidates:
        raise ValueError("No player remaining for UTIL slot")
    util_player = util_candidates[0]
    if str(util_player.get("player_type", "")).lower() == "pitcher":
        raise ValueError("Pitcher cannot occupy UTIL slot")

    ordered = [r["fd_player_id"] for r in assigned] + [util_player["fd_player_id"]]
    return ordered


def lineups_to_fanduel_upload(lineups: Sequence[LineupResult]) -> pd.DataFrame:
    if not lineups:
        return pd.DataFrame(columns=FANDUEL_UPLOAD_COLUMNS)
    rows: List[List[str]] = []
    skipped = 0
    for lineup in lineups:
        try:
            rows.append(_lineup_to_row(lineup.dataframe))
        except ValueError:
            skipped += 1
    if skipped:
        import sys
        print(f"Warning: skipped {skipped}/{len(lineups)} lineups with invalid position assignments", file=sys.stderr)
    upload_df = pd.DataFrame(rows, columns=FANDUEL_UPLOAD_COLUMNS)
    return upload_df


def write_fanduel_upload(lineups: Sequence[LineupResult], output_path: Path | str) -> pd.DataFrame:
    upload_df = lineups_to_fanduel_upload(lineups)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    upload_df.to_csv(output, index=False)
    return upload_df


def extract_template_entries(csv_path: Path | str) -> Optional[pd.DataFrame]:
    """Extract entry metadata rows from a FanDuel upload template CSV.

    Returns a DataFrame with entry_id, contest_id, contest_name, entry_fee
    columns, or None if the file is not an upload template.
    """
    csv_path = Path(csv_path)
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        if not header or header[0].strip().lower() != "entry_id":
            return None
        entries: List[Dict[str, str]] = []
        for row in reader:
            if len(row) < 4:
                continue
            entry_id = row[0].strip().strip('"')
            if not entry_id:
                continue
            entries.append({
                "entry_id": entry_id,
                "contest_id": row[1].strip().strip('"'),
                "contest_name": row[2].strip().strip('"'),
                "entry_fee": row[3].strip().strip('"'),
            })
    if not entries:
        return None
    return pd.DataFrame(entries)


def lineups_to_fanduel_template(
    lineups: Sequence[LineupResult],
    template_entries: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build a FanDuel-ready upload CSV with entry metadata.

    If *template_entries* is provided (from extract_template_entries), the
    output includes entry_id/contest_id/contest_name/entry_fee columns so
    the CSV can be uploaded directly to FanDuel without manual copy-paste.
    Lineups are assigned round-robin to entries when there are more entries
    than lineups.
    """
    base_df = lineups_to_fanduel_upload(lineups)
    if base_df.empty:
        return base_df

    if template_entries is None or template_entries.empty:
        return base_df

    n_entries = len(template_entries)
    n_lineups = len(base_df)

    # Repeat lineups round-robin if fewer lineups than entries
    if n_lineups < n_entries:
        repeats = (n_entries // n_lineups) + 1
        base_df = pd.concat([base_df] * repeats, ignore_index=True).iloc[:n_entries]
    elif n_lineups > n_entries:
        base_df = base_df.iloc[:n_entries]

    base_df = base_df.reset_index(drop=True)
    template_entries = template_entries.reset_index(drop=True)

    result = pd.concat([template_entries, base_df], axis=1)
    return result


__all__ = [
    "FANDUEL_UPLOAD_COLUMNS",
    "extract_template_entries",
    "lineups_to_fanduel_upload",
    "lineups_to_fanduel_template",
    "write_fanduel_upload",
]
