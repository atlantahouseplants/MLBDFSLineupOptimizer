"""FanDuel lineup export helpers bound to optimizer outputs."""
from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def validate_fanduel_lineups(lineups: Sequence[LineupResult]) -> List[Tuple[int, str]]:
    errors: List[Tuple[int, str]] = []
    for lineup_number, lineup in enumerate(lineups, start=1):
        try:
            _lineup_to_row(lineup.dataframe)
        except ValueError as exc:
            errors.append((lineup_number, str(exc)))
    return errors


def lineups_to_fanduel_upload(lineups: Sequence[LineupResult], *, strict: bool = False) -> pd.DataFrame:
    if not lineups:
        return pd.DataFrame(columns=FANDUEL_UPLOAD_COLUMNS)
    rows: List[List[str]] = []
    errors: List[Tuple[int, str]] = []
    for lineup_number, lineup in enumerate(lineups, start=1):
        try:
            rows.append(_lineup_to_row(lineup.dataframe))
        except ValueError as exc:
            errors.append((lineup_number, str(exc)))
    if errors:
        first_lineup, first_error = errors[0]
        message = (
            f"Skipped {len(errors)} FanDuel-invalid lineup(s) during export. "
            f"First skipped lineup #{first_lineup}: {first_error}"
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    # Invalid lineups are dropped only after warning or raising in strict mode.
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


def assign_lineups_to_contests(
    n_lineups: int,
    template_entries: pd.DataFrame,
) -> Dict[str, List[Optional[int]]]:
    """Deal lineup indexes across the contests in an entries template.

    Phase 1 deals unique lineups round-robin across contests so every contest
    inherits the portfolio's exposure profile. Phase 2 (shortage) reuses
    lineups across contests — legal on FanDuel — but never repeats a lineup
    within the same contest. Entries that still cannot be filled get None.

    Returns {contest_id: [lineup_index or None, ...]} with one slot per entry,
    contests in first-appearance order.
    """
    contest_ids = list(dict.fromkeys(template_entries["contest_id"].astype(str)))
    entry_counts = template_entries["contest_id"].astype(str).value_counts().to_dict()
    assigned: Dict[str, List[Optional[int]]] = {cid: [] for cid in contest_ids}

    next_lineup = 0
    active = [cid for cid in contest_ids if entry_counts.get(cid, 0) > 0]
    while active and next_lineup < n_lineups:
        for cid in list(active):
            if next_lineup >= n_lineups:
                break
            assigned[cid].append(next_lineup)
            next_lineup += 1
            if len(assigned[cid]) >= entry_counts[cid]:
                active.remove(cid)

    # Shortage: reuse lineups from other contests, skipping any already in this one.
    for cid in contest_ids:
        used = set(assigned[cid])
        candidates = (idx for idx in range(n_lineups) if idx not in used)
        while len(assigned[cid]) < entry_counts.get(cid, 0):
            idx = next(candidates, None)
            if idx is None:
                break
            assigned[cid].append(idx)
        while len(assigned[cid]) < entry_counts.get(cid, 0):
            assigned[cid].append(None)
    return assigned


def lineups_to_fanduel_template(
    lineups: Sequence[LineupResult],
    template_entries: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build a FanDuel-ready upload CSV with entry metadata.

    If *template_entries* is provided (from extract_template_entries), the
    output includes entry_id/contest_id/contest_name/entry_fee columns so the
    CSV can be uploaded directly to FanDuel. When the template spans multiple
    contests, lineups are dealt round-robin across contests and a lineup is
    never assigned twice within the same contest; entries that cannot be
    filled without an in-contest duplicate are omitted from the output.
    """
    base_df = lineups_to_fanduel_upload(lineups)
    if base_df.empty:
        return base_df

    if template_entries is None or template_entries.empty:
        return base_df

    template_entries = template_entries.reset_index(drop=True)
    assignment = assign_lineups_to_contests(len(base_df), template_entries)

    # Walk entries in template order, consuming each contest's dealt lineups.
    # Rows are built positionally because the upload columns contain
    # duplicate labels (three OF slots).
    cursors: Dict[str, int] = {cid: 0 for cid in assignment}
    rows: List[List] = []
    for _, entry in template_entries.iterrows():
        cid = str(entry["contest_id"])
        slot = cursors.get(cid, 0)
        cursors[cid] = slot + 1
        contest_slots = assignment.get(cid, [])
        lineup_idx = contest_slots[slot] if slot < len(contest_slots) else None
        if lineup_idx is None:
            continue
        rows.append(list(entry.values) + list(base_df.iloc[lineup_idx].values))

    columns = list(template_entries.columns) + list(base_df.columns)
    return pd.DataFrame(rows, columns=columns)


__all__ = [
    "FANDUEL_UPLOAD_COLUMNS",
    "assign_lineups_to_contests",
    "extract_template_entries",
    "lineups_to_fanduel_upload",
    "lineups_to_fanduel_template",
    "validate_fanduel_lineups",
    "write_fanduel_upload",
]
