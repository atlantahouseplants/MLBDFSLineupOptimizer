"""Learn field construction and duplication tendencies from contest exports."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class FieldDuplicationProfile:
    samples: int
    unique_lineups: int
    duplicate_lineup_pct: float
    avg_duplicate_count: float
    max_duplicate_count: int
    avg_salary_used: float
    full_salary_pct: float
    top_stack_share: float
    source: str = "contest_csv"

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def learn_field_duplication_profile(
    contest_df: pd.DataFrame,
    optimizer_df: pd.DataFrame | None = None,
    source: str = "contest_csv",
) -> FieldDuplicationProfile:
    lineup_players = extract_contest_lineup_players(contest_df)
    if lineup_players.empty:
        return FieldDuplicationProfile(0, 0, 0.0, 1.0, 1, 0.0, 0.0, 0.0, source)

    signatures = lineup_players["signature"].astype(str)
    counts = signatures.value_counts()
    dupe_counts = signatures.map(counts).astype(int)
    samples = int(len(lineup_players))
    unique_lineups = int(counts.size)
    duplicate_lineup_pct = float((dupe_counts > 1).mean()) if samples else 0.0
    avg_duplicate_count = float(dupe_counts.mean()) if samples else 1.0
    max_duplicate_count = int(dupe_counts.max()) if samples else 1

    salary_used = pd.Series(dtype=float)
    top_stack_share = 0.0
    if optimizer_df is not None and not optimizer_df.empty:
        meta = _player_meta(optimizer_df)
        enriched = _enrich_lineup_players(lineup_players, meta)
        if "salary_used" in enriched.columns:
            salary_used = pd.to_numeric(enriched["salary_used"], errors="coerce").dropna()
        if "primary_stack_team" in enriched.columns and not enriched["primary_stack_team"].empty:
            top_stack_share = float(enriched["primary_stack_team"].value_counts(normalize=True).iloc[0])

    avg_salary = float(salary_used.mean()) if not salary_used.empty else 0.0
    full_salary_pct = float((salary_used >= 34_700).mean()) if not salary_used.empty else 0.0
    return FieldDuplicationProfile(
        samples=samples,
        unique_lineups=unique_lineups,
        duplicate_lineup_pct=duplicate_lineup_pct,
        avg_duplicate_count=avg_duplicate_count,
        max_duplicate_count=max_duplicate_count,
        avg_salary_used=avg_salary,
        full_salary_pct=full_salary_pct,
        top_stack_share=top_stack_share,
        source=source,
    )


def extract_contest_lineup_players(contest_df: pd.DataFrame) -> pd.DataFrame:
    if contest_df is None or contest_df.empty:
        return pd.DataFrame(columns=["row_id", "players", "signature"])
    player_cols = _player_columns(contest_df.columns)
    rows = []
    for idx, row in contest_df.iterrows():
        players: list[str] = []
        if player_cols:
            for col in player_cols:
                players.extend(_split_player_cell(row.get(col)))
        else:
            for value in row.tolist():
                players.extend(_split_player_cell(value))
        players = [_canonical_player_name(player) for player in players if _canonical_player_name(player)]
        players = sorted(set(players))
        if len(players) >= 6:
            rows.append({"row_id": idx, "players": players, "signature": "|".join(players)})
    return pd.DataFrame(rows)


def compute_actual_ownership(
    contest_df: pd.DataFrame,
    optimizer_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Exact actual ownership from a FanDuel contest-results export.

    Every entrant's lineup is in the export, so ownership is simply
    appearances / entries — no projection involved. Returns a DataFrame with
    ``player_name``, ``actual_ownership_pct`` (decimals, 0-1) and, when
    *optimizer_df* is provided, ``fd_player_id`` resolved by canonical name.
    """
    lineup_players = extract_contest_lineup_players(contest_df)
    if lineup_players.empty:
        return pd.DataFrame(columns=["fd_player_id", "player_name", "actual_ownership_pct"])

    entries = len(lineup_players)
    counts: dict[str, int] = {}
    for players in lineup_players["players"]:
        for player in players:
            counts[player] = counts.get(player, 0) + 1

    ownership = pd.DataFrame(
        {
            "_name": list(counts.keys()),
            "actual_ownership_pct": [count / entries for count in counts.values()],
        }
    )

    if optimizer_df is not None and not optimizer_df.empty and "full_name" in optimizer_df.columns:
        meta = optimizer_df[["full_name"] + (["fd_player_id"] if "fd_player_id" in optimizer_df.columns else [])].copy()
        meta["_name"] = meta["full_name"].astype(str).map(_canonical_player_name)
        meta = meta[meta["_name"] != ""].drop_duplicates("_name")
        ownership = ownership.merge(meta, on="_name", how="left")
        ownership["player_name"] = ownership["full_name"].fillna(ownership["_name"])
        ownership = ownership.drop(columns=[col for col in ("full_name",) if col in ownership.columns])
    else:
        ownership["player_name"] = ownership["_name"]

    if "fd_player_id" not in ownership.columns:
        ownership["fd_player_id"] = pd.NA
    ownership["fd_player_id"] = ownership["fd_player_id"].astype("string")
    return ownership.drop(columns=["_name"]).sort_values(
        "actual_ownership_pct", ascending=False
    ).reset_index(drop=True)[["fd_player_id", "player_name", "actual_ownership_pct"]]


def _parse_money(value) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).replace("$", "").replace(",", "").strip()
    if not text or text.lower() in {"nan", "none", ""}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def match_lineups_to_contest_entries(
    contest_df: pd.DataFrame,
    lineup_df: pd.DataFrame,
) -> pd.DataFrame:
    """Find our generated lineups inside a contest-results export by player set.

    Matches on the canonical 9-player signature, so it needs no username or
    entry-id knowledge. Returns a DataFrame with lineup_id plus whichever of
    rank / payout the export carries (best rank / payout among duplicate
    field copies of the same lineup — duplicates score identically anyway).
    """
    empty = pd.DataFrame(columns=["lineup_id", "rank", "payout"])
    if contest_df is None or contest_df.empty or lineup_df is None or lineup_df.empty:
        return empty
    if "lineup_id" not in lineup_df.columns or "full_name" not in lineup_df.columns:
        return empty

    entries = extract_contest_lineup_players(contest_df)
    if entries.empty:
        return empty

    rank_col = points_col = money_col = None
    for col in contest_df.columns:
        norm = _normalize_col(col)
        if rank_col is None and "rank" in norm:
            rank_col = col
        elif money_col is None and any(tok in norm for tok in ("winning", "prize", "won", "payout")):
            money_col = col
        elif points_col is None and norm in {"points", "score", "fpts", "total_points"}:
            points_col = col

    by_signature: dict[str, dict] = {}
    for _, entry in entries.iterrows():
        row = contest_df.loc[entry["row_id"]]
        rank = pd.to_numeric(row.get(rank_col), errors="coerce") if rank_col else None
        payout = _parse_money(row.get(money_col)) if money_col else None
        bucket = by_signature.setdefault(entry["signature"], {"rank": None, "payout": None})
        if rank is not None and not pd.isna(rank):
            bucket["rank"] = int(rank) if bucket["rank"] is None else min(bucket["rank"], int(rank))
        if payout is not None:
            bucket["payout"] = payout if bucket["payout"] is None else max(bucket["payout"], payout)

    # Inverted index for subset matching: exports sometimes surface fewer than
    # 9 recognizable names per entry, so exact-signature equality is backed up
    # by "entry players are a subset of the lineup" (still >= 6 shared names).
    lineup_names: dict = {}
    name_to_lineups: dict[str, set] = {}
    for lineup_id, group in lineup_df.groupby("lineup_id"):
        names = {
            _canonical_player_name(name)
            for name in group["full_name"].astype(str)
            if _canonical_player_name(name)
        }
        lineup_names[lineup_id] = names
        for name in names:
            name_to_lineups.setdefault(name, set()).add(lineup_id)
    signature_by_lineup = {
        lineup_id: "|".join(sorted(names)) for lineup_id, names in lineup_names.items()
    }

    matches: dict = {}
    for lineup_id in lineup_names:
        match = by_signature.get(signature_by_lineup[lineup_id])
        if match is not None:
            matches[lineup_id] = match

    # Subset fallback for entries whose extraction surfaced fewer than 9
    # recognizable names: walk entries once, seed candidate lineups from the
    # inverted name index, accept if all extracted players are in the lineup.
    unmatched = set(lineup_names) - set(matches)
    if unmatched:
        for signature, match in by_signature.items():
            if not unmatched:
                break
            players = set(signature.split("|"))
            if len(players) < 6:
                continue
            seed = next(iter(players))
            for lineup_id in name_to_lineups.get(seed, ()):  # only lineups sharing a player
                if lineup_id in unmatched and players <= lineup_names[lineup_id]:
                    matches[lineup_id] = match
                    unmatched.discard(lineup_id)
                    break

    rows = [
        {"lineup_id": lineup_id, "rank": match["rank"], "payout": match["payout"]}
        for lineup_id, match in matches.items()
    ]
    return pd.DataFrame(rows, columns=["lineup_id", "rank", "payout"]) if rows else empty


def save_field_duplication_profile(profile: FieldDuplicationProfile, path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")


def load_field_duplication_profile(path: Path | str) -> dict[str, float | int | str]:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _player_columns(columns: Iterable) -> list[str]:
    roster_tokens = {"p", "c", "c_1b", "1b", "2b", "3b", "ss", "of", "util"}
    # Duplicate roster columns come through with suffixes (OF.1 -> of_1).
    roster_pattern = re.compile(r"^(p|c|c_1b|1b|2b|3b|ss|of|util)(_?\d+)?$")
    result = []
    for col in columns:
        norm = _normalize_col(col)
        if norm in roster_tokens or roster_pattern.match(norm) or norm.startswith("player") or norm.endswith("_player"):
            if not any(skip in norm for skip in ["rank", "score", "points", "owner", "entry"]):
                result.append(col)
    return result


def _split_player_cell(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    pieces = re.split(r"[|;/]", text)
    if len(pieces) == 1 and "," in text and len(text.split(",")) >= 6:
        pieces = text.split(",")
    return [piece.strip() for piece in pieces if piece.strip()]


def _canonical_player_name(value: str) -> str:
    text = re.sub(r"\([^)]*\)", "", str(value))
    text = re.sub(r"\$?\d+(?:\.\d+)?%?", "", text)
    text = re.sub(r"[^a-zA-Z .'\\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    if len(text) < 3 or text in {"p", "of", "util"}:
        return ""
    return text


def _normalize_col(value) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _player_meta(optimizer_df: pd.DataFrame) -> pd.DataFrame:
    meta = optimizer_df.copy()
    meta["_name"] = meta.get("full_name", "").astype(str).map(_canonical_player_name)
    keep = ["_name", "salary", "team_code", "player_type"]
    return meta[[col for col in keep if col in meta.columns]].drop_duplicates("_name")


def _enrich_lineup_players(lineups: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    meta_map = meta.set_index("_name").to_dict(orient="index") if not meta.empty else {}
    rows = []
    for _, row in lineups.iterrows():
        salary = 0.0
        team_counts: dict[str, int] = {}
        for player in row.get("players", []):
            info = meta_map.get(player) or {}
            salary += float(info.get("salary") or 0.0)
            if str(info.get("player_type", "")).lower() != "pitcher":
                team = str(info.get("team_code", "")).upper()
                if team:
                    team_counts[team] = team_counts.get(team, 0) + 1
        primary_team = ""
        primary_size = 0
        if team_counts:
            primary_team, primary_size = sorted(team_counts.items(), key=lambda item: item[1], reverse=True)[0]
        rows.append(
            {
                "row_id": row.get("row_id"),
                "salary_used": salary if salary > 0 else np.nan,
                "primary_stack_team": primary_team if primary_size >= 3 else "",
                "primary_stack_size": primary_size,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "FieldDuplicationProfile",
    "compute_actual_ownership",
    "extract_contest_lineup_players",
    "learn_field_duplication_profile",
    "load_field_duplication_profile",
    "match_lineups_to_contest_entries",
    "save_field_duplication_profile",
]
