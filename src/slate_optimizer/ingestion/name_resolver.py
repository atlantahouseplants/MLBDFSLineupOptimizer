"""Player-name alias helpers shared across projection and ownership ingestion."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from .text_utils import canonicalize_series, canonicalize_string


def build_player_name_lookup(
    players_df: pd.DataFrame,
    id_col: str = "fd_player_id",
    name_col: str = "full_name",
) -> pd.Series:
    """Return canonical player-name aliases mapped to FanDuel IDs.

    This intentionally strips suffixes/accents and adds first-initial aliases so
    sources like "Jazz Chisholm" can match "Jazz Chisholm Jr." and "J. Chisholm".
    Ambiguous aliases are dropped instead of guessed.
    """

    if players_df is None or players_df.empty or id_col not in players_df.columns or name_col not in players_df.columns:
        return pd.Series(dtype=str)

    rows: list[dict[str, str]] = []
    source = players_df[[id_col, name_col]].dropna().copy()
    for _, row in source.iterrows():
        player_id = str(row[id_col]).strip()
        full_name = str(row[name_col]).strip()
        if not player_id or not full_name:
            continue
        for alias in _aliases_for_name(full_name):
            rows.append({"alias": alias, "fd_player_id": player_id})

    if not rows:
        return pd.Series(dtype=str)
    aliases = pd.DataFrame(rows).drop_duplicates()
    counts = aliases.groupby("alias")["fd_player_id"].nunique()
    unambiguous = counts[counts == 1].index
    lookup = (
        aliases[aliases["alias"].isin(unambiguous)]
        .drop_duplicates("alias")
        .set_index("alias")["fd_player_id"]
        .astype(str)
    )
    return lookup


def resolve_name_series(names: pd.Series, lookup: pd.Series) -> pd.Series:
    """Resolve a series of player names against a canonical alias lookup."""

    if names is None or lookup is None or lookup.empty:
        return pd.Series(pd.NA, index=getattr(names, "index", None), dtype="string")
    canonical = canonicalize_series(names)
    resolved = canonical.map(lookup)
    missing = resolved.isna()
    if missing.any():
        initial_last = canonical.map(_initial_last_alias)
        resolved.loc[missing] = initial_last.loc[missing].map(lookup)
    return resolved.astype("string")


def _aliases_for_name(name: str) -> Iterable[str]:
    canonical = canonicalize_string(name)
    if canonical:
        yield canonical
        initial_last = _initial_last_alias(canonical)
        if initial_last and initial_last != canonical:
            yield initial_last


def _initial_last_alias(canonical_name: str) -> str:
    parts = str(canonical_name or "").split()
    if len(parts) < 2:
        return ""
    return f"{parts[0][0]} {parts[-1]}"


__all__ = ["build_player_name_lookup", "resolve_name_series"]
