"""Utilities to build combined slate datasets from BallparkPal + FanDuel."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import pandas as pd

from .aliases import CanonicalMap, apply_aliases
from .ballparkpal import BallparkPalBundle
from .text_utils import canonicalize_series


def _normalize_team(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.upper().str.strip()


def _prepare_fanduel_players(
    df: pd.DataFrame, alias_map: CanonicalMap | None = None
) -> pd.DataFrame:
    prepared = df.copy()
    names = prepared.get("full_name")
    if names is None:
        names = (
            prepared.get("first_name", "").fillna("")
            + " "
            + prepared.get("last_name", "").fillna("")
        )
    canonical = canonicalize_series(names)
    if alias_map:
        canonical = apply_aliases(canonical, alias_map)
    prepared["canonical_name"] = canonical

    last_names = prepared.get("last_name")
    if last_names is None:
        last_names = pd.Series(["" for _ in range(len(prepared))])
    prepared["canonical_last_name"] = canonicalize_series(last_names)

    prepared["team_code"] = _normalize_team(prepared.get("team", ""))
    prepared["opponent_code"] = _normalize_team(prepared.get("opponent", ""))
    prepared["player_type"] = prepared["position"].apply(
        lambda pos: "pitcher" if str(pos).upper() == "P" else "batter"
    )
    return prepared


def _prepare_ballparkpal_table(
    df: pd.DataFrame,
    teams_df: pd.DataFrame,
    games_df: pd.DataFrame,
    alias_map: CanonicalMap | None = None,
) -> pd.DataFrame:
    prepared = df.copy()
    name_column = None
    for candidate in ("full_name", "fullname", "player", "name"):
        if candidate in prepared.columns:
            name_column = candidate
            break
    if name_column is None:
        raise KeyError("Could not find a name column in BallparkPal data")
    canonical = canonicalize_series(prepared[name_column])
    if alias_map:
        canonical = apply_aliases(canonical, alias_map)
    prepared["canonical_name"] = canonical

    last_column = None
    for candidate in ("last_name", "lastname", "surname"):
        if candidate in prepared.columns:
            last_column = candidate
            break
    if last_column is not None:
        prepared["canonical_last_name"] = canonicalize_series(prepared[last_column])
    else:
        prepared["canonical_last_name"] = canonicalize_series(pd.Series(["" for _ in range(len(prepared))]))

    prepared["team_code"] = _normalize_team(prepared.get("team", ""))
    prepared["opponent_code"] = _normalize_team(prepared.get("opponent", ""))

    prepared = _attach_team_meta(prepared, teams_df)
    prepared = _attach_game_meta(prepared, games_df)
    return prepared


def _attach_team_meta(prepared: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    if teams_df is None or teams_df.empty:
        return prepared
    if "game_pk" not in prepared.columns or "team_code" not in prepared.columns:
        return prepared

    teams = teams_df.copy()
    if "game_pk" not in teams.columns or "team" not in teams.columns:
        return prepared
    teams["team_code"] = _normalize_team(teams["team"])
    teams_cols = ["game_pk", "team_code"] + [
        col for col in teams.columns if col not in {"game_pk", "team", "team_code"}
    ]
    existing = set(prepared.columns)
    meta_cols = ["game_pk", "team_code"] + [
        col for col in teams_cols if col not in {"game_pk", "team_code"} and col not in existing
    ]
    if len(meta_cols) <= 2:
        return prepared
    return prepared.merge(teams[meta_cols], on=["game_pk", "team_code"], how="left")


def _attach_game_meta(prepared: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df is None or games_df.empty:
        return prepared
    if "game_pk" not in prepared.columns or "game_pk" not in games_df.columns:
        return prepared
    existing = set(prepared.columns)
    meta_cols = ["game_pk"] + [
        col for col in games_df.columns if col != "game_pk" and col not in existing
    ]
    if len(meta_cols) <= 1:
        return prepared
    return prepared.merge(games_df[meta_cols].drop_duplicates("game_pk"), on="game_pk", how="left")


def _prefix_except(df: pd.DataFrame, prefix: str, exclude: Iterable[str]) -> pd.DataFrame:
    excluded = set(exclude)
    rename_map = {col: f"{prefix}{col}" for col in df.columns if col not in excluded}
    return df.rename(columns=rename_map)


def _apply_last_name_fallback(
    merged_df: pd.DataFrame,
    fd_df: pd.DataFrame,
    bpp_pref: pd.DataFrame,
) -> pd.DataFrame:
    if "canonical_last_name" not in fd_df.columns:
        return merged_df
    if "canonical_last_name" not in bpp_pref.columns:
        return merged_df

    bpp_columns = [col for col in bpp_pref.columns if col.startswith("bpp_")]
    if not bpp_columns:
        return merged_df

    key_counts = (
        bpp_pref.groupby(["team_code", "canonical_last_name"])
        .size()
        .reset_index(name="count")
    )
    valid_keys = key_counts[key_counts["count"] == 1][["team_code", "canonical_last_name"]]
    if valid_keys.empty:
        return merged_df

    unique_lookup = bpp_pref.merge(valid_keys, on=["team_code", "canonical_last_name"], how="inner")
    fallback = fd_df[["fd_player_id", "team_code", "canonical_last_name"]].merge(
        unique_lookup[["team_code", "canonical_last_name"] + bpp_columns],
        how="left",
        on=["team_code", "canonical_last_name"],
    )
    rename_map = {col: f"{col}_fallback" for col in bpp_columns}
    fallback = fallback.drop(columns=["team_code", "canonical_last_name"])
    fallback = fallback.rename(columns=rename_map)

    merged = merged_df.merge(fallback, on="fd_player_id", how="left")
    for col in bpp_columns:
        fallback_col = f"{col}_fallback"
        if fallback_col in merged.columns:
            merged[col] = merged[col].fillna(merged[fallback_col])
            merged = merged.drop(columns=[fallback_col])
    return merged


@dataclass
class MergeDiagnostics:
    hitters_total: int
    hitters_matched: int
    pitchers_total: int
    pitchers_matched: int

    @property
    def hitters_missing(self) -> int:
        return self.hitters_total - self.hitters_matched

    @property
    def pitchers_missing(self) -> int:
        return self.pitchers_total - self.pitchers_matched

    def as_dict(self) -> Dict[str, int]:
        return {
            "hitters_total": self.hitters_total,
            "hitters_matched": self.hitters_matched,
            "hitters_missing": self.hitters_missing,
            "pitchers_total": self.pitchers_total,
            "pitchers_matched": self.pitchers_matched,
            "pitchers_missing": self.pitchers_missing,
        }


def build_player_dataset(
    bundle: BallparkPalBundle, fanduel_players: pd.DataFrame, alias_map: CanonicalMap | None = None
) -> Tuple[pd.DataFrame, MergeDiagnostics]:
    fd_prepared = _prepare_fanduel_players(fanduel_players, alias_map=alias_map)
    fd_hitters = fd_prepared[fd_prepared["player_type"] == "batter"].copy()
    fd_pitchers = fd_prepared[fd_prepared["player_type"] == "pitcher"].copy()

    bpp_hitters = _prepare_ballparkpal_table(
        bundle.batters, bundle.teams, bundle.games, alias_map=alias_map
    )
    bpp_pitchers = _prepare_ballparkpal_table(
        bundle.pitchers, bundle.teams, bundle.games, alias_map=alias_map
    )

    pref_hitters = _prefix_except(
        bpp_hitters,
        "bpp_",
        {"canonical_name", "team_code", "canonical_last_name", "opponent_code"},
    )
    pref_pitchers = _prefix_except(
        bpp_pitchers,
        "bpp_",
        {"canonical_name", "team_code", "canonical_last_name", "opponent_code"},
    )

    hitters_merged = fd_hitters.merge(
        pref_hitters,
        how="left",
        left_on=["canonical_name", "team_code"],
        right_on=["canonical_name", "team_code"],
        suffixes=("", "_bpp"),
    )
    pitchers_merged = fd_pitchers.merge(
        pref_pitchers,
        how="left",
        left_on=["canonical_name", "team_code"],
        right_on=["canonical_name", "team_code"],
        suffixes=("", "_bpp"),
    )

    hitters_merged = _apply_last_name_fallback(hitters_merged, fd_hitters, pref_hitters)
    pitchers_merged = _apply_last_name_fallback(pitchers_merged, fd_pitchers, pref_pitchers)

    hitters_matched = int(hitters_merged["bpp_full_name"].notna().sum())
    pitchers_matched = int(pitchers_merged["bpp_full_name"].notna().sum())

    diagnostics = MergeDiagnostics(
        hitters_total=len(fd_hitters),
        hitters_matched=hitters_matched,
        pitchers_total=len(fd_pitchers),
        pitchers_matched=pitchers_matched,
    )

    combined = pd.concat([hitters_merged, pitchers_merged], ignore_index=True, sort=False)
    combined = combined.sort_values(["player_type", "team", "last_name"], na_position="last")
    return combined, diagnostics
