"""Ownership projection blending utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

_COLUMN_MAP = {
    "fd_player_id": "fd_player_id",
    "player_id": "fd_player_id",
    "id": "fd_player_id",
    "playerid": "fd_player_id",
    "player id": "fd_player_id",
    "fanduel id": "fd_player_id",
    "fdid": "fd_player_id",
    "name": "full_name",
    "player": "full_name",
    "player_name": "full_name",
    "full_name": "full_name",
    "ownership": "proj_fd_ownership",
    "projected ownership": "proj_fd_ownership",
    "projected_ownership": "proj_fd_ownership",
    "proj_ownership": "proj_fd_ownership",
    "proj fd ownership": "proj_fd_ownership",
    "own": "proj_fd_ownership",
    "fd_own": "proj_fd_ownership",
    "proj_fd_ownership": "proj_fd_ownership",
}


def _normalize_column_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _standardize_source(df: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for column in df.columns:
        normalized = column.strip().lower()
        if normalized in _COLUMN_MAP:
            rename[column] = _COLUMN_MAP[normalized]
        else:
            normalized = column.strip().lower().replace(" ", "_")
            if normalized in _COLUMN_MAP:
                rename[column] = _COLUMN_MAP[normalized]
    standardized = df.rename(columns=rename)
    return standardized


def _clean_ownership(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    max_val = values.max()
    if pd.notna(max_val) and max_val > 1.5:
        values = values / 100.0
    return values.clip(lower=0.0)


def _create_name_lookup(players_df: pd.DataFrame) -> pd.Series:
    if "full_name" not in players_df.columns:
        return pd.Series(dtype=str)
    lookup = (
        players_df[["fd_player_id", "full_name"]]
        .dropna()
        .assign(full_name=lambda df: df["full_name"].astype(str).str.strip().str.lower())
        .drop_duplicates("full_name")
        .set_index("full_name")["fd_player_id"]
    )
    lookup = lookup.astype(str)
    return lookup


def _load_source(path: Path, name_lookup: pd.Series) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _standardize_source(df)
    if "fd_player_id" not in df.columns and "full_name" in df.columns and not name_lookup.empty:
        df["fd_player_id"] = (
            df["full_name"].astype(str).str.strip().str.lower().map(name_lookup)
        )
    if "fd_player_id" not in df.columns:
        raise ValueError(f"Ownership file {path} missing fd_player_id column")
    if "proj_fd_ownership" not in df.columns:
        raise ValueError(f"Ownership file {path} missing ownership column")

    df["fd_player_id"] = df["fd_player_id"].astype(str).str.strip()
    df["proj_fd_ownership"] = _clean_ownership(df["proj_fd_ownership"])
    df = df.dropna(subset=["fd_player_id", "proj_fd_ownership"])
    grouped = df.groupby("fd_player_id")["proj_fd_ownership"].mean().reset_index()
    return grouped


def _estimate_fallback(
    players_df: pd.DataFrame,
    projections_df: pd.DataFrame,
    model_config: OwnershipModelConfig | None = None,
) -> pd.Series:
    meta_cols = [
        col
        for col in (
            "fd_player_id",
            "salary",
            "position",
            "team_code",
            "vegas_team_total",
            "full_name",
            "fppg",
        )
        if col in players_df.columns
    ]
    meta = players_df[meta_cols].copy()
    if meta.empty:
        meta = pd.DataFrame({"fd_player_id": projections_df["fd_player_id"].astype(str)})
        meta["salary"] = 0
        meta["position"] = "UTIL"
    meta["fd_player_id"] = meta["fd_player_id"].astype(str)
    meta = meta.drop_duplicates("fd_player_id")

    proj = projections_df[["fd_player_id", "proj_fd_mean"]].copy()
    proj["fd_player_id"] = proj["fd_player_id"].astype(str)
    merged = meta.merge(proj, on="fd_player_id", how="left")
    merged["salary"] = pd.to_numeric(merged["salary"], errors="coerce").fillna(
        merged["salary"].median()
    )
    merged["proj_fd_mean"] = pd.to_numeric(merged["proj_fd_mean"], errors="coerce").fillna(0.0)

    merged["salary_rank"] = merged["salary"].rank(pct=True)
    merged["projection_rank"] = merged["proj_fd_mean"].rank(pct=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        value = merged["proj_fd_mean"] / merged["salary"].replace(0, np.nan)
    merged["value_rank"] = value.rank(pct=True).fillna(0.5)

    team_totals_raw = merged.get("vegas_team_total")
    if isinstance(team_totals_raw, pd.Series):
        team_totals = pd.to_numeric(team_totals_raw, errors="coerce")
    else:
        team_totals = pd.Series(np.nan, index=merged.index)
    if team_totals.notna().any():
        merged["team_rank"] = team_totals.rank(pct=True)
    else:
        merged["team_rank"] = 0.5

    name_signal = pd.to_numeric(merged.get("fppg"), errors="coerce")
    if name_signal.notna().any():
        merged["name_rank"] = name_signal.rank(pct=True)
    else:
        merged["name_rank"] = merged["proj_fd_mean"].rank(pct=True)

    pos_counts = merged["position"].fillna("UTIL").value_counts()
    max_count = pos_counts.max() if not pos_counts.empty else 1
    merged["pos_scarcity"] = 1 - merged["position"].fillna("UTIL").map(pos_counts) / max_count
    merged["pos_scarcity"] = merged["pos_scarcity"].fillna(0.0)

    cfg = model_config or OwnershipModelConfig()
    normalized_weights = cfg.normalized_weights()
    raw = pd.Series(0.0, index=merged.index, dtype=float)
    for column, weight in normalized_weights.items():
        raw = raw.add(weight * merged[column], fill_value=0.0)
    raw = raw.fillna(raw.mean() if raw.notna().any() else 0.1)
    raw = raw.clip(lower=0.0)
    if raw.max() and raw.max() > 0:
        raw = raw / raw.max()
    min_pct, max_pct = cfg.clamp_range()
    span = max_pct - min_pct
    fallback = (min_pct + span * raw).clip(lower=min_pct, upper=max_pct)
    return pd.Series(fallback.values, index=merged["fd_player_id"], name="proj_fd_ownership")


@dataclass
class OwnershipSourceDetail:
    name: str
    weight: float
    matched_players: int


@dataclass
class OwnershipBlendResult:
    """Blended ownership output with metadata."""

    ownership: pd.Series
    source_count: int
    covered_players: int
    sources: list[OwnershipSourceDetail]


def compute_ownership_series(
    players_df: pd.DataFrame,
    projections_df: pd.DataFrame,
    source_paths: Sequence[Path] | None = None,
    weights: Sequence[float] | None = None,
    model_config: OwnershipModelConfig | None = None,
) -> OwnershipBlendResult:
    source_paths = list(source_paths or [])
    if weights and len(weights) != len(source_paths):
        raise ValueError("Number of ownership weights must match ownership sources")

    name_lookup = _create_name_lookup(players_df)
    unique_ids = projections_df["fd_player_id"].astype(str).drop_duplicates().tolist()

    prepared_sources: List[pd.DataFrame] = []
    for path in source_paths:
        prepared_sources.append(_load_source(Path(path), name_lookup))

    source_details: List[OwnershipSourceDetail] = []
    normalized_weights: List[float] = []
    covered_players = 0

    if prepared_sources:
        if weights:
            normalized_weights = [float(w) for w in weights]
        else:
            normalized_weights = [1.0] * len(prepared_sources)
        total = sum(normalized_weights)
        if total <= 0:
            raise ValueError("Ownership weights must sum to a positive value")
        normalized_weights = [w / total for w in normalized_weights]

        blended = pd.Series(0.0, index=unique_ids, dtype=float)
        coverage = pd.Series(False, index=unique_ids, dtype=bool)
        for weight, source, path_obj in zip(normalized_weights, prepared_sources, source_paths):
            src_series = source.set_index("fd_player_id")["proj_fd_ownership"]
            coverage.loc[src_series.index] = True
            blended = blended.add(src_series * weight, fill_value=0.0)
            matched = int(src_series.index.nunique())
            source_details.append(
                OwnershipSourceDetail(
                    name=Path(path_obj).name,
                    weight=float(weight),
                    matched_players=matched,
                )
            )

        fallback = _estimate_fallback(players_df, projections_df, model_config=model_config).reindex(unique_ids).fillna(
            0.05
        )
        blended = blended.where(coverage, fallback)
        covered_players = int(coverage.sum())
        fallback_gap = int(len(unique_ids) - covered_players)
        if fallback_gap > 0:
            remaining_weight = max(0.0, 1.0 - sum(detail.weight for detail in source_details))
            source_details.append(
                OwnershipSourceDetail(
                    name="fallback_model",
                    weight=remaining_weight,
                    matched_players=fallback_gap,
                )
            )
    else:
        fallback = _estimate_fallback(players_df, projections_df, model_config=model_config)
        blended = fallback.reindex(unique_ids).fillna(0.05)
        covered_players = len(unique_ids)
        source_details.append(
            OwnershipSourceDetail(name="fallback_model", weight=1.0, matched_players=len(unique_ids))
        )

    blended = blended.clip(lower=0.0, upper=1.0)
    ownership_series = pd.Series(blended.values, index=unique_ids, name="proj_fd_ownership")
    return OwnershipBlendResult(
        ownership=ownership_series,
        source_count=len(prepared_sources),
        covered_players=covered_players,
        sources=source_details,
    )



__all__ = [
    "compute_ownership_series",
    "OwnershipBlendResult",
    "OwnershipSourceDetail",
    "OwnershipModelConfig",
]
@dataclass
class OwnershipModelConfig:
    salary_weight: float = 0.2
    projection_weight: float = 0.25
    value_weight: float = 0.2
    team_weight: float = 0.15
    name_weight: float = 0.1
    position_weight: float = 0.1
    min_pct: float = 0.05
    max_pct: float = 1.0

    def normalized_weights(self) -> dict[str, float]:
        weights = {
            "salary_rank": max(0.0, float(self.salary_weight)),
            "projection_rank": max(0.0, float(self.projection_weight)),
            "value_rank": max(0.0, float(self.value_weight)),
            "team_rank": max(0.0, float(self.team_weight)),
            "name_rank": max(0.0, float(self.name_weight)),
            "pos_scarcity": max(0.0, float(self.position_weight)),
        }
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Ownership model weights must sum to a positive value")
        return {key: value / total for key, value in weights.items()}

    def clamp_range(self) -> tuple[float, float]:
        min_pct = max(0.0, float(self.min_pct))
        max_pct = max(min_pct + 1e-6, float(self.max_pct))
        return min_pct, max_pct
