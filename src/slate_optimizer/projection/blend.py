"""Utilities for blending external projection sources with the baseline model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

PROJECTION_COLUMNS = ["proj_fd_mean", "proj_fd_floor", "proj_fd_ceiling"]

_COLUMN_MAP = {
    "fd_player_id": "fd_player_id",
    "player_id": "fd_player_id",
    "playerid": "fd_player_id",
    "id": "fd_player_id",
    "fdid": "fd_player_id",
    "name": "full_name",
    "player_name": "full_name",
    "full_name": "full_name",
    "projection": "proj_fd_mean",
    "proj": "proj_fd_mean",
    "proj_fd_mean": "proj_fd_mean",
    "mean_projection": "proj_fd_mean",
    "projected_points": "proj_fd_mean",
    "points": "proj_fd_mean",
    "mean": "proj_fd_mean",
    "proj_mean": "proj_fd_mean",
    "floor": "proj_fd_floor",
    "proj_floor": "proj_fd_floor",
    "proj_fd_floor": "proj_fd_floor",
    "ceiling": "proj_fd_ceiling",
    "proj_ceiling": "proj_fd_ceiling",
    "proj_fd_ceiling": "proj_fd_ceiling",
}


def _normalize_column(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _standardize_projection_source(df: pd.DataFrame) -> pd.DataFrame:
    rename: Dict[str, str] = {}
    for column in df.columns:
        normalized = _normalize_column(column)
        mapped = _COLUMN_MAP.get(normalized)
        if mapped:
            rename[column] = mapped
    standardized = df.rename(columns=rename)
    return standardized


def _create_name_lookup(players_df: pd.DataFrame, projections_df: pd.DataFrame) -> pd.Series:
    if "full_name" in players_df.columns:
        source = players_df
    else:
        source = projections_df
    lookup = (
        source[["fd_player_id", "full_name"]]
        .dropna()
        .assign(full_name=lambda frame: frame["full_name"].astype(str).str.strip().str.lower())
        .drop_duplicates("full_name")
        .set_index("full_name")["fd_player_id"]
    )
    return lookup.astype(str)


def _load_projection_source(path: Path, name_lookup: pd.Series) -> tuple[pd.DataFrame, int, bool, bool]:
    df = pd.read_csv(path)
    df = _standardize_projection_source(df)
    if "fd_player_id" not in df.columns and "full_name" in df.columns and not name_lookup.empty:
        df["fd_player_id"] = (
            df["full_name"].astype(str).str.strip().str.lower().map(name_lookup)
        )
    if "fd_player_id" not in df.columns:
        raise ValueError(f"Projection file {path} missing fd_player_id column")
    if "proj_fd_mean" not in df.columns:
        raise ValueError(f"Projection file {path} missing projection column")

    df["fd_player_id"] = df["fd_player_id"].astype(str).str.strip()
    df = df.dropna(subset=["fd_player_id", "proj_fd_mean"])
    provided_floor = "proj_fd_floor" in df.columns
    provided_ceiling = "proj_fd_ceiling" in df.columns

    numeric_cols = [col for col in PROJECTION_COLUMNS if col in df.columns]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["proj_fd_mean"])

    agg_map: Dict[str, str] = {"proj_fd_mean": "mean"}
    if "proj_fd_floor" in df.columns:
        agg_map["proj_fd_floor"] = "mean"
    if "proj_fd_ceiling" in df.columns:
        agg_map["proj_fd_ceiling"] = "mean"
    grouped = df.groupby("fd_player_id").agg(agg_map).reset_index()

    if "proj_fd_floor" not in grouped.columns:
        grouped["proj_fd_floor"] = grouped["proj_fd_mean"]
    if "proj_fd_ceiling" not in grouped.columns:
        grouped["proj_fd_ceiling"] = grouped["proj_fd_mean"]

    matched = int(grouped["fd_player_id"].nunique())
    return grouped, matched, provided_floor, provided_ceiling


@dataclass
class ProjectionSourceDetail:
    name: str
    weight: float
    matched_players: int
    has_floor: bool
    has_ceiling: bool


@dataclass
class ProjectionBlendResult:
    baseline_share: float
    source_count: int
    total_players: int
    sources: List[ProjectionSourceDetail]


def blend_projection_sources(
    players_df: pd.DataFrame,
    projections_df: pd.DataFrame,
    source_paths: Sequence[Path] | None = None,
    weights: Sequence[float] | None = None,
    baseline_weight: float | None = 1.0,
) -> tuple[pd.DataFrame, ProjectionBlendResult]:
    source_paths = list(source_paths or [])
    if weights and len(weights) != len(source_paths):
        raise ValueError("Number of projection weights must match projection sources")

    base_weight = float(baseline_weight if baseline_weight is not None else 1.0)
    if base_weight < 0:
        raise ValueError("Baseline projection weight must be non-negative")

    unique_ids = projections_df["fd_player_id"].astype(str).tolist()
    total_players = len(unique_ids)

    if not source_paths:
        result = ProjectionBlendResult(
            baseline_share=1.0,
            source_count=0,
            total_players=total_players,
            sources=[],
        )
        return projections_df, result

    name_lookup = _create_name_lookup(players_df, projections_df)
    prepared_sources: List[pd.DataFrame] = []
    source_details: List[ProjectionSourceDetail] = []

    for path in source_paths:
        source_df, matched, has_floor, has_ceiling = _load_projection_source(Path(path), name_lookup)
        prepared_sources.append(source_df)
        source_details.append(
            ProjectionSourceDetail(
                name=Path(path).name,
                weight=0.0,
                matched_players=matched,
                has_floor=has_floor,
                has_ceiling=has_ceiling,
            )
        )

    if weights:
        ext_weights = [float(value) for value in weights]
    else:
        ext_weights = [1.0 for _ in prepared_sources]
    if any(weight < 0 for weight in ext_weights):
        raise ValueError("Projection weights must be non-negative")
    ext_sum = sum(ext_weights)
    if ext_sum <= 0:
        raise ValueError("Projection weights must sum to a positive value")
    total_weight = base_weight + ext_sum
    if total_weight <= 0:
        raise ValueError("Total projection weight must be positive")

    baseline_share = base_weight / total_weight if total_weight > 0 else 0.0
    source_shares = [weight / total_weight for weight in ext_weights]

    base_indexed = projections_df.set_index("fd_player_id")
    base_indexed.index = base_indexed.index.astype(str)
    blended = projections_df.copy()

    for column in PROJECTION_COLUMNS:
        coverage = pd.Series(baseline_share > 0, index=base_indexed.index, dtype=bool)
        base_series = pd.to_numeric(base_indexed.get(column), errors="coerce")
        total_series = pd.Series(0.0, index=base_indexed.index, dtype=float)
        if baseline_share > 0 and base_series is not None:
            total_series = total_series.add(base_series * baseline_share, fill_value=0.0)

        for share, source_df in zip(source_shares, prepared_sources):
            if share <= 0:
                continue
            src_indexed = source_df.set_index("fd_player_id")
            if column in src_indexed.columns:
                src_series = src_indexed[column]
            else:
                src_series = src_indexed.get("proj_fd_mean")
            if src_series is None:
                continue
            total_series = total_series.add(src_series * share, fill_value=0.0)
            coverage.loc[src_series.index] = True

        if base_series is None:
            continue
        total_series = total_series.reindex(base_indexed.index)
        total_series = total_series.where(coverage, base_series)
        mapped = blended["fd_player_id"].astype(str).map(total_series)
        base_mapped = blended["fd_player_id"].astype(str).map(base_series)
        blended[column] = mapped.fillna(base_mapped)

    salary_series = pd.to_numeric(blended.get("salary"), errors="coerce").replace(0, pd.NA)
    value_series = pd.to_numeric(blended["proj_fd_mean"] / salary_series, errors="coerce").fillna(0.0)
    blended["value_score"] = value_series * 1000.0

    for detail, share in zip(source_details, source_shares):
        detail.weight = share

    result = ProjectionBlendResult(
        baseline_share=baseline_share,
        source_count=len(prepared_sources),
        total_players=total_players,
        sources=source_details,
    )
    return blended, result


__all__ = [
    "ProjectionBlendResult",
    "ProjectionSourceDetail",
    "blend_projection_sources",
]
