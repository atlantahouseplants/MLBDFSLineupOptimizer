"""Utilities for blending external projection sources with the baseline model."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from slate_optimizer.ingestion.name_resolver import build_player_name_lookup, resolve_name_series

PROJECTION_COLUMNS = ["proj_fd_mean", "proj_fd_floor", "proj_fd_ceiling"]
OPTIONAL_SIM_COLUMNS = [
    "proj_fd_median",
    "proj_fd_bust_rate",
    "proj_fd_upside",
    "proj_fd_pts_per_salary",
    "proj_fd_median_per_salary",
    "proj_fd_upside_per_salary",
    "bpp_stack_count",
]

_COLUMN_MAP = {
    "fd_player_id": "fd_player_id",
    "player_id": "fd_player_id",
    "playerid": "fd_player_id",
    "id": "fd_player_id",
    "fdid": "fd_player_id",
    "name": "full_name",
    "player_name": "full_name",
    "full_name": "full_name",
    "fullname": "full_name",
    "first_name": "first_name",
    "firstname": "first_name",
    "last_name": "last_name",
    "lastname": "last_name",
    "projection": "proj_fd_mean",
    "points": "proj_fd_mean",
    "proj": "proj_fd_mean",
    "proj_fd_mean": "proj_fd_mean",
    "mean_projection": "proj_fd_mean",
    "projected_points": "proj_fd_mean",
    "mean": "proj_fd_mean",
    "proj_mean": "proj_fd_mean",
    "median": "proj_fd_median",
    "proj_median": "proj_fd_median",
    "proj_fd_median": "proj_fd_median",
    "floor": "proj_fd_floor",
    "proj_floor": "proj_fd_floor",
    "proj_fd_floor": "proj_fd_floor",
    "ceiling": "proj_fd_ceiling",
    "proj_ceiling": "proj_fd_ceiling",
    "proj_fd_ceiling": "proj_fd_ceiling",
    "upside": "proj_fd_upside",
    "bust": "proj_fd_bust_rate",
    "bust_rate": "proj_fd_bust_rate",
    "bust_pct": "proj_fd_bust_rate",
    "pts/$": "proj_fd_pts_per_salary",
    "pts_per_salary": "proj_fd_pts_per_salary",
    "med/$": "proj_fd_median_per_salary",
    "median_per_salary": "proj_fd_median_per_salary",
    "ups/$": "proj_fd_upside_per_salary",
    "upside_per_salary": "proj_fd_upside_per_salary",
    "stackcount": "bpp_stack_count",
    "stack_count": "bpp_stack_count",
    "tm": "team_code",
    "team": "team_code",
    "pos": "position",
    "player_type": "player_type",
    "playertype": "player_type",
    "slatename": "slate_name",
    "slate_name": "slate_name",
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


def _read_projection_source(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        raw = pd.read_excel(path)
        first_columns = [str(col).lower() for col in raw.columns]
        first_title = first_columns[0] if first_columns else ""
        if (
            first_title
            and (
                "ballpark dfs optimizer" in first_title
                or ("ballpark dfs" in first_title and "ballpark pal" in first_title)
            )
            and len(raw) > 0
        ):
            raw = pd.read_excel(path, header=1)
        return raw
    return pd.read_csv(path)


def _create_name_lookup(players_df: pd.DataFrame, projections_df: pd.DataFrame) -> pd.Series:
    if "full_name" in players_df.columns:
        source = players_df
    else:
        source = projections_df
    return build_player_name_lookup(source)


def _load_projection_source(path: Path, name_lookup: pd.Series) -> tuple[pd.DataFrame, int, bool, bool, bool, List[str]]:
    df = _read_projection_source(path)
    df = _standardize_projection_source(df)
    if "fd_player_id" not in df.columns and "full_name" in df.columns and not name_lookup.empty:
        df["fd_player_id"] = resolve_name_series(df["full_name"], name_lookup)
    if "fd_player_id" not in df.columns:
        raise ValueError(f"Projection file {path} missing fd_player_id column")
    if "proj_fd_mean" not in df.columns:
        raise ValueError(f"Projection file {path} missing projection column")

    df["fd_player_id"] = df["fd_player_id"].astype("string").str.strip()
    missing_ids = df["fd_player_id"].isna() | df["fd_player_id"].str.lower().isin({"", "nan", "none", "<na>"})
    unmatched_names: List[str] = []
    if "full_name" in df.columns:
        unmatched_names = (
            df.loc[missing_ids.fillna(True), "full_name"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .drop_duplicates()
            .head(25)
            .tolist()
        )
    df.loc[missing_ids.fillna(True), "fd_player_id"] = pd.NA
    df = df.dropna(subset=["fd_player_id", "proj_fd_mean"])
    provided_floor = "proj_fd_floor" in df.columns
    provided_ceiling = "proj_fd_ceiling" in df.columns
    provided_upside = "proj_fd_upside" in df.columns
    if "proj_fd_upside" in df.columns and "proj_fd_ceiling" not in df.columns:
        df["proj_fd_ceiling"] = df["proj_fd_upside"]
        provided_ceiling = True

    numeric_cols = [col for col in PROJECTION_COLUMNS + OPTIONAL_SIM_COLUMNS if col in df.columns]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if "proj_fd_bust_rate" in df.columns and df["proj_fd_bust_rate"].max() > 1.5:
        df["proj_fd_bust_rate"] = df["proj_fd_bust_rate"] / 100.0
    if "proj_fd_upside" in df.columns and "proj_fd_ceiling" in df.columns:
        df["proj_fd_ceiling"] = df["proj_fd_ceiling"].fillna(df["proj_fd_upside"])
    df = df.dropna(subset=["proj_fd_mean"])

    agg_map: Dict[str, str] = {"proj_fd_mean": "mean"}
    for col in PROJECTION_COLUMNS[1:] + OPTIONAL_SIM_COLUMNS:
        if col in df.columns:
            agg_map[col] = "mean"
    if "proj_fd_floor" in df.columns:
        agg_map["proj_fd_floor"] = "mean"
    if "proj_fd_ceiling" in df.columns:
        agg_map["proj_fd_ceiling"] = "mean"
    grouped = df.groupby("fd_player_id").agg(agg_map).reset_index()
    metadata_cols = [col for col in ("full_name", "team_code", "position") if col in df.columns]
    if metadata_cols:
        metadata = df.groupby("fd_player_id")[metadata_cols].first().reset_index()
        grouped = grouped.merge(metadata, on="fd_player_id", how="left")

    if "proj_fd_floor" not in grouped.columns:
        grouped["proj_fd_floor"] = grouped["proj_fd_mean"]
    if "proj_fd_ceiling" not in grouped.columns:
        grouped["proj_fd_ceiling"] = grouped["proj_fd_mean"]
    if "proj_fd_upside" not in grouped.columns and "proj_fd_ceiling" in grouped.columns:
        grouped["proj_fd_upside"] = grouped["proj_fd_ceiling"]

    matched = int(grouped["fd_player_id"].nunique())
    return grouped, matched, provided_floor, provided_ceiling, provided_upside, unmatched_names


def _index_by_unique_player(df: pd.DataFrame) -> pd.DataFrame:
    keyed = df.copy()
    keyed["fd_player_id"] = keyed["fd_player_id"].astype(str)
    if not keyed["fd_player_id"].duplicated().any():
        return keyed.set_index("fd_player_id")

    agg_map: Dict[str, str] = {}
    for col in keyed.columns:
        if col == "fd_player_id":
            continue
        if pd.api.types.is_bool_dtype(keyed[col]):
            agg_map[col] = "max"
        elif pd.api.types.is_numeric_dtype(keyed[col]):
            agg_map[col] = "mean"
        else:
            agg_map[col] = "first"
    return keyed.groupby("fd_player_id", as_index=True, dropna=False).agg(agg_map)


@dataclass
class ProjectionSourceDetail:
    name: str
    weight: float
    matched_players: int
    has_floor: bool
    has_ceiling: bool
    has_bust: bool = False
    has_median: bool = False
    has_upside: bool = False
    source_players: int = 0
    unmatched_players: List[str] = field(default_factory=list)


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
    target_ids = set(unique_ids)
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
        source_df, _, has_floor, has_ceiling, has_upside, missing_name_matches = _load_projection_source(Path(path), name_lookup)
        source_ids = source_df["fd_player_id"].astype(str)
        source_players = int(source_ids.nunique()) + len(missing_name_matches)
        slate_match_mask = source_ids.isin(target_ids)
        matched = int(source_ids[slate_match_mask].nunique())
        unmatched = _format_unmatched_projection_players(source_df.loc[~slate_match_mask], missing_name_matches)
        prepared_sources.append(source_df)
        source_details.append(
            ProjectionSourceDetail(
                name=Path(path).name,
                weight=0.0,
                matched_players=matched,
                has_floor=has_floor,
                has_ceiling=has_ceiling,
                has_bust="proj_fd_bust_rate" in source_df.columns,
                has_median="proj_fd_median" in source_df.columns,
                has_upside=has_upside or "proj_fd_upside" in source_df.columns,
                source_players=source_players,
                unmatched_players=unmatched,
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

    base_indexed = _index_by_unique_player(projections_df)
    blended = projections_df.copy()

    for column in PROJECTION_COLUMNS:
        base_series = pd.to_numeric(base_indexed.get(column), errors="coerce")
        total_series = pd.Series(0.0, index=base_indexed.index, dtype=float)
        total_share = pd.Series(0.0, index=base_indexed.index, dtype=float)
        if baseline_share > 0 and base_series is not None:
            covered = base_series.dropna().index
            total_series.loc[covered] += base_series.loc[covered] * baseline_share
            total_share.loc[covered] += baseline_share

        for share, source_df in zip(source_shares, prepared_sources):
            if share <= 0:
                continue
            src_indexed = _index_by_unique_player(source_df)
            if column in src_indexed.columns:
                src_series = src_indexed[column]
            else:
                src_series = src_indexed.get("proj_fd_mean")
            if src_series is None:
                continue
            src_series = pd.to_numeric(src_series, errors="coerce")
            covered = src_series.dropna().index.intersection(total_share.index)
            total_series.loc[covered] += src_series.loc[covered] * share
            total_share.loc[covered] += share

        if base_series is None:
            continue
        normalized = total_series.copy()
        covered_mask = total_share > 0
        normalized.loc[covered_mask] = total_series.loc[covered_mask] / total_share.loc[covered_mask]
        normalized.loc[~covered_mask] = base_series.loc[~covered_mask]
        mapped = blended["fd_player_id"].astype(str).map(normalized)
        base_mapped = blended["fd_player_id"].astype(str).map(base_series)
        blended[column] = mapped.fillna(base_mapped)

    for column in OPTIONAL_SIM_COLUMNS:
        total_series = pd.Series(0.0, index=base_indexed.index, dtype=float)
        total_share = pd.Series(0.0, index=base_indexed.index, dtype=float)
        coverage = pd.Series(False, index=base_indexed.index, dtype=bool)
        if column in projections_df.columns and baseline_share > 0:
            base_numeric = pd.to_numeric(base_indexed[column], errors="coerce")
            total_series = total_series.add(base_numeric * baseline_share, fill_value=0.0)
            covered = base_numeric.dropna().index
            total_share.loc[covered] += baseline_share
            coverage.loc[covered] = True
        for share, source_df in zip(source_shares, prepared_sources):
            if share <= 0 or column not in source_df.columns:
                continue
            src_indexed = _index_by_unique_player(source_df)
            src_series = pd.to_numeric(src_indexed[column], errors="coerce")
            total_series = total_series.add(src_series * share, fill_value=0.0)
            covered = [idx for idx in src_series.dropna().index if idx in total_share.index]
            total_share.loc[covered] += share
            coverage.loc[covered] = True
        if coverage.any():
            normalized = total_series.copy()
            covered_mask = total_share > 0
            normalized.loc[covered_mask] = total_series.loc[covered_mask] / total_share.loc[covered_mask]
            normalized.loc[~covered_mask] = pd.NA
            mapped = blended["fd_player_id"].astype(str).map(normalized.where(coverage))
            blended[column] = mapped

    if "proj_fd_upside" not in blended.columns:
        blended["proj_fd_upside"] = blended["proj_fd_ceiling"]
    else:
        blended["proj_fd_upside"] = pd.to_numeric(blended["proj_fd_upside"], errors="coerce").fillna(blended["proj_fd_ceiling"])

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


def _format_unmatched_projection_players(source_df: pd.DataFrame, missing_name_matches: List[str]) -> List[str]:
    values: List[str] = []
    for name in missing_name_matches:
        clean = str(name).strip()
        if clean and clean not in values:
            values.append(clean)
    if source_df is not None and not source_df.empty:
        for _, row in source_df.head(25).iterrows():
            name = str(row.get("full_name", "") or "").strip()
            team = str(row.get("team_code", "") or "").strip()
            fd_id = str(row.get("fd_player_id", "") or "").strip()
            label = name or fd_id
            if team and label:
                label = f"{label} ({team})"
            if label and label not in values:
                values.append(label)
    return values[:25]


__all__ = [
    "ProjectionBlendResult",
    "ProjectionSourceDetail",
    "blend_projection_sources",
]
