"""Structural ownership model: budget-constrained softmax over slate features.

FanDuel MLB lineups carry exactly 1 pitcher and 8 hitters, so across the
field pitcher ownership sums to ~100% and hitter ownership to ~800%. This
model scores every player log-linearly on the drivers of public ownership
(value, projection, team implied runs, batting order, HR upside, salary
tier, name recognition) and allocates those fixed budgets with a softmax
whose concentration scales with slate size.

Weights start as hand-set priors and are replaced by learned values from
``config/ownership_model.json`` once ``scripts/fit_ownership_model.py`` has
enough recorded slates (predicted vs actual ownership) to fit against.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

DEFAULT_PARAMS_PATH = Path(__file__).resolve().parents[3] / "config" / "ownership_model.json"

# Feature weights apply to z-scores computed within each pool (pitchers and
# hitters separately), so magnitudes are comparable across features.
DEFAULT_HITTER_WEIGHTS: Dict[str, float] = {
    "value": 1.00,
    "proj_mean": 0.75,
    "team_total": 0.70,
    "batting_order_score": 0.50,
    "hr_prob": 0.40,
    "salary": 0.30,
    "fppg": 0.20,
    "confirmed": 0.25,
}
DEFAULT_PITCHER_WEIGHTS: Dict[str, float] = {
    "proj_mean": 1.00,
    "value": 0.80,
    "k_proj": 0.70,
    "win_pct": 0.40,
    "salary": 0.40,
}

# One pitcher slot and eight hitter slots per lineup.
PITCHER_BUDGET = 1.0
HITTER_BUDGET = 8.0

# Per-player ownership caps before redistribution (decimals).
PITCHER_CAP = 0.55
HITTER_CAP = 0.45

# Bench/unprojected players still soak up a sliver of the field.
NONVIABLE_OWNERSHIP = 0.002
NONVIABLE_BUDGET_SHARE_MAX = 0.10


@dataclass
class StructuralOwnershipParams:
    hitter_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_HITTER_WEIGHTS))
    pitcher_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PITCHER_WEIGHTS))
    hitter_temperature: float = 3.40
    pitcher_temperature: float = 2.00
    slate_gamma: float = 0.25
    fitted_at: str = ""
    slates_used: int = 0

    def to_dict(self) -> Dict:
        return {
            "hitter": {"weights": self.hitter_weights, "temperature": self.hitter_temperature},
            "pitcher": {"weights": self.pitcher_weights, "temperature": self.pitcher_temperature},
            "slate_gamma": self.slate_gamma,
            "fitted_at": self.fitted_at,
            "slates_used": self.slates_used,
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "StructuralOwnershipParams":
        params = cls()
        hitter = payload.get("hitter") or {}
        pitcher = payload.get("pitcher") or {}
        if isinstance(hitter.get("weights"), dict):
            params.hitter_weights = {str(k): float(v) for k, v in hitter["weights"].items()}
        if isinstance(pitcher.get("weights"), dict):
            params.pitcher_weights = {str(k): float(v) for k, v in pitcher["weights"].items()}
        params.hitter_temperature = float(hitter.get("temperature", params.hitter_temperature))
        params.pitcher_temperature = float(pitcher.get("temperature", params.pitcher_temperature))
        params.slate_gamma = float(payload.get("slate_gamma", params.slate_gamma))
        params.fitted_at = str(payload.get("fitted_at", ""))
        params.slates_used = int(payload.get("slates_used", 0) or 0)
        return params


def load_ownership_params(path: Path | str | None = None) -> StructuralOwnershipParams:
    """Load learned params from JSON, falling back to hand-set priors."""
    target = Path(path) if path else DEFAULT_PARAMS_PATH
    if not target.exists():
        return StructuralOwnershipParams()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return StructuralOwnershipParams()
    if not isinstance(payload, dict):
        return StructuralOwnershipParams()
    return StructuralOwnershipParams.from_dict(payload)


def save_ownership_params(params: StructuralOwnershipParams, path: Path | str | None = None) -> None:
    target = Path(path) if path else DEFAULT_PARAMS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(params.to_dict(), indent=2), encoding="utf-8")


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _first_numeric(df: pd.DataFrame, *columns: str) -> pd.Series:
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for column in columns:
        values = _numeric(df, column)
        result = result.combine_first(values)
    return result


def estimate_slate_games(players_df: pd.DataFrame) -> int:
    """Best-effort count of games on the slate."""
    for column in ("bpp_game_pk", "game_pk", "game"):
        if column in players_df.columns:
            values = players_df[column].dropna()
            if not values.empty:
                return max(1, int(values.astype(str).nunique()))
    if "team_code" in players_df.columns:
        teams = players_df["team_code"].dropna().astype(str).str.strip()
        teams = teams[teams != ""]
        if not teams.empty:
            return max(1, int(round(teams.nunique() / 2)))
    return 8


def build_ownership_features(
    players_df: pd.DataFrame,
    projections_df: pd.DataFrame,
) -> pd.DataFrame:
    """One row per player with the raw features the structural model scores.

    Also used to persist a training snapshot per slate, so keep the columns
    stable: fitting code reads them back from ``ownership_history``.
    """
    meta_cols = [
        col
        for col in (
            "fd_player_id",
            "full_name",
            "player_type",
            "team_code",
            "position",
            "salary",
            "fppg",
            "is_confirmed_lineup",
            "batting_order_position",
            "bpp_batting_position",
            "bpp_runs",
            "vegas_team_total",
            "bpp_home_run_probability",
            "bpp_strikeouts",
            "bpp_win_pct",
            "bpp_quality_start",
        )
        if col in players_df.columns
    ]
    meta = players_df[meta_cols].copy()
    if "fd_player_id" not in meta.columns:
        raise ValueError("players_df is missing fd_player_id")
    meta["fd_player_id"] = meta["fd_player_id"].astype(str)
    meta = meta.drop_duplicates("fd_player_id")

    proj = projections_df.copy()
    proj["fd_player_id"] = proj["fd_player_id"].astype(str)
    proj_cols = ["fd_player_id", "proj_fd_mean"]
    for optional in ("proj_fd_ceiling", "proj_fd_upside", "vegas_team_total", "batting_order_position"):
        if optional in proj.columns and optional not in proj_cols:
            proj_cols.append(optional)
    proj = proj[proj_cols].groupby("fd_player_id", as_index=False).mean(numeric_only=True)

    merged = meta.merge(proj, on="fd_player_id", how="left", suffixes=("", "_proj"))

    features = pd.DataFrame(index=merged.index)
    features["fd_player_id"] = merged["fd_player_id"]
    features["full_name"] = merged.get("full_name", pd.Series("", index=merged.index)).fillna("")
    player_type = merged.get("player_type", pd.Series("", index=merged.index)).fillna("").astype(str).str.lower()
    if not player_type.isin(["pitcher"]).any() and "position" in merged.columns:
        player_type = np.where(
            merged["position"].fillna("").astype(str).str.upper() == "P", "pitcher", "batter"
        )
        player_type = pd.Series(player_type, index=merged.index)
    features["player_type"] = player_type.where(player_type == "pitcher", "batter")
    features["team_code"] = merged.get("team_code", pd.Series("", index=merged.index)).fillna("")

    salary = _numeric(merged, "salary")
    features["salary"] = salary.fillna(salary.median())
    proj_mean = _numeric(merged, "proj_fd_mean").fillna(0.0)
    features["proj_mean"] = proj_mean
    with np.errstate(divide="ignore", invalid="ignore"):
        value = proj_mean / features["salary"].replace(0, np.nan) * 1000.0
    features["value"] = value.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features["fppg"] = _numeric(merged, "fppg").fillna(0.0)

    # Team implied runs: BPP sims first, Vegas total as fallback.
    team_total = _first_numeric(merged, "bpp_runs", "vegas_team_total", "vegas_team_total_proj")
    features["team_total"] = team_total.fillna(team_total.median() if team_total.notna().any() else 4.5)

    # Batting order 1 is best; unknown order sits below the confirmed top-6.
    order = _first_numeric(merged, "batting_order_position", "batting_order_position_proj", "bpp_batting_position")
    order = order.where((order >= 1) & (order <= 9))
    order_score = (10.0 - order) / 9.0
    features["batting_order_score"] = order_score.fillna(0.30)

    features["hr_prob"] = _numeric(merged, "bpp_home_run_probability").fillna(0.0)
    if "is_confirmed_lineup" in merged.columns:
        features["confirmed"] = merged["is_confirmed_lineup"].fillna(False).astype(bool).astype(float)
    else:
        features["confirmed"] = 0.0

    features["k_proj"] = _numeric(merged, "bpp_strikeouts").fillna(0.0)
    features["win_pct"] = _numeric(merged, "bpp_win_pct").fillna(0.0)
    features["quality_start"] = _numeric(merged, "bpp_quality_start").fillna(0.0)

    # Viability gate: FanDuel lists whole rosters, but the field only rosters
    # players with a real projection signal. Bench bats without BPP coverage,
    # a batting order, or a confirmed-lineup flag would otherwise distort the
    # z-scores every starter is measured against.
    is_pitcher_flag = features["player_type"] == "pitcher"
    has_signal = (
        (features["hr_prob"] > 0)
        | (features["confirmed"] > 0)
        | order.notna()
    )
    viable = (features["proj_mean"] > 0.1) & (is_pitcher_flag | has_signal)
    # If the gate would wipe out a pool (e.g. no BPP data at all), keep everyone.
    for mask in (is_pitcher_flag, ~is_pitcher_flag):
        if mask.any() and not (viable & mask).any():
            viable = viable | mask
    features["viable"] = viable.astype(float)

    features["games"] = estimate_slate_games(players_df)
    return features.reset_index(drop=True)


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std < 1e-9:
        return pd.Series(0.0, index=series.index)
    # Winsorize so a single extreme value play can't run away with the softmax.
    return ((values - float(values.mean())) / std).clip(lower=-3.0, upper=3.0)


def _pool_ownership(
    pool: pd.DataFrame,
    weights: Dict[str, float],
    temperature: float,
    budget: float,
    cap: float,
) -> pd.Series:
    if pool.empty:
        return pd.Series(dtype=float)

    # Bench/unprojected players get a fixed sliver; the softmax runs over the
    # viable pool only so their zeroed features don't stretch the z-scores.
    if "viable" in pool.columns:
        viable_mask = pool["viable"].fillna(0.0) > 0
    else:
        viable_mask = pd.Series(True, index=pool.index)
    if not viable_mask.any():
        viable_mask = pd.Series(True, index=pool.index)
    nonviable = pool[~viable_mask]
    nonviable_total = min(
        NONVIABLE_OWNERSHIP * len(nonviable), budget * NONVIABLE_BUDGET_SHARE_MAX
    )
    nonviable_own = pd.Series(
        (nonviable_total / len(nonviable)) if len(nonviable) else 0.0, index=nonviable.index
    )
    viable = pool[viable_mask]
    budget = max(0.0, budget - nonviable_total)

    score = pd.Series(0.0, index=viable.index)
    for feature_name, weight in weights.items():
        if feature_name not in viable.columns or not weight:
            continue
        score = score + float(weight) * _zscore(viable[feature_name])

    temperature = max(0.05, float(temperature))
    scaled = (score - score.max()) / temperature
    exp = np.exp(scaled.clip(lower=-60.0))
    share = exp / exp.sum()
    ownership = share * budget

    # Cap runaway chalk and hand the excess to everyone else proportionally.
    for _ in range(4):
        over = ownership > cap
        if not over.any():
            break
        excess = float((ownership[over] - cap).sum())
        ownership[over] = cap
        under = ~over
        under_total = float(ownership[under].sum())
        if excess <= 0 or under_total <= 0:
            break
        ownership[under] = ownership[under] * (1.0 + excess / under_total)
    ownership = ownership.clip(lower=0.0, upper=1.0)
    return pd.concat([ownership, nonviable_own]).reindex(pool.index).fillna(0.0)


def estimate_structural_ownership(
    players_df: pd.DataFrame,
    projections_df: pd.DataFrame,
    params: Optional[StructuralOwnershipParams] = None,
    features: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Project FanDuel ownership (decimals) for every player on the slate.

    Returns a Series indexed by fd_player_id named ``proj_fd_ownership``.
    Pitchers and hitters are allocated separately against their roster-slot
    budgets (100% / 800% of the field).
    """
    params = params or load_ownership_params()
    if features is None:
        features = build_ownership_features(players_df, projections_df)
    if features.empty:
        return pd.Series(dtype=float, name="proj_fd_ownership")

    games = int(features["games"].iloc[0]) if "games" in features.columns else 8
    slate_factor = float(np.clip((games / 8.0) ** params.slate_gamma, 0.70, 1.35))

    is_pitcher = features["player_type"] == "pitcher"
    pitchers = features[is_pitcher]
    hitters = features[~is_pitcher]

    pitcher_own = _pool_ownership(
        pitchers,
        params.pitcher_weights,
        params.pitcher_temperature * slate_factor,
        PITCHER_BUDGET,
        PITCHER_CAP,
    )
    hitter_own = _pool_ownership(
        hitters,
        params.hitter_weights,
        params.hitter_temperature * slate_factor,
        HITTER_BUDGET,
        HITTER_CAP,
    )

    combined = pd.concat([pitcher_own, hitter_own]).reindex(features.index).fillna(0.0)
    result = pd.Series(
        combined.values,
        index=features["fd_player_id"].astype(str).values,
        name="proj_fd_ownership",
    )
    return result[~result.index.duplicated(keep="first")]


__all__ = [
    "DEFAULT_PARAMS_PATH",
    "HITTER_BUDGET",
    "PITCHER_BUDGET",
    "StructuralOwnershipParams",
    "build_ownership_features",
    "estimate_slate_games",
    "estimate_structural_ownership",
    "load_ownership_params",
    "save_ownership_params",
]
