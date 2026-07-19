"""Team stack exposure recommendations for MLB GPP portfolios."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


RUN_BUCKETS = [f"bpp_runs{i}" for i in range(16)]
HR_BUCKETS = [f"bpp_home_runs{i}" for i in range(6)]


@dataclass(frozen=True)
class StackExposureConfig:
    """Controls for translating team sims into portfolio exposure targets."""

    top_hitters: int = 5
    total_stack_exposure: float = 1.35
    max_team_exposure: float = 0.35
    min_team_exposure: float = 0.0
    min_viable_target: float = 0.02


def build_stack_exposure_plan(
    optimizer_df: pd.DataFrame,
    config: StackExposureConfig | None = None,
) -> pd.DataFrame:
    """Return team-level stack targets from projections, ownership, and BPP sims."""

    config = config or StackExposureConfig()
    if optimizer_df is None or optimizer_df.empty:
        return _empty_plan()

    df = optimizer_df.copy()
    if "player_type" not in df.columns or "team_code" not in df.columns:
        return _empty_plan()

    df["player_type"] = df["player_type"].astype(str).str.lower()
    df["team_code"] = df["team_code"].astype(str).str.upper().str.strip()
    hitters = df[(df["player_type"] != "pitcher") & df["team_code"].ne("")].copy()
    if hitters.empty:
        return _empty_plan()

    for col in (
        "proj_fd_mean",
        "proj_fd_ownership",
        "proj_fd_upside",
        "proj_fd_bust_rate",
        "team_leverage_score",
        "vegas_team_total",
        "bpp_runs",
        "bpp_home_runs",
    ):
        if col not in hitters.columns:
            hitters[col] = 0.0
        hitters[col] = pd.to_numeric(hitters[col], errors="coerce").fillna(0.0)
    if hitters["proj_fd_ownership"].max() > 1.5:
        hitters["proj_fd_ownership"] = hitters["proj_fd_ownership"] / 100.0
    hitters["proj_fd_upside"] = hitters["proj_fd_upside"].where(
        hitters["proj_fd_upside"] > 0,
        hitters["proj_fd_mean"],
    )

    rows = []
    for team, group in hitters.groupby("team_code"):
        top = group.sort_values("proj_fd_mean", ascending=False).head(max(1, config.top_hitters))
        run_dist = _probability_distribution(group, RUN_BUCKETS)
        hr_dist = _probability_distribution(group, HR_BUCKETS)
        rows.append(
            {
                "team_code": team,
                "eligible_hitters": int(len(group)),
                "confirmed_hitters": _confirmed_count(group),
                "team_runs": _first_positive(group, "vegas_team_total", "bpp_runs"),
                "bpp_runs": _first_positive(group, "bpp_runs"),
                "vegas_team_total": _first_positive(group, "vegas_team_total"),
                "win_percent": _first_positive(group, "bpp_win_percent"),
                "home_runs": _first_positive(group, "bpp_home_runs"),
                "p_runs_ge_5": _bucket_tail(run_dist, 5),
                "p_runs_ge_6": _bucket_tail(run_dist, 6),
                "p_runs_ge_8": _bucket_tail(run_dist, 8),
                "p90_runs": _bucket_percentile(run_dist, 0.90),
                "p_hr_ge_2": _bucket_tail(hr_dist, 2),
                "p_hr_ge_3": _bucket_tail(hr_dist, 3),
                "top5_projection": float(top["proj_fd_mean"].sum()),
                "top5_upside": float(top["proj_fd_upside"].sum()),
                "top5_bust_rate": float(top["proj_fd_bust_rate"].mean()),
                "stack_ownership": float(top["proj_fd_ownership"].sum()),
                "avg_hitter_ownership": float(top["proj_fd_ownership"].mean()),
                "chalk_hitters": int((top["proj_fd_ownership"] >= 0.20).sum()),
                "team_leverage_score": float(group["team_leverage_score"].mean()),
            }
        )

    plan = pd.DataFrame(rows)
    if plan.empty:
        return _empty_plan()

    plan["sim_upside_score"] = _ranked_average(
        plan,
        {
            "team_runs": 0.25,
            "p_runs_ge_5": 0.18,
            "p_runs_ge_6": 0.18,
            "p_runs_ge_8": 0.10,
            "p90_runs": 0.08,
            "p_hr_ge_2": 0.08,
            "top5_projection": 0.08,
            "top5_upside": 0.05,
        },
    )
    ownership_rank = _pct_rank(plan["stack_ownership"])
    plan["stack_leverage_score"] = (
        plan["sim_upside_score"]
        - ownership_rank
        + pd.to_numeric(plan["team_leverage_score"], errors="coerce").fillna(0.0) * 0.15
    )
    bust_penalty = pd.to_numeric(plan["top5_bust_rate"], errors="coerce").fillna(0.0).clip(0.0, 1.0) * 0.15
    ownership_discount = (1.0 - ownership_rank) * plan["sim_upside_score"]
    raw_score = (
        0.70 * plan["sim_upside_score"]
        + 0.20 * ownership_discount
        + 0.25 * _pct_rank(plan["stack_leverage_score"])
        - bust_penalty
    )
    raw_score = raw_score.fillna(0.0).clip(lower=0.0)
    plan["stack_score"] = raw_score
    plan["target_exposure"] = _allocate_targets(raw_score, config)
    plan["recommended_min_exposure"] = np.where(
        plan["target_exposure"] >= 0.08,
        (plan["target_exposure"] * 0.45).clip(config.min_team_exposure, 0.12),
        0.0,
    )
    plan["recommended_max_exposure"] = (
        (plan["target_exposure"] * 1.65 + 0.04)
        .clip(lower=0.08, upper=config.max_team_exposure)
        .fillna(0.08)
    )
    plan["tier"] = plan.apply(_recommendation_tier, axis=1)
    plan["recommendation"] = plan.apply(_recommendation_reason, axis=1)
    plan["stack_weight"] = _normalize_weights(plan["stack_score"])

    numeric_cols = [
        "team_runs",
        "bpp_runs",
        "vegas_team_total",
        "win_percent",
        "home_runs",
        "p_runs_ge_5",
        "p_runs_ge_6",
        "p_runs_ge_8",
        "p90_runs",
        "p_hr_ge_2",
        "p_hr_ge_3",
        "top5_projection",
        "top5_upside",
        "top5_bust_rate",
        "stack_ownership",
        "avg_hitter_ownership",
        "team_leverage_score",
        "sim_upside_score",
        "stack_leverage_score",
        "stack_score",
        "target_exposure",
        "recommended_min_exposure",
        "recommended_max_exposure",
        "stack_weight",
    ]
    for col in numeric_cols:
        if col in plan.columns:
            plan[col] = pd.to_numeric(plan[col], errors="coerce").fillna(0.0)

    return plan.sort_values(
        ["target_exposure", "stack_score", "team_runs"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def stack_plan_weights(plan: pd.DataFrame) -> Dict[str, float]:
    """Return team weights suitable for final portfolio selection."""

    if plan is None or plan.empty or "team_code" not in plan.columns:
        return {}
    score_col = "stack_weight" if "stack_weight" in plan.columns else "stack_score"
    weights = pd.to_numeric(plan.get(score_col), errors="coerce").fillna(0.0)
    return {
        str(team): float(weight)
        for team, weight in zip(plan["team_code"].astype(str), weights)
        if float(weight) > 0
    }


def _empty_plan() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "team_code",
            "eligible_hitters",
            "confirmed_hitters",
            "team_runs",
            "p_runs_ge_5",
            "p_runs_ge_6",
            "p_runs_ge_8",
            "p90_runs",
            "p_hr_ge_2",
            "top5_projection",
            "top5_upside",
            "stack_ownership",
            "sim_upside_score",
            "stack_leverage_score",
            "stack_score",
            "target_exposure",
            "recommended_min_exposure",
            "recommended_max_exposure",
            "stack_weight",
            "tier",
            "recommendation",
        ]
    )


def _confirmed_count(group: pd.DataFrame) -> int:
    if "is_confirmed_lineup" not in group.columns:
        return 0
    return int(group["is_confirmed_lineup"].fillna(False).astype(bool).sum())


def _first_positive(group: pd.DataFrame, *columns: str) -> float:
    for column in columns:
        if column not in group.columns:
            continue
        values = pd.to_numeric(group[column], errors="coerce")
        values = values[values > 0]
        if not values.empty:
            return float(values.iloc[0])
    return 0.0


def _probability_distribution(group: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [col for col in columns if col in group.columns]
    if not available:
        return pd.Series(dtype=float)
    first = group[available].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if first.empty:
        return pd.Series(dtype=float)
    values = first.iloc[0].fillna(0.0).astype(float)
    total = float(values.sum())
    if total > 1.5:
        values = values / 100.0
        total = float(values.sum())
    if total <= 0:
        return pd.Series(dtype=float)
    return values / total


def _bucket_tail(dist: pd.Series, threshold: int) -> float:
    if dist.empty:
        return 0.0
    total = 0.0
    for key, value in dist.items():
        try:
            bucket = int(str(key).split("runs")[-1].split("home_runs")[-1])
        except ValueError:
            bucket = _trailing_int(str(key))
        if bucket >= threshold:
            total += float(value)
    return float(total)


def _bucket_percentile(dist: pd.Series, quantile: float) -> float:
    if dist.empty:
        return 0.0
    running = 0.0
    for key, value in dist.items():
        bucket = _trailing_int(str(key))
        running += float(value)
        if running >= quantile:
            return float(bucket)
    return float(_trailing_int(str(dist.index[-1])))


def _trailing_int(value: str) -> int:
    digits = []
    for char in reversed(value):
        if char.isdigit():
            digits.append(char)
        elif digits:
            break
    return int("".join(reversed(digits))) if digits else 0


def _pct_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if len(values) <= 1 or values.nunique() <= 1:
        return pd.Series(0.5, index=values.index, dtype=float)
    return values.rank(pct=True)


def _ranked_average(plan: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    total_weight = sum(max(0.0, weight) for weight in weights.values())
    if total_weight <= 0:
        return pd.Series(0.5, index=plan.index, dtype=float)
    score = pd.Series(0.0, index=plan.index, dtype=float)
    for column, weight in weights.items():
        if weight <= 0:
            continue
        score += _pct_rank(plan.get(column, pd.Series(0.0, index=plan.index))) * weight
    return score / total_weight


def _allocate_targets(score: pd.Series, config: StackExposureConfig) -> pd.Series:
    values = pd.to_numeric(score, errors="coerce").fillna(0.0).clip(lower=0.0)
    if values.sum() <= 0:
        return pd.Series(0.0, index=values.index, dtype=float)
    shifted = values - values.max()
    raw = np.exp(shifted * 2.0)
    raw = pd.Series(raw, index=values.index, dtype=float)
    targets = raw / raw.sum() * max(0.0, float(config.total_stack_exposure))
    targets = targets.clip(lower=config.min_team_exposure, upper=config.max_team_exposure)
    targets = targets.where(targets >= config.min_viable_target, 0.0)
    return targets.fillna(0.0)


def _normalize_weights(score: pd.Series) -> pd.Series:
    values = pd.to_numeric(score, errors="coerce").fillna(0.0).clip(lower=0.0)
    if values.sum() <= 0:
        return pd.Series(0.0, index=values.index, dtype=float)
    return values / values.max()


def _recommendation_tier(row: pd.Series) -> str:
    target = float(row.get("target_exposure", 0.0) or 0.0)
    leverage = float(row.get("stack_leverage_score", 0.0) or 0.0)
    ownership = float(row.get("stack_ownership", 0.0) or 0.0)
    upside = float(row.get("sim_upside_score", 0.0) or 0.0)
    if upside < 0.45:
        return "Underweight"
    if target >= 0.16 and leverage >= 0 and upside >= 0.65:
        return "Core overweight"
    if target >= 0.10 and leverage > 0.05 and upside >= 0.55:
        return "Leverage overweight"
    if ownership >= 0.85 and upside >= 0.70:
        return "Controlled chalk"
    if target <= 0.03:
        return "Underweight"
    return "Neutral"


def _recommendation_reason(row: pd.Series) -> str:
    tier = str(row.get("tier", "Neutral"))
    if tier == "Core overweight":
        return "Strong run/HR sim profile with enough ownership leverage to anchor stacks."
    if tier == "Leverage overweight":
        return "Comparable stack ceiling at lower expected field ownership."
    if tier == "Controlled chalk":
        return "High-upside team, but ownership calls for a cap instead of a full fade."
    if tier == "Underweight":
        return "Lower stack ceiling or insufficient ownership discount."
    return "Playable, but best used as part of portfolio balance."


__all__ = [
    "StackExposureConfig",
    "build_stack_exposure_plan",
    "stack_plan_weights",
]
