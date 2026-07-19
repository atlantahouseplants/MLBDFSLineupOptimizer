"""Portfolio frontier comparisons from a simulated candidate set."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class PortfolioFrontierReport:
    summary: pd.DataFrame
    recommendation: str


def build_portfolio_frontier_report(
    contest_df: Optional[pd.DataFrame],
    lineup_df: Optional[pd.DataFrame] = None,
    target_lineups: int = 300,
) -> PortfolioFrontierReport:
    if contest_df is None or contest_df.empty:
        return PortfolioFrontierReport(pd.DataFrame(), "")

    df = contest_df.copy()
    if "lineup_id" not in df.columns:
        return PortfolioFrontierReport(pd.DataFrame(), "")
    for col in [
        "payout_ev",
        "expected_roi",
        "top_1pct_rate",
        "leverage_score",
        "total_ownership",
        "duplication_score",
        "field_duplication_rate",
        "uniqueness_score",
        "ownership_scenario_score",
    ]:
        if col not in df.columns:
            df[col] = 0.0 if col not in {"uniqueness_score", "ownership_scenario_score"} else 1.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0 if col not in {"uniqueness_score", "ownership_scenario_score"} else 1.0)

    profiles = {
        "Highest EV": _z(df["payout_ev"]) + 0.45 * _z(df["top_1pct_rate"]),
        "Leverage heavy": _z(df["payout_ev"]) + 0.85 * _z(df["leverage_score"]) - 0.45 * _z(df["total_ownership"]),
        "Low duplication": _z(df["payout_ev"]) + 0.80 * _z(df["uniqueness_score"]) - 0.70 * _z(df[["duplication_score", "field_duplication_rate"]].max(axis=1)),
        "High ceiling": _z(df["top_1pct_rate"]) + 0.35 * _z(df["payout_ev"]) + 0.25 * _z(df["leverage_score"]),
        "Balanced GPP": _z(df["payout_ev"]) + 0.45 * _z(df["leverage_score"]) - 0.30 * _z(df["total_ownership"]) - 0.35 * _z(df[["duplication_score", "field_duplication_rate"]].max(axis=1)) + 0.25 * _z(df["ownership_scenario_score"]),
    }

    rows = []
    target = max(1, min(int(target_lineups or 300), len(df)))
    for label, score in profiles.items():
        picks = df.assign(_frontier_score=score).sort_values("_frontier_score", ascending=False).head(target)
        lineup_ids = _normalize_ids(picks["lineup_id"], lineup_df)
        rows.append(
            {
                "profile": label,
                "lineups": int(len(picks)),
                "avg_payout_ev": float(picks["payout_ev"].mean()),
                "avg_roi": float(picks["expected_roi"].mean()),
                "avg_top_1pct": float(picks["top_1pct_rate"].mean()),
                "avg_leverage": float(picks["leverage_score"].mean()),
                "avg_ownership": float(picks["total_ownership"].mean()),
                "avg_duplication": float(picks[["duplication_score", "field_duplication_rate"]].max(axis=1).mean()),
                "avg_uniqueness": float(picks["uniqueness_score"].mean()),
                "lineup_ids": ",".join(str(int(v)) for v in lineup_ids[:20]),
            }
        )
    summary = pd.DataFrame(rows)
    recommendation = _recommend(summary)
    return PortfolioFrontierReport(summary, recommendation)


def _z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    std = float(values.std())
    if std <= 1e-9:
        return pd.Series(0.0, index=values.index)
    return (values - float(values.mean())) / std


def _normalize_ids(ids: pd.Series, lineup_df: Optional[pd.DataFrame]) -> list[int]:
    values = pd.to_numeric(ids, errors="coerce").dropna().astype(int).tolist()
    if not values:
        return []
    if lineup_df is not None and not lineup_df.empty and "lineup_id" in lineup_df.columns:
        lineup_ids = pd.to_numeric(lineup_df["lineup_id"], errors="coerce").dropna()
        if lineup_ids.min() >= 1 and min(values) == 0:
            return [v + 1 for v in values]
    return values


def _recommend(summary: pd.DataFrame) -> str:
    if summary.empty:
        return ""
    temp = summary.copy()
    for col in ["avg_payout_ev", "avg_top_1pct", "avg_leverage", "avg_uniqueness"]:
        temp[col + "_rank"] = temp[col].rank(pct=True)
    for col in ["avg_ownership", "avg_duplication"]:
        temp[col + "_rank"] = 1.0 - temp[col].rank(pct=True)
    temp["_score"] = (
        0.30 * temp["avg_payout_ev_rank"]
        + 0.20 * temp["avg_top_1pct_rank"]
        + 0.20 * temp["avg_leverage_rank"]
        + 0.15 * temp["avg_duplication_rank"]
        + 0.10 * temp["avg_ownership_rank"]
        + 0.05 * temp["avg_uniqueness_rank"]
    )
    row = temp.sort_values("_score", ascending=False).iloc[0]
    return f"Recommended frontier: {row['profile']} for the current large-field GPP tradeoff."


__all__ = ["PortfolioFrontierReport", "build_portfolio_frontier_report"]
