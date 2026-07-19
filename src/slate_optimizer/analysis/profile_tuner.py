"""Tune slate-size profile numbers from recorded lineup results.

Reads ``lineup_results`` from slates.db (written by Step 6), groups slates by
the slate-size profile that produced them (recorded in strategy_config_json),
and checks whether the profile's chalk/ceiling posture matched how lineups
actually finished:

- If high-ownership lineups consistently outscored low-ownership ones, the
  ownership penalty for that profile is too aggressive — nudge it down.
- If leverage lineups outscored chalk, nudge the penalty up.
- Same test for upside (ceiling weight) using lineup total Upside.

All adjustments are bounded single steps gated on a minimum sample of slates,
so one hot night can never rewrite the strategy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from slate_optimizer.data.storage import SlateDatabase

MIN_LINEUPS_PER_SLATE = 5
CORR_THRESHOLD = 0.15

OWNERSHIP_STEP = 2.0
OWNERSHIP_FLOOR = 4.0
OWNERSHIP_CAP = 20.0
CEILING_STEP = 5.0
CEILING_FLOOR = 20.0
CEILING_CAP = 50.0


@dataclass
class ProfileTuningReport:
    per_profile: pd.DataFrame
    recommendations: pd.DataFrame
    adjustments: Dict[str, Dict[str, float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _slate_correlation(group: pd.DataFrame, metric: str) -> float | None:
    values = pd.to_numeric(group.get(metric), errors="coerce")
    actual = pd.to_numeric(group.get("total_actual_points"), errors="coerce")
    mask = values.notna() & actual.notna()
    if mask.sum() < MIN_LINEUPS_PER_SLATE:
        return None
    values, actual = values[mask], actual[mask]
    if values.std(ddof=0) < 1e-9 or actual.std(ddof=0) < 1e-9:
        return None
    return float(np.corrcoef(values, actual)[0, 1])


def _extract_profile_context(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        "profile": payload.get("slate_profile"),
        "games": payload.get("slate_games"),
    }


def build_profile_tuning_report(
    db_path: Path | str,
    start_date: str,
    end_date: str,
    current_profiles: Dict[str, Dict[str, Any]],
    min_slates: int = 8,
) -> ProfileTuningReport:
    db = SlateDatabase(Path(db_path))
    results = db.fetch_lineup_results_range(start_date, end_date)
    db.close()

    if results.empty:
        return ProfileTuningReport(
            pd.DataFrame(),
            pd.DataFrame(),
            notes=["No lineup results recorded in the selected window — process slates in Step 6 first."],
        )

    context = results["strategy_config_json"].map(_extract_profile_context)
    results = results.assign(
        profile=[c.get("profile") for c in context],
        games=[c.get("games") for c in context],
    )
    untagged = int(results["profile"].isna().sum() + (results["profile"] == "").sum())
    results = results[results["profile"].notna() & (results["profile"] != "")]

    notes: List[str] = []
    if untagged:
        notes.append(
            f"{untagged} recorded lineups predate profile tagging and were skipped "
            "(results from before the tuner shipped)."
        )
    if results.empty:
        notes.append("No profile-tagged results yet — tuning starts once new slates are processed in Step 6.")
        return ProfileTuningReport(pd.DataFrame(), pd.DataFrame(), notes=notes)

    slate_rows = []
    for (date, profile), group in results.groupby(["date", "profile"]):
        slate_rows.append(
            {
                "date": date,
                "profile": profile,
                "lineups": len(group),
                "avg_actual": float(pd.to_numeric(group["total_actual_points"], errors="coerce").mean()),
                "avg_roi": float(pd.to_numeric(group.get("roi"), errors="coerce").mean()),
                "own_corr": _slate_correlation(group, "total_ownership"),
                "upside_corr": _slate_correlation(group, "total_upside"),
            }
        )
    slates = pd.DataFrame(slate_rows)

    per_profile = (
        slates.groupby("profile")
        .agg(
            slates=("date", "nunique"),
            lineups=("lineups", "sum"),
            avg_actual=("avg_actual", "mean"),
            avg_roi=("avg_roi", "mean"),
            own_corr=("own_corr", "mean"),
            upside_corr=("upside_corr", "mean"),
        )
        .reset_index()
    )

    recommendations: List[Dict[str, Any]] = []
    adjustments: Dict[str, Dict[str, float]] = {}

    for _, row in per_profile.iterrows():
        profile = str(row["profile"])
        settings = (current_profiles.get(profile) or {}).get("optimizer") or {}
        if int(row["slates"]) < min_slates:
            notes.append(
                f"Profile '{profile}': {int(row['slates'])}/{min_slates} slates recorded — "
                "no recommendation until the sample fills out."
            )
            continue

        own_corr = row["own_corr"]
        if own_corr is not None and not pd.isna(own_corr):
            current = float(settings.get("ownership_penalty_weight", 10.0) or 0.0)
            if own_corr >= CORR_THRESHOLD and current > OWNERSHIP_FLOOR:
                new_value = max(OWNERSHIP_FLOOR, current - OWNERSHIP_STEP)
                _recommend(
                    recommendations, adjustments, profile, "ownership_penalty_weight", current, new_value,
                    f"Chalk lineups outscored leverage builds (avg corr {own_corr:+.2f}) — fade less.",
                )
            elif own_corr <= -CORR_THRESHOLD and current < OWNERSHIP_CAP:
                new_value = min(OWNERSHIP_CAP, current + OWNERSHIP_STEP)
                _recommend(
                    recommendations, adjustments, profile, "ownership_penalty_weight", current, new_value,
                    f"Low-owned lineups outscored chalk (avg corr {own_corr:+.2f}) — fade harder.",
                )

        upside_corr = row["upside_corr"]
        if upside_corr is not None and not pd.isna(upside_corr):
            current = float(settings.get("ceiling_weight", 35.0) or 0.0)
            if upside_corr >= CORR_THRESHOLD and current < CEILING_CAP:
                new_value = min(CEILING_CAP, current + CEILING_STEP)
                _recommend(
                    recommendations, adjustments, profile, "ceiling_weight", current, new_value,
                    f"High-Upside builds converted (avg corr {upside_corr:+.2f}) — weight ceiling more.",
                )
            elif upside_corr <= -CORR_THRESHOLD and current > CEILING_FLOOR:
                new_value = max(CEILING_FLOOR, current - CEILING_STEP)
                _recommend(
                    recommendations, adjustments, profile, "ceiling_weight", current, new_value,
                    f"High-Upside builds underperformed (avg corr {upside_corr:+.2f}) — weight ceiling less.",
                )

    if not recommendations and not notes:
        notes.append("Profiles look calibrated against the recorded window — no changes recommended.")

    return ProfileTuningReport(
        per_profile=per_profile,
        recommendations=pd.DataFrame(recommendations),
        adjustments=adjustments,
        notes=notes,
    )


def _recommend(
    recommendations: List[Dict[str, Any]],
    adjustments: Dict[str, Dict[str, float]],
    profile: str,
    setting: str,
    current: float,
    new_value: float,
    reason: str,
) -> None:
    if abs(current - new_value) < 1e-9:
        return
    recommendations.append(
        {
            "profile": profile,
            "setting": setting,
            "current": current,
            "recommended": new_value,
            "reason": reason,
        }
    )
    adjustments.setdefault(profile, {})[setting] = new_value


__all__ = ["ProfileTuningReport", "build_profile_tuning_report"]
