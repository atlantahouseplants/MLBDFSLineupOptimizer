"""Historical optimizer weight tuning from slate replay results."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .slate_replay import build_slate_replay_report


@dataclass
class OptimizerWeightTuningReport:
    candidates: pd.DataFrame
    recommended_profile: str
    recommended_state: dict[str, float | str | bool]
    lessons: list[str]


PROFILE_STATE_MAP: dict[str, dict[str, float | str | bool]] = {
    "Projection only": {
        "selection_metric": "payout_ev",
        "selection_leverage_weight": 0.06,
        "selection_ownership_weight": 0.04,
        "selection_duplication_weight": 0.12,
        "selection_scenario_weight": 0.06,
        "selection_marginal_value_weight": 0.10,
        "stack_target_weight": 0.12,
        "risk_profile": "Chalk-neutral unique",
    },
    "Balanced leverage": {
        "selection_metric": "payout_ev",
        "selection_leverage_weight": 0.20,
        "selection_ownership_weight": 0.13,
        "selection_duplication_weight": 0.24,
        "selection_scenario_weight": 0.16,
        "selection_marginal_value_weight": 0.24,
        "stack_target_weight": 0.24,
        "risk_profile": "Balanced leverage",
    },
    "Chalk fade": {
        "selection_metric": "payout_ev",
        "selection_leverage_weight": 0.32,
        "selection_ownership_weight": 0.28,
        "selection_duplication_weight": 0.40,
        "selection_scenario_weight": 0.26,
        "selection_marginal_value_weight": 0.34,
        "stack_target_weight": 0.28,
        "risk_profile": "High leverage",
    },
    "Stack upside": {
        "selection_metric": "payout_ev",
        "selection_leverage_weight": 0.24,
        "selection_ownership_weight": 0.14,
        "selection_duplication_weight": 0.28,
        "selection_scenario_weight": 0.18,
        "selection_marginal_value_weight": 0.30,
        "stack_target_weight": 0.46,
        "risk_profile": "Stack-heavy",
        "use_stack_exposure_engine": True,
        "use_stack_auto_caps": True,
    },
    "Boom/bust leverage": {
        "selection_metric": "payout_ev",
        "selection_leverage_weight": 0.36,
        "selection_ownership_weight": 0.20,
        "selection_duplication_weight": 0.34,
        "selection_scenario_weight": 0.24,
        "selection_marginal_value_weight": 0.36,
        "stack_target_weight": 0.32,
        "risk_profile": "Extreme top-heavy GPP",
    },
}


def build_optimizer_weight_tuning_report(
    db_path: Path | str,
    start_date: str,
    end_date: str,
    data_dir: Path | str | None = None,
) -> OptimizerWeightTuningReport:
    replay = build_slate_replay_report(db_path, start_date, end_date, data_dir=data_dir)
    candidates = replay.summary.copy()
    if candidates.empty:
        return OptimizerWeightTuningReport(pd.DataFrame(), "", {}, ["No replay rows available for optimizer tuning."])

    candidates["stability_score"] = (
        pd.to_numeric(candidates.get("avg_replay_score"), errors="coerce").fillna(0.0)
        + pd.to_numeric(candidates.get("boom_hit_rate"), errors="coerce").fillna(0.0) * 1.5
        + pd.to_numeric(candidates.get("stack_hit_rate"), errors="coerce").fillna(0.0) * 1.0
        - pd.to_numeric(candidates.get("chalk_selected_pct"), errors="coerce").fillna(0.0) * 0.35
    )
    candidates = candidates.sort_values("stability_score", ascending=False).reset_index(drop=True)
    recommended_profile = str(candidates.iloc[0]["strategy_profile"])
    recommended_state = dict(PROFILE_STATE_MAP.get(recommended_profile, PROFILE_STATE_MAP["Balanced leverage"]))
    lessons = _lessons(candidates, recommended_profile)
    return OptimizerWeightTuningReport(
        candidates=candidates,
        recommended_profile=recommended_profile,
        recommended_state=recommended_state,
        lessons=lessons,
    )


def _lessons(candidates: pd.DataFrame, recommended_profile: str) -> list[str]:
    notes = [
        f"Recommended weight profile: {recommended_profile}.",
    ]
    if len(candidates) >= 2:
        leader = float(candidates.iloc[0].get("stability_score", 0.0) or 0.0)
        runner = float(candidates.iloc[1].get("stability_score", 0.0) or 0.0)
        if abs(leader - runner) < 0.35:
            notes.append("Top profiles are close; treat this as a gentle nudge rather than a hard switch.")
    chalk = float(candidates.iloc[0].get("chalk_selected_pct", 0.0) or 0.0)
    if chalk >= 0.35:
        notes.append("Winning replay profile still selected plenty of chalk, so fades should be stack-specific rather than automatic.")
    return notes


__all__ = ["OptimizerWeightTuningReport", "PROFILE_STATE_MAP", "build_optimizer_weight_tuning_report"]
