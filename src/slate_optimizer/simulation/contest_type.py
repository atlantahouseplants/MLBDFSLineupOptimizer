"""Contest-shape profiles for tuning final portfolio construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .payouts import PayoutBand


@dataclass(frozen=True)
class ContestTypeProfile:
    name: str
    description: str
    settings: Mapping[str, float | str | bool]


CONTEST_TYPE_PROFILES: dict[str, ContestTypeProfile] = {
    "Massive 150/300-max GPP": ContestTypeProfile(
        "Massive 150/300-max GPP",
        "Large-field multi-entry build: lower duplication, wider stack spread, and stronger leverage.",
        {
            "selection_metric": "payout_ev",
            "diversity_weight": 0.44,
            "max_batter_exposure": 0.34,
            "max_pitcher_exposure": 0.52,
            "selection_leverage_weight": 0.30,
            "selection_ownership_weight": 0.22,
            "selection_duplication_weight": 0.36,
            "selection_scenario_weight": 0.24,
            "selection_marginal_value_weight": 0.34,
            "stack_target_weight": 0.32,
            "field_quality_shark_pct": 0.16,
            "field_quality_rec_pct": 0.62,
            "field_quality_random_pct": 0.22,
            "field_realism_value_bias": 0.28,
            "field_realism_train_bias": 0.24,
            "risk_profile": "High leverage",
            "use_ownership_scenarios": True,
            "use_stack_exposure_engine": True,
            "use_stack_auto_caps": True,
        },
    ),
    "Top-heavy large-field GPP": ContestTypeProfile(
        "Top-heavy large-field GPP",
        "First-place-heavy payout structure: more ceiling, uniqueness, and ownership uncertainty protection.",
        {
            "selection_metric": "payout_ev",
            "diversity_weight": 0.48,
            "max_batter_exposure": 0.30,
            "max_pitcher_exposure": 0.48,
            "selection_leverage_weight": 0.36,
            "selection_ownership_weight": 0.28,
            "selection_duplication_weight": 0.42,
            "selection_scenario_weight": 0.30,
            "selection_marginal_value_weight": 0.38,
            "stack_target_weight": 0.34,
            "field_quality_shark_pct": 0.18,
            "field_quality_rec_pct": 0.62,
            "field_quality_random_pct": 0.20,
            "field_realism_value_bias": 0.30,
            "field_realism_train_bias": 0.26,
            "risk_profile": "Extreme top-heavy GPP",
            "use_ownership_scenarios": True,
            "use_stack_exposure_engine": True,
            "use_stack_auto_caps": True,
        },
    ),
    "Large-field GPP": ContestTypeProfile(
        "Large-field GPP",
        "Balanced large-field posture for normal payout ladders and moderate ownership concentration.",
        {
            "selection_metric": "payout_ev",
            "diversity_weight": 0.36,
            "max_batter_exposure": 0.38,
            "max_pitcher_exposure": 0.56,
            "selection_leverage_weight": 0.22,
            "selection_ownership_weight": 0.16,
            "selection_duplication_weight": 0.26,
            "selection_scenario_weight": 0.18,
            "selection_marginal_value_weight": 0.26,
            "stack_target_weight": 0.24,
            "field_quality_shark_pct": 0.12,
            "field_quality_rec_pct": 0.62,
            "field_quality_random_pct": 0.26,
            "field_realism_value_bias": 0.22,
            "field_realism_train_bias": 0.18,
            "risk_profile": "Balanced leverage",
            "use_ownership_scenarios": True,
            "use_stack_exposure_engine": True,
        },
    ),
    "Small-field GPP": ContestTypeProfile(
        "Small-field GPP",
        "Smaller contest field: keep more projection and reduce forced contrarian pressure.",
        {
            "selection_metric": "payout_ev",
            "diversity_weight": 0.24,
            "max_batter_exposure": 0.50,
            "max_pitcher_exposure": 0.68,
            "selection_leverage_weight": 0.12,
            "selection_ownership_weight": 0.08,
            "selection_duplication_weight": 0.14,
            "selection_scenario_weight": 0.08,
            "selection_marginal_value_weight": 0.16,
            "stack_target_weight": 0.18,
            "field_quality_shark_pct": 0.08,
            "field_quality_rec_pct": 0.58,
            "field_quality_random_pct": 0.34,
            "field_realism_value_bias": 0.16,
            "field_realism_train_bias": 0.10,
            "risk_profile": "Chalk-neutral unique",
            "use_ownership_scenarios": True,
        },
    ),
    "Single-entry / 3-max": ContestTypeProfile(
        "Single-entry / 3-max",
        "Few entries: projection and lineup quality matter more than broad portfolio spread.",
        {
            "selection_metric": "payout_ev",
            "diversity_weight": 0.08,
            "max_batter_exposure": 1.0,
            "max_pitcher_exposure": 1.0,
            "selection_leverage_weight": 0.08,
            "selection_ownership_weight": 0.04,
            "selection_duplication_weight": 0.10,
            "selection_scenario_weight": 0.06,
            "selection_marginal_value_weight": 0.0,
            "stack_target_weight": 0.12,
            "field_quality_shark_pct": 0.20,
            "field_quality_rec_pct": 0.62,
            "field_quality_random_pct": 0.18,
            "field_realism_value_bias": 0.20,
            "field_realism_train_bias": 0.20,
            "risk_profile": "Chalk-neutral unique",
            "use_ownership_scenarios": True,
            "use_portfolio_optimizer": True,
        },
    ),
}


def recommend_contest_type_profile(
    contest_entries: int,
    num_lineups: int,
    payout_bands: Sequence[PayoutBand] | None = None,
) -> ContestTypeProfile:
    entries = max(1, int(contest_entries or 0))
    lineups = max(1, int(num_lineups or 1))
    top_heavy = payout_top_heaviness(payout_bands or [], entries)

    if lineups <= 3:
        return CONTEST_TYPE_PROFILES["Single-entry / 3-max"]
    if entries >= 40_000 and lineups >= 100:
        if top_heavy >= 0.34:
            return CONTEST_TYPE_PROFILES["Top-heavy large-field GPP"]
        return CONTEST_TYPE_PROFILES["Massive 150/300-max GPP"]
    if entries >= 8_000:
        if top_heavy >= 0.38:
            return CONTEST_TYPE_PROFILES["Top-heavy large-field GPP"]
        return CONTEST_TYPE_PROFILES["Large-field GPP"]
    return CONTEST_TYPE_PROFILES["Small-field GPP"]


def apply_contest_type_to_state(
    state: dict,
    profile_name: str = "Auto",
    payout_bands: Sequence[PayoutBand] | None = None,
) -> ContestTypeProfile:
    if profile_name == "Auto":
        profile = recommend_contest_type_profile(
            int(state.get("contest_entries", 50_000) or 50_000),
            int(state.get("num_candidates", 300) or 300),
            payout_bands=payout_bands,
        )
    else:
        profile = CONTEST_TYPE_PROFILES.get(profile_name) or CONTEST_TYPE_PROFILES["Large-field GPP"]
    state.update(dict(profile.settings))
    state["contest_type_profile"] = profile.name
    state["use_contest_type_brain"] = True
    return profile


def payout_top_heaviness(payout_bands: Iterable[PayoutBand], contest_entries: int) -> float:
    bands = list(payout_bands or [])
    if not bands:
        return 0.25
    total_prizes = 0.0
    top_prize = 0.0
    paid_entries = 0
    for band in bands:
        count = max(0, int(band.max_rank) - int(band.min_rank) + 1)
        total_prizes += count * float(band.payout)
        paid_entries = max(paid_entries, int(band.max_rank))
        if int(band.min_rank) <= 1 <= int(band.max_rank):
            top_prize = max(top_prize, float(band.payout))
    if total_prizes <= 0:
        return 0.25
    top_share = top_prize / total_prizes
    paid_share = paid_entries / max(1, int(contest_entries or paid_entries or 1))
    paid_pressure = max(0.0, 0.22 - paid_share)
    return float(min(1.0, top_share * 2.3 + paid_pressure))


__all__ = [
    "CONTEST_TYPE_PROFILES",
    "ContestTypeProfile",
    "apply_contest_type_to_state",
    "payout_top_heaviness",
    "recommend_contest_type_profile",
]
