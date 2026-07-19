"""Portfolio risk budget profiles that tune multiple engine knobs together."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RiskBudgetProfile:
    name: str
    description: str
    settings: Mapping[str, float | str | bool]


RISK_BUDGET_PROFILES: dict[str, RiskBudgetProfile] = {
    "Balanced leverage": RiskBudgetProfile(
        "Balanced leverage",
        "Default 300-entry GPP posture: meaningful leverage without forcing every lineup off the board.",
        {
            "selection_metric": "payout_ev",
            "diversity_weight": 0.34,
            "max_batter_exposure": 0.40,
            "max_pitcher_exposure": 0.58,
            "selection_leverage_weight": 0.18,
            "selection_ownership_weight": 0.12,
            "selection_duplication_weight": 0.22,
            "selection_scenario_weight": 0.14,
            "selection_marginal_value_weight": 0.22,
            "stack_target_weight": 0.22,
            "use_ownership_scenarios": True,
        },
    ),
    "High leverage": RiskBudgetProfile(
        "High leverage",
        "Fades more chalk and prioritizes robust low-duplication lineups for large-field top-heavy contests.",
        {
            "selection_metric": "payout_ev",
            "diversity_weight": 0.42,
            "max_batter_exposure": 0.34,
            "max_pitcher_exposure": 0.52,
            "selection_leverage_weight": 0.30,
            "selection_ownership_weight": 0.22,
            "selection_duplication_weight": 0.34,
            "selection_scenario_weight": 0.22,
            "selection_marginal_value_weight": 0.30,
            "stack_target_weight": 0.30,
            "use_ownership_scenarios": True,
        },
    ),
    "Extreme top-heavy GPP": RiskBudgetProfile(
        "Extreme top-heavy GPP",
        "Maximizes unique first-place paths. Use when payout is very top-heavy and you accept more bust risk.",
        {
            "selection_metric": "payout_ev",
            "diversity_weight": 0.50,
            "max_batter_exposure": 0.28,
            "max_pitcher_exposure": 0.45,
            "selection_leverage_weight": 0.38,
            "selection_ownership_weight": 0.30,
            "selection_duplication_weight": 0.42,
            "selection_scenario_weight": 0.28,
            "selection_marginal_value_weight": 0.38,
            "stack_target_weight": 0.34,
            "use_ownership_scenarios": True,
        },
    ),
    "Chalk-neutral unique": RiskBudgetProfile(
        "Chalk-neutral unique",
        "Allows some good chalk but pushes for uncommon combinations, lower duplicate risk, and cleaner pivots.",
        {
            "selection_metric": "payout_ev",
            "diversity_weight": 0.36,
            "max_batter_exposure": 0.42,
            "max_pitcher_exposure": 0.58,
            "selection_leverage_weight": 0.14,
            "selection_ownership_weight": 0.10,
            "selection_duplication_weight": 0.38,
            "selection_scenario_weight": 0.22,
            "selection_marginal_value_weight": 0.30,
            "stack_target_weight": 0.18,
            "use_ownership_scenarios": True,
        },
    ),
    "Stack-heavy": RiskBudgetProfile(
        "Stack-heavy",
        "Lets the Ballpark/team stack plan drive more of the portfolio while still respecting duplication.",
        {
            "selection_metric": "payout_ev",
            "diversity_weight": 0.38,
            "max_batter_exposure": 0.36,
            "max_pitcher_exposure": 0.55,
            "selection_leverage_weight": 0.22,
            "selection_ownership_weight": 0.14,
            "selection_duplication_weight": 0.26,
            "selection_scenario_weight": 0.18,
            "selection_marginal_value_weight": 0.26,
            "stack_target_weight": 0.45,
            "use_ownership_scenarios": True,
            "use_stack_exposure_engine": True,
            "use_stack_auto_caps": True,
        },
    ),
}


def apply_risk_budget_to_state(state: dict, profile_name: str) -> dict:
    profile = RISK_BUDGET_PROFILES.get(profile_name)
    if not profile:
        return state
    state.update(dict(profile.settings))
    state["risk_profile"] = profile.name
    return state


__all__ = ["RISK_BUDGET_PROFILES", "RiskBudgetProfile", "apply_risk_budget_to_state"]
