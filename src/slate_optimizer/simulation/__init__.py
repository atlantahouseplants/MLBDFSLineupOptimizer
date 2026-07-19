"""Simulation package for probabilistic contest modeling."""
from .config import SimulationConfig
from .contest_simulator import ContestSimResult, LineupSimResult, simulate_contest
from .contest_type import (
    CONTEST_TYPE_PROFILES,
    ContestTypeProfile,
    apply_contest_type_to_state,
    payout_top_heaviness,
    recommend_contest_type_profile,
)
from .correlation import CorrelationConfig, CorrelationModel, build_correlation_matrix
from .distributions import PlayerDistribution, fit_player_distributions
from .duplication import DuplicationEstimate, estimate_duplication_dict, estimate_lineup_duplication
from .field_simulator import FieldQualityMix, FieldRealismProfile, SimulatedField, simulate_field
from .field_learning import (
    FieldDuplicationProfile,
    compute_actual_ownership,
    extract_contest_lineup_players,
    learn_field_duplication_profile,
    load_field_duplication_profile,
    match_lineups_to_contest_entries,
    save_field_duplication_profile,
)
from .lineup_selector import PortfolioSelection, select_portfolio
from .ownership_scenarios import DEFAULT_OWNERSHIP_SCENARIOS, OwnershipScenario, apply_ownership_scenario_metrics
from .payouts import (
    DEFAULT_PAYOUT_LADDER_TEXT,
    PayoutBand,
    parse_payout_ladder_dataframe,
    parse_payout_ladder_text,
    payout_ladder_to_dicts,
    payout_ladder_to_text,
)
from .risk_budget import RISK_BUDGET_PROFILES, RiskBudgetProfile, apply_risk_budget_to_state
from .slate_simulator import SlateSimulation, simulate_slate
from .variance_reduction import (
    antithetic_uniforms,
    control_variate_adjustment,
    stratified_uniforms,
)

__all__ = [
    "SimulationConfig",
    "PlayerDistribution",
    "fit_player_distributions",
    "DuplicationEstimate",
    "estimate_duplication_dict",
    "estimate_lineup_duplication",
    "CorrelationConfig",
    "CONTEST_TYPE_PROFILES",
    "ContestTypeProfile",
    "apply_contest_type_to_state",
    "payout_top_heaviness",
    "recommend_contest_type_profile",
    "CorrelationModel",
    "build_correlation_matrix",
    "SlateSimulation",
    "simulate_slate",
    "SimulatedField",
    "FieldQualityMix",
    "FieldRealismProfile",
    "simulate_field",
    "FieldDuplicationProfile",
    "compute_actual_ownership",
    "extract_contest_lineup_players",
    "learn_field_duplication_profile",
    "load_field_duplication_profile",
    "match_lineups_to_contest_entries",
    "save_field_duplication_profile",
    "LineupSimResult",
    "ContestSimResult",
    "simulate_contest",
    "PortfolioSelection",
    "select_portfolio",
    "DEFAULT_OWNERSHIP_SCENARIOS",
    "OwnershipScenario",
    "apply_ownership_scenario_metrics",
    "RISK_BUDGET_PROFILES",
    "RiskBudgetProfile",
    "apply_risk_budget_to_state",
    "DEFAULT_PAYOUT_LADDER_TEXT",
    "PayoutBand",
    "parse_payout_ladder_dataframe",
    "parse_payout_ladder_text",
    "payout_ladder_to_dicts",
    "payout_ladder_to_text",
    "antithetic_uniforms",
    "stratified_uniforms",
    "control_variate_adjustment",
]
