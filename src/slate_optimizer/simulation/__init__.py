"""Simulation package for probabilistic contest modeling."""
from .config import SimulationConfig
from .contest_simulator import ContestSimResult, LineupSimResult, simulate_contest
from .correlation import CorrelationConfig, CorrelationModel, build_correlation_matrix
from .distributions import PlayerDistribution, fit_player_distributions
from .field_simulator import FieldQualityMix, SimulatedField, simulate_field
from .lineup_selector import PortfolioSelection, select_portfolio
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
    "CorrelationConfig",
    "CorrelationModel",
    "build_correlation_matrix",
    "SlateSimulation",
    "simulate_slate",
    "SimulatedField",
    "FieldQualityMix",
    "simulate_field",
    "LineupSimResult",
    "ContestSimResult",
    "simulate_contest",
    "PortfolioSelection",
    "select_portfolio",
    "antithetic_uniforms",
    "stratified_uniforms",
    "control_variate_adjustment",
]
