from __future__ import annotations

import numpy as np

__all__ = [
    "antithetic_uniforms",
    "stratified_uniforms",
    "control_variate_adjustment",
]


def antithetic_uniforms(uniforms: np.ndarray) -> np.ndarray:
    """Stack uniform draws with their antithetic counterparts."""
    return np.vstack([uniforms, 1.0 - uniforms])


def stratified_uniforms(
    num_strata: int,
    num_per_stratum: int,
    num_players: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate stratified uniforms for the first dimension."""
    if num_strata <= 1:
        return rng.random((num_per_stratum, num_players))
    draws = []
    strata_edges = np.linspace(0.0, 1.0, num_strata + 1)
    for i in range(num_strata):
        low, high = strata_edges[i], strata_edges[i + 1]
        first_dim = rng.uniform(low, high, size=(num_per_stratum, 1))
        remaining = rng.random((num_per_stratum, num_players - 1))
        draws.append(np.hstack([first_dim, remaining]))
    return np.vstack(draws)


def control_variate_adjustment(
    raw_estimates: np.ndarray,
    control_values: np.ndarray,
    control_expected: float,
) -> np.ndarray:
    """Apply control variate adjustment to reduce estimator variance."""
    control_var = np.var(control_values)
    if control_var < 1e-10:
        return raw_estimates
    cov = np.cov(raw_estimates, control_values)[0, 1]
    beta = cov / control_var
    return raw_estimates - beta * (control_values - control_expected)
