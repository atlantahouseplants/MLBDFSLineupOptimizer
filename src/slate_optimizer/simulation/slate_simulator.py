from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import t as student_t

from .distributions import PlayerDistribution
from .correlation import CorrelationModel

__all__ = [
    "SlateSimulation",
    "simulate_slate",
]


@dataclass
class SlateSimulation:
    scores: np.ndarray
    player_ids: List[str]

    def __post_init__(self) -> None:
        self.player_id_to_index = {pid: idx for idx, pid in enumerate(self.player_ids)}

    @property
    def num_simulations(self) -> int:
        return int(self.scores.shape[0])

    @property
    def num_players(self) -> int:
        return int(self.scores.shape[1])

    def player_scores(self, fd_player_id: str) -> np.ndarray:
        idx = self.player_id_to_index[fd_player_id]
        return self.scores[:, idx]

    def lineup_scores(self, player_ids: List[str]) -> np.ndarray:
        indices = [self.player_id_to_index[pid] for pid in player_ids]
        return self.scores[:, indices].sum(axis=1)

    def percentile_score(self, player_ids: List[str], pct: float) -> float:
        return float(np.percentile(self.lineup_scores(player_ids), pct))

    def summary_stats(self, player_ids: List[str]) -> Dict[str, float]:
        scores = self.lineup_scores(player_ids)
        return {
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "std": float(scores.std()),
            "p10": float(np.percentile(scores, 10)),
            "p25": float(np.percentile(scores, 25)),
            "p75": float(np.percentile(scores, 75)),
            "p90": float(np.percentile(scores, 90)),
            "p99": float(np.percentile(scores, 99)),
            "max": float(scores.max()),
            "min": float(scores.min()),
        }


def simulate_slate(
    distributions: Dict[str, PlayerDistribution],
    correlation_model: CorrelationModel,
    num_simulations: int = 10_000,
    seed: Optional[int] = None,
    use_antithetic: bool = True,
) -> SlateSimulation:
    if not distributions:
        raise ValueError("No player distributions provided")

    rng = np.random.default_rng(seed)
    player_ids = correlation_model.player_ids
    missing = [pid for pid in player_ids if pid not in distributions]
    if missing:
        raise KeyError(f"Missing distributions for players: {missing[:5]}")

    draws = int(num_simulations)
    if draws <= 0:
        raise ValueError("num_simulations must be positive")

    if use_antithetic:
        base_draws = (draws + 1) // 2
    else:
        base_draws = draws

    uniforms = _student_t_copula_draws(
        correlation_model,
        base_draws,
        rng,
    )

    if use_antithetic:
        anti = 1.0 - uniforms
        uniforms = np.vstack([uniforms, anti])
    uniforms = uniforms[:draws]

    scores = np.zeros((uniforms.shape[0], uniforms.shape[1]), dtype=np.float64)
    for j, pid in enumerate(player_ids):
        dist = distributions[pid]
        scores[:, j] = dist.ppf(uniforms[:, j])
    return SlateSimulation(scores=scores, player_ids=player_ids)


def _student_t_copula_draws(
    correlation_model: CorrelationModel,
    num_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    p = len(correlation_model.player_ids)
    z = rng.standard_normal(size=(num_draws, p))
    correlated = z @ correlation_model.cholesky.T
    chi2 = rng.chisquare(df=correlation_model.nu, size=num_draws)
    scaled = correlated / np.sqrt((chi2 / correlation_model.nu)[:, None])
    uniforms = student_t.cdf(scaled, df=correlation_model.nu)
    return np.clip(uniforms, 1e-6, 1 - 1e-6)
