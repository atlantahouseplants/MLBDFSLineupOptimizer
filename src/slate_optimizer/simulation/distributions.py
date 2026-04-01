from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from scipy.stats import lognorm, truncnorm

__all__ = [
    "PlayerDistribution",
    "fit_player_distributions",
]


_DTYPE = np.float64


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class PlayerDistribution:
    """Parameterized fantasy point distribution for one player."""

    fd_player_id: str
    full_name: str
    player_type: Literal["batter", "pitcher"]
    dist_type: Literal["lognormal", "truncated_normal"]
    mu: float
    sigma: float
    shift: float
    proj_mean: float
    proj_floor: float
    proj_ceiling: float
    salary: int
    bust_rate: float = 0.15

    def _lognormal(self):
        return lognorm(s=self.sigma, scale=math.exp(self.mu), loc=self.shift)

    def _truncnorm(self):
        a = (0.0 - self.mu) / self.sigma
        return truncnorm(a=a, b=np.inf, loc=self.mu, scale=self.sigma)

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        if self.dist_type == "lognormal":
            draws = rng.lognormal(mean=self.mu, sigma=self.sigma, size=n)
            return draws.astype(_DTYPE) + self.shift
        if self.dist_type == "truncated_normal":
            dist = self._truncnorm()
            return dist.rvs(size=n, random_state=rng).astype(_DTYPE)
        raise ValueError(f"Unknown distribution type: {self.dist_type}")

    def _prepare_quantiles(self, quantiles: np.ndarray | float) -> np.ndarray:
        arr = np.asarray(quantiles, dtype=_DTYPE)
        return np.clip(arr, 1e-6, 1 - 1e-6)

    def ppf(self, quantiles: np.ndarray | float) -> np.ndarray:
        qs = self._prepare_quantiles(quantiles)
        if self.dist_type == "lognormal":
            dist = self._lognormal()
            return dist.ppf(qs)
        dist = self._truncnorm()
        return dist.ppf(qs)

    def cdf(self, values: np.ndarray | float) -> np.ndarray:
        arr = np.asarray(values, dtype=_DTYPE)
        if self.dist_type == "lognormal":
            dist = self._lognormal()
            return dist.cdf(arr)
        dist = self._truncnorm()
        return dist.cdf(arr)

    def mean(self) -> float:
        if self.dist_type == "lognormal":
            return float(math.exp(self.mu + 0.5 * self.sigma ** 2) + self.shift)
        dist = self._truncnorm()
        return float(dist.mean())

    def percentile(self, p: float) -> float:
        return float(self.ppf(np.array([_clamp(p, 0.0, 100.0) / 100.0], dtype=_DTYPE))[0])


def fit_player_distributions(
    optimizer_df: pd.DataFrame,
    volatility_scale: float = 1.0,
    gpp_mode: bool = False,
) -> Dict[str, PlayerDistribution]:
    """Fit fantasy score distributions for every player in the optimizer dataset."""

    volatility_scale = max(0.05, float(volatility_scale))
    dists: Dict[str, PlayerDistribution] = {}
    for record in optimizer_df.to_dict("records"):
        player_id = str(record.get("fd_player_id"))
        if not player_id:
            continue
        full_name = str(record.get("full_name", ""))
        player_type = str(record.get("player_type", "batter")).lower()
        mean = float(record.get("proj_fd_mean", 0.0) or 0.0)
        floor = float(record.get("proj_fd_floor", mean * 0.8) or mean * 0.8)
        ceiling = float(record.get("proj_fd_ceiling", mean * 1.2) or mean * 1.2)
        salary = int(record.get("salary", 0) or 0)
        bust_rate = float(record.get("bust_rate", 0.15) or 0.15)

        if player_type == "pitcher":
            mu, sigma, shift = _fit_pitcher_normal(mean, floor, ceiling)
            dist_type: Literal["lognormal", "truncated_normal"] = "truncated_normal"
        else:
            mu, sigma, shift = _fit_batter_lognormal(mean, floor, ceiling, salary, gpp_mode=gpp_mode)
            player_type = "batter"
            dist_type = "lognormal"

        sigma *= volatility_scale
        dists[player_id] = PlayerDistribution(
            fd_player_id=player_id,
            full_name=full_name,
            player_type=player_type,
            dist_type=dist_type,
            mu=float(mu),
            sigma=max(1e-3, float(sigma)),
            shift=float(shift),
            proj_mean=mean,
            proj_floor=floor,
            proj_ceiling=ceiling,
            salary=salary,
            bust_rate=bust_rate,
        )
    return dists


def _fit_batter_lognormal(
    mean: float, floor: float, ceiling: float, salary: int, gpp_mode: bool = False,
) -> tuple[float, float, float]:
    shift = max(0.0, float(floor) * 0.5)
    adj_mean = max(mean - shift, 0.1)
    adj_ceiling = max(ceiling - shift, adj_mean * 1.05)
    ratio = adj_ceiling / adj_mean
    salary = max(int(salary) if salary else 0, 1)
    if gpp_mode:
        # GPP: wider distributions for cheap players (more boom/bust)
        salary_factor = _clamp(3500.0 / salary, 0.8, 1.5)
    else:
        salary_factor = _clamp(4000.0 / salary, 0.7, 1.3)
    if ratio <= 1.05:
        sigma = 0.6 * salary_factor
    else:
        sigma = math.sqrt(max(1e-6, math.log(ratio))) * salary_factor
    mu = math.log(max(adj_mean, 1e-3)) - 0.5 * sigma ** 2
    return mu, sigma, shift


def _fit_pitcher_normal(mean: float, floor: float, ceiling: float) -> tuple[float, float, float]:
    spread = max(ceiling - floor, 1.0)
    sigma = spread / (2.0 * 1.645)
    if not math.isfinite(sigma) or sigma <= 0:
        sigma = max(abs(mean) * 0.1, 3.0)
    return mean, sigma, 0.0
