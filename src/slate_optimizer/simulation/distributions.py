from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Literal, Optional

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist
from scipy.stats import lognorm, truncnorm

from slate_optimizer.projection.score_distributions import fit_player_distributions as calculate_player_metrics, _get_env_flag

if TYPE_CHECKING:  # pragma: no cover
    from .calibration import SimulationCalibration

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
    dist_type: Literal["lognormal", "truncated_normal", "hurdle_gamma"]
    mu: float
    sigma: float
    shift: float
    proj_mean: float
    proj_floor: float
    proj_ceiling: float
    salary: int
    proj_median: float = 0.0
    bust_rate: float = 0.0
    # hurdle_gamma only: P(score == 0), and the gamma that sits on top of
    # `shift` for the non-zero scores.
    p_zero: float = 0.0
    gamma_shape: float = 1.0
    gamma_scale: float = 1.0
    # truncated_normal only: score range (legacy model: 0 to +infinity).
    lower_bound: float = 0.0
    upper_bound: float = float("inf")

    def _lognormal(self):
        return lognorm(s=self.sigma, scale=math.exp(self.mu), loc=self.shift)

    def _truncnorm(self):
        a = (self.lower_bound - self.mu) / self.sigma
        b = (self.upper_bound - self.mu) / self.sigma if math.isfinite(self.upper_bound) else np.inf
        return truncnorm(a=a, b=b, loc=self.mu, scale=self.sigma)

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        if self.dist_type == "hurdle_gamma":
            return np.asarray(self.ppf(rng.uniform(size=n)), dtype=_DTYPE)
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
        if self.dist_type == "hurdle_gamma":
            # Zero with probability p_zero; otherwise shift + gamma, read
            # at the quantile rescaled to the non-zero part.
            p_zero = min(max(self.p_zero, 0.0), 1.0 - 1e-9)
            inner = np.clip((qs - p_zero) / (1.0 - p_zero), 1e-9, 1 - 1e-9)
            positive = self.shift + gamma_dist.ppf(inner, a=self.gamma_shape, scale=self.gamma_scale)
            return np.where(qs <= p_zero, 0.0, positive)
        if self.dist_type == "lognormal":
            dist = self._lognormal()
            return dist.ppf(qs)
        dist = self._truncnorm()
        return dist.ppf(qs)

    def cdf(self, values: np.ndarray | float) -> np.ndarray:
        arr = np.asarray(values, dtype=_DTYPE)
        if self.dist_type == "hurdle_gamma":
            positive = gamma_dist.cdf(arr - self.shift, a=self.gamma_shape, scale=self.gamma_scale)
            result = self.p_zero + (1.0 - self.p_zero) * positive
            return np.where(arr < 0, 0.0, result)
        if self.dist_type == "lognormal":
            dist = self._lognormal()
            return dist.cdf(arr)
        dist = self._truncnorm()
        return dist.cdf(arr)

    def mean(self) -> float:
        if self.dist_type == "hurdle_gamma":
            positive_mean = self.shift + self.gamma_shape * self.gamma_scale
            return float((1.0 - self.p_zero) * positive_mean)
        if self.dist_type == "lognormal":
            return float(math.exp(self.mu + 0.5 * self.sigma ** 2) + self.shift)
        dist = self._truncnorm()
        return float(dist.mean())

    def percentile(self, p: float) -> float:
        return float(self.ppf(np.array([_clamp(p, 0.0, 100.0) / 100.0], dtype=_DTYPE))[0])


def fit_player_distributions(
    optimizer_df: pd.DataFrame,
    volatility_scale: float = 1.0,
    calibration: Optional["SimulationCalibration"] = None,
) -> Dict[str, PlayerDistribution]:
    """Fit fantasy score distributions for every player in the optimizer dataset.

    With ``calibration`` (see ``simulation/calibration.py``) every player gets
    the outcome-fitted model: a zero-inflated lognormal for hitters and a
    full-width normal for pitchers, both centred on ``proj_fd_mean``.  Without
    it the legacy floor/ceiling heuristics below are used unchanged.
    """

    volatility_scale = max(0.05, float(volatility_scale))
    if calibration is not None:
        return _fit_calibrated_distributions(optimizer_df, volatility_scale, calibration)
    dists: Dict[str, PlayerDistribution] = {}
    computed_boom_bust_enabled = _get_env_flag("COMPUTED_BOOM_BUST", False)
    for record in optimizer_df.to_dict("records"):
        player_id = str(record.get("fd_player_id"))
        if not player_id:
            continue
        full_name = str(record.get("full_name", ""))
        player_type = str(record.get("player_type", "batter")).lower()
        mean = float(record.get("proj_fd_mean", 0.0) or 0.0)
        salary = int(record.get("salary", 0) or 0)

        if computed_boom_bust_enabled:
            player_metrics = calculate_player_metrics(record, player_type)
            median = player_metrics['median_fd_points']
            floor = player_metrics['floor_fd_points']
            ceiling = player_metrics['ceiling_fd_points']
            bust_rate = player_metrics['bust_percentage']
        else:
            floor = float(record.get("proj_fd_floor", mean * 0.8) or mean * 0.8)
            ceiling = float(record.get("proj_fd_ceiling", mean * 1.2) or mean * 1.2)
            upside = float(record.get("proj_fd_upside", ceiling) or ceiling)
            ceiling = max(ceiling, upside)
            median_raw = record.get("proj_fd_median", None)
            median = float(median_raw) if median_raw not in (None, "") else 0.0
            if not math.isfinite(median) or median <= 0:
                median = mean
            bust_rate = float(record.get("proj_fd_bust_rate", 0.0) or 0.0)
            if not math.isfinite(bust_rate):
                bust_rate = 0.0
            bust_rate = _clamp(bust_rate, 0.0, 1.0)

        if player_type == "pitcher":
            mu, sigma, shift = _fit_pitcher_normal(mean, floor, ceiling, bust_rate)
            dist_type: Literal["lognormal", "truncated_normal"] = "truncated_normal"
        else:
            mu, sigma, shift = _fit_batter_lognormal(mean, floor, ceiling, salary, median, bust_rate)
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
            proj_median=median,
            bust_rate=bust_rate,
        )
    return dists


def _fit_calibrated_distributions(
    optimizer_df: pd.DataFrame,
    volatility_scale: float,
    calibration: "SimulationCalibration",
) -> Dict[str, PlayerDistribution]:
    dists: Dict[str, PlayerDistribution] = {}
    for record in optimizer_df.to_dict("records"):
        player_id = str(record.get("fd_player_id"))
        if not player_id:
            continue
        raw_mean = record.get("proj_fd_mean", 0.0)
        mean = float(raw_mean) if raw_mean not in (None, "") else 0.0
        if not math.isfinite(mean):
            mean = 0.0
        salary = int(record.get("salary", 0) or 0)
        full_name = str(record.get("full_name", ""))
        is_pitcher = str(record.get("player_type", "batter")).lower() == "pitcher"
        if is_pitcher:
            sigma = calibration.pitcher_sd(mean, volatility_scale)
            lower = float(calibration.pitcher_lower_bound)
            upper = mean + float(calibration.pitcher_upper_sd) * sigma
            # Cutting the tails moves the mean a little; shift the centre
            # so the simulated mean still equals the projection.
            centre = mean
            for _ in range(3):
                a = (lower - centre) / sigma
                b = (upper - centre) / sigma
                centre += mean - float(truncnorm.mean(a, b, loc=centre, scale=sigma))
            dist = PlayerDistribution(
                fd_player_id=player_id,
                full_name=full_name,
                player_type="pitcher",
                dist_type="truncated_normal",
                mu=centre,
                sigma=sigma,
                shift=0.0,
                proj_mean=mean,
                proj_floor=mean - 1.2816 * sigma,
                proj_ceiling=mean + 1.2816 * sigma,
                salary=salary,
                proj_median=mean,
                bust_rate=0.0,
                lower_bound=lower,
                upper_bound=upper,
            )
        else:
            p_zero, shape, scale, shift = calibration.hitter_parameters(mean, volatility_scale)
            dist = PlayerDistribution(
                fd_player_id=player_id,
                full_name=full_name,
                player_type="batter",
                dist_type="hurdle_gamma",
                mu=shape * scale,
                sigma=max(1e-3, math.sqrt(shape) * scale),
                shift=shift,
                proj_mean=mean,
                proj_floor=0.0,
                proj_ceiling=mean,
                salary=salary,
                proj_median=mean,
                bust_rate=p_zero,
                p_zero=p_zero,
                gamma_shape=shape,
                gamma_scale=scale,
            )
            floor, median, ceiling = (float(v) for v in dist.ppf(np.array([0.10, 0.50, 0.90])))
            dist = PlayerDistribution(**{**dist.__dict__, "proj_floor": floor, "proj_median": median, "proj_ceiling": ceiling})
        dists[player_id] = dist
    return dists


def _fit_batter_lognormal(
    mean: float,
    floor: float,
    ceiling: float,
    salary: int,
    median: float,
    bust_rate: float,
) -> tuple[float, float, float]:
    bust_shift_discount = _clamp(1.0 - bust_rate * 0.8, 0.25, 1.0)
    shift = max(0.0, float(floor) * 0.5 * bust_shift_discount)
    adj_mean = max(mean - shift, 0.1)
    adj_ceiling = max(ceiling - shift, adj_mean * 1.05)
    adj_median = max(median - shift, 0.05)
    ratio = adj_ceiling / adj_mean
    salary = max(int(salary) if salary else 0, 1)
    salary_factor = _clamp(4000.0 / salary, 0.7, 1.3)
    if ratio <= 1.05:
        sigma = 0.6 * salary_factor
    else:
        sigma = math.sqrt(max(1e-6, math.log(ratio))) * salary_factor
    if adj_mean > adj_median:
        median_sigma = math.sqrt(max(1e-6, 2.0 * math.log(adj_mean / adj_median)))
        sigma = max(sigma, median_sigma)
    sigma *= _clamp(0.85 + bust_rate * 0.9, 0.85, 1.45)
    mu = math.log(max(adj_mean, 1e-3)) - 0.5 * sigma ** 2
    return mu, sigma, shift


def _fit_pitcher_normal(mean: float, floor: float, ceiling: float, bust_rate: float) -> tuple[float, float, float]:
    spread = max(ceiling - floor, 1.0)
    sigma = spread / (2.0 * 1.645)
    if not math.isfinite(sigma) or sigma <= 0:
        sigma = max(abs(mean) * 0.1, 3.0)
    sigma *= _clamp(0.85 + bust_rate * 0.8, 0.85, 1.35)
    return mean, sigma, 0.0
