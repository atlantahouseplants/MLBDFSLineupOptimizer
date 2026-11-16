"""Simulation settings fitted to real box-score outcomes.

The hand-set simulator produced hitter scores with about half the real
spread (SD 4.5 vs 9.2 fantasy points), pitchers with under a third (4.3 vs
14.3), and never let a hitter score zero - which 23% of real starts do.  It
also assumed teammates were about twice as correlated as they are.

``scripts/calibrate_simulation.py`` measures those quantities from
``data/backtest/player_games.csv.gz`` and writes them to
``config/simulation_calibration.json``.  When that file exists the pipeline
and dashboard simulate with it; delete it to return to the hand-set model.

Hitter model (a hurdle):
    P(score = 0)      = exp(-projection / hitter_zero_scale), clipped
    score | score > 0 = hitter_positive_shift + Gamma
                        with SD = hitter_positive_cv x E[score | score > 0]
    The overall mean is held equal to the projection.

Pitcher model:
    Normal(projection, pitcher_sd_intercept + pitcher_sd_slope x projection),
    cut off below ``pitcher_lower_bound`` (pitchers do post negative scores)
    and above projection + ``pitcher_upper_sd`` SDs (a start has a hard
    ceiling: nine innings and a finite number of strikeouts).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

from .correlation import CorrelationConfig

DEFAULT_SIM_CALIBRATION_PATH = Path(__file__).resolve().parents[3] / "config" / "simulation_calibration.json"


@dataclass
class SimulationCalibration:
    # Hitter hurdle model.
    hitter_zero_scale: float = 6.2
    hitter_zero_min: float = 0.05
    hitter_zero_max: float = 0.60
    hitter_positive_cv: float = 0.78
    hitter_positive_shift: float = 3.0
    # Pitcher normal model.
    pitcher_sd_intercept: float = 14.0
    pitcher_sd_slope: float = 0.0
    pitcher_lower_bound: float = -20.0
    pitcher_upper_sd: float = 2.3
    # Copula parameters (latent correlations that reproduce the measured ones).
    teammate_base: float = 0.16
    teammate_adjacent_bonus: float = 0.0
    same_game_opponent: float = 0.0
    pitcher_own_hitters: float = 0.07
    pitcher_vs_opposing: float = -0.36
    copula_nu: int = 5
    # Provenance.
    source: str = "defaults"
    slates_used: int = 0
    hitter_starts: int = 0
    pitcher_starts: int = 0
    fitted_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "SimulationCalibration":
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in known})

    # -- hitters ---------------------------------------------------------
    def hitter_zero_probability(self, projection: float) -> float:
        """Chance a starting hitter with this projection scores exactly zero."""
        if projection <= 0:
            return 1.0
        shift = max(0.0, float(self.hitter_positive_shift))
        base = math.exp(-float(projection) / max(float(self.hitter_zero_scale), 1e-6))
        base = min(max(base, self.hitter_zero_min), self.hitter_zero_max)
        # A non-zero score is at least `shift`, so a tiny projection forces a
        # high zero rate to keep the mean equal to the projection.
        needed = 1.0 - float(projection) / (shift + 0.5)
        return float(min(0.995, max(base, needed)))

    def hitter_parameters(self, projection: float, volatility_scale: float = 1.0) -> tuple[float, float, float, float]:
        """(p_zero, gamma shape, gamma scale, shift) for a hitter's hurdle distribution."""
        projection = max(float(projection), 0.0)
        shift = max(0.0, float(self.hitter_positive_shift))
        p_zero = self.hitter_zero_probability(projection)
        if projection <= 0 or p_zero >= 0.995:
            return 0.995, 1.0, 0.5, shift
        mean_positive = projection / (1.0 - p_zero)
        excess_mean = max(mean_positive - shift, 0.25)
        sd_positive = max(self.hitter_positive_cv * mean_positive * max(volatility_scale, 0.05), 0.05)
        shape = (excess_mean / sd_positive) ** 2
        scale = sd_positive**2 / excess_mean
        return p_zero, shape, scale, shift

    # -- pitchers --------------------------------------------------------
    def pitcher_sd(self, projection: float, volatility_scale: float = 1.0) -> float:
        sd = self.pitcher_sd_intercept + self.pitcher_sd_slope * max(float(projection), 0.0)
        return float(max(sd, 2.0) * max(volatility_scale, 0.05))

    # -- correlation -----------------------------------------------------
    def correlation_config(self) -> CorrelationConfig:
        """Copula settings matching the measured correlations.

        The batting-order adjacency bonus, the "high priority stack" bonus and
        the Vegas-total scaling are switched off: measured teammate correlation
        is flat across batting-order distance, and the other two were never
        supported by data.
        """
        return CorrelationConfig(
            teammate_base=float(self.teammate_base),
            teammate_adjacent_bonus=float(self.teammate_adjacent_bonus),
            same_game_opponent=float(self.same_game_opponent),
            pitcher_own_hitters=float(self.pitcher_own_hitters),
            pitcher_vs_opposing=float(self.pitcher_vs_opposing),
            cross_game=0.0,
            copula_nu=int(self.copula_nu),
            stack_priority_bonus=0.0,
            vegas_scale_floor=1.0,
            vegas_scale_cap=1.0,
        )


def load_simulation_calibration(path: Path | str | None = None) -> Optional[SimulationCalibration]:
    """Fitted simulation settings, or ``None`` when no calibration file exists."""
    target = Path(path) if path else DEFAULT_SIM_CALIBRATION_PATH
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return SimulationCalibration.from_dict(payload)
    except (TypeError, ValueError):
        return None


def save_simulation_calibration(calibration: SimulationCalibration, path: Path | str | None = None) -> None:
    target = Path(path) if path else DEFAULT_SIM_CALIBRATION_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(calibration.to_dict(), indent=2) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_SIM_CALIBRATION_PATH",
    "SimulationCalibration",
    "load_simulation_calibration",
    "save_simulation_calibration",
]
