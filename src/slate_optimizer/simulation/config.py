from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

from .correlation import CorrelationConfig

__all__ = ["SimulationConfig"]


@dataclass
class SimulationConfig:
    """Container for all simulation parameters."""

    num_simulations: int = 10_000
    seed: Optional[int] = None
    use_antithetic: bool = True
    use_stratified: bool = False
    num_strata: int = 10

    volatility_scale: float = 1.0

    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)

    num_field_lineups: int = 1000
    field_quality_shark_pct: float = 0.10
    field_quality_rec_pct: float = 0.60
    field_quality_random_pct: float = 0.30

    entry_fee: float = 20.0
    payout_structure: Optional[Dict[str, float]] = None

    num_candidates: int = 500
    selection_metric: str = "top_1pct_rate"
    max_overlap: int = 5
    max_batter_exposure: float = 0.40
    max_pitcher_exposure: float = 0.60
    diversity_weight: float = 0.3

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["correlation"] = asdict(self.correlation)
        return data

    @classmethod
    def load(cls, path: Path | str) -> "SimulationConfig":
        data = json.loads(Path(path).read_text())
        corr_data = data.pop("correlation", None)
        config = cls(**data)
        if corr_data:
            config.correlation = CorrelationConfig(**corr_data)
        return config

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def gpp_preset(cls) -> "SimulationConfig":
        return cls(
            num_simulations=20_000,
            volatility_scale=1.1,
            num_field_lineups=2000,
            selection_metric="top_1pct_rate",
            diversity_weight=0.4,
            max_batter_exposure=0.30,
            max_pitcher_exposure=0.50,
        )

    @classmethod
    def cash_preset(cls) -> "SimulationConfig":
        return cls(
            num_simulations=10_000,
            volatility_scale=0.8,
            num_field_lineups=500,
            selection_metric="cash_rate",
            diversity_weight=0.1,
            max_batter_exposure=0.80,
            max_pitcher_exposure=0.80,
        )

    @classmethod
    def single_entry_preset(cls) -> "SimulationConfig":
        return cls(
            num_simulations=20_000,
            volatility_scale=1.0,
            num_field_lineups=2000,
            selection_metric="win_rate",
            diversity_weight=0.0,
            max_batter_exposure=1.0,
            max_pitcher_exposure=1.0,
        )
