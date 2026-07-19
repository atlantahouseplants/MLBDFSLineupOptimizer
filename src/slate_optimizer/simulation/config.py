from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

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
    use_field_realism: bool = True
    field_realism_value_bias: float = 0.22
    field_realism_train_bias: float = 0.18
    field_realism_upside_bias: float = 0.10
    field_salary_target_shark: float = 0.970
    field_salary_target_rec: float = 0.930

    entry_fee: float = 20.0
    contest_entries: int = 50_000
    payout_structure: Optional[Any] = None
    contest_type_profile: str = "Auto"
    use_contest_type_brain: bool = True

    num_candidates: int = 500
    selection_metric: str = "top_1pct_rate"
    max_overlap: int = 5
    max_batter_exposure: float = 0.40
    max_pitcher_exposure: float = 0.60
    min_batter_exposure: float = 0.0
    min_pitcher_exposure: float = 0.0
    min_stack_exposure: float = 0.0
    max_stack_exposure: float = 1.0
    diversity_weight: float = 0.3
    selection_leverage_weight: float = 0.15
    selection_ownership_weight: float = 0.10
    selection_duplication_weight: float = 0.15
    selection_scenario_weight: float = 0.10
    selection_marginal_value_weight: float = 0.20
    use_ownership_scenarios: bool = True
    risk_profile: str = "Balanced leverage"
    calibration_adjustment_strength: float = 0.0
    use_stack_exposure_engine: bool = True
    stack_target_weight: float = 0.20
    use_stack_auto_caps: bool = True
    use_stack_auto_mins: bool = False
    use_portfolio_optimizer: bool = True
    portfolio_time_limit_seconds: int = 45

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
            max_stack_exposure=0.50,
            selection_leverage_weight=0.20,
            selection_ownership_weight=0.15,
            selection_duplication_weight=0.20,
            selection_scenario_weight=0.15,
            selection_marginal_value_weight=0.25,
            contest_type_profile="Massive 150/300-max GPP",
            use_contest_type_brain=True,
            use_field_realism=True,
            use_stack_exposure_engine=True,
            stack_target_weight=0.25,
            use_stack_auto_caps=True,
            use_stack_auto_mins=False,
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
            selection_leverage_weight=0.0,
            selection_ownership_weight=0.0,
            selection_duplication_weight=0.0,
            selection_scenario_weight=0.0,
            selection_marginal_value_weight=0.05,
            contest_type_profile="Small-field GPP",
            use_contest_type_brain=False,
            use_field_realism=False,
            use_stack_exposure_engine=False,
            stack_target_weight=0.0,
            use_stack_auto_caps=False,
            use_stack_auto_mins=False,
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
            selection_leverage_weight=0.0,
            selection_ownership_weight=0.0,
            selection_duplication_weight=0.0,
            selection_scenario_weight=0.0,
            selection_marginal_value_weight=0.0,
            contest_type_profile="Single-entry / 3-max",
            use_contest_type_brain=True,
            use_field_realism=True,
            use_stack_exposure_engine=False,
            stack_target_weight=0.0,
            use_stack_auto_caps=False,
            use_stack_auto_mins=False,
        )
