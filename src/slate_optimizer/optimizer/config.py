"""Load and apply optimizer configuration."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

@dataclass
class OptimizerConfig:
    salary_cap: Optional[int] = None
    min_stack_size: Optional[int] = None
    stack_player_types: Optional[List[str]] = None
    stack_templates: Optional[List[int]] = None
    player_exposure_overrides: Dict[str, float] = field(default_factory=dict)
    max_lineup_ownership: Optional[float] = None
    chalk_threshold: Optional[float] = None
    chalk_exposure_cap: Optional[float] = None
    player_ownership_caps: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "OptimizerConfig":
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def apply_exposure_overrides(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if not self.player_exposure_overrides:
            return dataset
        df = dataset.copy()
        overrides = self.player_exposure_overrides
        df["default_max_exposure"] = df.apply(
            lambda row: overrides.get(str(row["fd_player_id"]), overrides.get(row.get("full_name"), row["default_max_exposure"])),
            axis=1,
        )
        return df

    def apply_ownership_strategy(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if "proj_fd_ownership" not in dataset.columns:
            return dataset
        df = dataset.copy()
        ownership = pd.to_numeric(df["proj_fd_ownership"], errors="coerce").fillna(0.0)
        if self.chalk_threshold is not None and self.chalk_exposure_cap is not None:
            mask = ownership >= self.chalk_threshold
            df.loc[mask, "default_max_exposure"] = df.loc[mask, "default_max_exposure"].clip(upper=self.chalk_exposure_cap)
        if self.player_ownership_caps:
            for key, cap in self.player_ownership_caps.items():
                try:
                    cap_value = float(cap)
                except (TypeError, ValueError):
                    continue
                mask = (df["fd_player_id"].astype(str) == str(key)) | (df["full_name"] == key)
                if mask.any():
                    df.loc[mask, "default_max_exposure"] = df.loc[mask, "default_max_exposure"].clip(upper=cap_value)
        return df

__all__ = ["OptimizerConfig"]


