"""Load and apply optimizer configuration."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class LeverageConfig:
    """Controls for the leverage-weighted tournament objective function.

    All contests are tournaments. Mode selects the contest-size profile:
    ``"single_entry"``, ``"small_field_gpp"``, or ``"large_field_gpp"``.
    """

    # How much to reward high-ceiling players
    ceiling_weight: float = 0.20

    # How much to penalize high-ownership players
    ownership_penalty: float = 0.30

    # How much to reward boom potential (ceiling-mean spread)
    boom_weight: float = 0.08

    # Stack leverage: bonus for stacking teams from "prime" or "good" environment games
    stack_leverage_bonus: float = 0.5

    # Minimum projection threshold — bottom N% of projections excluded
    min_viable_projection_pct: float = 0.40

    # Chalk ceiling — extra penalty above this ownership %
    chalk_threshold: float = 0.20
    chalk_extra_penalty: float = 0.08

    # Within-stack chalk penalty
    within_stack_chalk_penalty: float = 0.06

    # Bring-back leverage bonus
    bring_back_leverage_bonus: float = 0.30

    # Portfolio-level targets
    target_avg_lineup_ownership: float = 0.08
    max_stack_exposure_gpp: float = 0.30
    max_batter_exposure_gpp: float = 0.30
    max_pitcher_exposure_gpp: float = 0.50

    # Contest mode — always a tournament type
    mode: str = "large_field_gpp"
    # Valid: "single_entry", "small_field_gpp", "large_field_gpp"

    # Number of lineups to generate (varies by contest type)
    default_num_lineups: int = 150

    # Portfolio diversity weight for lineup selector
    portfolio_diversity_weight: float = 0.40

    # Pitcher fade bonus — rewards non-chalk pitchers (set by slate adjustments)
    pitcher_fade_bonus: float = 0.0

    # Minimum primary stack size (overridden to 4 on small slates)
    min_primary_stack_size: int = 3

    # Bring-back settings (can be overridden by slate adjustments)
    bring_back_enabled: bool = True
    bring_back_count: int = 1

    @property
    def is_gpp(self) -> bool:
        """Return True for all tournament modes (always True now)."""
        return True

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "LeverageConfig":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def load(cls, path: Path | str) -> "LeverageConfig":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def single_entry_preset(cls) -> "LeverageConfig":
        return cls(
            mode="single_entry",
            ceiling_weight=0.22,
            ownership_penalty=0.15,
            boom_weight=0.15,
            chalk_threshold=0.30,
            chalk_extra_penalty=0.04,
            within_stack_chalk_penalty=0.05,
            target_avg_lineup_ownership=0.12,
            max_stack_exposure_gpp=1.0,
            max_batter_exposure_gpp=1.0,
            max_pitcher_exposure_gpp=1.0,
            min_viable_projection_pct=0.35,
            default_num_lineups=1,
            portfolio_diversity_weight=0.0,
        )

    @classmethod
    def small_field_gpp_preset(cls) -> "LeverageConfig":
        return cls(
            mode="small_field_gpp",
            ceiling_weight=0.18,
            ownership_penalty=0.20,
            boom_weight=0.10,
            chalk_threshold=0.25,
            chalk_extra_penalty=0.05,
            within_stack_chalk_penalty=0.08,
            target_avg_lineup_ownership=0.11,
            max_stack_exposure_gpp=0.45,
            max_batter_exposure_gpp=0.40,
            max_pitcher_exposure_gpp=0.60,
            min_viable_projection_pct=0.35,
            default_num_lineups=20,
            portfolio_diversity_weight=0.25,
        )

    @classmethod
    def large_field_gpp_preset(cls) -> "LeverageConfig":
        return cls(
            mode="large_field_gpp",
            ceiling_weight=0.20,
            ownership_penalty=0.30,
            boom_weight=0.08,
            chalk_threshold=0.20,
            chalk_extra_penalty=0.08,
            within_stack_chalk_penalty=0.06,
            target_avg_lineup_ownership=0.08,
            max_stack_exposure_gpp=0.30,
            max_batter_exposure_gpp=0.30,
            max_pitcher_exposure_gpp=0.50,
            min_viable_projection_pct=0.40,
            default_num_lineups=150,
            portfolio_diversity_weight=0.40,
        )

@dataclass
class SlateProfile:
    """Describes the size and shape of the current slate."""

    num_games: int
    num_batters: int
    num_pitchers: int
    slate_type: str  # "small", "medium", "large"
    recommended_stacks: int
    stack_templates: List[tuple]


def detect_slate_profile(optimizer_df: pd.DataFrame) -> SlateProfile:
    """Auto-detect slate size from the optimizer dataset."""
    games = int(optimizer_df["game_key"].nunique()) if "game_key" in optimizer_df.columns else 0
    is_batter = optimizer_df["player_type"].astype(str).str.lower() == "batter"
    batters = int(is_batter.sum())
    pitchers = int((~is_batter).sum())

    if games <= 3:
        return SlateProfile(
            num_games=games,
            num_batters=batters,
            num_pitchers=pitchers,
            slate_type="small",
            recommended_stacks=max(games, 1),
            stack_templates=[(4, 4)] if games <= 2 else [(4, 3, 1), (4, 4)],
        )
    elif games <= 8:
        return SlateProfile(
            num_games=games,
            num_batters=batters,
            num_pitchers=pitchers,
            slate_type="medium",
            recommended_stacks=min(4, games - 1),
            stack_templates=[(4, 3, 1), (4, 2, 2), (3, 3, 2)],
        )
    else:
        return SlateProfile(
            num_games=games,
            num_batters=batters,
            num_pitchers=pitchers,
            slate_type="large",
            recommended_stacks=min(5, games // 2),
            stack_templates=[(4, 2, 2), (4, 3, 1), (4, 2, 1, 1), (3, 3, 2)],
        )


def apply_slate_adjustments(config: LeverageConfig, profile: SlateProfile) -> LeverageConfig:
    """Layer slate-specific tweaks on top of a contest-size preset.

    Returns a *new* config — does not mutate the input.
    """
    import copy
    cfg = copy.copy(config)

    if profile.slate_type == "small":
        _apply_small_slate(cfg)
    elif profile.slate_type == "large":
        _apply_large_slate(cfg)
    else:
        _apply_medium_slate(cfg)
    return cfg


def _apply_small_slate(cfg: LeverageConfig) -> None:
    cfg.ownership_penalty *= 1.15
    cfg.chalk_threshold *= 0.8
    cfg.chalk_extra_penalty *= 1.2
    cfg.pitcher_fade_bonus = 0.20
    cfg.within_stack_chalk_penalty *= 1.5
    cfg.min_primary_stack_size = 4
    cfg.bring_back_enabled = True
    cfg.bring_back_count = 1
    cfg.portfolio_diversity_weight *= 0.7


def _apply_medium_slate(cfg: LeverageConfig) -> None:
    cfg.bring_back_enabled = True
    cfg.bring_back_count = 1


def _apply_large_slate(cfg: LeverageConfig) -> None:
    cfg.ownership_penalty *= 0.85
    cfg.chalk_threshold *= 1.15
    cfg.stack_leverage_bonus *= 1.4
    cfg.min_viable_projection_pct *= 0.85
    cfg.pitcher_fade_bonus = 0.10
    cfg.portfolio_diversity_weight *= 1.2
    cfg.max_stack_exposure_gpp *= 0.85


@dataclass
class OptimizerConfig:
    salary_cap: Optional[int] = None
    min_stack_size: Optional[int] = None
    stack_player_types: Optional[List[str]] = None
    stack_templates: Optional[List[int]] = None
    player_exposure_overrides: Dict[str, float] = field(default_factory=dict)
    max_lineup_ownership: Optional[float] = None
    # Batter chalk controls
    chalk_threshold: Optional[float] = None
    chalk_exposure_cap: Optional[float] = None
    # Pitcher chalk controls (fall back to batter values if None)
    pitcher_chalk_threshold: Optional[float] = None
    pitcher_chalk_exposure_cap: Optional[float] = None
    player_ownership_caps: Dict[str, float] = field(default_factory=dict)
    bring_back_enabled: bool = False
    bring_back_count: int = 1
    min_game_total_for_stacks: Optional[float] = None

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
        player_type = df["player_type"].astype(str).str.lower() if "player_type" in df.columns else pd.Series("batter", index=df.index)
        is_pitcher = player_type == "pitcher"
        is_batter = ~is_pitcher

        # Apply batter chalk controls
        if self.chalk_threshold is not None and self.chalk_exposure_cap is not None:
            batter_chalk = is_batter & (ownership >= self.chalk_threshold)
            df.loc[batter_chalk, "default_max_exposure"] = df.loc[batter_chalk, "default_max_exposure"].clip(upper=self.chalk_exposure_cap)

        # Apply pitcher chalk controls (fall back to batter values if not set)
        p_threshold = self.pitcher_chalk_threshold if self.pitcher_chalk_threshold is not None else self.chalk_threshold
        p_cap = self.pitcher_chalk_exposure_cap if self.pitcher_chalk_exposure_cap is not None else self.chalk_exposure_cap
        if p_threshold is not None and p_cap is not None:
            pitcher_chalk = is_pitcher & (ownership >= p_threshold)
            df.loc[pitcher_chalk, "default_max_exposure"] = df.loc[pitcher_chalk, "default_max_exposure"].clip(upper=p_cap)

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

__all__ = ["OptimizerConfig", "LeverageConfig", "SlateProfile", "detect_slate_profile", "apply_slate_adjustments"]


