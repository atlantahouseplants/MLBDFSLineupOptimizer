from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["SimulatedField", "FieldQualityMix", "simulate_field"]


ROSTER_SLOTS = [
    ("P", ("P",)),
    ("C1B", ("C/1B", "C", "1B")),
    ("2B", ("2B",)),
    ("3B", ("3B",)),
    ("SS", ("SS",)),
    ("OF", ("OF",)),
    ("OF", ("OF",)),
    ("OF", ("OF",)),
    ("UTIL", ("P", "C", "1B", "C/1B", "2B", "3B", "SS", "OF")),
]


@dataclass
class FieldQualityMix:
    shark_pct: float = 0.10
    rec_pct: float = 0.60
    random_pct: float = 0.30

    def normalized(self) -> np.ndarray:
        weights = np.array([self.shark_pct, self.rec_pct, self.random_pct], dtype=float)
        total = weights.sum()
        if total <= 0:
            return np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        return weights / total


@dataclass
class SimulatedField:
    lineups: np.ndarray
    player_ids: List[str]
    ownership_used: np.ndarray

    @property
    def num_lineups(self) -> int:
        return int(self.lineups.shape[0])


def simulate_field(
    optimizer_df: pd.DataFrame,
    num_opponent_lineups: int = 1000,
    salary_cap: int = 35_000,
    seed: Optional[int] = None,
    position_constraints: bool = True,
    quality_mix: Optional[FieldQualityMix] = None,
) -> SimulatedField:
    df = optimizer_df.reset_index(drop=True).copy()
    df["fd_player_id"] = df["fd_player_id"].astype(str)
    df["position"] = df.get("position", "").astype(str)
    df["player_type"] = df.get("player_type", "").astype(str).str.lower()
    df["proj_fd_ownership"] = pd.to_numeric(df.get("proj_fd_ownership"), errors="coerce").fillna(0.0)
    df["proj_fd_mean"] = pd.to_numeric(df.get("proj_fd_mean"), errors="coerce").fillna(0.0)
    df["salary"] = pd.to_numeric(df.get("salary"), errors="coerce").fillna(0).astype(int)

    player_ids = df["fd_player_id"].tolist()
    ownership_used = df["proj_fd_ownership"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    quality_mix = quality_mix or FieldQualityMix()
    mix_weights = quality_mix.normalized()
    quality_types = np.array(["shark", "rec", "random"])

    pitcher_mask = df["player_type"] == "pitcher"
    hitter_mask = ~pitcher_mask
    pitcher_indices = np.flatnonzero(pitcher_mask.values)
    hitter_indices = np.flatnonzero(hitter_mask.values)

    if len(pitcher_indices) == 0 or len(hitter_indices) < 8:
        raise ValueError("Insufficient players to simulate field lineups")

    lineups: List[np.ndarray] = []
    max_attempts = num_opponent_lineups * 10
    attempts = 0

    while len(lineups) < num_opponent_lineups and attempts < max_attempts:
        attempts += 1
        tier = rng.choice(quality_types, p=mix_weights)
        if position_constraints:
            lineup = _build_structured_lineup(df, rng, tier)
        else:
            lineup = _build_simple_lineup(df, rng, tier)
        if lineup is None:
            continue
        total_salary = df.iloc[lineup]["salary"].sum()
        if total_salary <= salary_cap:
            lineups.append(lineup)
            continue
        # Try resampling UTIL slot to fix salary overage
        success = False
        util_idx = len(ROSTER_SLOTS) - 1
        for _ in range(50):
            new_player = _sample_slot_candidate(df, rng, lineup, ROSTER_SLOTS[util_idx], tier)
            if new_player is None:
                break
            temp = lineup.copy()
            temp[util_idx] = new_player
            if df.iloc[temp]["salary"].sum() <= salary_cap:
                lineups.append(temp)
                success = True
                break
        if not success:
            continue

    if not lineups:
        raise RuntimeError("Failed to generate any field lineups")

    return SimulatedField(
        lineups=np.vstack(lineups),
        player_ids=player_ids,
        ownership_used=ownership_used,
    )


def _build_structured_lineup(df: pd.DataFrame, rng: np.random.Generator, tier: str) -> Optional[np.ndarray]:
    selected: List[int] = []
    for slot in ROSTER_SLOTS:
        idx = _sample_slot_candidate(df, rng, selected, slot, tier)
        if idx is None:
            return None
        selected.append(idx)
    return np.array(selected, dtype=int)


def _build_simple_lineup(df: pd.DataFrame, rng: np.random.Generator, tier: str) -> Optional[np.ndarray]:
    lineup: List[int] = []
    pitcher = _sample_from_pool(df, rng, [], slot_positions=("P",), tier=tier)
    if pitcher is None:
        return None
    lineup.append(pitcher)
    hitters = []
    for _ in range(8):
        idx = _sample_from_pool(df, rng, lineup + hitters, slot_positions=("H",), tier=tier, hitters_only=True)
        if idx is None:
            return None
        hitters.append(idx)
    lineup.extend(hitters)
    return np.array(lineup, dtype=int)


def _sample_slot_candidate(
    df: pd.DataFrame,
    rng: np.random.Generator,
    used_indices: Sequence[int],
    slot: tuple[str, tuple[str, ...]],
    tier: str,
) -> Optional[int]:
    slot_name, elig_positions = slot
    hitters_only = slot_name != "P"
    return _sample_from_pool(
        df,
        rng,
        used_indices,
        slot_positions=elig_positions,
        tier=tier,
        hitters_only=hitters_only,
    )


def _sample_from_pool(
    df: pd.DataFrame,
    rng: np.random.Generator,
    used_indices: Sequence[int],
    slot_positions: tuple[str, ...],
    tier: str,
    hitters_only: bool = False,
) -> Optional[int]:
    mask = (~df.index.isin(used_indices))
    if hitters_only:
        mask &= df["player_type"] != "pitcher"
    positions = df["position"].str.upper().fillna("")
    if slot_positions == ("H",):
        mask &= df["player_type"] != "pitcher"
    elif "P" in slot_positions and len(slot_positions) == 1:
        mask &= df["player_type"] == "pitcher"
    else:
        mask &= positions.apply(lambda pos: _position_match(pos, slot_positions))
    eligible = df[mask]
    if eligible.empty:
        return None
    probs = _selection_probabilities(eligible, tier)
    choice = rng.choice(eligible.index.to_numpy(), p=probs)
    return int(choice)


def _position_match(position_text: str, targets: tuple[str, ...]) -> bool:
    tokens = [p.strip().upper() for p in position_text.split("/") if p]
    token_set = set(tokens)
    for target in targets:
        if target == "UTIL":
            return True
        if target in token_set:
            return True
    return False


def _selection_probabilities(eligible: pd.DataFrame, tier: str) -> np.ndarray:
    ownership = eligible["proj_fd_ownership"].to_numpy(dtype=float)
    ownership = np.clip(ownership, 0.005, 0.40)
    if tier == "shark":
        proj = eligible["proj_fd_mean"].to_numpy(dtype=float)
        proj = proj - proj.min() if proj.size else proj
        if proj.size:
            proj_norm = (proj - proj.min()) / (proj.ptp() + 1e-6)
            ownership *= (1.0 + 0.5 * proj_norm)
    elif tier == "rec":
        salary = eligible["salary"].to_numpy(dtype=float)
        if salary.size:
            sal_norm = (salary - salary.min()) / (salary.ptp() + 1e-6)
            ownership *= (0.5 + sal_norm)
    elif tier == "random":
        ownership = np.ones_like(ownership)
    total = ownership.sum()
    if total <= 0:
        ownership = np.ones_like(ownership)
        total = ownership.sum()
    return ownership / total
