from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["SimulatedField", "FieldQualityMix", "FieldRealismProfile", "simulate_field"]


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

FIELD_STACK_TEMPLATES = {
    "shark": [(4, 3, 1), (4, 4), (4, 2, 2), (3, 3, 2)],
    "rec": [(4, 3, 1), (4, 2, 2), (3, 3, 2), (3, 2, 2, 1)],
    "random": [(3, 3, 2), (3, 2, 2, 1), (2, 2, 2, 2)],
}
STACK_BUILD_PROBABILITY = {"shark": 0.90, "rec": 0.70, "random": 0.35}


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


@dataclass(frozen=True)
class FieldRealismProfile:
    """Controls how aggressively the simulated field chases common DFS constructions."""

    shark_stack_probability: float = 0.92
    rec_stack_probability: float = 0.72
    random_stack_probability: float = 0.35
    shark_salary_target: float = 0.970
    rec_salary_target: float = 0.930
    shark_salary_softness: float = 0.016
    rec_salary_softness: float = 0.030
    value_chalk_bias: float = 0.22
    optimizer_train_bias: float = 0.18
    upside_chalk_bias: float = 0.10

    def stack_probability(self, tier: str) -> float:
        if tier == "shark":
            return float(np.clip(self.shark_stack_probability, 0.0, 1.0))
        if tier == "rec":
            return float(np.clip(self.rec_stack_probability, 0.0, 1.0))
        return float(np.clip(self.random_stack_probability, 0.0, 1.0))


@dataclass
class SimulatedField:
    lineups: np.ndarray
    player_ids: List[str]
    ownership_used: np.ndarray

    @property
    def num_lineups(self) -> int:
        return int(self.lineups.shape[0])


def _realistic_field_ownership(df: pd.DataFrame, profile: FieldRealismProfile) -> pd.Series:
    ownership = pd.to_numeric(df["proj_fd_ownership"], errors="coerce").fillna(0.0).clip(0.001, 0.60)
    projection = pd.to_numeric(df["proj_fd_mean"], errors="coerce").fillna(0.0)
    salary = pd.to_numeric(df["salary"], errors="coerce").fillna(0.0).clip(lower=1.0)
    upside = pd.to_numeric(df.get("proj_fd_upside", projection), errors="coerce").fillna(projection)

    value_rank = _rank01(projection / salary * 1000.0)
    proj_rank = _rank01(projection)
    upside_rank = _rank01(upside)
    chalk_rank = _rank01(ownership)

    value_signal = (0.50 * value_rank + 0.35 * proj_rank + 0.15 * upside_rank) - 0.50
    train_signal = np.maximum(0.0, chalk_rank - 0.55)
    upside_signal = np.maximum(0.0, upside_rank - 0.60)
    multiplier = (
        1.0
        + float(profile.value_chalk_bias) * value_signal
        + float(profile.optimizer_train_bias) * train_signal
        + float(profile.upside_chalk_bias) * upside_signal
    )
    adjusted = ownership.to_numpy(dtype=float) * np.clip(multiplier, 0.55, 1.70)
    return pd.Series(np.clip(adjusted, 0.001, 0.70), index=df.index, dtype=float)


def _rank01(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if len(values) <= 1 or values.nunique() <= 1:
        return np.full(len(values), 0.5, dtype=float)
    return values.rank(pct=True).to_numpy(dtype=float)


def simulate_field(
    optimizer_df: pd.DataFrame,
    num_opponent_lineups: int = 1000,
    salary_cap: int = 35_000,
    seed: Optional[int] = None,
    position_constraints: bool = True,
    quality_mix: Optional[FieldQualityMix] = None,
    realism_profile: Optional[FieldRealismProfile] = None,
) -> SimulatedField:
    df = optimizer_df.reset_index(drop=True).copy()
    df["fd_player_id"] = df["fd_player_id"].astype(str)
    if "position" not in df.columns:
        df["position"] = ""
    df["position"] = df["position"].astype(str)
    if "roster_position" in df.columns:
        df["roster_position"] = df["roster_position"].astype(str)
    if "player_type" not in df.columns:
        df["player_type"] = ""
    df["player_type"] = df["player_type"].astype(str).str.lower()
    if "proj_fd_ownership" not in df.columns:
        df["proj_fd_ownership"] = 0.0
    if "proj_fd_mean" not in df.columns:
        df["proj_fd_mean"] = 0.0
    if "salary" not in df.columns:
        df["salary"] = 0
    if "team_code" not in df.columns:
        df["team_code"] = ""
    df["proj_fd_ownership"] = pd.to_numeric(df["proj_fd_ownership"], errors="coerce").fillna(0.0)
    if df["proj_fd_ownership"].max() > 1.5:
        df["proj_fd_ownership"] = df["proj_fd_ownership"] / 100.0
    df["proj_fd_mean"] = pd.to_numeric(df["proj_fd_mean"], errors="coerce").fillna(0.0)
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)
    for col in ("vegas_team_total", "team_leverage_score", "proj_fd_upside", "proj_fd_bust_rate"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["proj_fd_upside"] = df["proj_fd_upside"].where(df["proj_fd_upside"] > 0, df["proj_fd_mean"])
    df["proj_fd_bust_rate"] = df["proj_fd_bust_rate"].clip(lower=0.0, upper=1.0)
    realism_profile = realism_profile or FieldRealismProfile()
    df["proj_fd_ownership"] = _realistic_field_ownership(df, realism_profile)

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
        stack_probability = realism_profile.stack_probability(str(tier))
        if position_constraints and rng.random() < stack_probability:
            lineup = _build_stacked_lineup(df, rng, str(tier))
            if lineup is None:
                lineup = _build_structured_lineup(df, rng, str(tier))
        elif position_constraints:
            lineup = _build_structured_lineup(df, rng, tier)
        else:
            lineup = _build_simple_lineup(df, rng, tier)
        if lineup is None:
            continue
        if _is_valid_field_lineup(df, lineup, salary_cap) and _accept_salary(df, lineup, salary_cap, str(tier), rng, realism_profile):
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
            if _is_valid_field_lineup(df, temp, salary_cap) and _accept_salary(df, temp, salary_cap, str(tier), rng, realism_profile):
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


def _build_stacked_lineup(df: pd.DataFrame, rng: np.random.Generator, tier: str) -> Optional[np.ndarray]:
    pitcher = _sample_from_pool(df, rng, [], slot_positions=("P",), tier=tier)
    if pitcher is None:
        return None

    selected: List[int] = [pitcher]
    selected_hitters: List[int] = []
    pitcher_row = df.iloc[pitcher]
    forbidden_hitter_team = str(pitcher_row.get("opponent_code", ""))
    template = _sample_stack_template(tier, rng)

    for stack_size in template:
        if stack_size < 2:
            continue
        team = _sample_stack_team(
            df,
            rng,
            tier,
            stack_size,
            selected_hitters,
            forbidden_hitter_team=forbidden_hitter_team,
        )
        if team is None:
            return None
        team_hitters = _sample_team_hitters(
            df,
            rng,
            tier,
            team,
            stack_size,
            selected + selected_hitters,
        )
        if len(team_hitters) != stack_size:
            return None
        selected_hitters.extend(team_hitters)

    while len(selected_hitters) < 8:
        hitter = _sample_one_off_hitter(
            df,
            rng,
            tier,
            selected + selected_hitters,
            forbidden_hitter_team=forbidden_hitter_team,
        )
        if hitter is None:
            return None
        selected_hitters.append(hitter)

    lineup = np.array(selected + selected_hitters[:8], dtype=int)
    if not _has_valid_position_assignment(df, lineup):
        return None
    return lineup


def _sample_stack_template(tier: str, rng: np.random.Generator) -> tuple[int, ...]:
    templates = FIELD_STACK_TEMPLATES.get(tier, FIELD_STACK_TEMPLATES["rec"])
    if tier == "shark":
        weights = np.array([0.35, 0.20, 0.25, 0.20], dtype=float)
    elif tier == "rec":
        weights = np.array([0.35, 0.20, 0.25, 0.20], dtype=float)
    else:
        weights = np.ones(len(templates), dtype=float)
    weights = weights / weights.sum()
    return templates[int(rng.choice(np.arange(len(templates)), p=weights))]


def _sample_stack_team(
    df: pd.DataFrame,
    rng: np.random.Generator,
    tier: str,
    stack_size: int,
    selected_hitters: Sequence[int],
    forbidden_hitter_team: str,
) -> Optional[str]:
    hitters = df[df["player_type"] != "pitcher"].copy()
    if forbidden_hitter_team:
        hitters = hitters[hitters["team_code"].astype(str) != forbidden_hitter_team]
    if selected_hitters:
        selected_teams = df.iloc[list(selected_hitters)]["team_code"].astype(str).value_counts().to_dict()
    else:
        selected_teams = {}

    team_rows = []
    for team, group in hitters.groupby("team_code"):
        team = str(team)
        already = int(selected_teams.get(team, 0))
        room = max(0, 4 - already)
        if room < stack_size:
            continue
        available = group[~group.index.isin(selected_hitters)]
        if len(available) < stack_size:
            continue
        team_rows.append((team, _team_stack_weight(available, tier)))
    if not team_rows:
        return None
    teams = [team for team, _ in team_rows]
    weights = np.array([weight for _, weight in team_rows], dtype=float)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(len(teams), dtype=float)
    weights = weights / weights.sum()
    return str(rng.choice(teams, p=weights))


def _team_stack_weight(group: pd.DataFrame, tier: str) -> float:
    top = group.sort_values("proj_fd_mean", ascending=False).head(5)
    ownership = top["proj_fd_ownership"].clip(0.003, 0.60).sum()
    projection = top["proj_fd_mean"].clip(lower=0).sum()
    upside = top["proj_fd_upside"].clip(lower=0).sum()
    avg_bust = top["proj_fd_bust_rate"].clip(0.0, 1.0).mean()
    vegas = top["vegas_team_total"].replace(0, np.nan).mean()
    leverage = top["team_leverage_score"].mean()
    vegas_mult = 1.0 if pd.isna(vegas) else max(0.7, min(1.4, float(vegas) / 4.5))
    leverage_mult = max(0.7, min(1.4, 1.0 + float(leverage or 0.0) * 0.25))
    upside_mult = max(0.8, min(1.5, 1.0 + (float(upside) / max(float(projection), 1.0) - 2.0) * 0.2))
    bust_mult = max(0.65, min(1.1, 1.05 - float(avg_bust or 0.0)))
    if tier == "shark":
        return float((ownership ** 1.15) * (1.0 + projection / 60.0) * upside_mult * bust_mult * vegas_mult)
    if tier == "rec":
        return float((ownership ** 1.35) * (1.0 + projection / 90.0) * vegas_mult)
    return float(max(0.01, ownership * leverage_mult * upside_mult))


def _sample_team_hitters(
    df: pd.DataFrame,
    rng: np.random.Generator,
    tier: str,
    team: str,
    count: int,
    used_indices: Sequence[int],
) -> List[int]:
    eligible = df[
        (df["player_type"] != "pitcher")
        & (df["team_code"].astype(str) == str(team))
        & (~df.index.isin(used_indices))
    ]
    if len(eligible) < count:
        return []
    probs = _selection_probabilities(eligible, tier)
    chosen = rng.choice(eligible.index.to_numpy(), size=count, replace=False, p=probs)
    return [int(idx) for idx in chosen]


def _sample_one_off_hitter(
    df: pd.DataFrame,
    rng: np.random.Generator,
    tier: str,
    used_indices: Sequence[int],
    forbidden_hitter_team: str,
) -> Optional[int]:
    used_set = set(int(idx) for idx in used_indices)
    hitters = df[(df["player_type"] != "pitcher") & (~df.index.isin(used_set))].copy()
    if forbidden_hitter_team:
        hitters = hitters[hitters["team_code"].astype(str) != forbidden_hitter_team]
    if hitters.empty:
        return None
    current_counts: Dict[str, int] = (
        df.iloc[list(used_set)]["team_code"].astype(str).value_counts().to_dict()
        if used_set
        else {}
    )
    hitters = hitters[hitters["team_code"].astype(str).map(lambda team: current_counts.get(str(team), 0) < 4)]
    if hitters.empty:
        return None
    probs = _selection_probabilities(hitters, tier)
    return int(rng.choice(hitters.index.to_numpy(), p=probs))


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
    position_col = "roster_position" if "roster_position" in df.columns else "position"
    positions = df[position_col].str.upper().fillna("")
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


def _is_valid_field_lineup(df: pd.DataFrame, lineup: np.ndarray, salary_cap: int) -> bool:
    if len(set(lineup.tolist())) != len(lineup):
        return False
    selected = df.iloc[lineup]
    if int(selected["salary"].sum()) > salary_cap:
        return False
    player_type = selected["player_type"].astype(str).str.lower()
    if int((player_type == "pitcher").sum()) != 1:
        return False
    hitters = selected[player_type != "pitcher"]
    if hitters.empty:
        return False
    team_counts = hitters["team_code"].astype(str).value_counts()
    if not team_counts.empty and int(team_counts.max()) > 4:
        return False
    if not _has_valid_position_assignment(df, lineup):
        return False
    return True


def _has_valid_position_assignment(df: pd.DataFrame, lineup: np.ndarray) -> bool:
    selected = df.iloc[lineup].to_dict("records")
    slots = [
        ("P", ("P",)),
        ("C1B", ("C", "1B", "C/1B")),
        ("2B", ("2B",)),
        ("3B", ("3B",)),
        ("SS", ("SS",)),
        ("OF1", ("OF",)),
        ("OF2", ("OF",)),
        ("OF3", ("OF",)),
    ]

    def positions(row: dict) -> set[str]:
        source = row.get("roster_position") or row.get("position") or ""
        return {token.strip().upper() for token in str(source).replace("-", "/").split("/") if token.strip()}

    def fits(row: dict, targets: tuple[str, ...]) -> bool:
        pos = positions(row)
        expanded = set(targets)
        if "C/1B" in pos:
            pos.update({"C", "1B"})
        return bool(pos & expanded)

    def backtrack(remaining: List[dict], slot_idx: int) -> bool:
        if slot_idx == len(slots):
            return True
        _, targets = slots[slot_idx]
        for idx, row in enumerate(remaining):
            if fits(row, targets):
                if backtrack(remaining[:idx] + remaining[idx + 1:], slot_idx + 1):
                    return True
        return False

    if not backtrack(selected, 0):
        return False
    return True


def _accept_salary(
    df: pd.DataFrame,
    lineup: np.ndarray,
    salary_cap: int,
    tier: str,
    rng: np.random.Generator,
    realism_profile: FieldRealismProfile,
) -> bool:
    total_salary = float(df.iloc[lineup]["salary"].sum())
    if total_salary > salary_cap:
        return False
    if tier == "random":
        return True
    target_ratio = realism_profile.shark_salary_target if tier == "shark" else realism_profile.rec_salary_target
    salary_ratio = total_salary / max(float(salary_cap), 1.0)
    softness = realism_profile.shark_salary_softness if tier == "shark" else realism_profile.rec_salary_softness
    probability = 1.0 / (1.0 + np.exp(-(salary_ratio - target_ratio) / softness))
    floor = 0.25 if tier == "shark" else 0.40
    return bool(rng.random() < max(floor, float(probability)))


def _selection_probabilities(eligible: pd.DataFrame, tier: str) -> np.ndarray:
    ownership = eligible["proj_fd_ownership"].to_numpy(dtype=float)
    ownership = np.clip(ownership, 0.005, 0.40)
    if tier == "shark":
        proj = eligible["proj_fd_mean"].to_numpy(dtype=float)
        upside = eligible["proj_fd_upside"].to_numpy(dtype=float)
        bust = eligible["proj_fd_bust_rate"].to_numpy(dtype=float)
        proj = proj - proj.min() if proj.size else proj
        if proj.size:
            proj_range = proj.max() - proj.min()
            proj_norm = (proj - proj.min()) / (proj_range + 1e-6)
            upside_norm = (upside - upside.min()) / ((upside.max() - upside.min()) + 1e-6)
            bust_discount = np.clip(1.05 - bust, 0.65, 1.05)
            ownership *= (1.0 + 0.4 * proj_norm + 0.25 * upside_norm) * bust_discount
    elif tier == "rec":
        salary = eligible["salary"].to_numpy(dtype=float)
        if salary.size:
            sal_range = salary.max() - salary.min()
            sal_norm = (salary - salary.min()) / (sal_range + 1e-6)
            ownership *= (0.5 + sal_norm)
    elif tier == "random":
        ownership = np.ones_like(ownership)
    total = ownership.sum()
    if total <= 0:
        ownership = np.ones_like(ownership)
        total = ownership.sum()
    return ownership / total
