"""Simple ILP-based MLB lineup solver."""
from __future__ import annotations

import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pulp import (
    LpBinary,
    LpMaximize,
    LpProblem,
    LpStatusOptimal,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
)

SALARY_CAP = 35000
TOTAL_PLAYERS = 9
MAX_HITTERS_PER_TEAM = 4  # FanDuel hard cap

POSITION_REQUIREMENTS = {
    "pitcher": ("P", 1, 1),
    "catcher_first": ("C/1B", 1, None),
    "second_base": ("2B", 1, None),
    "third_base": ("3B", 1, None),
    "shortstop": ("SS", 1, None),
    "outfield": ("OF", 3, None),
}

# Maximum players whose ONLY non-UTIL eligibility is within this group.
_SLOT_CAPS = {
    "C/1B": 2,   # C/1B slot + UTIL
    "2B": 2,     # 2B slot + UTIL
    "3B": 2,     # 3B slot + UTIL
    "SS": 2,     # SS slot + UTIL
    "OF": 4,     # 3 OF slots + UTIL
}

FANDUEL_ASSIGNMENT_SLOTS = {
    "P": ("P",),
    "C1B": ("C", "1B"),
    "2B": ("2B",),
    "3B": ("3B",),
    "SS": ("SS",),
    "OF1": ("OF",),
    "OF2": ("OF",),
    "OF3": ("OF",),
    "UTIL": ("C", "1B", "2B", "3B", "SS", "OF"),
}

# ──────────────────────────────────────────────────────────────────────
# Stack presets: each tuple sums to 8 (total batters in a FanDuel lineup).
# Groups of 1 need no LP constraint — only groups >= 2 are enforced.
# ──────────────────────────────────────────────────────────────────────
STACK_PRESETS: OrderedDict[str, Optional[Tuple[int, ...]]] = OrderedDict([
    ("Auto (optimizer's choice)", None),
    ("4-4 (two big stacks)",      (4, 4)),
    ("4-3-1",                     (4, 3, 1)),
    ("4-2-2",                     (4, 2, 2)),
    ("4-2-1-1",                   (4, 2, 1, 1)),
    ("3-3-2",                     (3, 3, 2)),
    ("3-3-1-1",                   (3, 3, 1, 1)),
    ("3-2-2-1",                   (3, 2, 2, 1)),
    ("3-2-1-1-1",                 (3, 2, 1, 1, 1)),
    ("2-2-2-2",                   (2, 2, 2, 2)),
    ("2-2-2-1-1",                 (2, 2, 2, 1, 1)),
])


def _parse_position_tokens(value: object) -> set[str]:
    if not isinstance(value, str):
        return set()
    parts = value.upper().replace("-", "/").split("/")
    return {part.strip() for part in parts if part.strip() and part.strip() != "UTIL"}


def _preferred_position_values(df: pd.DataFrame) -> pd.Series:
    values = df.get("position", pd.Series("", index=df.index)).fillna("").astype(str)
    if "roster_position" not in df.columns:
        return values
    roster = df["roster_position"].fillna("").astype(str)
    roster_clean = roster.str.strip().str.upper()
    has_roster = ~roster_clean.isin(("", "NAN", "NONE"))
    return values.mask(has_roster, roster)


def _position_mask(df: pd.DataFrame, keyword: str) -> pd.Series:
    keyword = keyword.upper()
    tokens = set(keyword.split("/"))
    position_values = _preferred_position_values(df)

    def matches(value: str) -> bool:
        parts = _parse_position_tokens(value)
        return bool(parts.intersection(tokens))

    return position_values.map(matches)


def _eligible_assignment_slots(row: pd.Series) -> List[str]:
    player_type = str(row.get("player_type", "")).lower()
    roster_pos = str(row.get("roster_position", "")).strip()
    roster_clean = roster_pos.upper()
    raw_position = roster_pos if roster_clean not in ("", "NAN", "NONE") else row.get("position", "")
    tokens = _parse_position_tokens(raw_position)

    if player_type == "pitcher":
        return ["P"] if "P" in tokens else []

    slots: List[str] = []
    for slot, slot_tokens in FANDUEL_ASSIGNMENT_SLOTS.items():
        if slot == "P":
            continue
        if slot == "UTIL":
            if tokens.intersection(slot_tokens):
                slots.append(slot)
            continue
        if tokens.intersection(slot_tokens):
            slots.append(slot)
    return slots


@dataclass
class LineupResult:
    dataframe: pd.DataFrame
    total_salary: int
    total_projection: float
    stack_template_used: Optional[Tuple[int, ...]] = None
    stacks_dropped: bool = False


def _max_usage(df: pd.DataFrame, num_lineups: int) -> Dict[str, int]:
    usage: Dict[str, int] = {}
    for _, row in df.iterrows():
        exposure = row.get("default_max_exposure", 1.0)
        try:
            exposure = float(exposure)
        except (TypeError, ValueError):
            exposure = 1.0
        if not np.isfinite(exposure):
            exposure = 1.0
        exposure = min(1.0, max(0.0, exposure))
        allowed = 0 if exposure <= 0 else max(1, int(np.floor(exposure * num_lineups + 1e-9)))
        usage[str(row["fd_player_id"])] = allowed
    return usage


# ──────────────────────────────────────────────────────────────────────
# Base LP construction (everything except stack constraints)
# ──────────────────────────────────────────────────────────────────────

def _build_base_lp(
    pool: pd.DataFrame,
    lineup_index: int,
    salary_cap: int,
    max_lineup_ownership: Optional[float],
    previous_lineups: List[List[str]],
    tag: str = "",
    leverage_weight: float = 0.0,
    randomness: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    locked_player_ids: Optional[set[str]] = None,
    ceiling_weight: float = 0.0,
    ownership_penalty_weight: float = 0.0,
    min_salary: Optional[int] = None,
    min_uniques: int = 1,
) -> Tuple[LpProblem, Dict[int, LpVariable]]:
    """Build the LP with all constraints except stack/bring-back."""
    label = f"mlb_lineup_{lineup_index}{tag}"
    prob = LpProblem(label, LpMaximize)
    decision_vars = {
        idx: LpVariable(f"x{tag}_{idx}", lowBound=0, upBound=1, cat=LpBinary)
        for idx in pool.index
    }

    # Objective: maximize a GPP score with optional noise:
    #   base = (1 - ceiling_weight) * mean + ceiling_weight * upside
    #   score = base * (1 + leverage_weight * leverage_score) * (1 + noise)
    #           - ownership_penalty_weight * ownership * mean_scale
    # ceiling_weight = 0 reproduces the legacy mean-only objective.
    has_leverage = "player_leverage_score" in pool.columns and leverage_weight > 0

    ceiling_weight = min(1.0, max(0.0, float(ceiling_weight)))
    upside_series = None
    if ceiling_weight > 0:
        for candidate in ("proj_fd_upside", "proj_fd_ceiling"):
            if candidate in pool.columns:
                upside_series = pd.to_numeric(pool[candidate], errors="coerce")
                break
        if upside_series is not None:
            upside_series = upside_series.where(upside_series > 0).fillna(
                pd.to_numeric(pool["proj_fd_mean"], errors="coerce").fillna(0.0)
            )

    ownership_series = None
    if ownership_penalty_weight > 0 and "proj_fd_ownership" in pool.columns:
        ownership_series = pd.to_numeric(pool["proj_fd_ownership"], errors="coerce").fillna(0.0)
        if float(ownership_series.max()) > 1.5:  # percent scale -> decimal
            ownership_series = ownership_series / 100.0
        ownership_series = ownership_series.clip(lower=0.0, upper=1.0)
    mean_scale = float(pd.to_numeric(pool["proj_fd_mean"], errors="coerce").fillna(0.0).mean()) or 1.0

    # Generate per-player noise for diversity across lineups
    if randomness > 0 and rng is not None:
        noise = rng.normal(0, randomness, size=len(pool))
    else:
        noise = np.zeros(len(pool))

    obj_coeffs = {}
    for i, (idx, var) in enumerate(decision_vars.items()):
        proj = pool.loc[idx, "proj_fd_mean"]
        if upside_series is not None:
            proj = (1.0 - ceiling_weight) * proj + ceiling_weight * float(upside_series.loc[idx])
        if has_leverage:
            proj = proj * (1.0 + leverage_weight * pool.loc[idx, "player_leverage_score"])
        coeff = proj * (1.0 + noise[i])
        if ownership_series is not None:
            coeff -= ownership_penalty_weight * float(ownership_series.loc[idx]) * mean_scale
        obj_coeffs[idx] = coeff

    prob += lpSum(obj_coeffs[idx] * var for idx, var in decision_vars.items())

    # Salary constraint
    prob += lpSum(pool.loc[idx, "salary"] * var for idx, var in decision_vars.items()) <= salary_cap
    if min_salary is not None and min_salary > 0:
        prob += lpSum(pool.loc[idx, "salary"] * var for idx, var in decision_vars.items()) >= int(min_salary)

    # Exactly 9 players
    prob += lpSum(var for var in decision_vars.values()) == TOTAL_PLAYERS

    if locked_player_ids:
        id_series = pool["fd_player_id"].astype(str)
        for locked_id in locked_player_ids:
            locked_indices = id_series[id_series == str(locked_id)].index.tolist()
            if locked_indices:
                prob += lpSum(decision_vars[idx] for idx in locked_indices) == 1

    assignment_vars: Dict[Tuple[int, str], LpVariable] = {}
    for idx, row in pool.iterrows():
        eligible_slots = _eligible_assignment_slots(row)
        if not eligible_slots:
            prob += decision_vars[idx] == 0
            continue
        player_slot_vars = []
        for slot in eligible_slots:
            var = LpVariable(f"slot{tag}_{idx}_{slot}", lowBound=0, upBound=1, cat=LpBinary)
            assignment_vars[(idx, slot)] = var
            player_slot_vars.append(var)
        prob += lpSum(player_slot_vars) == decision_vars[idx]

    first_decision_var = next(iter(decision_vars.values()))
    for slot in FANDUEL_ASSIGNMENT_SLOTS:
        slot_vars = [var for (idx, slot_name), var in assignment_vars.items() if slot_name == slot]
        if slot_vars:
            prob += lpSum(slot_vars) == 1
        else:
            prob += 0 * first_decision_var == 1

    # Position constraints
    pitcher_mask = _position_mask(pool, "P")
    prob += lpSum(decision_vars[idx] for idx in pitcher_mask[pitcher_mask].index) == 1

    for _, (keyword, minimum, maximum) in POSITION_REQUIREMENTS.items():
        if keyword == "P":
            continue
        mask = _position_mask(pool, keyword)
        if mask.any():
            prob += lpSum(decision_vars[idx] for idx in mask[mask].index) >= minimum
            if maximum:
                prob += lpSum(decision_vars[idx] for idx in mask[mask].index) <= maximum

    # Exactly 8 hitters
    hitters_mask = pool["player_type"].str.lower() != "pitcher"
    prob += lpSum(decision_vars[idx] for idx in hitters_mask[hitters_mask].index) == TOTAL_PLAYERS - 1

    # Slot caps for position-limited players
    if "roster_position" in pool.columns:
        _slot_groups = {"C": "C/1B", "1B": "C/1B", "2B": "2B", "3B": "3B", "SS": "SS", "OF": "OF"}
        for group_label, cap in _SLOT_CAPS.items():
            group_tokens = set(group_label.split("/"))

            def _is_limited_to_group(roster_pos: str, _tokens=group_tokens, _label=group_label) -> bool:
                parts = {p.strip() for p in str(roster_pos).upper().replace("-", "/").split("/")}
                parts.discard("UTIL")
                parts.discard("")
                if not parts:
                    return False
                mapped = {_slot_groups.get(p, p) for p in parts}
                return mapped == {_label}

            limited_mask = pool["roster_position"].map(_is_limited_to_group)
            if limited_mask.any():
                prob += lpSum(decision_vars[idx] for idx in limited_mask[limited_mask].index) <= cap

    # No opposing hitters for the selected pitcher
    batter_mask = pool["player_type"].str.lower() == "batter"
    for idx in pitcher_mask[pitcher_mask].index:
        opponent_team = str(pool.loc[idx, "opponent_code"] or "")
        if not opponent_team:
            continue
        opp_hitters = pool.index[(pool["team_code"] == opponent_team) & batter_mask]
        if opp_hitters.empty:
            continue
        prob += lpSum(decision_vars[j] for j in opp_hitters) <= (1 - decision_vars[idx]) * len(opp_hitters)

    # FanDuel rule: max 4 hitters from the same team
    for team_code in pool.loc[batter_mask, "team_code"].dropna().unique():
        team_hitter_indices = pool.index[(pool["team_code"] == team_code) & batter_mask]
        if len(team_hitter_indices) > MAX_HITTERS_PER_TEAM:
            prob += lpSum(decision_vars[idx] for idx in team_hitter_indices) <= MAX_HITTERS_PER_TEAM

    # Ownership cap
    if max_lineup_ownership is not None and "proj_fd_ownership" in pool.columns:
        prob += lpSum(
            pool.loc[idx, "proj_fd_ownership"] * var for idx, var in decision_vars.items()
        ) <= max_lineup_ownership

    # Exclusion constraints for previous lineups.
    # min_uniques = N means each new lineup must differ from every previous
    # lineup by at least N players (legacy behavior is N=1: merely not identical).
    min_uniques = max(1, int(min_uniques))
    max_shared = TOTAL_PLAYERS - min_uniques
    player_id_to_index = {
        str(pid): idx for idx, pid in pool["fd_player_id"].astype(str).items()
    }
    for lineup in previous_lineups:
        indices = [
            player_id_to_index[str(pid)]
            for pid in lineup
            if str(pid) in player_id_to_index
        ]
        if indices and len(indices) > max_shared:
            prob += lpSum(decision_vars[idx] for idx in indices) <= max_shared

    return prob, decision_vars


# ──────────────────────────────────────────────────────────────────────
# Stack constraints (assignment-based formulation)
# ──────────────────────────────────────────────────────────────────────

def _add_stack_constraints(
    prob: LpProblem,
    pool: pd.DataFrame,
    decision_vars: Dict[int, LpVariable],
    stack_template: Tuple[int, ...],
    min_game_total: Optional[float] = None,
    leverage_weight: float = 0.0,
) -> List[Tuple[str, str, LpVariable]]:
    """Add assignment-based stack constraints ensuring distinct teams per group.

    When leverage_weight > 0 and the pool has team_leverage_score, a bonus is
    added to the objective for assigning high-leverage teams to stack slots.
    The bonus scales with leverage_weight, team score, and slot size — nudging
    the solver toward low-owned, high-run-expectancy teams for stacks.
    """
    # Only constrain groups with >= 2 batters
    constrained = [(slot_idx, size) for slot_idx, size in enumerate(stack_template) if size >= 2]
    if not constrained:
        return []

    type_mask = pool["player_type"].str.lower() == "batter"
    cols = ["team_code", "opponent_code"]
    if "vegas_game_total" in pool.columns:
        cols.append("vegas_game_total")
    hitters = pool.loc[type_mask, cols].copy()
    hitters["team_code"] = hitters["team_code"].fillna("")
    hitters["opponent_code"] = hitters["opponent_code"].fillna("")
    if "vegas_game_total" not in hitters.columns:
        hitters["vegas_game_total"] = float("nan")
    hitters["vegas_game_total"] = pd.to_numeric(hitters["vegas_game_total"], errors="coerce")

    team_meta = hitters.drop_duplicates("team_code").set_index("team_code")
    team_codes = [code for code in team_meta.index if code]

    if min_game_total is not None:
        team_codes = [
            code for code in team_codes
            if pd.notna(team_meta.loc[code, "vegas_game_total"])
            and team_meta.loc[code, "vegas_game_total"] >= min_game_total
        ]
    if not team_codes:
        return []

    # Count available batters per team to skip teams that can't fill any slot
    team_batter_count: Dict[str, int] = {}
    team_batter_indices: Dict[str, pd.Index] = {}
    for tc in team_codes:
        indices = pool.index[(pool["team_code"] == tc) & type_mask]
        team_batter_count[tc] = len(indices)
        team_batter_indices[tc] = indices

    assign_vars: Dict[Tuple[str, int], LpVariable] = {}
    stack_details: List[Tuple[str, str, LpVariable]] = []

    for tc in team_codes:
        team_sum = lpSum(decision_vars[idx] for idx in team_batter_indices[tc])
        slots_for_team = []

        for slot_idx, group_size in constrained:
            capped_size = min(group_size, MAX_HITTERS_PER_TEAM)
            if team_batter_count[tc] < capped_size:
                continue  # team can't fill this slot
            var = LpVariable(f"assign_{tc}_s{slot_idx}", cat=LpBinary)
            assign_vars[(tc, slot_idx)] = var
            slots_for_team.append(var)

            # If this team is assigned to this slot, enforce >= group_size batters
            prob += team_sum >= capped_size * var

        # Each team can fill at most one slot
        if len(slots_for_team) > 1:
            prob += lpSum(slots_for_team) <= 1

    # Each slot must be filled by exactly one team
    for slot_idx, group_size in constrained:
        slot_vars = [assign_vars[(tc, slot_idx)] for tc in team_codes if (tc, slot_idx) in assign_vars]
        if slot_vars:
            prob += lpSum(slot_vars) == 1
        else:
            # No team can fill this slot — template is infeasible
            return []

    # ── Team leverage bonus ───────────────────────────────────────────────────
    # Add objective bonus for assigning high-leverage teams to stack slots.
    # Bonus = leverage_weight * team_leverage_score * slot_size * mean_proj
    # This nudges the solver toward low-owned, high-run-expectancy teams.
    if leverage_weight > 0 and "team_leverage_score" in pool.columns and assign_vars:
        mean_proj = float(pool["proj_fd_mean"].mean()) if "proj_fd_mean" in pool.columns else 35.0
        # Build per-team leverage score (use the team's own value from any player row)
        team_leverage: Dict[str, float] = {}
        for tc in team_codes:
            idx = team_batter_indices[tc]
            if len(idx) > 0 and "team_leverage_score" in pool.columns:
                team_leverage[tc] = float(pool.loc[idx[0], "team_leverage_score"])

        slot_size_map = {slot_idx: size for slot_idx, size in constrained}
        bonus_terms = []
        for (tc, slot_idx), var in assign_vars.items():
            score = team_leverage.get(tc, 0.0)
            slot_size = slot_size_map.get(slot_idx, 1)
            bonus = leverage_weight * score * slot_size * mean_proj
            if bonus != 0.0:
                bonus_terms.append(bonus * var)
        if bonus_terms:
            prob.objective += lpSum(bonus_terms)

    # Build stack_details for bring-back (use the primary/largest slot)
    primary_slot = constrained[0][0]
    for tc in team_codes:
        if (tc, primary_slot) in assign_vars:
            opp = team_meta.loc[tc, "opponent_code"] if tc in team_meta.index else ""
            opp = opp if isinstance(opp, str) else ""
            stack_details.append((tc, opp, assign_vars[(tc, primary_slot)]))

    return stack_details


# ──────────────────────────────────────────────────────────────────────
# Main lineup generation
# ──────────────────────────────────────────────────────────────────────

def generate_lineups(
    dataset: pd.DataFrame,
    num_lineups: int = 20,
    salary_cap: int = SALARY_CAP,
    min_stack_size: int = 0,
    stack_player_types: Sequence[str] = ("batter",),
    stack_templates: Optional[Sequence[int]] = None,
    stack_template: Optional[Tuple[int, ...]] = None,
    stack_rotation: Optional[List[Optional[Tuple[int, ...]]]] = None,
    max_lineup_ownership: Optional[float] = None,
    bring_back_enabled: bool = False,
    bring_back_count: int = 1,
    min_game_total_for_stacks: Optional[float] = None,
    leverage_weight: float = 0.0,
    randomness: float = 0.05,
    locked_player_ids: Optional[Sequence[str]] = None,
    ceiling_weight: float = 0.0,
    ownership_penalty_weight: float = 0.0,
    min_salary: Optional[int] = None,
    min_uniques: int = 1,
) -> List[LineupResult]:
    df = dataset.copy()
    df["fd_player_id"] = df["fd_player_id"].astype(str).str.strip()
    df["player_type"] = df["player_type"].astype(str)
    df["proj_fd_mean"] = pd.to_numeric(df["proj_fd_mean"], errors="coerce").fillna(0.0)
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)
    rng = np.random.default_rng()

    # Resolve stack template: rotation > single template > legacy fallback
    if stack_rotation is not None:
        # Multi-template mode: cycle through the rotation list
        pass  # handled per-lineup below
    elif stack_template is None and stack_templates:
        stack_template = tuple(min(s, MAX_HITTERS_PER_TEAM) for s in stack_templates if s and s > 0)
        if not stack_template:
            stack_template = None
    elif stack_template is None and min_stack_size and min_stack_size > 1:
        stack_template = (min(int(min_stack_size), MAX_HITTERS_PER_TEAM),)

    bring_back_count = max(1, int(bring_back_count))

    usage_limits = _max_usage(df, num_lineups)
    usage_counts: Dict[str, int] = {pid: 0 for pid in usage_limits}
    previous_lineups: List[List[str]] = []
    results: List[LineupResult] = []
    _seen_sets: set = set()
    locked_id_set = {str(pid).strip() for pid in (locked_player_ids or []) if str(pid).strip()}

    max_attempts = num_lineups * 4 + 20
    for lineup_index in range(max_attempts):
        if len(results) >= num_lineups:
            break
        eligible_mask = df["fd_player_id"].map(
            lambda pid: usage_counts.get(str(pid), 0) < usage_limits.get(str(pid), 0)
        )
        pool = df[eligible_mask].reset_index(drop=True)
        if len(pool) < TOTAL_PLAYERS:
            break

        # Build base LP (no stacks)
        prob, decision_vars = _build_base_lp(
            pool, lineup_index, salary_cap, max_lineup_ownership, previous_lineups,
            leverage_weight=leverage_weight,
            randomness=randomness, rng=rng,
            locked_player_ids=locked_id_set,
            ceiling_weight=ceiling_weight,
            ownership_penalty_weight=ownership_penalty_weight,
            min_salary=min_salary,
            min_uniques=min_uniques,
        )

        # Pick the template for this lineup (rotation or single)
        if stack_rotation:
            current_template = stack_rotation[len(results) % len(stack_rotation)]
        else:
            current_template = stack_template

        # Try adding stack constraints
        stack_info: List[Tuple[str, str, LpVariable]] = []
        used_stacks = current_template is not None
        requires_stack_constraints = bool(
            current_template and any(size >= 2 for size in current_template)
        )
        stack_constraints_unavailable = False
        if used_stacks:
            stack_info = _add_stack_constraints(
                prob, pool, decision_vars, current_template,
                min_game_total=min_game_total_for_stacks,
                leverage_weight=leverage_weight,
            )
            stack_constraints_unavailable = requires_stack_constraints and not stack_info

        # Bring-back constraints (only when stacks applied)
        if bring_back_enabled and stack_info:
            batter_mask = pool["player_type"].str.lower() == "batter"
            for team_code, opponent_code, stack_var in stack_info:
                if not opponent_code:
                    continue
                opp_indices = pool.index[(pool["team_code"] == opponent_code) & batter_mask]
                if opp_indices.empty:
                    continue
                prob += lpSum(decision_vars[idx] for idx in opp_indices) >= bring_back_count * stack_var

        fell_back_to_no_stacks = False
        if stack_constraints_unavailable:
            warnings.warn(
                f"Lineup {lineup_index}: stack template {current_template} could not be applied, solving without stacks.",
                stacklevel=2,
            )
            prob, decision_vars = _build_base_lp(
                pool, lineup_index, salary_cap, max_lineup_ownership, previous_lineups, tag="_ns_empty",
                leverage_weight=leverage_weight,
                randomness=randomness, rng=rng,
                locked_player_ids=locked_id_set,
                ceiling_weight=ceiling_weight,
                ownership_penalty_weight=ownership_penalty_weight,
                min_salary=min_salary,
                min_uniques=min_uniques,
            )
            status = prob.solve(PULP_CBC_CMD(msg=False))
            fell_back_to_no_stacks = True
        else:
            status = prob.solve(PULP_CBC_CMD(msg=False))

        # Fallback: if stacks made it infeasible, solve without stacks
        if status != LpStatusOptimal and used_stacks and not fell_back_to_no_stacks:
            warnings.warn(
                f"Lineup {lineup_index}: stack template {current_template} infeasible, solving without stacks.",
                stacklevel=2,
            )
            prob, decision_vars = _build_base_lp(
                pool, lineup_index, salary_cap, max_lineup_ownership, previous_lineups, tag="_ns",
                leverage_weight=leverage_weight,
                randomness=randomness, rng=rng,
                locked_player_ids=locked_id_set,
                ceiling_weight=ceiling_weight,
                ownership_penalty_weight=ownership_penalty_weight,
                min_salary=min_salary,
                min_uniques=min_uniques,
            )
            status = prob.solve(PULP_CBC_CMD(msg=False))
            fell_back_to_no_stacks = True

        if status != LpStatusOptimal:
            break

        selected_indices = [idx for idx, var in decision_vars.items() if var.varValue == 1]
        lineup_df = pool.loc[selected_indices].copy()
        lineup_df = lineup_df.sort_values(by=["player_type", "position"], ascending=[True, True])

        player_ids = lineup_df["fd_player_id"].astype(str).tolist()
        player_set = frozenset(player_ids)
        if player_set in _seen_sets:
            previous_lineups.append(player_ids)
            continue
        _seen_sets.add(player_set)

        for pid in player_ids:
            usage_counts[pid] = usage_counts.get(pid, 0) + 1

        previous_lineups.append(player_ids)
        results.append(
            LineupResult(
                dataframe=lineup_df,
                total_salary=int(lineup_df["salary"].sum()),
                total_projection=float(lineup_df["proj_fd_mean"].sum()),
                stack_template_used=current_template if not fell_back_to_no_stacks else None,
                stacks_dropped=fell_back_to_no_stacks,
            )
        )

    return results


__all__ = ["generate_lineups", "LineupResult", "STACK_PRESETS"]
