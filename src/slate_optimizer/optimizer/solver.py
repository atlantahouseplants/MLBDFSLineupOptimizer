"""Simple ILP-based MLB lineup solver."""
from __future__ import annotations

import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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


def _position_mask(df: pd.DataFrame, keyword: str) -> pd.Series:
    keyword = keyword.upper()
    tokens = keyword.split("/")
    pos_col = "roster_position" if "roster_position" in df.columns else "position"

    def matches(value: str) -> bool:
        text = str(value).upper()
        parts = text.replace("-", "/").split("/")
        parts = [p for p in parts if p != "UTIL"]
        return any(tok in parts for tok in tokens)

    return df[pos_col].map(matches)


@dataclass
class LineupResult:
    dataframe: pd.DataFrame
    total_salary: int
    total_projection: float


def _max_usage(df: pd.DataFrame, num_lineups: int) -> Dict[str, int]:
    usage: Dict[str, int] = {}
    for _, row in df.iterrows():
        exposure = row.get("default_max_exposure", 1.0)
        try:
            exposure = float(exposure)
        except (TypeError, ValueError):
            exposure = 1.0
        allowed = max(1, int(round(exposure * num_lineups)))
        usage[row["fd_player_id"]] = allowed
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
) -> Tuple[LpProblem, Dict[int, LpVariable]]:
    """Build the LP with all constraints except stack/bring-back."""
    label = f"mlb_lineup_{lineup_index}{tag}"
    prob = LpProblem(label, LpMaximize)
    decision_vars = {
        idx: LpVariable(f"x{tag}_{idx}", lowBound=0, upBound=1, cat=LpBinary)
        for idx in pool.index
    }

    # Objective: maximize projected score
    prob += lpSum(pool.loc[idx, "proj_fd_mean"] * var for idx, var in decision_vars.items())

    # Salary constraint
    prob += lpSum(pool.loc[idx, "salary"] * var for idx, var in decision_vars.items()) <= salary_cap

    # Exactly 9 players
    prob += lpSum(var for var in decision_vars.values()) == TOTAL_PLAYERS

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

    # Exclusion constraints for previous lineups
    for lineup in previous_lineups:
        indices = [
            pool.index[pool["fd_player_id"] == pid][0]
            for pid in lineup
            if pid in pool["fd_player_id"].values
        ]
        if indices:
            prob += lpSum(decision_vars[idx] for idx in indices) <= len(indices) - 1

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
) -> List[Tuple[str, str, LpVariable]]:
    """Add assignment-based stack constraints ensuring distinct teams per group."""
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
            prob += lpSum(slot_vars) >= 1
        else:
            # No team can fill this slot — template is infeasible
            return []

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
) -> List[LineupResult]:
    df = dataset.copy()
    df["proj_fd_mean"] = pd.to_numeric(df["proj_fd_mean"], errors="coerce").fillna(0.0)
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)

    # Resolve stack template: rotation > single template > legacy fallback
    if stack_rotation is not None:
        # Multi-template mode: cycle through the rotation list
        pass  # handled per-lineup below
    elif stack_template is None and stack_templates:
        stack_template = tuple(min(s, MAX_HITTERS_PER_TEAM) for s in stack_templates if s and s > 0)
        if not stack_template:
            stack_template = None

    bring_back_count = max(1, int(bring_back_count))

    usage_limits = _max_usage(df, num_lineups)
    usage_counts: Dict[str, int] = {pid: 0 for pid in usage_limits}
    previous_lineups: List[List[str]] = []
    results: List[LineupResult] = []
    _seen_sets: set = set()

    max_attempts = num_lineups * 4 + 20
    for lineup_index in range(max_attempts):
        if len(results) >= num_lineups:
            break
        eligible_mask = df["fd_player_id"].map(
            lambda pid: usage_counts.get(pid, 0) < usage_limits.get(pid, 0)
        )
        pool = df[eligible_mask].reset_index(drop=True)
        if len(pool) < TOTAL_PLAYERS:
            break

        # Build base LP (no stacks)
        prob, decision_vars = _build_base_lp(
            pool, lineup_index, salary_cap, max_lineup_ownership, previous_lineups,
        )

        # Pick the template for this lineup (rotation or single)
        if stack_rotation:
            current_template = stack_rotation[len(results) % len(stack_rotation)]
        else:
            current_template = stack_template

        # Try adding stack constraints
        stack_info: List[Tuple[str, str, LpVariable]] = []
        used_stacks = current_template is not None
        if used_stacks:
            stack_info = _add_stack_constraints(
                prob, pool, decision_vars, current_template,
                min_game_total=min_game_total_for_stacks,
            )

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

        status = prob.solve(PULP_CBC_CMD(msg=False))

        # Fallback: if stacks made it infeasible, solve without stacks
        if status != LpStatusOptimal and used_stacks:
            warnings.warn(
                f"Lineup {lineup_index}: stack template {current_template} infeasible, solving without stacks.",
                stacklevel=2,
            )
            prob, decision_vars = _build_base_lp(
                pool, lineup_index, salary_cap, max_lineup_ownership, previous_lineups, tag="_ns",
            )
            status = prob.solve(PULP_CBC_CMD(msg=False))

        if status != LpStatusOptimal:
            break

        selected_indices = [idx for idx, var in decision_vars.items() if var.varValue == 1]
        lineup_df = pool.loc[selected_indices].copy()
        lineup_df = lineup_df.sort_values(by=["player_type", "position"], ascending=[True, True])

        player_set = frozenset(lineup_df["fd_player_id"].tolist())
        if player_set in _seen_sets:
            previous_lineups.append(lineup_df["fd_player_id"].tolist())
            continue
        _seen_sets.add(player_set)

        for pid in lineup_df["fd_player_id"]:
            usage_counts[pid] = usage_counts.get(pid, 0) + 1

        previous_lineups.append(lineup_df["fd_player_id"].tolist())
        results.append(
            LineupResult(
                dataframe=lineup_df,
                total_salary=int(lineup_df["salary"].sum()),
                total_projection=float(lineup_df["proj_fd_mean"].sum()),
            )
        )

    return results


__all__ = ["generate_lineups", "LineupResult", "STACK_PRESETS"]
