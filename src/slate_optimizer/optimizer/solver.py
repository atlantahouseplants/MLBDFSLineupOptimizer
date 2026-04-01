"""Simple ILP-based MLB lineup solver."""
from __future__ import annotations

import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

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

if TYPE_CHECKING:
    from .config import LeverageConfig

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
# GPP scoring helper
# ──────────────────────────────────────────────────────────────────────

def _compute_gpp_score(
    pool: pd.DataFrame,
    idx: int,
    leverage_config: "LeverageConfig",
    pitcher_top2_avg_own: float = 0.0,
) -> float:
    """Compute leverage-weighted objective score for a single player."""
    mean = float(pool.loc[idx, "proj_fd_mean"])
    ceiling = float(pool.loc[idx, "proj_fd_ceiling"]) if "proj_fd_ceiling" in pool.columns else mean * 1.2
    ownership = max(float(pool.loc[idx, "proj_fd_ownership"]) if "proj_fd_ownership" in pool.columns else 0.05, 0.005)

    score = mean
    score += leverage_config.ceiling_weight * ceiling

    # Ownership penalty scaled relative to the player's mean projection.
    # This ensures a 25%-owned ace still scores well, while a 25%-owned
    # mediocre player gets penalized more proportionally.
    relative_ownership_cost = 0.0
    if mean > 0:
        relative_ownership_cost = ownership / max(mean, 1.0)
        score -= leverage_config.ownership_penalty * relative_ownership_cost * mean * 0.5

    # Extra chalk penalty (also relative)
    if ownership > leverage_config.chalk_threshold and mean > 0:
        score -= leverage_config.chalk_extra_penalty * relative_ownership_cost * mean * 0.3

    # Boom potential: (ceiling - mean) / mean gives upside percentage
    if mean > 0:
        boom_pct = (ceiling - mean) / mean
        score += leverage_config.boom_weight * boom_pct * mean

    # Pitcher fade bonus: reward non-chalk pitchers
    if leverage_config.pitcher_fade_bonus > 0 and pitcher_top2_avg_own > 0:
        player_type = str(pool.loc[idx, "player_type"]).lower() if "player_type" in pool.columns else ""
        if player_type == "pitcher" and ownership < pitcher_top2_avg_own * 0.6:
            score += leverage_config.pitcher_fade_bonus * (pitcher_top2_avg_own - ownership)

    return score


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
    leverage_config: Optional["LeverageConfig"] = None,
) -> Tuple[LpProblem, Dict[int, LpVariable]]:
    """Build the LP with all constraints except stack/bring-back."""
    label = f"mlb_lineup_{lineup_index}{tag}"
    prob = LpProblem(label, LpMaximize)
    decision_vars = {
        idx: LpVariable(f"x{tag}_{idx}", lowBound=0, upBound=1, cat=LpBinary)
        for idx in pool.index
    }

    # Objective: maximize projected score (leverage-weighted in GPP mode)
    if leverage_config is not None and leverage_config.is_gpp:
        # Precompute pitcher top-2 avg ownership for pitcher fade logic
        _p2_avg = 0.0
        if leverage_config.pitcher_fade_bonus > 0 and "proj_fd_ownership" in pool.columns:
            pitcher_pool = pool[pool["player_type"].astype(str).str.lower() == "pitcher"]
            if len(pitcher_pool) >= 2:
                _p2_avg = float(pitcher_pool["proj_fd_ownership"].nlargest(2).mean())
        prob += lpSum(
            _compute_gpp_score(pool, idx, leverage_config, pitcher_top2_avg_own=_p2_avg) * var
            for idx, var in decision_vars.items()
        )
    else:
        prob += lpSum(pool.loc[idx, "proj_fd_mean"] * var for idx, var in decision_vars.items())

    # Salary constraint
    prob += lpSum(pool.loc[idx, "salary"] * var for idx, var in decision_vars.items()) <= salary_cap

    # Minimum salary usage — don't waste cap space
    # For a $35K cap, require at least $32K usage (91% utilization)
    min_salary = int(salary_cap * 0.91)
    prob += lpSum(pool.loc[idx, "salary"] * var for idx, var in decision_vars.items()) >= min_salary

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
# Smart auto-stack selection
# ──────────────────────────────────────────────────────────────────────

def _select_auto_stack_template(
    pool: pd.DataFrame,
    slate_type: str = "medium",
    num_lineups: int = 20,
) -> Tuple[int, ...]:
    """Choose a stack template based on slate size and game environment tiers.

    Returns a tuple like (4, 4) or (4, 3, 1) depending on how many
    attractive ("prime" / "good") games exist on the slate.
    """
    num_games = pool["game_key"].nunique() if "game_key" in pool.columns else 0
    tiers = pool.drop_duplicates("game_key")["environment_tier"].value_counts() if "environment_tier" in pool.columns else pd.Series(dtype=int)
    prime_count = int(tiers.get("prime", 0))
    good_count = int(tiers.get("good", 0))

    # ── Small slate (2-3 games) ──────────────────────────────────────
    if slate_type == "small" or num_games <= 3:
        if num_games <= 2:
            return (4, 4)
        # 3-game slate: pick based on game quality
        if prime_count >= 1:
            return (4, 3, 1)
        return (4, 4)

    # ── Large slate (9+ games) ───────────────────────────────────────
    if slate_type == "large" or num_games >= 9:
        if prime_count >= 2:
            return (4, 3, 1)
        if prime_count == 1 and good_count >= 1:
            return (4, 2, 2)
        return (4, 2, 1, 1)

    # ── Medium slate (default) ───────────────────────────────────────
    if prime_count >= 2:
        return (4, 4)
    if prime_count >= 1:
        return (4, 3, 1)
    if good_count >= 2:
        return (4, 3, 1)  # 4-batter primary stack is always better in baseball
    if good_count >= 1:
        return (4, 2, 2)
    return (3, 3, 2)  # absolute fallback only


def _build_large_slate_rotation(num_lineups: int) -> List[Tuple[int, ...]]:
    """Build a template rotation for large slates with 100+ lineups."""
    prime_templates = [(4, 3, 1), (4, 2, 2)]
    secondary_templates = [(3, 3, 2), (4, 2, 1, 1)]
    wild_templates = [(3, 2, 2, 1), (3, 2, 1, 1, 1)]

    prime_count = int(num_lineups * 0.60)
    secondary_count = int(num_lineups * 0.30)
    wild_count = num_lineups - prime_count - secondary_count

    rotation: List[Tuple[int, ...]] = []
    for t in prime_templates:
        rotation.extend([t] * max(1, prime_count // len(prime_templates)))
    for t in secondary_templates:
        rotation.extend([t] * max(1, secondary_count // len(secondary_templates)))
    for t in wild_templates:
        rotation.extend([t] * max(1, wild_count // len(wild_templates)))
    return rotation


def _add_stack_leverage_bonus(
    prob: LpProblem,
    pool: pd.DataFrame,
    stack_info: List[Tuple[str, str, LpVariable]],
    leverage_bonus: float,
) -> None:
    """Add a soft objective bonus that tilts stack assignment toward high-leverage teams."""
    if not stack_info or leverage_bonus <= 0 or "team_gpp_leverage" not in pool.columns:
        return
    # Compute per-team leverage
    is_batter = pool["player_type"].astype(str).str.lower() == "batter"
    team_leverage = (
        pool.loc[is_batter]
        .groupby("team_code")["team_gpp_leverage"]
        .first()
    )
    if team_leverage.empty:
        return
    # Normalize to 0-1 range
    lev_max = team_leverage.max()
    if lev_max and lev_max > 0:
        team_leverage = team_leverage / lev_max

    for team_code, _opp, assign_var in stack_info:
        lev = float(team_leverage.get(team_code, 0.0))
        if lev > 0:
            prob += leverage_bonus * lev * assign_var


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
    leverage_config: Optional["LeverageConfig"] = None,
) -> List[LineupResult]:
    df = dataset.copy()
    df["proj_fd_mean"] = pd.to_numeric(df["proj_fd_mean"], errors="coerce").fillna(0.0)
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)

    # Resolve stack template: rotation > single template > legacy fallback
    _gpp_mode = leverage_config is not None and leverage_config.is_gpp
    if stack_rotation is not None:
        # Multi-template mode: cycle through the rotation list
        pass  # handled per-lineup below
    elif stack_template is None and stack_templates:
        stack_template = tuple(min(s, MAX_HITTERS_PER_TEAM) for s in stack_templates if s and s > 0)
        if not stack_template:
            stack_template = None
    # Smart auto-stack: when in GPP mode and no template specified, pick one
    if stack_template is None and stack_rotation is None and _gpp_mode:
        # Detect slate type for template selection
        _num_games = df["game_key"].nunique() if "game_key" in df.columns else 0
        _slate_type = "small" if _num_games <= 3 else ("large" if _num_games >= 9 else "medium")
        # Large slate + many lineups → use rotation for template diversity
        if _slate_type == "large" and num_lineups >= 100:
            stack_rotation = _build_large_slate_rotation(num_lineups)
        else:
            stack_template = _select_auto_stack_template(df, slate_type=_slate_type, num_lineups=num_lineups)

    # Override bring-back from leverage config when available
    if _gpp_mode and leverage_config is not None:
        if leverage_config.bring_back_enabled:
            bring_back_enabled = True
        bring_back_count = max(bring_back_count, leverage_config.bring_back_count)
    bring_back_count = max(1, int(bring_back_count))

    # ── Section 5.4: Player floor filter (GPP mode) ──────────────────
    if _gpp_mode and leverage_config is not None:
        projection_cutoff = df["proj_fd_mean"].quantile(leverage_config.min_viable_projection_pct)
        is_pitcher = df["player_type"].str.lower() == "pitcher"
        viable_mask = is_pitcher | (df["proj_fd_mean"] >= projection_cutoff)
        # Also keep boom candidates (high ceiling even if mean is low)
        if "proj_fd_ceiling" in df.columns:
            boom_mask = df["proj_fd_ceiling"] >= df["proj_fd_ceiling"].quantile(0.6)
            viable_mask = viable_mask | boom_mask
        df = df[viable_mask].reset_index(drop=True)

    # ── Pitcher salary floor: exclude relievers from pitcher pool ────
    # On FanDuel, starting pitchers are generally $6,500+. Relievers at
    # $5,500-$6,000 should never be selected as the lineup's starting pitcher.
    # This applies in ALL modes, not just GPP.
    PITCHER_MIN_SALARY = 6500
    pitcher_mask_filter = (df["player_type"].str.lower() == "pitcher") & (df["salary"] < PITCHER_MIN_SALARY)
    if pitcher_mask_filter.any():
        df = df[~pitcher_mask_filter].reset_index(drop=True)

    # ── Pitcher must be a confirmed starter when lineup data is available ──
    if "is_confirmed_lineup" in df.columns:
        # For pitchers, require is_confirmed_lineup=True
        # This prevents relievers from being selected even if they pass the salary floor
        unconfirmed_pitcher = (
            (df["player_type"].str.lower() == "pitcher")
            & (~df["is_confirmed_lineup"].astype(bool))
        )
        if unconfirmed_pitcher.any() and df["is_confirmed_lineup"].astype(bool).any():
            # Only filter if we have SOME confirmed data (avoid filtering everything)
            confirmed_pitchers = df[
                (df["player_type"].str.lower() == "pitcher")
                & (df["is_confirmed_lineup"].astype(bool))
            ]
            if len(confirmed_pitchers) >= 2:
                # We have enough confirmed pitchers — remove unconfirmed ones
                df = df[~unconfirmed_pitcher].reset_index(drop=True)

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
            leverage_config=leverage_config,
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

        # Stack leverage bonus (GPP mode)
        if _gpp_mode and stack_info and leverage_config is not None:
            _add_stack_leverage_bonus(
                prob, pool, stack_info, leverage_config.stack_leverage_bonus,
            )

            # Section 5.5: Within-stack chalk penalty — prefer underowned batters on stacked teams
            if leverage_config.within_stack_chalk_penalty > 0 and "proj_fd_ownership" in pool.columns:
                batter_mask_ws = pool["player_type"].str.lower() == "batter"
                for team_code, _opp, assign_var in stack_info:
                    team_batter_idx = pool.index[
                        (pool["team_code"] == team_code) & batter_mask_ws
                    ]
                    for bidx in team_batter_idx:
                        player_own = float(pool.loc[bidx, "proj_fd_ownership"])
                        if player_own > leverage_config.chalk_threshold:
                            prob += (
                                -leverage_config.within_stack_chalk_penalty
                                * player_own
                                * decision_vars[bidx]
                                * assign_var
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

                # Section 6: Bring-back leverage preference — favor low-owned bring-back players
                if _gpp_mode and leverage_config is not None and "proj_fd_ownership" in pool.columns:
                    bb_bonus = leverage_config.bring_back_leverage_bonus
                    if bb_bonus > 0:
                        median_own = float(pool.loc[opp_indices, "proj_fd_ownership"].median())
                        for oi in opp_indices:
                            own = float(pool.loc[oi, "proj_fd_ownership"])
                            if own < median_own:
                                prob += bb_bonus * (median_own - own) * decision_vars[oi]

        status = prob.solve(PULP_CBC_CMD(msg=False))

        # Fallback: if stacks made it infeasible, try a simpler stack first
        if status != LpStatusOptimal and used_stacks:
            # Try a relaxed 3-3-2 before giving up on stacks entirely
            warnings.warn(
                f"Lineup {lineup_index}: template {current_template} infeasible, trying 3-3-2.",
                stacklevel=2,
            )
            prob, decision_vars = _build_base_lp(
                pool, lineup_index, salary_cap, max_lineup_ownership, previous_lineups, tag="_relax",
                leverage_config=leverage_config,
            )
            relaxed_info = _add_stack_constraints(
                prob, pool, decision_vars, (3, 3, 2),
                min_game_total=None,  # Drop game total filter for fallback
            )
            if relaxed_info:
                if _gpp_mode and leverage_config is not None:
                    _add_stack_leverage_bonus(prob, pool, relaxed_info, leverage_config.stack_leverage_bonus)
            status = prob.solve(PULP_CBC_CMD(msg=False))

            # Only if 3-3-2 also fails, go truly unconstrained
            if status != LpStatusOptimal:
                warnings.warn(
                    f"Lineup {lineup_index}: all stacks infeasible, solving without stacks.",
                    stacklevel=2,
                )
                prob, decision_vars = _build_base_lp(
                    pool, lineup_index, salary_cap, max_lineup_ownership, previous_lineups, tag="_ns",
                    leverage_config=leverage_config,
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
