"""Simple ILP-based MLB lineup solver."""
from __future__ import annotations

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

POSITION_REQUIREMENTS = {
    "pitcher": ("P", 1, 1),
    "catcher_first": ("C/1B", 1, None),
    "second_base": ("2B", 1, None),
    "third_base": ("3B", 1, None),
    "shortstop": ("SS", 1, None),
    "outfield": ("OF", 3, None),
}

# Maximum players whose ONLY non-UTIL eligibility is within this group.
# E.g. a pure 1B (roster_position C/1B/UTIL) can only fill C/1B or UTIL.
# With 1 C/1B slot + 1 UTIL slot, max 2 such players.
_SLOT_CAPS = {
    "C/1B": 2,   # C/1B slot + UTIL
    "2B": 2,     # 2B slot + UTIL
    "3B": 2,     # 3B slot + UTIL
    "SS": 2,     # SS slot + UTIL
    "OF": 4,     # 3 OF slots + UTIL
}


def _position_mask(df: pd.DataFrame, keyword: str) -> pd.Series:
    keyword = keyword.upper()
    tokens = keyword.split("/")

    # Use roster_position (full FanDuel eligibility) when available
    pos_col = "roster_position" if "roster_position" in df.columns else "position"

    def matches(value: str) -> bool:
        text = str(value).upper()
        parts = text.replace("-", "/").split("/")
        # Exclude UTIL from position matching — UTIL is implicit for all hitters
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


def _team_stack_constraints(
    prob: LpProblem,
    pool: pd.DataFrame,
    decision_vars: Dict[int, LpVariable],
    stack_sizes: Sequence[int],
    stack_player_types: Sequence[str],
    min_game_total: Optional[float] = None,
) -> List[Tuple[str, str, LpVariable]]:
    valid_sizes = [size for size in stack_sizes if size and size > 0]
    if not valid_sizes:
        return []

    stack_types = {t.lower() for t in stack_player_types}
    type_mask = pool["player_type"].str.lower().isin(stack_types)
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
            code
            for code in team_codes
            if pd.notna(team_meta.loc[code, "vegas_game_total"]) and team_meta.loc[code, "vegas_game_total"] >= min_game_total
        ]
    if not team_codes:
        return []

    stack_details: List[Tuple[str, str, LpVariable]] = []
    for size in valid_sizes:
        stack_vars = {}
        for team_code in team_codes:
            indices = pool.index[(pool["team_code"] == team_code) & type_mask]
            if indices.empty:
                continue
            stack_var = LpVariable(f"stack_{size}_{team_code}", lowBound=0, upBound=1, cat=LpBinary)
            stack_vars[team_code] = stack_var
            prob += (
                lpSum(decision_vars[idx] for idx in indices)
                - size * stack_var
                >= 0
            )
            opponent_code = team_meta.loc[team_code, "opponent_code"] if team_code in team_meta.index else ""
            opponent_code = opponent_code if isinstance(opponent_code, str) else ""
            stack_details.append((team_code, opponent_code, stack_var))
        if stack_vars:
            prob += lpSum(stack_vars.values()) >= 1
    return stack_details


def generate_lineups(
    dataset: pd.DataFrame,
    num_lineups: int = 20,
    salary_cap: int = SALARY_CAP,
    min_stack_size: int = 4,
    stack_player_types: Sequence[str] = ("batter",),
    stack_templates: Optional[Sequence[int]] = None,
    max_lineup_ownership: Optional[float] = None,
    bring_back_enabled: bool = False,
    bring_back_count: int = 1,
    min_game_total_for_stacks: Optional[float] = None,
) -> List[LineupResult]:
    df = dataset.copy()
    df["proj_fd_mean"] = pd.to_numeric(df["proj_fd_mean"], errors="coerce").fillna(0.0)
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)

    MAX_HITTERS_PER_TEAM = 4  # FanDuel hard cap
    stack_sizes = tuple(stack_templates) if stack_templates else ((min_stack_size,) if min_stack_size else tuple())
    stack_sizes = tuple(min(s, MAX_HITTERS_PER_TEAM) for s in stack_sizes)
    bring_back_count = max(1, int(bring_back_count))

    usage_limits = _max_usage(df, num_lineups)
    usage_counts: Dict[str, int] = {pid: 0 for pid in usage_limits}
    previous_lineups: List[List[str]] = []
    results: List[LineupResult] = []
    _seen_sets: set = set()

    max_attempts = num_lineups * 2
    for lineup_index in range(max_attempts):
        if len(results) >= num_lineups:
            break
        eligible_mask = df["fd_player_id"].map(
            lambda pid: usage_counts.get(pid, 0) < usage_limits.get(pid, 0)
        )
        pool = df[eligible_mask].reset_index(drop=True)
        if len(pool) < TOTAL_PLAYERS:
            break

        prob = LpProblem(f"mlb_lineup_{lineup_index}", LpMaximize)
        decision_vars = {
            idx: LpVariable(f"x_{idx}", lowBound=0, upBound=1, cat=LpBinary)
            for idx in pool.index
        }

        prob += lpSum(pool.loc[idx, "proj_fd_mean"] * var for idx, var in decision_vars.items())
        prob += lpSum(pool.loc[idx, "salary"] * var for idx, var in decision_vars.items()) <= salary_cap
        prob += lpSum(var for var in decision_vars.values()) == TOTAL_PLAYERS

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

        # Ensure at least 8 hitters (= 9 total - 1 pitcher)
        hitters_mask = pool["player_type"].str.lower() != "pitcher"
        prob += lpSum(decision_vars[idx] for idx in hitters_mask[hitters_mask].index) == TOTAL_PLAYERS - 1

        # Cap players whose eligibility is limited to a single position group.
        # Prevents e.g. 3 pure-1B players when only C/1B + UTIL slots exist (max 2).
        if "roster_position" in pool.columns:
            _slot_groups = {"C": "C/1B", "1B": "C/1B", "2B": "2B", "3B": "3B", "SS": "SS", "OF": "OF"}
            for group_label, cap in _SLOT_CAPS.items():
                group_tokens = set(group_label.split("/"))

                def _is_limited_to_group(roster_pos: str) -> bool:
                    parts = {p.strip() for p in str(roster_pos).upper().replace("-", "/").split("/")}
                    parts.discard("UTIL")
                    parts.discard("")
                    if not parts:
                        return False
                    mapped = {_slot_groups.get(p, p) for p in parts}
                    return mapped == {group_label}

                limited_mask = pool["roster_position"].map(_is_limited_to_group)
                if limited_mask.any():
                    prob += lpSum(decision_vars[idx] for idx in limited_mask[limited_mask].index) <= cap

        stack_info = _team_stack_constraints(
            prob,
            pool,
            decision_vars,
            stack_sizes,
            stack_player_types,
            min_game_total=min_game_total_for_stacks,
        )

        if bring_back_enabled and stack_info:
            batter_mask = pool["player_type"].str.lower() == "batter"
            for team_code, opponent_code, stack_var in stack_info:
                if not opponent_code:
                    continue
                opp_indices = pool.index[(pool["team_code"] == opponent_code) & batter_mask]
                if opp_indices.empty:
                    continue
                prob += lpSum(decision_vars[idx] for idx in opp_indices) >= bring_back_count * stack_var

        batter_mask = pool["player_type"].str.lower() == "batter"
        for idx in pitcher_mask[pitcher_mask].index:
            opponent_team = str(pool.loc[idx, "opponent_code"] or "")
            if not opponent_team:
                continue
            opp_hitters = pool.index[(pool["team_code"] == opponent_team) & batter_mask]
            if opp_hitters.empty:
                continue
            prob += lpSum(decision_vars[j] for j in opp_hitters) <= (1 - decision_vars[idx]) * len(opp_hitters)

        # FanDuel rule: max 4 hitters from the same team (pitcher excluded)
        for team_code in pool.loc[batter_mask, "team_code"].dropna().unique():
            team_hitter_indices = pool.index[(pool["team_code"] == team_code) & batter_mask]
            if len(team_hitter_indices) > 4:
                prob += lpSum(decision_vars[idx] for idx in team_hitter_indices) <= 4

        if max_lineup_ownership is not None and "proj_fd_ownership" in pool.columns:
            prob += lpSum(
                pool.loc[idx, "proj_fd_ownership"] * var for idx, var in decision_vars.items()
            ) <= max_lineup_ownership

        for lineup in previous_lineups:
            indices = [
                pool.index[pool["fd_player_id"] == pid][0]
                for pid in lineup
                if pid in pool["fd_player_id"].values
            ]
            if indices:
                prob += lpSum(decision_vars[idx] for idx in indices) <= len(indices) - 1

        status = prob.solve(PULP_CBC_CMD(msg=False))
        if status != LpStatusOptimal:
            if lineup_index == 0 and stack_sizes:
                # First lineup infeasible with stack constraints — retry without
                import warnings
                warnings.warn(
                    "LP infeasible with stack templates; retrying without stack constraints.",
                    stacklevel=2,
                )
                prob2 = LpProblem(f"mlb_lineup_{lineup_index}_nostacks", LpMaximize)
                vars2 = {
                    idx: LpVariable(f"y_{idx}", lowBound=0, upBound=1, cat=LpBinary)
                    for idx in pool.index
                }
                prob2 += lpSum(pool.loc[idx, "proj_fd_mean"] * v for idx, v in vars2.items())
                prob2 += lpSum(pool.loc[idx, "salary"] * v for idx, v in vars2.items()) <= salary_cap
                prob2 += lpSum(v for v in vars2.values()) == TOTAL_PLAYERS
                pitcher_mask2 = _position_mask(pool, "P")
                prob2 += lpSum(vars2[idx] for idx in pitcher_mask2[pitcher_mask2].index) == 1
                for _, (keyword, minimum, maximum) in POSITION_REQUIREMENTS.items():
                    if keyword == "P":
                        continue
                    mask2 = _position_mask(pool, keyword)
                    if mask2.any():
                        prob2 += lpSum(vars2[idx] for idx in mask2[mask2].index) >= minimum
                        if maximum:
                            prob2 += lpSum(vars2[idx] for idx in mask2[mask2].index) <= maximum
                hitters_mask2 = pool["player_type"].str.lower() != "pitcher"
                prob2 += lpSum(vars2[idx] for idx in hitters_mask2[hitters_mask2].index) == TOTAL_PLAYERS - 1
                batter_mask2 = pool["player_type"].str.lower() == "batter"
                for team_code2 in pool.loc[batter_mask2, "team_code"].dropna().unique():
                    ti = pool.index[(pool["team_code"] == team_code2) & batter_mask2]
                    if len(ti) > 4:
                        prob2 += lpSum(vars2[idx] for idx in ti) <= 4
                if max_lineup_ownership is not None and "proj_fd_ownership" in pool.columns:
                    prob2 += lpSum(pool.loc[idx, "proj_fd_ownership"] * v for idx, v in vars2.items()) <= max_lineup_ownership
                status2 = prob2.solve(PULP_CBC_CMD(msg=False))
                if status2 == LpStatusOptimal:
                    # Disable stacks for all future iterations
                    stack_sizes = ()
                    decision_vars = vars2
                    # Fall through to extract results
                else:
                    break
            else:
                break

        selected_indices = [idx for idx, var in decision_vars.items() if var.varValue == 1]
        lineup_df = pool.loc[selected_indices].copy()
        lineup_df = lineup_df.sort_values(by=["player_type", "position"], ascending=[True, True])

        player_set = frozenset(lineup_df["fd_player_id"].tolist())
        if player_set in _seen_sets:
            # Still add exclusion constraint so LP doesn't repeat it
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


__all__ = ["generate_lineups", "LineupResult"]
