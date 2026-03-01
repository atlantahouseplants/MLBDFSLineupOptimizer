"""Simple ILP-based MLB lineup solver."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

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


def _position_mask(df: pd.DataFrame, keyword: str) -> pd.Series:
    keyword = keyword.upper()
    tokens = keyword.split("/")

    def matches(value: str) -> bool:
        text = str(value).upper()
        parts = text.replace("-", "/").split("/")
        return any(tok in parts for tok in tokens)

    return df["position"].map(matches)


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
):
    valid_sizes = [size for size in stack_sizes if size and size > 0]
    if not valid_sizes:
        return

    stack_types = {t.lower() for t in stack_player_types}
    type_mask = pool["player_type"].str.lower().isin(stack_types)
    team_codes = pool.loc[type_mask, "team_code"].fillna("").unique()

    for size in valid_sizes:
        stack_vars = {}
        for team_code in team_codes:
            if not team_code:
                continue
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
        if stack_vars:
            prob += lpSum(stack_vars.values()) >= 1


def generate_lineups(
    dataset: pd.DataFrame,
    num_lineups: int = 20,
    salary_cap: int = SALARY_CAP,
    min_stack_size: int = 4,
    stack_player_types: Sequence[str] = ("batter",),
    stack_templates: Optional[Sequence[int]] = None,
    max_lineup_ownership: Optional[float] = None,
) -> List[LineupResult]:
    df = dataset.copy()
    df["proj_fd_mean"] = pd.to_numeric(df["proj_fd_mean"], errors="coerce").fillna(0.0)
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)

    stack_sizes = tuple(stack_templates) if stack_templates else ((min_stack_size,) if min_stack_size else tuple())

    usage_limits = _max_usage(df, num_lineups)
    usage_counts: Dict[str, int] = {pid: 0 for pid in usage_limits}
    previous_lineups: List[List[str]] = []
    results: List[LineupResult] = []

    for lineup_index in range(num_lineups):
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

        _team_stack_constraints(prob, pool, decision_vars, stack_sizes, stack_player_types)

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
            break

        selected_indices = [idx for idx, var in decision_vars.items() if var.varValue == 1]
        lineup_df = pool.loc[selected_indices].copy()
        lineup_df = lineup_df.sort_values(by=["player_type", "position"], ascending=[True, True])

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
