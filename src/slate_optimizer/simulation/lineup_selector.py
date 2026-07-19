from __future__ import annotations

import itertools
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set

import numpy as np
import pandas as pd
from pulp import LpBinary, LpMaximize, LpProblem, LpStatusOptimal, LpVariable, PULP_CBC_CMD, lpSum

from .contest_simulator import ContestSimResult, LineupSimResult

__all__ = ["PortfolioSelection", "select_portfolio"]


@dataclass
class PortfolioSelection:
    selected: List[LineupSimResult]
    num_selected: int
    portfolio_win_rate: float
    portfolio_top1pct_rate: float
    portfolio_cash_rate: float
    portfolio_expected_roi: float
    portfolio_total_cost: float
    avg_pairwise_overlap: float
    unique_players_used: int
    max_player_exposure: float
    stack_exposure: Dict[str, float] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(lineup) for lineup in self.selected])


def select_portfolio(
    contest_result: ContestSimResult,
    num_lineups: int = 20,
    selection_metric: str = "top_1pct_rate",
    max_overlap: int = 5,
    max_batter_exposure: float = 0.40,
    max_pitcher_exposure: float = 0.60,
    min_batter_exposure: float = 0.0,
    min_pitcher_exposure: float = 0.0,
    pitcher_ids: Optional[Set[str]] = None,
    diversity_weight: float = 0.3,
    min_stack_exposure: float = 0.0,
    max_stack_exposure: float = 1.0,
    selection_leverage_weight: float = 0.0,
    selection_ownership_weight: float = 0.0,
    selection_duplication_weight: float = 0.0,
    selection_scenario_weight: float = 0.0,
    selection_marginal_value_weight: float = 0.0,
    stack_team_weights: Optional[Dict[str, float]] = None,
    stack_target_weight: float = 0.0,
    stack_team_min_exposures: Optional[Dict[str, float]] = None,
    stack_team_max_exposures: Optional[Dict[str, float]] = None,
    use_portfolio_optimizer: bool = True,
    portfolio_time_limit_seconds: int = 45,
    # Legacy support
    max_player_exposure: Optional[float] = None,
) -> PortfolioSelection:
    # Legacy: if caller passes old single value, use it for both
    if max_player_exposure is not None:
        max_batter_exposure = max_player_exposure
        max_pitcher_exposure = max_player_exposure
    pitcher_set: FrozenSet[str] = frozenset(pitcher_ids or set())

    candidates = contest_result.rank_by(selection_metric)
    if not candidates:
        raise ValueError("No candidates available for portfolio selection")

    if use_portfolio_optimizer and selection_marginal_value_weight <= 0:
        selected = _select_portfolio_ilp(
            candidates,
            num_lineups=num_lineups,
            selection_metric=selection_metric,
            max_overlap=max_overlap,
            max_batter_exposure=max_batter_exposure,
            max_pitcher_exposure=max_pitcher_exposure,
            min_batter_exposure=min_batter_exposure,
            min_pitcher_exposure=min_pitcher_exposure,
            pitcher_ids=pitcher_set,
            min_stack_exposure=min_stack_exposure,
            max_stack_exposure=max_stack_exposure,
            selection_leverage_weight=selection_leverage_weight,
            selection_ownership_weight=selection_ownership_weight,
            selection_duplication_weight=selection_duplication_weight,
            selection_scenario_weight=selection_scenario_weight,
            stack_team_weights=stack_team_weights,
            stack_target_weight=stack_target_weight,
            stack_team_min_exposures=stack_team_min_exposures,
            stack_team_max_exposures=stack_team_max_exposures,
            time_limit_seconds=portfolio_time_limit_seconds,
        )
        if selected:
            return _build_portfolio_selection(selected, contest_result.entry_fee)

    selected = _select_portfolio_greedy(
        candidates,
        num_lineups=num_lineups,
        selection_metric=selection_metric,
        max_overlap=max_overlap,
        max_batter_exposure=max_batter_exposure,
        max_pitcher_exposure=max_pitcher_exposure,
        min_batter_exposure=min_batter_exposure,
        min_pitcher_exposure=min_pitcher_exposure,
        pitcher_ids=pitcher_set,
        diversity_weight=diversity_weight,
        min_stack_exposure=min_stack_exposure,
        max_stack_exposure=max_stack_exposure,
        selection_leverage_weight=selection_leverage_weight,
        selection_ownership_weight=selection_ownership_weight,
        selection_duplication_weight=selection_duplication_weight,
        selection_scenario_weight=selection_scenario_weight,
        selection_marginal_value_weight=selection_marginal_value_weight,
        stack_team_weights=stack_team_weights,
        stack_target_weight=stack_target_weight,
        stack_team_min_exposures=stack_team_min_exposures,
        stack_team_max_exposures=stack_team_max_exposures,
    )
    return _build_portfolio_selection(selected, contest_result.entry_fee)


def _select_portfolio_greedy(
    candidates: List[LineupSimResult],
    num_lineups: int,
    selection_metric: str,
    max_overlap: int,
    max_batter_exposure: float,
    max_pitcher_exposure: float,
    min_batter_exposure: float,
    min_pitcher_exposure: float,
    pitcher_ids: FrozenSet[str],
    diversity_weight: float,
    min_stack_exposure: float,
    max_stack_exposure: float,
    selection_leverage_weight: float,
    selection_ownership_weight: float,
    selection_duplication_weight: float,
    selection_scenario_weight: float,
    selection_marginal_value_weight: float,
    stack_team_weights: Optional[Dict[str, float]],
    stack_target_weight: float,
    stack_team_min_exposures: Optional[Dict[str, float]],
    stack_team_max_exposures: Optional[Dict[str, float]],
) -> List[LineupSimResult]:
    metric_component = _score_component(
        candidates,
        lambda lineup: float(getattr(lineup, selection_metric, 0.0) or 0.0),
        higher_is_better=True,
    )
    leverage_component = _score_component(
        candidates,
        lambda lineup: float(lineup.leverage_score or 0.0),
        higher_is_better=True,
    )
    ownership_component = _score_component(
        candidates,
        lambda lineup: float(lineup.total_ownership or 0.0),
        higher_is_better=False,
    )
    duplication_component = _score_component(
        candidates,
        lambda lineup: max(
            float(getattr(lineup, "field_duplication_rate", 0.0) or 0.0),
            float(getattr(lineup, "duplication_score", 0.0) or 0.0),
        ),
        higher_is_better=False,
    )
    stack_component = _score_component(
        candidates,
        lambda lineup: _stack_plan_lineup_score(lineup, stack_team_weights),
        higher_is_better=True,
    )
    scenario_component = _score_component(
        candidates,
        lambda lineup: float(getattr(lineup, "ownership_scenario_score", 1.0) or 0.0),
        higher_is_better=True,
    )

    # Collect all stack teams and all player IDs that appear across candidates
    all_stack_teams: Set[str] = set()
    all_player_ids: Set[str] = set()
    for c in candidates:
        all_stack_teams.update(c.stack_teams)
        all_player_ids.update(c.player_ids)

    selected: List[LineupSimResult] = []
    player_counts: Counter[str] = Counter()
    stack_counts: Counter[str] = Counter()

    # Compute minimum counts needed
    min_player_counts: Dict[str, int] = {}
    if min_batter_exposure > 0 or min_pitcher_exposure > 0:
        for pid in all_player_ids:
            is_pitcher = pid in pitcher_ids
            min_pct = min_pitcher_exposure if is_pitcher else min_batter_exposure
            if min_pct > 0:
                min_player_counts[pid] = max(1, int(np.ceil(min_pct * num_lineups)))

    min_stack_count = max(1, int(np.ceil(min_stack_exposure * num_lineups))) if min_stack_exposure > 0 else 0

    # Two-pass greedy selection
    while len(selected) < num_lineups:
        remaining = num_lineups - len(selected)

        # Determine what still needs filling
        needed_players: Set[str] = set()
        for pid, min_ct in min_player_counts.items():
            if player_counts[pid] < min_ct and (min_ct - player_counts[pid]) >= remaining * 0.1:
                needed_players.add(pid)

        needed_stacks: Set[str] = set()
        if min_stack_count > 0:
            for team in all_stack_teams:
                if stack_counts[team] < min_stack_count:
                    needed_stacks.add(team)

        has_needs = bool(needed_players or needed_stacks)

        best_candidate = None
        best_score = -np.inf
        for candidate in candidates:
            if candidate in selected:
                continue
            if not _respects_overlap(candidate, selected, max_overlap):
                continue
            if not _respects_max_exposure(candidate, player_counts, num_lineups, max_batter_exposure, max_pitcher_exposure, pitcher_ids):
                continue
            if not _respects_max_stack_exposure(candidate, stack_counts, num_lineups, max_stack_exposure):
                continue
            if not _respects_team_stack_caps(candidate, stack_counts, num_lineups, stack_team_max_exposures):
                continue

            metric_value = metric_component.get(candidate.lineup_id, 0.0)
            overlap_penalty = _average_overlap(candidate, selected) / 9 if selected else 0.0
            combined = metric_value
            combined += selection_leverage_weight * leverage_component.get(candidate.lineup_id, 0.0)
            combined += selection_ownership_weight * ownership_component.get(candidate.lineup_id, 0.0)
            combined += selection_duplication_weight * duplication_component.get(candidate.lineup_id, 0.0)
            combined += stack_target_weight * stack_component.get(candidate.lineup_id, 0.0)
            combined += selection_scenario_weight * scenario_component.get(candidate.lineup_id, 0.0)
            combined += selection_marginal_value_weight * _marginal_value_score(
                candidate,
                selected,
                player_counts,
                stack_counts,
                num_lineups,
                stack_team_weights,
            )
            combined *= (1 - diversity_weight * overlap_penalty)

            # Bonus for satisfying minimums that are still needed
            if has_needs:
                need_bonus = 0.0
                cand_players = set(candidate.player_ids)
                cand_stacks = set(candidate.stack_teams)
                player_hits = len(cand_players & needed_players)
                stack_hits = len(cand_stacks & needed_stacks)
                need_bonus = (player_hits + stack_hits) * 0.5
                combined += need_bonus

            if combined > best_score:
                best_score = combined
                best_candidate = candidate

        if best_candidate is None:
            break
        selected.append(best_candidate)
        for pid in best_candidate.player_ids:
            player_counts[pid] += 1
        for team in best_candidate.stack_teams:
            stack_counts[team] += 1

    return selected


def _select_portfolio_ilp(
    candidates: List[LineupSimResult],
    num_lineups: int,
    selection_metric: str,
    max_overlap: int,
    max_batter_exposure: float,
    max_pitcher_exposure: float,
    min_batter_exposure: float,
    min_pitcher_exposure: float,
    pitcher_ids: FrozenSet[str],
    min_stack_exposure: float,
    max_stack_exposure: float,
    selection_leverage_weight: float,
    selection_ownership_weight: float,
    selection_duplication_weight: float,
    selection_scenario_weight: float,
    stack_team_weights: Optional[Dict[str, float]],
    stack_target_weight: float,
    stack_team_min_exposures: Optional[Dict[str, float]],
    stack_team_max_exposures: Optional[Dict[str, float]],
    time_limit_seconds: int,
) -> Optional[List[LineupSimResult]]:
    target = min(max(0, int(num_lineups)), len(candidates))
    if target <= 0:
        return []

    objective_scores = _portfolio_objective_scores(
        candidates,
        selection_metric,
        selection_leverage_weight,
        selection_ownership_weight,
        selection_duplication_weight,
        selection_scenario_weight,
        stack_team_weights,
        stack_target_weight,
    )

    prob = LpProblem("dfs_portfolio_selection", LpMaximize)
    x = {
        idx: LpVariable(f"lineup_{idx}", lowBound=0, upBound=1, cat=LpBinary)
        for idx in range(len(candidates))
    }

    prob += lpSum(objective_scores[idx] * x[idx] for idx in x)
    prob += lpSum(x.values()) == target

    player_to_lineups: Dict[str, List[int]] = {}
    stack_to_lineups: Dict[str, List[int]] = {}
    lineup_player_sets: List[Set[str]] = []

    for idx, lineup in enumerate(candidates):
        player_set = {str(pid) for pid in lineup.player_ids}
        lineup_player_sets.append(player_set)
        for pid in player_set:
            player_to_lineups.setdefault(pid, []).append(idx)
        for team in lineup.stack_teams:
            stack_to_lineups.setdefault(str(team), []).append(idx)

    _add_player_exposure_constraints(
        prob,
        x,
        player_to_lineups,
        target,
        pitcher_ids,
        max_batter_exposure,
        max_pitcher_exposure,
        min_batter_exposure,
        min_pitcher_exposure,
    )
    _add_stack_exposure_constraints(
        prob,
        x,
        stack_to_lineups,
        target,
        min_stack_exposure,
        max_stack_exposure,
        stack_team_min_exposures,
        stack_team_max_exposures,
    )
    _add_overlap_constraints(prob, x, lineup_player_sets, max_overlap)

    try:
        status = prob.solve(
            PULP_CBC_CMD(msg=False, timeLimit=max(5, int(time_limit_seconds)))
        )
    except Exception as exc:  # pragma: no cover - solver availability fallback
        warnings.warn(f"Portfolio optimizer failed; falling back to greedy selection: {exc}", stacklevel=2)
        return None

    if status != LpStatusOptimal:
        warnings.warn(
            "Portfolio optimizer could not satisfy the full constraint set; falling back to greedy selection.",
            stacklevel=2,
        )
        return None

    selected_indices = [idx for idx, var in x.items() if (var.varValue or 0) > 0.5]
    if len(selected_indices) != target:
        return None
    selected_indices.sort(key=lambda idx: objective_scores[idx], reverse=True)
    return [candidates[idx] for idx in selected_indices]


def _portfolio_objective_scores(
    candidates: List[LineupSimResult],
    selection_metric: str,
    selection_leverage_weight: float,
    selection_ownership_weight: float,
    selection_duplication_weight: float,
    selection_scenario_weight: float,
    stack_team_weights: Optional[Dict[str, float]] = None,
    stack_target_weight: float = 0.0,
) -> Dict[int, float]:
    metric_component = _score_component(
        candidates,
        lambda lineup: float(getattr(lineup, selection_metric, 0.0) or 0.0),
        higher_is_better=True,
    )
    leverage_component = _score_component(
        candidates,
        lambda lineup: float(lineup.leverage_score or 0.0),
        higher_is_better=True,
    )
    ownership_component = _score_component(
        candidates,
        lambda lineup: float(lineup.total_ownership or 0.0),
        higher_is_better=False,
    )
    duplication_component = _score_component(
        candidates,
        lambda lineup: max(
            float(getattr(lineup, "field_duplication_rate", 0.0) or 0.0),
            float(getattr(lineup, "duplication_score", 0.0) or 0.0),
        ),
        higher_is_better=False,
    )
    stack_component = _score_component(
        candidates,
        lambda lineup: _stack_plan_lineup_score(lineup, stack_team_weights),
        higher_is_better=True,
    )
    scenario_component = _score_component(
        candidates,
        lambda lineup: float(getattr(lineup, "ownership_scenario_score", 1.0) or 0.0),
        higher_is_better=True,
    )
    return {
        idx: (
            metric_component.get(lineup.lineup_id, 0.0)
            + selection_leverage_weight * leverage_component.get(lineup.lineup_id, 0.0)
            + selection_ownership_weight * ownership_component.get(lineup.lineup_id, 0.0)
            + selection_duplication_weight * duplication_component.get(lineup.lineup_id, 0.0)
            + selection_scenario_weight * scenario_component.get(lineup.lineup_id, 0.0)
            + stack_target_weight * stack_component.get(lineup.lineup_id, 0.0)
        )
        for idx, lineup in enumerate(candidates)
    }


def _stack_plan_lineup_score(
    lineup: LineupSimResult,
    stack_team_weights: Optional[Dict[str, float]],
) -> float:
    if not stack_team_weights:
        return 0.0
    return float(
        sum(float(stack_team_weights.get(str(team), 0.0) or 0.0) for team in lineup.stack_teams)
    )


def _marginal_value_score(
    lineup: LineupSimResult,
    selected: List[LineupSimResult],
    player_counts: Counter[str],
    stack_counts: Counter[str],
    num_lineups: int,
    stack_team_weights: Optional[Dict[str, float]],
) -> float:
    if not selected:
        return float(getattr(lineup, "ownership_scenario_score", 1.0) or 0.0)
    player_set = {str(pid) for pid in lineup.player_ids}
    unseen_players = sum(1 for pid in player_set if player_counts[pid] == 0)
    new_player_share = unseen_players / max(1, len(player_set))
    overlap_score = 1.0 - min(1.0, _average_overlap(lineup, selected) / 9.0)
    new_stack_bonus = 0.0
    if lineup.stack_teams:
        stack_values = []
        for team in lineup.stack_teams:
            team_key = str(team)
            underused = 1.0 - min(1.0, stack_counts[team_key] / max(1, num_lineups))
            plan_weight = 0.5 + 0.5 * float((stack_team_weights or {}).get(team_key, 0.0) or 0.0)
            stack_values.append(underused * plan_weight)
        new_stack_bonus = float(np.mean(stack_values)) if stack_values else 0.0
    uniqueness = 1.0 - min(
        1.0,
        max(
            float(getattr(lineup, "duplication_score", 0.0) or 0.0),
            float(getattr(lineup, "worst_case_duplication_score", 0.0) or 0.0),
        ),
    )
    scenario = float(getattr(lineup, "ownership_scenario_score", 1.0) or 0.0)
    return float(
        np.clip(
            0.30 * new_player_share
            + 0.25 * overlap_score
            + 0.20 * new_stack_bonus
            + 0.15 * uniqueness
            + 0.10 * scenario,
            0.0,
            1.0,
        )
    )


def _add_player_exposure_constraints(
    prob: LpProblem,
    x: Dict[int, LpVariable],
    player_to_lineups: Dict[str, List[int]],
    target: int,
    pitcher_ids: FrozenSet[str],
    max_batter_exposure: float,
    max_pitcher_exposure: float,
    min_batter_exposure: float,
    min_pitcher_exposure: float,
) -> None:
    min_requirements: Dict[str, int] = {}
    total_min_slots = 0
    for pid, lineup_indices in player_to_lineups.items():
        is_pitcher = pid in pitcher_ids
        max_pct = max_pitcher_exposure if is_pitcher else max_batter_exposure
        if max_pct < 1.0:
            limit = max(1, int(np.floor(max_pct * target)))
            prob += lpSum(x[idx] for idx in lineup_indices) <= limit

        min_pct = min_pitcher_exposure if is_pitcher else min_batter_exposure
        if min_pct > 0:
            min_count = max(1, int(np.ceil(min_pct * target)))
            if len(lineup_indices) >= min_count:
                min_requirements[pid] = min_count
                total_min_slots += min_count

    if total_min_slots > target * 9:
        warnings.warn(
            "Skipping player exposure floors because they exceed available roster slots.",
            stacklevel=2,
        )
        return
    for pid, min_count in min_requirements.items():
        prob += lpSum(x[idx] for idx in player_to_lineups[pid]) >= min_count


def _add_stack_exposure_constraints(
    prob: LpProblem,
    x: Dict[int, LpVariable],
    stack_to_lineups: Dict[str, List[int]],
    target: int,
    min_stack_exposure: float,
    max_stack_exposure: float,
    stack_team_min_exposures: Optional[Dict[str, float]] = None,
    stack_team_max_exposures: Optional[Dict[str, float]] = None,
) -> None:
    min_requirements: Dict[str, int] = {}
    total_min_stacks = 0
    stack_team_min_exposures = stack_team_min_exposures or {}
    stack_team_max_exposures = stack_team_max_exposures or {}
    for team, lineup_indices in stack_to_lineups.items():
        team_max = float(stack_team_max_exposures.get(str(team), max_stack_exposure))
        if team_max < 1.0:
            limit = max(1, int(np.floor(team_max * target)))
            prob += lpSum(x[idx] for idx in lineup_indices) <= limit
        team_min = float(stack_team_min_exposures.get(str(team), min_stack_exposure))
        if team_min > 0:
            min_count = max(1, int(np.ceil(team_min * target)))
            if len(lineup_indices) >= min_count:
                min_requirements[team] = min_count
                total_min_stacks += min_count

    # Most MLB lineups have one or two 3+ hitter stacks. Avoid forcing an
    # impossible "every team gets a floor" slate when the UI floor is broad.
    if total_min_stacks > target * 2:
        warnings.warn(
            "Skipping stack exposure floors because they exceed practical portfolio capacity.",
            stacklevel=2,
        )
        return
    for team, min_count in min_requirements.items():
        prob += lpSum(x[idx] for idx in stack_to_lineups[team]) >= min_count


def _add_overlap_constraints(
    prob: LpProblem,
    x: Dict[int, LpVariable],
    lineup_player_sets: List[Set[str]],
    max_overlap: int,
) -> None:
    if max_overlap >= 9:
        return
    for i, lineup_a in enumerate(lineup_player_sets):
        for j in range(i + 1, len(lineup_player_sets)):
            if len(lineup_a & lineup_player_sets[j]) > max_overlap:
                prob += x[i] + x[j] <= 1


def _build_portfolio_selection(
    selected: List[LineupSimResult],
    entry_fee: float,
) -> PortfolioSelection:
    total_entries = max(1, len(selected))
    player_counts: Counter[str] = Counter()
    stack_counts: Counter[str] = Counter()
    for lineup in selected:
        for pid in lineup.player_ids:
            player_counts[pid] += 1
        for team in lineup.stack_teams:
            stack_counts[team] += 1

    portfolio = _portfolio_metrics(selected, entry_fee)

    unique_players = len(player_counts)
    max_exposure_val = 0.0
    if player_counts:
        max_exposure_val = max(count / total_entries for count in player_counts.values())

    stack_exposure_pcts: Dict[str, float] = {}
    for team, count in stack_counts.items():
        stack_exposure_pcts[team] = count / total_entries

    return PortfolioSelection(
        selected=selected,
        num_selected=len(selected),
        portfolio_win_rate=portfolio["win_rate"],
        portfolio_top1pct_rate=portfolio["top1"],
        portfolio_cash_rate=portfolio["cash"],
        portfolio_expected_roi=portfolio["roi"],
        portfolio_total_cost=entry_fee * len(selected),
        avg_pairwise_overlap=_average_pairwise_overlap(selected),
        unique_players_used=unique_players,
        max_player_exposure=max_exposure_val,
        stack_exposure=stack_exposure_pcts,
    )


def _respects_overlap(
    candidate: LineupSimResult,
    selected: List[LineupSimResult],
    max_overlap: int,
) -> bool:
    if not selected:
        return True
    cand_set = set(candidate.player_ids)
    for lineup in selected:
        if len(cand_set & set(lineup.player_ids)) > max_overlap:
            return False
    return True


def _score_component(
    candidates: List[LineupSimResult],
    value_fn,
    higher_is_better: bool,
) -> Dict[int, float]:
    values = np.array([value_fn(candidate) for candidate in candidates], dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    if len(values) == 0:
        return {}
    if np.allclose(values, values[0]):
        return {candidate.lineup_id: 0.5 for candidate in candidates}
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    scores = ranks / max(1, len(values) - 1)
    if not higher_is_better:
        scores = 1.0 - scores
    return {
        candidate.lineup_id: float(score)
        for candidate, score in zip(candidates, scores)
    }


def _respects_max_exposure(
    candidate: LineupSimResult,
    counts: Counter,
    num_lineups: int,
    max_batter_exposure: float,
    max_pitcher_exposure: float,
    pitcher_ids: FrozenSet[str],
) -> bool:
    for pid in candidate.player_ids:
        is_pitcher = pid in pitcher_ids
        limit_pct = max_pitcher_exposure if is_pitcher else max_batter_exposure
        if limit_pct >= 1.0:
            continue
        limit = max(1, int(np.floor(limit_pct * num_lineups)))
        if counts[pid] + 1 > limit:
            return False
    return True


def _respects_max_stack_exposure(
    candidate: LineupSimResult,
    stack_counts: Counter,
    num_lineups: int,
    max_stack_exposure: float,
) -> bool:
    if max_stack_exposure >= 1.0:
        return True
    limit = max(1, int(np.floor(max_stack_exposure * num_lineups)))
    for team in candidate.stack_teams:
        if stack_counts[team] + 1 > limit:
            return False
    return True


def _respects_team_stack_caps(
    candidate: LineupSimResult,
    stack_counts: Counter,
    num_lineups: int,
    stack_team_max_exposures: Optional[Dict[str, float]],
) -> bool:
    if not stack_team_max_exposures:
        return True
    for team in candidate.stack_teams:
        max_pct = float(stack_team_max_exposures.get(str(team), 1.0))
        if max_pct >= 1.0:
            continue
        limit = max(1, int(np.floor(max_pct * num_lineups)))
        if stack_counts[team] + 1 > limit:
            return False
    return True


def _average_overlap(candidate: LineupSimResult, selected: List[LineupSimResult]) -> float:
    if not selected:
        return 0.0
    cand_set = set(candidate.player_ids)
    overlaps = [len(cand_set & set(lineup.player_ids)) for lineup in selected]
    return float(np.mean(overlaps)) if overlaps else 0.0


def _portfolio_metrics(selected: List[LineupSimResult], entry_fee: float) -> Dict[str, float]:
    if not selected:
        return {"win_rate": 0.0, "top1": 0.0, "cash": 0.0, "roi": 0.0}
    win_rates = np.array([lineup.win_rate for lineup in selected])
    top1 = np.array([lineup.top_1pct_rate for lineup in selected])
    cash_rates = np.array([lineup.cash_rate for lineup in selected])
    roi_values = np.array([lineup.expected_roi for lineup in selected])

    portfolio_win = 1 - np.prod(1 - win_rates)
    portfolio_top1 = 1 - np.prod(1 - top1)
    portfolio_cash = 1 - np.prod(1 - cash_rates)
    portfolio_roi = float(roi_values.sum())
    return {"win_rate": float(portfolio_win), "top1": float(portfolio_top1), "cash": float(portfolio_cash), "roi": portfolio_roi}


def _average_pairwise_overlap(selected: List[LineupSimResult]) -> float:
    if len(selected) < 2:
        return 0.0
    overlaps = []
    for lineup_a, lineup_b in itertools.combinations(selected, 2):
        overlaps.append(len(set(lineup_a.player_ids) & set(lineup_b.player_ids)))
    return float(np.mean(overlaps)) if overlaps else 0.0
