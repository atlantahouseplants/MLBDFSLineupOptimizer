from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set

import numpy as np
import pandas as pd

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

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(lineup) for lineup in self.selected])


def select_portfolio(
    contest_result: ContestSimResult,
    num_lineups: int = 20,
    selection_metric: str = "top_1pct_rate",
    max_overlap: int = 5,
    max_batter_exposure: float = 0.40,
    max_pitcher_exposure: float = 0.60,
    pitcher_ids: Optional[Set[str]] = None,
    diversity_weight: float = 0.3,
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

    selected: List[LineupSimResult] = []
    player_counts: Counter[str] = Counter()

    while len(selected) < num_lineups:
        best_candidate = None
        best_score = -np.inf
        for candidate in candidates:
            if candidate in selected:
                continue
            if not _respects_overlap(candidate, selected, max_overlap):
                continue
            if not _respects_exposure(candidate, player_counts, num_lineups, max_batter_exposure, max_pitcher_exposure, pitcher_set):
                continue
            metric_value = getattr(candidate, selection_metric, 0.0)
            overlap_penalty = _average_overlap(candidate, selected) / 9 if selected else 0.0
            combined = metric_value * (1 - diversity_weight * overlap_penalty)
            if combined > best_score:
                best_score = combined
                best_candidate = candidate
        if best_candidate is None:
            break
        selected.append(best_candidate)
        for pid in best_candidate.player_ids:
            player_counts[pid] += 1

    exposures = player_counts.copy()
    total_entries = max(1, len(selected))
    portfolio = _portfolio_metrics(selected, contest_result.entry_fee)

    unique_players = len(player_counts)
    max_exposure_val = 0.0
    if player_counts:
        max_exposure_val = max(count / total_entries for count in player_counts.values())

    return PortfolioSelection(
        selected=selected,
        num_selected=len(selected),
        portfolio_win_rate=portfolio["win_rate"],
        portfolio_top1pct_rate=portfolio["top1"],
        portfolio_cash_rate=portfolio["cash"],
        portfolio_expected_roi=portfolio["roi"],
        portfolio_total_cost=contest_result.entry_fee * len(selected),
        avg_pairwise_overlap=_average_pairwise_overlap(selected),
        unique_players_used=unique_players,
        max_player_exposure=max_exposure_val,
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


def _respects_exposure(
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
