from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from slate_optimizer.optimizer.solver import LineupResult

from .field_simulator import SimulatedField
from .slate_simulator import SlateSimulation

__all__ = [
    "LineupSimResult",
    "ContestSimResult",
    "simulate_contest",
]

DEFAULT_GPP_PAYOUTS = [
    (99.9, 100.0, 100.0),
    (99.5, 99.9, 20.0),
    (99.0, 99.5, 10.0),
    (95.0, 99.0, 3.0),
    (80.0, 95.0, 1.5),
    (0.0, 80.0, 0.0),
]


@dataclass
class LineupSimResult:
    lineup_id: int
    player_ids: List[str]
    mean_score: float
    median_score: float
    std_score: float
    p10_score: float
    p25_score: float
    p75_score: float
    p90_score: float
    p99_score: float
    max_score: float
    win_rate: float
    top_1pct_rate: float
    top_10pct_rate: float
    cash_rate: float
    expected_roi: float
    total_ownership: float
    leverage_score: float
    field_duplication_rate: float


@dataclass
class ContestSimResult:
    lineup_results: List[LineupSimResult]
    num_simulations: int
    num_field_lineups: int
    num_candidates: int
    entry_fee: float

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(result) for result in self.lineup_results])

    def rank_by(self, metric: str = "top_1pct_rate") -> List[LineupSimResult]:
        return sorted(
            self.lineup_results,
            key=lambda res: getattr(res, metric, 0.0),
            reverse=True,
        )


def simulate_contest(
    candidates: Sequence[LineupResult],
    slate_sim: SlateSimulation,
    field_sim: SimulatedField,
    entry_fee: float = 20.0,
    payout_structure: Optional[Dict[tuple, float]] = None,
) -> ContestSimResult:
    if not candidates:
        raise ValueError("No candidate lineups provided for simulation")

    payout_structure = _prepare_payout_structure(payout_structure)
    player_index = slate_sim.player_id_to_index

    candidate_indices = _lineup_indices(candidates, player_index)
    candidate_scores = slate_sim.scores[:, candidate_indices]
    candidate_scores = candidate_scores.sum(axis=2)  # (N, C)

    # Compute field scores in chunks to avoid a huge intermediate array
    # (N × M × 9 at full expansion would exceed memory on cloud hosts)
    num_sims = slate_sim.num_simulations
    num_field = field_sim.num_lineups
    field_scores = np.empty((num_sims, num_field), dtype=np.float64)
    _CHUNK = 100
    for start in range(0, num_field, _CHUNK):
        end = min(start + _CHUNK, num_field)
        field_scores[:, start:end] = slate_sim.scores[:, field_sim.lineups[start:end]].sum(axis=2)

    all_scores = np.concatenate([field_scores, candidate_scores], axis=1)
    ranks = _rank_scores(all_scores)
    percentiles = (ranks + 1) / all_scores.shape[1] * 100.0

    field_sets = _field_lineup_sets(field_sim)

    lineup_results: List[LineupSimResult] = []
    for lineup_id, (candidate, lineup_idx) in enumerate(zip(candidates, candidate_indices)):
        lineup_scores = candidate_scores[:, lineup_id]
        lineup_percentiles = percentiles[:, field_scores.shape[1] + lineup_id]
        metrics = _aggregate_metrics(
            lineup_scores,
            lineup_percentiles,
            payout_structure,
            entry_fee,
        )
        lineup_df = candidate.dataframe
        player_ids = lineup_df["fd_player_id"].astype(str).tolist()
        if "proj_fd_ownership" in lineup_df.columns:
            total_ownership = float(pd.to_numeric(lineup_df["proj_fd_ownership"], errors="coerce").fillna(0.0).sum())
        else:
            total_ownership = 0.0
        if "player_leverage_score" in lineup_df.columns:
            leverage_score = float(pd.to_numeric(lineup_df["player_leverage_score"], errors="coerce").fillna(0.0).sum())
        else:
            leverage_score = 0.0
        dup_rate = _duplication_rate(player_ids, field_sets)
        lineup_results.append(
            LineupSimResult(
                lineup_id=lineup_id,
                player_ids=player_ids,
                mean_score=metrics["mean"],
                median_score=metrics["median"],
                std_score=metrics["std"],
                p10_score=metrics["p10"],
                p25_score=metrics["p25"],
                p75_score=metrics["p75"],
                p90_score=metrics["p90"],
                p99_score=metrics["p99"],
                max_score=metrics["max"],
                win_rate=metrics["win_rate"],
                top_1pct_rate=metrics["top_1pct"],
                top_10pct_rate=metrics["top_10pct"],
                cash_rate=metrics["cash_rate"],
                expected_roi=metrics["expected_roi"],
                total_ownership=total_ownership,
                leverage_score=leverage_score,
                field_duplication_rate=dup_rate,
            )
        )

    return ContestSimResult(
        lineup_results=lineup_results,
        num_simulations=slate_sim.num_simulations,
        num_field_lineups=field_sim.num_lineups,
        num_candidates=len(candidates),
        entry_fee=entry_fee,
    )


def _lineup_indices(
    candidates: Sequence[LineupResult],
    player_index: Dict[str, int],
) -> np.ndarray:
    indices = []
    for candidate in candidates:
        ids = candidate.dataframe["fd_player_id"].astype(str).tolist()
        try:
            indices.append([player_index[pid] for pid in ids])
        except KeyError as exc:
            raise KeyError(f"Lineup references unknown player {exc}") from exc
    return np.array(indices, dtype=int)


def _rank_scores(all_scores: np.ndarray) -> np.ndarray:
    order = np.argsort(all_scores, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    ranks[rows, order] = np.arange(order.shape[1])
    return ranks


def _aggregate_metrics(
    lineup_scores: np.ndarray,
    percentiles: np.ndarray,
    payout_structure: List[tuple[float, float, float]],
    entry_fee: float,
) -> Dict[str, float]:
    mean = float(lineup_scores.mean())
    median = float(np.median(lineup_scores))
    std = float(lineup_scores.std())
    p10 = float(np.percentile(lineup_scores, 10))
    p25 = float(np.percentile(lineup_scores, 25))
    p75 = float(np.percentile(lineup_scores, 75))
    p90 = float(np.percentile(lineup_scores, 90))
    p99 = float(np.percentile(lineup_scores, 99))
    max_score = float(lineup_scores.max())

    win_rate = float(np.mean(percentiles >= 100.0 - 1e-9))
    top_1pct = float(np.mean(percentiles >= 99.0))
    top_10pct = float(np.mean(percentiles >= 90.0))
    cash_rate = float(np.mean(percentiles >= 80.0))

    multipliers = _payout_multipliers(percentiles, payout_structure)
    roi = (multipliers - 1.0)
    expected_roi = float(roi.mean())

    return {
        "mean": mean,
        "median": median,
        "std": std,
        "p10": p10,
        "p25": p25,
        "p75": p75,
        "p90": p90,
        "p99": p99,
        "max": max_score,
        "win_rate": win_rate,
        "top_1pct": top_1pct,
        "top_10pct": top_10pct,
        "cash_rate": cash_rate,
        "expected_roi": expected_roi,
    }


def _payout_multipliers(
    percentiles: np.ndarray,
    payout_structure: List[tuple[float, float, float]],
) -> np.ndarray:
    multipliers = np.zeros_like(percentiles, dtype=float)
    for lower, upper, value in payout_structure:
        mask = (percentiles >= lower) & (percentiles < upper)
        multipliers[mask] = value
    # Handle exact 100th percentile
    multipliers[percentiles >= 100.0] = payout_structure[0][2]
    return multipliers


def _prepare_payout_structure(payout_structure: Optional[Dict[tuple, float]]):
    if payout_structure:
        bands = []
        for (lower, upper), value in payout_structure.items():
            bands.append((float(lower), float(upper), float(value)))
        bands.sort(key=lambda x: x[0], reverse=True)
        return bands
    return DEFAULT_GPP_PAYOUTS


def _field_lineup_sets(field_sim: SimulatedField) -> List[set[str]]:
    player_ids = np.array(field_sim.player_ids, dtype=object)
    return [set(player_ids[lineup]) for lineup in field_sim.lineups]


def _duplication_rate(player_ids: List[str], field_sets: List[set[str]]) -> float:
    lineup_set = set(player_ids)
    dup_count = 0
    for opponent in field_sets:
        if len(lineup_set & opponent) >= 6:
            dup_count += 1
    if not field_sets:
        return 0.0
    return dup_count / len(field_sets)
