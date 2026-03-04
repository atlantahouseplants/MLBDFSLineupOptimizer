from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "CorrelationConfig",
    "CorrelationModel",
    "build_correlation_matrix",
]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass
class CorrelationConfig:
    teammate_base: float = 0.25
    teammate_adjacent_bonus: float = 0.10
    same_game_opponent: float = 0.10
    pitcher_own_hitters: float = 0.05
    pitcher_vs_opposing: float = -0.15
    cross_game: float = 0.0
    copula_nu: int = 5
    stack_priority_bonus: float = 0.05
    vegas_scaling_denominator: float = 8.5
    vegas_scale_cap: float = 1.3


@dataclass
class CorrelationModel:
    player_ids: List[str]
    matrix: np.ndarray
    nu: int
    cholesky: np.ndarray

    def validate(self) -> bool:
        if self.matrix.shape[0] != self.matrix.shape[1]:
            return False
        if self.matrix.shape[0] != len(self.player_ids):
            return False
        symmetric = np.allclose(self.matrix, self.matrix.T, atol=1e-8)
        positive = np.all(np.linalg.eigvalsh(self.matrix) > -1e-8)
        return symmetric and positive


def build_correlation_matrix(
    optimizer_df: pd.DataFrame,
    config: Optional[CorrelationConfig] = None,
) -> CorrelationModel:
    config = config or CorrelationConfig()
    df = optimizer_df.copy()
    df["player_type"] = df.get("player_type", "").astype(str).str.lower()
    df["stack_key"] = df.get("stack_key", "").astype(str)
    df["game_key"] = df.get("game_key", "").astype(str)
    df["team_code"] = df.get("team_code", "").astype(str)
    df["opponent_code"] = df.get("opponent_code", "").astype(str)
    df["stack_priority"] = df.get("stack_priority", "mid").astype(str).str.lower()
    df["vegas_game_total"] = pd.to_numeric(df.get("vegas_game_total"), errors="coerce")
    df["batting_order_position"] = pd.to_numeric(
        df.get("batting_order_position"), errors="coerce"
    )

    player_ids = df["fd_player_id"].astype(str).tolist()
    num_players = len(player_ids)
    matrix = np.eye(num_players, dtype=np.float64)

    for i in range(num_players):
        row_i = df.iloc[i]
        for j in range(i + 1, num_players):
            row_j = df.iloc[j]
            corr = _pair_correlation(row_i, row_j, config)
            matrix[i, j] = matrix[j, i] = corr

    matrix = _ensure_positive_semidefinite(matrix)
    cholesky = _compute_cholesky(matrix)
    return CorrelationModel(
        player_ids=player_ids,
        matrix=matrix,
        nu=max(config.copula_nu, 2),
        cholesky=cholesky,
    )


def _pair_correlation(row_i: pd.Series, row_j: pd.Series, config: CorrelationConfig) -> float:
    same_game = bool(row_i.get("game_key") and row_i.get("game_key") == row_j.get("game_key"))
    same_stack = bool(row_i.get("stack_key") and row_i.get("stack_key") == row_j.get("stack_key"))
    type_i = row_i.get("player_type")
    type_j = row_j.get("player_type")

    if same_stack and type_i == type_j == "batter":
        corr = config.teammate_base
        if _are_adjacent(row_i.get("batting_order_position"), row_j.get("batting_order_position")):
            corr += config.teammate_adjacent_bonus
        if "high" in {row_i.get("stack_priority"), row_j.get("stack_priority")}:
            corr += config.stack_priority_bonus
        vegas_total = _pair_vegas_total(row_i, row_j)
        if vegas_total and vegas_total > 0:
            scale = _clamp(
                vegas_total / config.vegas_scaling_denominator,
                1.0,
                config.vegas_scale_cap,
            )
            corr *= scale
        return float(_clamp(corr, -0.95, 0.95))

    if same_game and type_i == type_j == "batter" and not same_stack:
        return float(_clamp(config.same_game_opponent, -0.95, 0.95))

    if type_i == "pitcher" and type_j == "batter":
        return _pitcher_batter_corr(row_i, row_j, config)
    if type_j == "pitcher" and type_i == "batter":
        return _pitcher_batter_corr(row_j, row_i, config)

    if not same_game:
        return float(config.cross_game)

    return float(config.cross_game)


def _pitcher_batter_corr(pitcher: pd.Series, hitter: pd.Series, config: CorrelationConfig) -> float:
    team_code = pitcher.get("team_code")
    if team_code and team_code == hitter.get("team_code"):
        return float(_clamp(config.pitcher_own_hitters, -0.95, 0.95))
    if team_code and team_code == hitter.get("opponent_code"):
        return float(_clamp(config.pitcher_vs_opposing, -0.95, 0.0))
    return float(config.cross_game)


def _are_adjacent(order_i, order_j) -> bool:
    if pd.isna(order_i) or pd.isna(order_j):
        return False
    return abs(float(order_i) - float(order_j)) <= 1.0


def _pair_vegas_total(row_i: pd.Series, row_j: pd.Series) -> Optional[float]:
    vals = []
    for row in (row_i, row_j):
        value = row.get("vegas_game_total")
        if pd.notna(value):
            vals.append(float(value))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _ensure_positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    eigvals = np.linalg.eigvalsh(matrix)
    if np.all(eigvals >= 0):
        return matrix
    return _nearest_positive_semidefinite(matrix)


def _nearest_positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    sym = (matrix + matrix.T) / 2
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, 1e-6, None)
    reconstructed = eigvecs @ np.diag(eigvals) @ eigvecs.T
    reconstructed = (reconstructed + reconstructed.T) / 2
    np.fill_diagonal(reconstructed, 1.0)
    return reconstructed


def _compute_cholesky(matrix: np.ndarray) -> np.ndarray:
    for jitter_exp in range(6):
        try:
            return np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            jitter = 10 ** (-6 + jitter_exp)
            matrix = matrix + np.eye(matrix.shape[0]) * jitter
    raise
