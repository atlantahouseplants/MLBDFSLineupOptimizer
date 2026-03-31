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
    """Build correlation matrix using vectorized operations."""
    config = config or CorrelationConfig()
    df = optimizer_df.copy()
    df["player_type"] = df.get("player_type", "").astype(str).str.lower()
    df["stack_key"] = df.get("stack_key", "").astype(str)
    df["game_key"] = df.get("game_key", "").astype(str)
    df["team_code"] = df.get("team_code", "").astype(str)
    df["opponent_code"] = df.get("opponent_code", "").astype(str)
    df["stack_priority"] = df.get("stack_priority", "mid").astype(str).str.lower()
    if "vegas_game_total" in df.columns:
        df["vegas_game_total"] = pd.to_numeric(df["vegas_game_total"], errors="coerce").fillna(0.0)
    else:
        df["vegas_game_total"] = 0.0
    if "batting_order_position" in df.columns:
        df["batting_order_position"] = pd.to_numeric(df["batting_order_position"], errors="coerce").fillna(0.0)
    else:
        df["batting_order_position"] = 0.0

    player_ids = df["fd_player_id"].astype(str).tolist()
    n = len(player_ids)
    matrix = np.eye(n, dtype=np.float64)

    # Extract arrays for vectorized comparison (use object dtype to avoid
    # pandas StringArray broadcasting issues on newer pandas versions)
    player_type = df["player_type"].to_numpy(dtype=object)
    stack_key = df["stack_key"].to_numpy(dtype=object)
    game_key = df["game_key"].to_numpy(dtype=object)
    team_code = df["team_code"].to_numpy(dtype=object)
    opponent_code = df["opponent_code"].to_numpy(dtype=object)
    stack_priority = df["stack_priority"].to_numpy(dtype=object)
    vegas_total = df["vegas_game_total"].values
    batting_order = df["batting_order_position"].to_numpy(dtype=np.float64, na_value=0.0)
    is_batter = player_type == "batter"
    is_pitcher = player_type == "pitcher"

    # Build boolean matrices (n x n) using broadcasting
    # Same stack: both have non-empty matching stack_key
    same_stack = (stack_key[:, None] == stack_key[None, :]) & (stack_key[:, None] != "")
    # Same game: both have non-empty matching game_key
    same_game = (game_key[:, None] == game_key[None, :]) & (game_key[:, None] != "")
    # Both batters
    both_batters = is_batter[:, None] & is_batter[None, :]
    # Pitcher-batter pairs
    pitcher_batter = is_pitcher[:, None] & is_batter[None, :]
    batter_pitcher = is_batter[:, None] & is_pitcher[None, :]
    # Pitcher on same team as batter
    pitcher_same_team = (team_code[:, None] == team_code[None, :])
    # Pitcher vs opposing hitters
    pitcher_vs_opp = (team_code[:, None] == opponent_code[None, :])

    # 1. Teammate batters (same stack, both batters)
    teammate_mask = same_stack & both_batters
    matrix[teammate_mask] = config.teammate_base

    # Adjacent batting order bonus
    order_diff = np.abs(batting_order[:, None] - batting_order[None, :])
    adjacent = (order_diff <= 1.0) & (batting_order[:, None] > 0) & (batting_order[None, :] > 0)
    adjacent_teammate = teammate_mask & adjacent
    matrix[adjacent_teammate] += config.teammate_adjacent_bonus

    # Stack priority bonus (either player is "high")
    is_high = stack_priority == "high"
    either_high = is_high[:, None] | is_high[None, :]
    high_teammate = teammate_mask & either_high
    matrix[high_teammate] += config.stack_priority_bonus

    # Vegas scaling for teammates
    avg_vegas = (vegas_total[:, None] + vegas_total[None, :]) / 2.0
    has_vegas = (vegas_total[:, None] > 0) & (vegas_total[None, :] > 0)
    vegas_scale = np.clip(avg_vegas / config.vegas_scaling_denominator, 1.0, config.vegas_scale_cap)
    scale_mask = teammate_mask & has_vegas
    matrix[scale_mask] *= vegas_scale[scale_mask]

    # 2. Same game, different team, both batters
    opp_game_mask = same_game & both_batters & ~same_stack
    matrix[opp_game_mask] = config.same_game_opponent

    # 3. Pitcher + own team's batter
    own_pitcher_mask = (pitcher_batter | batter_pitcher) & pitcher_same_team
    matrix[own_pitcher_mask] = config.pitcher_own_hitters

    # 4. Pitcher vs opposing batter
    opp_pitcher_mask = (pitcher_batter & pitcher_vs_opp) | (batter_pitcher & (opponent_code[:, None] == team_code[None, :]))
    matrix[opp_pitcher_mask] = config.pitcher_vs_opposing

    # Clamp off-diagonal to [-0.95, 0.95]
    np.clip(matrix, -0.95, 0.95, out=matrix)
    np.fill_diagonal(matrix, 1.0)

    # Symmetrize (some masks may have applied asymmetrically)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 1.0)

    # Ensure PSD and compute Cholesky
    matrix = _ensure_positive_semidefinite(matrix)
    cholesky = _compute_cholesky(matrix)
    return CorrelationModel(
        player_ids=player_ids,
        matrix=matrix,
        nu=max(config.copula_nu, 2),
        cholesky=cholesky,
    )


def _ensure_positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    """Force the matrix to be PSD with a robust minimum eigenvalue."""
    sym = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    min_eig = eigvals.min()
    if min_eig >= 1e-4:
        return sym
    # Clip eigenvalues with a strong floor
    eigvals = np.clip(eigvals, 1e-4, None)
    reconstructed = eigvecs @ np.diag(eigvals) @ eigvecs.T
    reconstructed = (reconstructed + reconstructed.T) / 2.0
    # Rescale so diagonal is exactly 1.0 (correlation matrix)
    d = np.sqrt(np.diag(reconstructed))
    d[d == 0] = 1.0
    reconstructed = reconstructed / (d[:, None] * d[None, :])
    np.fill_diagonal(reconstructed, 1.0)
    return reconstructed


def _compute_cholesky(matrix: np.ndarray) -> np.ndarray:
    """Cholesky with increasing jitter fallback."""
    last_error = None
    m = matrix.copy()
    for attempt in range(8):
        try:
            return np.linalg.cholesky(m)
        except np.linalg.LinAlgError as exc:
            last_error = exc
            jitter = 10 ** (-6 + attempt)
            m = m + np.eye(m.shape[0]) * jitter
    raise np.linalg.LinAlgError(
        f"Cholesky decomposition failed after 8 jitter attempts: {last_error}"
    )
