"""Data structures for optimizer-ready player rows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class OptimizerPlayer:
    fd_player_id: str
    full_name: str
    position: str
    player_type: str
    team_code: str
    opponent_code: str
    game_pk: Optional[str]
    stack_key: str
    game_key: str
    salary: int
    proj_fd_mean: float
    proj_fd_floor: float
    proj_fd_ceiling: float
    stack_priority: str
    default_max_exposure: float


__all__ = ["OptimizerPlayer"]
