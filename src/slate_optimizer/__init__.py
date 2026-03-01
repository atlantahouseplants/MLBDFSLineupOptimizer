"""Core package for the next-gen MLB slate optimizer."""

from .data.storage import SlateDatabase
from .ingestion.ballparkpal import (
    BallparkPalBundle,
    BallparkPalLoader,
    BallparkPalPaths,
)
from .ingestion.fanduel import FanduelCSVLoader, FanduelPlayerList
from .ingestion.slate_builder import build_player_dataset
from .optimizer import (
    OPTIMIZER_COLUMNS,
    LineupResult,
    OptimizerPlayer,
    build_optimizer_dataset,
    generate_lineups,
)
from .projection import PROJECTION_COLUMNS, compute_baseline_projections

__all__ = [
    "BallparkPalBundle",
    "BallparkPalLoader",
    "BallparkPalPaths",
    "FanduelCSVLoader",
    "FanduelPlayerList",
    "build_player_dataset",
    "SlateDatabase",
    "PROJECTION_COLUMNS",
    "compute_baseline_projections",
    "OPTIMIZER_COLUMNS",
    "build_optimizer_dataset",
    "OptimizerPlayer",
    "generate_lineups",
    "LineupResult",
]
