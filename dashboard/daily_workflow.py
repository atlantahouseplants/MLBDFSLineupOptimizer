"""Requirements: streamlit>=1.30, pandas.

Unified Streamlit workflow for the MLB slate optimizer (Steps 1-4).
"""
from __future__ import annotations

import hashlib
import io
import json
import tempfile

from dotenv import load_dotenv
load_dotenv()
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from zoneinfo import ZoneInfo

from slate_optimizer.analysis import backtest
from slate_optimizer.analysis.boom_bust_calibration import build_boom_bust_calibration_report
from slate_optimizer.analysis.calibration_feedback import build_calibration_feedback
from slate_optimizer.analysis.exposure_tuner import build_exposure_recommendations, player_overrides_text, stack_cap_maps
from slate_optimizer.analysis.failure_attribution import build_failure_attribution_report
from slate_optimizer.analysis.late_news import build_late_news_report
from slate_optimizer.analysis.lineup_scorecard import build_lineup_scorecard
from slate_optimizer.analysis.model_calibration import build_model_calibration_report
from slate_optimizer.analysis.optimizer_tuning import build_optimizer_weight_tuning_report
from slate_optimizer.analysis.post_slate_learning import build_post_slate_learning_report
from slate_optimizer.analysis.portfolio_audit import audit_report_to_csv, build_portfolio_audit
from slate_optimizer.analysis.candidate_pool_quality import build_candidate_pool_quality_report
from slate_optimizer.analysis.portfolio_frontier import build_portfolio_frontier_report
from slate_optimizer.analysis.portfolio_stress import build_portfolio_stress_report
from slate_optimizer.analysis.preset_recommender import recommend_slate_preset
from slate_optimizer.analysis.profile_tuner import build_profile_tuning_report
from slate_optimizer.analysis.run_audit import build_run_audit_report
from slate_optimizer.analysis.slate_replay import build_slate_replay_report
from slate_optimizer.analysis.stack_exposure import build_stack_exposure_plan, stack_plan_weights
from slate_optimizer.analysis.strategy_backtest import build_strategy_backtest_report
from slate_optimizer.data.storage import SlateDatabase

from slate_optimizer.ingestion.ballparkpal import BallparkPalLoader
from slate_optimizer.ingestion.batting_orders import BattingOrderLoader
from slate_optimizer.ingestion.fanduel import FanduelCSVLoader
from slate_optimizer.ingestion.handedness import HandednessLoader
from slate_optimizer.ingestion.recent_stats import RecentStatsLoader
from slate_optimizer.ingestion.slate_builder import build_player_dataset
from slate_optimizer.ingestion.vegas import VegasLoader
from slate_optimizer.ingestion.vegas_api import fetch_vegas_lines
from slate_optimizer.ingestion.mlb_lineups_api import fetch_batting_orders, lineup_fetch_status
from slate_optimizer.optimizer import build_optimizer_dataset, generate_lineups
from slate_optimizer.optimizer.solver import STACK_PRESETS
from slate_optimizer.optimizer.config import OptimizerConfig
from slate_optimizer.optimizer.dataset import OPTIMIZER_COLUMNS
from slate_optimizer.optimizer.export import (
    FANDUEL_UPLOAD_COLUMNS,
    assign_lineups_to_contests,
    extract_template_entries,
    lineups_to_fanduel_template,
    lineups_to_fanduel_upload,
    validate_fanduel_lineups,
)
from slate_optimizer.projection import (
    OwnershipModelConfig,
    blend_projection_sources,
    compute_baseline_projections,
    compute_ownership_series,
)
from slate_optimizer.projection.ownership_model import build_ownership_features
from slate_optimizer.simulation import (
    CONTEST_TYPE_PROFILES,
    DEFAULT_PAYOUT_LADDER_TEXT,
    FieldQualityMix,
    FieldRealismProfile,
    SimulationConfig,
    apply_contest_type_to_state,
    build_correlation_matrix,
    compute_actual_ownership,
    fit_player_distributions,
    learn_field_duplication_profile,
    match_lineups_to_contest_entries,
    load_field_duplication_profile,
    apply_ownership_scenario_metrics,
    parse_payout_ladder_dataframe,
    parse_payout_ladder_text,
    payout_ladder_to_dicts,
    payout_ladder_to_text,
    save_field_duplication_profile,
    select_portfolio,
    simulate_contest,
    simulate_field,
    simulate_slate,
)
from slate_optimizer.simulation.risk_budget import RISK_BUDGET_PROFILES, apply_risk_budget_to_state
from scipy.stats import norm

WORKFLOW_KEY = "workflow_state"
NAV_KEY = "workflow_nav"
CONFIG_KEY = "optimizer_config"
LINEUPS_KEY = "lineup_results"
SIM_CONFIG_KEY = "simulation_config"
SIM_RESULTS_KEY = "simulation_results"
RUN_HISTORY_KEY = "run_history"
UI_DAILY_MODE_KEY = "ui_daily_build_mode"
UI_ADVANCED_KEY = "ui_show_advanced_controls"
DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "slates.db"
FIELD_DUPLICATION_PROFILE_PATH = Path(__file__).resolve().parents[1] / "data" / "field_duplication_profile.json"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "uploads"
UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_MANIFEST = UPLOAD_CACHE_DIR / "manifest.json"
_RUNTIME_SIM_CACHE: Dict[str, Dict[str, object]] = {}

SLATE_RUN_PRESETS: Dict[str, Dict[str, Dict[str, object]]] = {
    "300-entry large-field GPP": {
        "description": "Deep candidate pool, payout EV selection, stack plan nudges, and stronger uniqueness controls.",
        "optimizer": {
            "num_lineups": 800,
            "stack_template_selections": ["4-3-1", "4-2-2", "3-3-2"],
            "template_counts": {"4-3-1": 360, "4-2-2": 260, "3-3-2": 180},
            "batter_chalk_threshold": 22.0,
            "pitcher_chalk_threshold": 32.0,
            "batter_chalk_exposure_cap": 28.0,
            "pitcher_chalk_exposure_cap": 48.0,
            "max_lineup_ownership": 0.0,
            "leverage_weight": 18.0,
            "randomness": 7.0,
            "ceiling_weight": 40.0,
            "ownership_penalty_weight": 12.0,
            "min_salary": 34000,
            "min_uniques": 2,
            "bring_back_enabled": False,
            "min_game_total": 0.0,
        },
        "simulation": {
            "num_candidates": 300,
            "num_simulations": 25000,
            "field_size": 3000,
            "selection_metric": "payout_ev",
            "diversity_weight": 0.40,
            "max_batter_exposure": 0.35,
            "max_pitcher_exposure": 0.50,
            "selection_leverage_weight": 0.25,
            "selection_ownership_weight": 0.18,
            "selection_duplication_weight": 0.30,
            "use_stack_exposure_engine": True,
            "stack_target_weight": 0.30,
            "use_stack_auto_caps": True,
            "use_stack_auto_mins": False,
            "use_portfolio_optimizer": True,
        },
    },
    "150-max large-field GPP": {
        "description": "Same leverage DNA with a tighter final portfolio.",
        "optimizer": {
            "num_lineups": 500,
            "stack_template_selections": ["4-3-1", "4-2-2", "3-3-2"],
            "template_counts": {"4-3-1": 220, "4-2-2": 170, "3-3-2": 110},
            "batter_chalk_threshold": 23.0,
            "pitcher_chalk_threshold": 33.0,
            "batter_chalk_exposure_cap": 30.0,
            "pitcher_chalk_exposure_cap": 50.0,
            "leverage_weight": 17.0,
            "randomness": 6.0,
            "ceiling_weight": 35.0,
            "ownership_penalty_weight": 10.0,
            "min_salary": 34000,
            "min_uniques": 2,
        },
        "simulation": {
            "num_candidates": 150,
            "num_simulations": 22000,
            "field_size": 2500,
            "selection_metric": "payout_ev",
            "diversity_weight": 0.35,
            "max_batter_exposure": 0.38,
            "max_pitcher_exposure": 0.55,
            "selection_leverage_weight": 0.22,
            "selection_ownership_weight": 0.15,
            "selection_duplication_weight": 0.25,
            "use_stack_exposure_engine": True,
            "stack_target_weight": 0.25,
            "use_stack_auto_caps": True,
            "use_stack_auto_mins": False,
        },
    },
    "Small-field GPP": {
        "description": "More projection and fewer pure fades for smaller fields.",
        "optimizer": {
            "num_lineups": 300,
            "stack_template_selections": ["4-3-1", "4-2-2"],
            "template_counts": {"4-3-1": 170, "4-2-2": 130},
            "batter_chalk_threshold": 28.0,
            "pitcher_chalk_threshold": 38.0,
            "batter_chalk_exposure_cap": 45.0,
            "pitcher_chalk_exposure_cap": 65.0,
            "leverage_weight": 10.0,
            "randomness": 4.0,
            "ceiling_weight": 25.0,
            "ownership_penalty_weight": 6.0,
            "min_salary": 34300,
            "min_uniques": 1,
        },
        "simulation": {
            "num_candidates": 60,
            "num_simulations": 18000,
            "field_size": 1500,
            "selection_metric": "expected_roi",
            "diversity_weight": 0.20,
            "max_batter_exposure": 0.55,
            "max_pitcher_exposure": 0.70,
            "selection_leverage_weight": 0.08,
            "selection_ownership_weight": 0.06,
            "selection_duplication_weight": 0.10,
            "use_stack_exposure_engine": True,
            "stack_target_weight": 0.15,
            "use_stack_auto_caps": True,
            "use_stack_auto_mins": False,
        },
    },
    "Chalk fade build": {
        "description": "Aggressive ownership and duplication penalties for slates with obvious chalk stacks.",
        "optimizer": {
            "num_lineups": 700,
            "stack_template_selections": ["4-3-1", "4-2-2", "3-3-2"],
            "template_counts": {"4-3-1": 280, "4-2-2": 250, "3-3-2": 170},
            "batter_chalk_threshold": 18.0,
            "pitcher_chalk_threshold": 28.0,
            "batter_chalk_exposure_cap": 20.0,
            "pitcher_chalk_exposure_cap": 42.0,
            "leverage_weight": 24.0,
            "randomness": 9.0,
            "ceiling_weight": 40.0,
            "ownership_penalty_weight": 18.0,
            "min_salary": 33800,
            "min_uniques": 2,
        },
        "simulation": {
            "num_candidates": 300,
            "num_simulations": 25000,
            "field_size": 3000,
            "selection_metric": "payout_ev",
            "diversity_weight": 0.45,
            "max_batter_exposure": 0.30,
            "max_pitcher_exposure": 0.48,
            "selection_leverage_weight": 0.35,
            "selection_ownership_weight": 0.28,
            "selection_duplication_weight": 0.40,
            "use_stack_exposure_engine": True,
            "stack_target_weight": 0.35,
            "use_stack_auto_caps": True,
            "use_stack_auto_mins": False,
        },
    },
    "Balanced leverage build": {
        "description": "A middle lane when projections, ownership, and team sims all look fairly efficient.",
        "optimizer": {
            "num_lineups": 600,
            "stack_template_selections": ["4-3-1", "4-2-2"],
            "template_counts": {"4-3-1": 340, "4-2-2": 260},
            "batter_chalk_threshold": 25.0,
            "pitcher_chalk_threshold": 35.0,
            "batter_chalk_exposure_cap": 35.0,
            "pitcher_chalk_exposure_cap": 55.0,
            "leverage_weight": 15.0,
            "randomness": 5.0,
            "ceiling_weight": 30.0,
            "ownership_penalty_weight": 8.0,
            "min_salary": 34000,
            "min_uniques": 1,
        },
        "simulation": {
            "num_candidates": 300,
            "num_simulations": 22000,
            "field_size": 2500,
            "selection_metric": "payout_ev",
            "diversity_weight": 0.32,
            "max_batter_exposure": 0.40,
            "max_pitcher_exposure": 0.58,
            "selection_leverage_weight": 0.18,
            "selection_ownership_weight": 0.12,
            "selection_duplication_weight": 0.22,
            "use_stack_exposure_engine": True,
            "stack_target_weight": 0.22,
            "use_stack_auto_caps": True,
            "use_stack_auto_mins": False,
        },
    },
}

# ──────────────────────────────────────────────────────────────────────
# Slate-size profiles: overlay adjustments applied ON TOP of the contest
# preset, keyed by number of games. A 2-game slate concentrates ownership
# and kills stack diversity; an 11-game slate spreads the field thin.
# Overlay keys win over the contest preset where they overlap.
# Numeric values can be overridden by config/slate_profiles.json, which the
# Step 6 profile tuner maintains from actual results.
# ──────────────────────────────────────────────────────────────────────
_DEFAULT_SLATE_SIZE_PROFILES: Dict[str, Dict[str, object]] = {
    "small": {
        "label": "Small slate (2–4 games)",
        "games": (1, 4),
        "reason": (
            "Few games = concentrated ownership and unavoidable chalk. Loosen chalk caps "
            "(you cannot dodge everyone), raise randomness and ceiling weight for separation, "
            "lean on game stacks with bring-backs."
        ),
        "optimizer": {
            "batter_chalk_threshold": 30.0,
            "batter_chalk_exposure_cap": 50.0,
            "pitcher_chalk_threshold": 40.0,
            "pitcher_chalk_exposure_cap": 60.0,
            "leverage_weight": 22.0,
            "randomness": 9.0,
            "ceiling_weight": 40.0,
            "ownership_penalty_weight": 15.0,
            "min_uniques": 1,
            "stack_template_selections": ["4-3-1", "4-4 (two big stacks)", "3-3-2"],
            "bring_back_enabled": True,
            "bring_back_count": 1,
            "min_game_total": 0.0,
        },
    },
    "medium": {
        "label": "Medium slate (5–9 games)",
        "games": (5, 9),
        "reason": (
            "Standard main-slate shape: enough stacks to be selective, standard chalk fades, "
            "bring-backs for correlation, 2-unique separation across the portfolio."
        ),
        "optimizer": {
            "batter_chalk_threshold": 25.0,
            "batter_chalk_exposure_cap": 30.0,
            "pitcher_chalk_threshold": 35.0,
            "pitcher_chalk_exposure_cap": 50.0,
            "leverage_weight": 17.0,
            "randomness": 6.0,
            "ceiling_weight": 35.0,
            "ownership_penalty_weight": 10.0,
            "min_uniques": 2,
            "stack_template_selections": ["4-3-1", "4-2-2", "3-3-2"],
            "bring_back_enabled": True,
            "bring_back_count": 1,
            "min_game_total": 0.0,
        },
    },
    "large": {
        "label": "Large slate (10+ games)",
        "games": (10, 99),
        "reason": (
            "Many games = the field spreads thin and obvious chalk gets extra-owned. Tighten "
            "chalk caps, filter stacks to high-total games, drop forced bring-backs (natural "
            "diversification), keep 2-unique separation."
        ),
        "optimizer": {
            "batter_chalk_threshold": 20.0,
            "batter_chalk_exposure_cap": 25.0,
            "pitcher_chalk_threshold": 30.0,
            "pitcher_chalk_exposure_cap": 45.0,
            "leverage_weight": 20.0,
            "randomness": 6.0,
            "ceiling_weight": 35.0,
            "ownership_penalty_weight": 12.0,
            "min_uniques": 2,
            "stack_template_selections": ["4-3-1", "4-2-2", "3-3-2"],
            "bring_back_enabled": False,
            "min_game_total": 8.5,
        },
    },
}

SLATE_PROFILES_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "slate_profiles.json"


def _load_slate_size_profiles() -> Dict[str, Dict[str, object]]:
    """Defaults overlaid with tuner-maintained overrides from config/slate_profiles.json."""
    import copy

    profiles = copy.deepcopy(_DEFAULT_SLATE_SIZE_PROFILES)
    if not SLATE_PROFILES_CONFIG_PATH.exists():
        return profiles
    try:
        overrides = json.loads(SLATE_PROFILES_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:  # pylint: disable=broad-except
        return profiles
    if not isinstance(overrides, dict):
        return profiles
    for key, payload in overrides.items():
        if key not in profiles or not isinstance(payload, dict):
            continue
        optimizer_overrides = payload.get("optimizer")
        if isinstance(optimizer_overrides, dict):
            profiles[key]["optimizer"].update(optimizer_overrides)
    return profiles


def _save_slate_profile_overrides(adjustments: Dict[str, Dict[str, float]]) -> None:
    """Merge {profile: {setting: value}} into the JSON override file."""
    existing: Dict[str, Dict[str, object]] = {}
    if SLATE_PROFILES_CONFIG_PATH.exists():
        try:
            existing = json.loads(SLATE_PROFILES_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:  # pylint: disable=broad-except
            existing = {}
    if not isinstance(existing, dict):
        existing = {}
    for profile_key, settings in adjustments.items():
        bucket = existing.setdefault(profile_key, {}).setdefault("optimizer", {})
        bucket.update({k: v for k, v in settings.items()})
    existing["updated_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    SLATE_PROFILES_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SLATE_PROFILES_CONFIG_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")


SLATE_SIZE_PROFILES: Dict[str, Dict[str, object]] = _load_slate_size_profiles()


def _slate_profile_key(num_games: int) -> str:
    for key, profile in SLATE_SIZE_PROFILES.items():
        low, high = profile["games"]
        if low <= num_games <= high:
            return key
    return "medium"


def _detect_slate_shape(optimizer_df: Optional[pd.DataFrame]) -> Dict[str, object]:
    if optimizer_df is None or optimizer_df.empty:
        return {"games": 0, "teams": 0, "profile": "medium"}
    games = int(optimizer_df.get("game_key", pd.Series(dtype=str)).dropna().nunique())
    teams = int(optimizer_df.get("team_code", pd.Series(dtype=str)).dropna().nunique())
    return {"games": games, "teams": teams, "profile": _slate_profile_key(games)}


MANUAL_WEIGHT_PRESET = "Manual weights"
DEFAULT_WEIGHT_PRESET = "Balanced (equal blend)"

DEFAULT_MODEL_PRESET = "Balanced (field median)"
OWNERSHIP_MODEL_CUSTOM = "Manual adjustments"

MODEL_DEFAULT_VALUES = asdict(OwnershipModelConfig())

def _preset_values(base: Dict[str, float], **updates) -> Dict[str, float]:
    data = dict(base)
    data.update(updates)
    return data


OWNERSHIP_MODEL_PRESETS = {
    DEFAULT_MODEL_PRESET: dict(MODEL_DEFAULT_VALUES),
    "Chalk leaning": _preset_values(
        MODEL_DEFAULT_VALUES,
        salary_weight=0.35,
        projection_weight=0.35,
        value_weight=0.1,
        team_weight=0.1,
        name_weight=0.05,
        position_weight=0.05,
        min_pct=0.08,
        max_pct=0.95,
    ),
    "Leverage hunting": _preset_values(
        MODEL_DEFAULT_VALUES,
        salary_weight=0.1,
        projection_weight=0.2,
        value_weight=0.35,
        team_weight=0.15,
        name_weight=0.05,
        position_weight=0.15,
        min_pct=0.03,
        max_pct=0.75,
    ),
}

OWNERSHIP_MODEL_OPTIONS = list(OWNERSHIP_MODEL_PRESETS.keys()) + [OWNERSHIP_MODEL_CUSTOM]

OWNERSHIP_MODEL_FIELDS = [
    ("salary_weight", "Salary weight", 0.0, 3.0, 0.05, "Higher salary rank tends to increase ownership."),
    ("projection_weight", "Projection weight", 0.0, 3.0, 0.05, "Effect of projection rank on ownership."),
    ("value_weight", "Value weight", 0.0, 3.0, 0.05, "Effect of value (points per dollar)."),
    ("team_weight", "Team total weight", 0.0, 3.0, 0.05, "Influence of team implied run totals."),
    ("name_weight", "Name recognition weight", 0.0, 3.0, 0.05, "Star power / recent performance signal."),
    ("position_weight", "Positional scarcity weight", 0.0, 3.0, 0.05, "Concentration at thin positions."),
    ("min_pct", "Min ownership (decimal)", 0.0, 0.5, 0.01, "0.05 = 5% baseline ownership floor."),
    ("max_pct", "Max ownership (decimal)", 0.1, 1.0, 0.01, "Upper cap for fallback ownership (0.25 = 25%)."),
]


def _ownership_model_preset_values(preset: str) -> Dict[str, float]:
    base = OWNERSHIP_MODEL_PRESETS.get(preset)
    if base is None:
        base = OWNERSHIP_MODEL_PRESETS[DEFAULT_MODEL_PRESET]
    values: Dict[str, float] = {}
    for field, *_ in OWNERSHIP_MODEL_FIELDS:
        values[field] = float(base.get(field, MODEL_DEFAULT_VALUES.get(field, 0.0)))
    return values


def _apply_ownership_model_preset(preset: str) -> None:
    values = _ownership_model_preset_values(preset)
    for field, value in values.items():
        st.session_state[f"ownership_model_{field}"] = value


def _preset_equal_weights(count: int) -> List[float]:
    return [1.0] * max(count, 1)


def _preset_primary_anchor(count: int) -> List[float]:
    if count <= 1:
        return [1.0] * max(count, 1)
    anchor = 0.5
    tail_share = (1.0 - anchor) / max(count - 1, 1)
    return [anchor] + [tail_share] * (count - 1)


def _preset_leverage_boost(count: int) -> List[float]:
    # Later uploads often reflect last-minute intel; emphasize them progressively.
    if count <= 0:
        return []
    return [float(idx + 1) for idx in range(count)]


WEIGHT_PRESETS = {
    DEFAULT_WEIGHT_PRESET: {
        "description": "Treat every uploaded ownership source equally.",
        "generator": _preset_equal_weights,
    },
    "Primary anchor (source order)": {
        "description": "Lean on the first uploaded source while still blending others.",
        "generator": _preset_primary_anchor,
    },
    "Leverage boost (latest source emphasis)": {
        "description": "Weight later uploads more heavily to capture late-breaking ownership moves.",
        "generator": _preset_leverage_boost,
    },
}

WEIGHT_PRESET_OPTIONS = list(WEIGHT_PRESETS.keys()) + [MANUAL_WEIGHT_PRESET]


def _get_session() -> Dict:
    if WORKFLOW_KEY not in st.session_state:
        st.session_state[WORKFLOW_KEY] = {}
    return st.session_state[WORKFLOW_KEY]


def _get_config_state() -> Dict:
    default_config = {
        "num_lineups": 500,
        "salary_cap": 35000,
        "stack_template_selections": ["4-3-1"],
        "batter_chalk_threshold": 25.0,
        "pitcher_chalk_threshold": 35.0,
        "batter_chalk_exposure_cap": 30.0,
        "pitcher_chalk_exposure_cap": 50.0,
        "max_lineup_ownership": 0.0,
        "leverage_weight": 15.0,
        "randomness": 5.0,
        "ceiling_weight": 35.0,
        "ownership_penalty_weight": 10.0,
        "min_salary": 34000,
        "min_uniques": 1,
        "player_overrides": "",
        "bring_back_enabled": False,
        "bring_back_count": 1,
        "min_game_total": 0.0,
    }
    return st.session_state.setdefault(CONFIG_KEY, default_config)


def _get_sim_config_state() -> Dict:
    default_config = {
        "num_simulations": 20000,
        "num_candidates": 300,
        "entry_fee": 20.0,
        "contest_entries": 50000,
        "payout_ladder_text": DEFAULT_PAYOUT_LADDER_TEXT,
        "volatility_scale": 1.0,
        "copula_nu": 5,
        "teammate_corr": 0.25,
        "pitcher_vs_opposing": -0.15,
        "field_size": 1000,
        "field_quality_shark_pct": 0.10,
        "field_quality_rec_pct": 0.60,
        "field_quality_random_pct": 0.30,
        "use_field_realism": True,
        "field_realism_value_bias": 0.22,
        "field_realism_train_bias": 0.18,
        "field_realism_upside_bias": 0.10,
        "field_salary_target_shark": 0.970,
        "field_salary_target_rec": 0.930,
        "contest_type_profile": "Auto",
        "use_contest_type_brain": True,
        "selection_metric": "top_1pct_rate",
        "diversity_weight": 0.3,
        "max_batter_exposure": 0.4,
        "max_pitcher_exposure": 0.6,
        "min_batter_exposure": 0.0,
        "min_pitcher_exposure": 0.0,
        "min_stack_exposure": 0.0,
        "max_stack_exposure": 1.0,
        "selection_leverage_weight": 0.15,
        "selection_ownership_weight": 0.10,
        "selection_duplication_weight": 0.15,
        "selection_scenario_weight": 0.10,
        "selection_marginal_value_weight": 0.20,
        "use_ownership_scenarios": True,
        "risk_profile": "Balanced leverage",
        "calibration_adjustment_strength": 0.0,
        "use_stack_exposure_engine": True,
        "stack_target_weight": 0.20,
        "use_stack_auto_caps": True,
        "use_stack_auto_mins": False,
        "use_portfolio_optimizer": True,
        "portfolio_time_limit_seconds": 45,
        "use_simulation_cache": True,
        "use_stratified": False,
    }
    return st.session_state.setdefault(SIM_CONFIG_KEY, default_config)


def _daily_mode_enabled() -> bool:
    return bool(st.session_state.get(UI_DAILY_MODE_KEY, True))


def _advanced_controls_enabled() -> bool:
    return (not _daily_mode_enabled()) or bool(st.session_state.get(UI_ADVANCED_KEY, False))


def _status_token(ready: bool) -> str:
    return "Ready" if ready else "Needed"


def _render_sidebar_daily_status(workflow: Dict) -> None:
    optimizer_ready = workflow.get("optimizer") is not None
    candidate_ready = workflow.get("lineups_df") is not None
    sim_results = workflow.get(SIM_RESULTS_KEY) or {}
    portfolio_ready = bool(sim_results.get("selected_lineup_objects")) or workflow.get("active_lineup_source") == "simulation"
    export_ready = bool(workflow.get("lineups"))
    st.sidebar.markdown("**Today build status**")
    st.sidebar.caption(f"1. Slate setup: {_status_token(optimizer_ready)}")
    st.sidebar.caption(f"2. Candidate pool: {_status_token(candidate_ready)}")
    st.sidebar.caption(f"3. Simulation portfolio: {_status_token(portfolio_ready)}")
    st.sidebar.caption(f"4. Review/export: {_status_token(export_ready)}")


def _render_daily_step_note(step_label: str) -> None:
    if not _daily_mode_enabled():
        return
    notes = {
        "setup": "Daily mode: upload or re-use the slate, then press Process Slate. Optional feeds and tuning controls are still available below, but you usually do not need to touch them.",
        "optimize": "Daily mode: pick the slate preset, confirm candidate pool size and stack templates, then run the optimizer. Use Show advanced controls in the sidebar when you want every engine knob.",
        "simulate": "Daily mode: apply the risk budget, choose 300 lineups, then run the simulation. The hidden settings still use your saved/preset leverage defaults.",
        "review": "Daily mode: focus on the final audit, why-these-lineups scorecard, pre-submit check, and FanDuel download. Advanced review panels are tucked below.",
    }
    note = notes.get(step_label)
    if note:
        st.info(note)


def _render_advanced_caption() -> None:
    if _daily_mode_enabled() and not _advanced_controls_enabled():
        st.caption("Showing the daily controls. Turn on **Show advanced controls** in the sidebar for the full research/tuning UI.")


def _apply_sim_preset(preset: SimulationConfig, state: Dict) -> None:
    state["num_simulations"] = preset.num_simulations
    state["volatility_scale"] = preset.volatility_scale
    state["field_size"] = preset.num_field_lineups
    state["field_quality_shark_pct"] = preset.field_quality_shark_pct
    state["field_quality_rec_pct"] = preset.field_quality_rec_pct
    state["field_quality_random_pct"] = preset.field_quality_random_pct
    state["use_field_realism"] = preset.use_field_realism
    state["field_realism_value_bias"] = preset.field_realism_value_bias
    state["field_realism_train_bias"] = preset.field_realism_train_bias
    state["field_realism_upside_bias"] = preset.field_realism_upside_bias
    state["field_salary_target_shark"] = preset.field_salary_target_shark
    state["field_salary_target_rec"] = preset.field_salary_target_rec
    state["contest_type_profile"] = preset.contest_type_profile
    state["use_contest_type_brain"] = preset.use_contest_type_brain
    state["selection_metric"] = preset.selection_metric
    state["diversity_weight"] = preset.diversity_weight
    state["max_batter_exposure"] = preset.max_batter_exposure
    state["max_pitcher_exposure"] = preset.max_pitcher_exposure
    state["min_batter_exposure"] = preset.min_batter_exposure
    state["min_pitcher_exposure"] = preset.min_pitcher_exposure
    state["min_stack_exposure"] = preset.min_stack_exposure
    state["max_stack_exposure"] = preset.max_stack_exposure
    state["selection_leverage_weight"] = preset.selection_leverage_weight
    state["selection_ownership_weight"] = preset.selection_ownership_weight
    state["selection_duplication_weight"] = preset.selection_duplication_weight
    state["selection_scenario_weight"] = preset.selection_scenario_weight
    state["selection_marginal_value_weight"] = preset.selection_marginal_value_weight
    state["use_ownership_scenarios"] = preset.use_ownership_scenarios
    state["risk_profile"] = preset.risk_profile
    state["use_stack_exposure_engine"] = preset.use_stack_exposure_engine
    state["stack_target_weight"] = preset.stack_target_weight
    state["use_stack_auto_caps"] = preset.use_stack_auto_caps
    state["use_stack_auto_mins"] = preset.use_stack_auto_mins


def _apply_slate_run_preset(label: str) -> None:
    preset = SLATE_RUN_PRESETS.get(label)
    if not preset:
        return
    optimizer_state = _get_config_state()
    sim_state = _get_sim_config_state()
    optimizer_state.update(preset.get("optimizer", {}))
    sim_state.update(preset.get("simulation", {}))
    st.session_state["active_slate_run_preset"] = label


def _apply_slate_size_profile(profile_key: str) -> None:
    profile = SLATE_SIZE_PROFILES.get(profile_key)
    if not profile:
        return
    _get_config_state().update(profile.get("optimizer", {}))
    st.session_state["active_slate_profile"] = profile_key


def _auto_configure_for_slate(workflow: Dict) -> None:
    """Auto-apply contest preset + slate-size overlay right after processing.

    Order matters: contest preset first (broad strategy), slate-size profile
    second (overrides the size-sensitive knobs). Everything remains editable
    in Step 3; the strategy banner shows what was applied and why.
    """
    optimizer_df = workflow.get("optimizer")
    shape = _detect_slate_shape(optimizer_df)
    try:
        stack_plan = build_stack_exposure_plan(optimizer_df) if optimizer_df is not None else None
        recommendation = recommend_slate_preset(optimizer_df, stack_plan)
    except Exception:  # pylint: disable=broad-except
        recommendation = None
    preset_label = recommendation.preset if recommendation else "150-max large-field GPP"
    if preset_label not in SLATE_RUN_PRESETS:
        preset_label = "150-max large-field GPP"
    _apply_slate_run_preset(preset_label)
    _apply_slate_size_profile(str(shape["profile"]))
    workflow["strategy_auto"] = {
        "games": shape["games"],
        "teams": shape["teams"],
        "profile": shape["profile"],
        "preset": preset_label,
        "confidence": getattr(recommendation, "confidence", None),
        "reasons": list(getattr(recommendation, "reasons", []) or []),
    }


def _render_strategy_banner(workflow: Dict) -> None:
    auto = workflow.get("strategy_auto")
    if not auto:
        return
    profile = SLATE_SIZE_PROFILES.get(str(auto.get("profile")), {})
    config_state = _get_config_state()
    conf = auto.get("confidence")
    conf_text = f" ({conf:.0%} confidence)" if isinstance(conf, float) else ""
    st.info(
        f"🎯 **{auto.get('games', '?')}-game slate detected → {profile.get('label', auto.get('profile'))}**  \n"
        f"Auto-applied contest preset **{auto.get('preset')}**{conf_text} plus the slate-size overlay: "
        f"ceiling {config_state.get('ceiling_weight', 0):.0f}%, "
        f"ownership penalty {config_state.get('ownership_penalty_weight', 0):.0f}%, "
        f"leverage {config_state.get('leverage_weight', 0):.0f}%, "
        f"randomness {config_state.get('randomness', 0):.0f}%, "
        f"min uniques {config_state.get('min_uniques', 1)}, "
        f"chalk caps {config_state.get('batter_chalk_exposure_cap', 0):.0f}% bat / "
        f"{config_state.get('pitcher_chalk_exposure_cap', 0):.0f}% pit, "
        f"bring-back {'on' if config_state.get('bring_back_enabled') else 'off'}.  \n"
        f"_{profile.get('reason', '')}_ Every value stays editable below."
    )


def _build_simulation_config(state: Dict) -> SimulationConfig:
    payout_bands = parse_payout_ladder_text(str(state.get("payout_ladder_text", "") or ""))
    config = SimulationConfig(
        num_simulations=int(state.get("num_simulations", 10000)),
        num_candidates=int(state.get("num_candidates", 300)),
        entry_fee=float(state.get("entry_fee", 20.0) or 20.0),
        contest_entries=int(state.get("contest_entries", 50000) or 50000),
        payout_structure=payout_ladder_to_dicts(payout_bands) if payout_bands else None,
        volatility_scale=float(state.get("volatility_scale", 1.0)),
        num_field_lineups=int(state.get("field_size", 1000)),
        field_quality_shark_pct=float(state.get("field_quality_shark_pct", 0.10) or 0.0),
        field_quality_rec_pct=float(state.get("field_quality_rec_pct", 0.60) or 0.0),
        field_quality_random_pct=float(state.get("field_quality_random_pct", 0.30) or 0.0),
        use_field_realism=bool(state.get("use_field_realism", True)),
        field_realism_value_bias=float(state.get("field_realism_value_bias", 0.22) or 0.0),
        field_realism_train_bias=float(state.get("field_realism_train_bias", 0.18) or 0.0),
        field_realism_upside_bias=float(state.get("field_realism_upside_bias", 0.10) or 0.0),
        field_salary_target_shark=float(state.get("field_salary_target_shark", 0.970) or 0.970),
        field_salary_target_rec=float(state.get("field_salary_target_rec", 0.930) or 0.930),
        contest_type_profile=str(state.get("contest_type_profile", "Auto")),
        use_contest_type_brain=bool(state.get("use_contest_type_brain", True)),
        selection_metric=str(state.get("selection_metric", "top_1pct_rate")),
        diversity_weight=float(state.get("diversity_weight", 0.3)),
        max_batter_exposure=float(state.get("max_batter_exposure", 0.4)),
        max_pitcher_exposure=float(state.get("max_pitcher_exposure", 0.6)),
        min_batter_exposure=float(state.get("min_batter_exposure", 0.0)),
        min_pitcher_exposure=float(state.get("min_pitcher_exposure", 0.0)),
        min_stack_exposure=float(state.get("min_stack_exposure", 0.0)),
        max_stack_exposure=float(state.get("max_stack_exposure", 1.0)),
        selection_leverage_weight=float(state.get("selection_leverage_weight", 0.15)),
        selection_ownership_weight=float(state.get("selection_ownership_weight", 0.10)),
        selection_duplication_weight=float(state.get("selection_duplication_weight", 0.15)),
        selection_scenario_weight=float(state.get("selection_scenario_weight", 0.10)),
        selection_marginal_value_weight=float(state.get("selection_marginal_value_weight", 0.20)),
        use_ownership_scenarios=bool(state.get("use_ownership_scenarios", True)),
        risk_profile=str(state.get("risk_profile", "Balanced leverage")),
        calibration_adjustment_strength=float(state.get("calibration_adjustment_strength", 0.0) or 0.0),
        use_stack_exposure_engine=bool(state.get("use_stack_exposure_engine", True)),
        stack_target_weight=float(state.get("stack_target_weight", 0.20)),
        use_stack_auto_caps=bool(state.get("use_stack_auto_caps", True)),
        use_stack_auto_mins=bool(state.get("use_stack_auto_mins", False)),
        use_portfolio_optimizer=bool(state.get("use_portfolio_optimizer", True)),
        portfolio_time_limit_seconds=int(state.get("portfolio_time_limit_seconds", 45)),
    )
    config.correlation.teammate_base = float(state.get("teammate_corr", config.correlation.teammate_base))
    config.correlation.pitcher_vs_opposing = float(state.get("pitcher_vs_opposing", config.correlation.pitcher_vs_opposing))
    config.correlation.copula_nu = int(state.get("copula_nu", config.correlation.copula_nu))
    config.use_stratified = bool(state.get("use_stratified", False))
    return config


def _select_simulation_portfolio(
    contest_result,
    optimizer_df: pd.DataFrame,
    lineups,
    lineup_df: Optional[pd.DataFrame],
    sim_config: SimulationConfig,
    stack_plan: Optional[pd.DataFrame] = None,
):
    stack_plan = stack_plan if stack_plan is not None else build_stack_exposure_plan(optimizer_df)
    team_stack_weights = (
        stack_plan_weights(stack_plan)
        if sim_config.use_stack_exposure_engine
        else None
    )
    stack_min_caps, stack_max_caps = stack_cap_maps(stack_plan)
    pitcher_ids = set(
        optimizer_df.loc[
            optimizer_df["player_type"].astype(str).str.lower() == "pitcher",
            "fd_player_id",
        ].astype(str).tolist()
    )
    portfolio = select_portfolio(
        contest_result,
        num_lineups=min(sim_config.num_candidates, len(lineups)),
        selection_metric=sim_config.selection_metric,
        max_overlap=sim_config.max_overlap,
        max_batter_exposure=sim_config.max_batter_exposure,
        max_pitcher_exposure=sim_config.max_pitcher_exposure,
        pitcher_ids=pitcher_ids,
        diversity_weight=sim_config.diversity_weight,
        min_batter_exposure=sim_config.min_batter_exposure,
        min_pitcher_exposure=sim_config.min_pitcher_exposure,
        min_stack_exposure=sim_config.min_stack_exposure,
        max_stack_exposure=sim_config.max_stack_exposure,
        selection_leverage_weight=sim_config.selection_leverage_weight,
        selection_ownership_weight=sim_config.selection_ownership_weight,
        selection_duplication_weight=sim_config.selection_duplication_weight,
        selection_scenario_weight=sim_config.selection_scenario_weight,
        selection_marginal_value_weight=sim_config.selection_marginal_value_weight,
        stack_team_weights=team_stack_weights,
        stack_target_weight=sim_config.stack_target_weight,
        stack_team_min_exposures=stack_min_caps if sim_config.use_stack_auto_mins else None,
        stack_team_max_exposures=stack_max_caps if sim_config.use_stack_auto_caps else None,
        use_portfolio_optimizer=sim_config.use_portfolio_optimizer,
        portfolio_time_limit_seconds=sim_config.portfolio_time_limit_seconds,
    )
    portfolio_df = portfolio.to_dataframe() if portfolio.selected else pd.DataFrame()
    raw_ids: List[int] = []
    if not portfolio_df.empty and "lineup_id" in portfolio_df.columns:
        raw_ids = [int(value) for value in portfolio_df["lineup_id"].tolist()]
    selected_ids = [idx + 1 for idx in raw_ids]
    selected_players = pd.DataFrame()
    if selected_ids and lineup_df is not None and not lineup_df.empty:
        selected_players = lineup_df[lineup_df["lineup_id"].isin(selected_ids)].copy()
    selected_objects = []
    if raw_ids:
        for idx in raw_ids:
            if 0 <= idx < len(lineups):
                selected_objects.append(lineups[idx])
    summary = {
        "win_rate": portfolio.portfolio_win_rate,
        "top1": portfolio.portfolio_top1pct_rate,
        "cash": portfolio.portfolio_cash_rate,
        "roi": portfolio.portfolio_expected_roi,
        "entry_fee": sim_config.entry_fee,
        "contest_entries": sim_config.contest_entries,
        "stack_exposure": portfolio.stack_exposure,
        "stack_engine_enabled": sim_config.use_stack_exposure_engine,
        "payout_ev_sum": float(sum(float(getattr(lineup, "payout_ev", 0.0) or 0.0) for lineup in portfolio.selected)),
        "avg_duplication_score": float(np.mean([float(getattr(lineup, "duplication_score", 0.0) or 0.0) for lineup in portfolio.selected])) if portfolio.selected else 0.0,
        "avg_estimated_dupes": float(np.mean([float(getattr(lineup, "estimated_dupes", 0.0) or 0.0) for lineup in portfolio.selected])) if portfolio.selected else 0.0,
        "avg_scenario_score": float(np.mean([float(getattr(lineup, "ownership_scenario_score", 0.0) or 0.0) for lineup in portfolio.selected])) if portfolio.selected else 0.0,
        "worst_case_duplication": float(np.max([float(getattr(lineup, "worst_case_duplication_score", 0.0) or 0.0) for lineup in portfolio.selected])) if portfolio.selected else 0.0,
    }
    return portfolio, portfolio_df, summary, selected_players, selected_ids, selected_objects


def _simulation_cache_key(optimizer_df: pd.DataFrame, sim_config: SimulationConfig, salary_cap: int) -> str:
    key_cols = [
        col
        for col in [
            "fd_player_id",
            "proj_fd_mean",
            "proj_fd_floor",
            "proj_fd_ceiling",
            "proj_fd_ownership",
            "team_code",
            "opponent_code",
            "player_type",
            "salary",
        ]
        if col in optimizer_df.columns
    ]
    slate_hash = pd.util.hash_pandas_object(
        optimizer_df[key_cols].astype(str).sort_values("fd_player_id") if "fd_player_id" in key_cols else optimizer_df[key_cols].astype(str),
        index=False,
    ).values.tobytes()
    payload = {
        "slate": hashlib.sha256(slate_hash).hexdigest(),
        "num_simulations": sim_config.num_simulations,
        "seed": sim_config.seed,
        "volatility_scale": sim_config.volatility_scale,
        "use_antithetic": sim_config.use_antithetic,
        "use_stratified": sim_config.use_stratified,
        "num_strata": sim_config.num_strata,
        "field_size": sim_config.num_field_lineups,
        "salary_cap": salary_cap,
        "correlation": asdict(sim_config.correlation),
        "field_quality": [
            sim_config.field_quality_shark_pct,
            sim_config.field_quality_rec_pct,
            sim_config.field_quality_random_pct,
        ],
        "field_realism": [
            sim_config.use_field_realism,
            sim_config.field_realism_value_bias,
            sim_config.field_realism_train_bias,
            sim_config.field_realism_upside_bias,
            sim_config.field_salary_target_shark,
            sim_config.field_salary_target_rec,
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _get_simulation_cache() -> Dict[str, Dict[str, object]]:
    try:
        return st.session_state.setdefault("simulation_runtime_cache", {})
    except Exception:
        return _RUNTIME_SIM_CACHE


def _store_simulation_cache(key: str, value: Dict[str, object], max_entries: int = 3) -> None:
    cache = _get_simulation_cache()
    cache[key] = value
    while len(cache) > max_entries:
        oldest = next(iter(cache.keys()))
        cache.pop(oldest, None)


def _run_simulation_stack(
    optimizer_df: pd.DataFrame,
    lineups,
    lineup_df: Optional[pd.DataFrame],
    sim_config: SimulationConfig,
    salary_cap: int,
    use_cache: bool = True,
):
    cache_key = _simulation_cache_key(optimizer_df, sim_config, salary_cap)
    cached = _get_simulation_cache().get(cache_key) if use_cache else None
    if cached:
        distributions = cached["distributions"]
        correlation_model = cached["correlation_model"]
        slate_sim = cached["slate_sim"]
        field_sim = cached["field_sim"]
        stack_plan = cached["stack_plan"]
    else:
        distributions = fit_player_distributions(optimizer_df, sim_config.volatility_scale)
        correlation_model = build_correlation_matrix(optimizer_df, sim_config.correlation)
        slate_sim = simulate_slate(
            distributions,
            correlation_model,
            num_simulations=sim_config.num_simulations,
            seed=sim_config.seed,
            use_antithetic=sim_config.use_antithetic,
            use_stratified=sim_config.use_stratified,
            num_strata=sim_config.num_strata,
        )
        quality_mix = FieldQualityMix(
            shark_pct=sim_config.field_quality_shark_pct,
            rec_pct=sim_config.field_quality_rec_pct,
            random_pct=sim_config.field_quality_random_pct,
        )
        realism_profile = FieldRealismProfile(
            value_chalk_bias=sim_config.field_realism_value_bias if sim_config.use_field_realism else 0.0,
            optimizer_train_bias=sim_config.field_realism_train_bias if sim_config.use_field_realism else 0.0,
            upside_chalk_bias=sim_config.field_realism_upside_bias if sim_config.use_field_realism else 0.0,
            shark_salary_target=sim_config.field_salary_target_shark,
            rec_salary_target=sim_config.field_salary_target_rec,
        )
        field_sim = simulate_field(
            optimizer_df,
            num_opponent_lineups=sim_config.num_field_lineups,
            salary_cap=salary_cap,
            seed=sim_config.seed,
            position_constraints=True,
            quality_mix=quality_mix,
            realism_profile=realism_profile,
        )
        stack_plan = build_stack_exposure_plan(optimizer_df)
        if use_cache:
            _store_simulation_cache(
                cache_key,
                {
                    "distributions": distributions,
                    "correlation_model": correlation_model,
                    "slate_sim": slate_sim,
                    "field_sim": field_sim,
                    "stack_plan": stack_plan,
                },
            )
    field_profile = load_field_duplication_profile(FIELD_DUPLICATION_PROFILE_PATH)
    contest_result = simulate_contest(
        lineups,
        slate_sim,
        field_sim,
        entry_fee=sim_config.entry_fee,
        payout_structure=sim_config.payout_structure,
        contest_entries=sim_config.contest_entries,
        field_duplication_profile=field_profile,
    )
    scenario_df = pd.DataFrame()
    if sim_config.use_ownership_scenarios:
        scenario_df = apply_ownership_scenario_metrics(
            contest_result,
            lineups,
            optimizer_df,
            field_size=sim_config.contest_entries,
            field_profile=field_profile,
        )
    contest_df = contest_result.to_dataframe().sort_values(
        sim_config.selection_metric, ascending=False
    )
    (
        portfolio,
        portfolio_df,
        summary,
        selected_players,
        selected_ids,
        selected_objects,
    ) = _select_simulation_portfolio(
        contest_result,
        optimizer_df,
        lineups,
        lineup_df,
        sim_config,
        stack_plan,
    )
    return (
        contest_df,
        portfolio_df,
        summary,
        selected_players,
        selected_ids,
        slate_sim,
        correlation_model,
        selected_objects,
        stack_plan,
        contest_result,
        scenario_df,
    )


def _strategy_comparison_configs(base_config: SimulationConfig) -> list[tuple[str, SimulationConfig]]:
    time_limit = min(int(getattr(base_config, "portfolio_time_limit_seconds", 45) or 45), 15)
    return [
        ("Current settings", replace(base_config, portfolio_time_limit_seconds=time_limit)),
        (
            "Payout EV balanced",
            replace(
                base_config,
                selection_metric="payout_ev",
                diversity_weight=0.35,
                selection_leverage_weight=0.18,
                selection_ownership_weight=0.12,
                selection_duplication_weight=0.22,
                selection_scenario_weight=0.14,
                selection_marginal_value_weight=0.22,
                stack_target_weight=0.22,
                portfolio_time_limit_seconds=time_limit,
            ),
        ),
        (
            "Aggressive leverage",
            replace(
                base_config,
                selection_metric="payout_ev",
                diversity_weight=0.42,
                max_batter_exposure=min(float(base_config.max_batter_exposure), 0.34),
                max_pitcher_exposure=min(float(base_config.max_pitcher_exposure), 0.52),
                selection_leverage_weight=0.35,
                selection_ownership_weight=0.20,
                selection_duplication_weight=0.28,
                selection_scenario_weight=0.22,
                selection_marginal_value_weight=0.30,
                stack_target_weight=0.30,
                use_stack_exposure_engine=True,
                portfolio_time_limit_seconds=time_limit,
            ),
        ),
        (
            "Chalk fade unique",
            replace(
                base_config,
                selection_metric="payout_ev",
                diversity_weight=0.48,
                max_batter_exposure=min(float(base_config.max_batter_exposure), 0.30),
                max_pitcher_exposure=min(float(base_config.max_pitcher_exposure), 0.48),
                selection_leverage_weight=0.28,
                selection_ownership_weight=0.30,
                selection_duplication_weight=0.45,
                selection_scenario_weight=0.28,
                selection_marginal_value_weight=0.36,
                stack_target_weight=0.28,
                use_stack_exposure_engine=True,
                portfolio_time_limit_seconds=time_limit,
            ),
        ),
        (
            "Stack plan overweight",
            replace(
                base_config,
                selection_metric="payout_ev",
                diversity_weight=0.35,
                selection_leverage_weight=0.22,
                selection_ownership_weight=0.12,
                selection_duplication_weight=0.25,
                selection_scenario_weight=0.18,
                selection_marginal_value_weight=0.30,
                stack_target_weight=0.45,
                use_stack_exposure_engine=True,
                use_stack_auto_caps=True,
                portfolio_time_limit_seconds=time_limit,
            ),
        ),
        (
            "Ceiling chase",
            replace(
                base_config,
                selection_metric="p99_score",
                diversity_weight=0.40,
                selection_leverage_weight=0.15,
                selection_ownership_weight=0.08,
                selection_duplication_weight=0.18,
                selection_scenario_weight=0.10,
                selection_marginal_value_weight=0.25,
                stack_target_weight=0.18,
                portfolio_time_limit_seconds=time_limit,
            ),
        ),
    ]


def _run_strategy_comparison(
    contest_result,
    optimizer_df: pd.DataFrame,
    lineups,
    lineup_df: Optional[pd.DataFrame],
    base_config: SimulationConfig,
    stack_plan: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    rows: list[dict[str, object]] = []
    portfolios: Dict[str, Dict[str, object]] = {}
    for label, config in _strategy_comparison_configs(base_config):
        (
            portfolio,
            portfolio_df,
            summary,
            selected_players,
            selected_ids,
            selected_objects,
        ) = _select_simulation_portfolio(
            contest_result,
            optimizer_df,
            lineups,
            lineup_df,
            config,
            stack_plan,
        )
        avg_own = float(pd.to_numeric(portfolio_df.get("total_ownership"), errors="coerce").mean()) if not portfolio_df.empty else 0.0
        avg_lev = float(pd.to_numeric(portfolio_df.get("leverage_score"), errors="coerce").mean()) if not portfolio_df.empty else 0.0
        avg_dup = float(pd.to_numeric(portfolio_df.get("duplication_score"), errors="coerce").mean()) if not portfolio_df.empty else 0.0
        avg_dupes = float(pd.to_numeric(portfolio_df.get("estimated_dupes"), errors="coerce").mean()) if not portfolio_df.empty else 0.0
        avg_scenario = float(pd.to_numeric(portfolio_df.get("ownership_scenario_score"), errors="coerce").mean()) if not portfolio_df.empty else 0.0
        rows.append(
            {
                "strategy": label,
                "selected": portfolio.num_selected,
                "selection_metric": config.selection_metric,
                "payout_ev_sum": summary.get("payout_ev_sum", 0.0),
                "top1_rate": portfolio.portfolio_top1pct_rate,
                "win_rate": portfolio.portfolio_win_rate,
                "cash_rate": portfolio.portfolio_cash_rate,
                "roi_sum": portfolio.portfolio_expected_roi,
                "avg_ownership": avg_own,
                "avg_leverage": avg_lev,
                "avg_dup_score": avg_dup,
                "avg_est_dupes": avg_dupes,
                "avg_scenario_score": avg_scenario,
                "max_player_exposure": portfolio.max_player_exposure,
                "avg_pairwise_overlap": portfolio.avg_pairwise_overlap,
                "top_stacks": _stack_exposure_label(portfolio.stack_exposure),
            }
        )
        portfolios[label] = {
            "portfolio_df": portfolio_df,
            "summary": summary,
            "selected_players": selected_players,
            "selected_ids": selected_ids,
            "selected_lineup_objects": selected_objects,
            "sim_config": config,
        }
    comparison_df = pd.DataFrame(rows)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values(["payout_ev_sum", "top1_rate"], ascending=False)
    return comparison_df, portfolios


def _stack_exposure_label(stack_exposure: Dict[str, float], limit: int = 4) -> str:
    if not stack_exposure:
        return ""
    pieces = []
    for team, pct in sorted(stack_exposure.items(), key=lambda item: item[1], reverse=True)[:limit]:
        pieces.append(f"{team} {pct:.0%}")
    return ", ".join(pieces)


def _render_strategy_comparison(
    workflow: Dict,
    sim_results: Dict,
    optimizer_df: pd.DataFrame,
) -> None:
    contest_result = sim_results.get("contest_result")
    base_config = sim_results.get("sim_config")
    stack_plan = sim_results.get("stack_plan")
    if contest_result is None or base_config is None or stack_plan is None:
        return
    lineups_for_comparison = sim_results.get("candidate_lineups") or workflow.get("lineups")
    lineup_df_for_comparison = sim_results.get("candidate_lineup_df")
    if lineup_df_for_comparison is None:
        lineup_df_for_comparison = workflow.get("lineups_df")
    with st.expander("Compare strategy builds", expanded=False):
        st.caption("Re-selects the final portfolio from the same simulation under several strategy profiles, without rerunning Monte Carlo.")
        if st.button("Compare Strategy Builds"):
            try:
                with st.spinner("Comparing final portfolio strategies..."):
                    comparison_df, portfolios = _run_strategy_comparison(
                        contest_result,
                        optimizer_df,
                        lineups_for_comparison,
                        lineup_df_for_comparison,
                        base_config,
                        stack_plan,
                    )
                sim_results["strategy_comparison"] = comparison_df
                sim_results["strategy_portfolios"] = portfolios
                st.success("Strategy comparison complete.")
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Strategy comparison failed: {exc}")
        comparison_df = sim_results.get("strategy_comparison")
        if isinstance(comparison_df, pd.DataFrame) and not comparison_df.empty:
            display = comparison_df.copy()
            st.dataframe(display, width="stretch", hide_index=True)
            portfolios = sim_results.get("strategy_portfolios") or {}
            choices = list(portfolios.keys())
            if choices:
                best_default = str(display.iloc[0]["strategy"]) if "strategy" in display.columns else choices[0]
                selected_strategy = st.selectbox(
                    "Strategy to activate",
                    choices,
                    index=choices.index(best_default) if best_default in choices else 0,
                )
                if st.button("Use Selected Strategy Portfolio", type="primary"):
                    chosen = portfolios.get(selected_strategy) or {}
                    sim_results.update(
                        {
                            "portfolio_df": chosen.get("portfolio_df", pd.DataFrame()),
                            "summary": chosen.get("summary", {}),
                            "selected_players": chosen.get("selected_players", pd.DataFrame()),
                            "selected_ids": chosen.get("selected_ids", []),
                            "selected_lineup_objects": chosen.get("selected_lineup_objects", []),
                            "sim_config": chosen.get("sim_config", base_config),
                            "active_strategy_build": selected_strategy,
                        }
                    )
                    st.success(f"Activated {selected_strategy}.")
                    st.rerun()


def _save_uploaded_file(uploaded, directory: Path) -> Path:
    path = directory / uploaded.name
    with open(path, "wb") as temp_file:
        temp_file.write(uploaded.getbuffer())
    return path


class _FileProxy:
    """Wraps saved bytes on disk so they look like a Streamlit UploadedFile."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self.name = path.name

    def getbuffer(self):
        return self._path.read_bytes()

    def read(self):
        return self._path.read_bytes()


def _persist_slate_files(
    fanduel_file,
    bpp_files,
    vegas_file,
    batting_file,
    handedness_file,
    recent_file,
    ownership_files,
    projection_files,
    lineup_paste: str,
) -> None:
    """Save all uploaded slate files to UPLOAD_CACHE_DIR and write a manifest."""
    import datetime, json, shutil

    # Clear old cached files before saving new ones (files only — never
    # touch subdirectories, and never die over a locked file)
    for f in UPLOAD_CACHE_DIR.iterdir():
        if f.is_file() and f.name != "manifest.json":
            try:
                f.unlink()
            except OSError:
                pass

    manifest: Dict[str, object] = {
        "saved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "fanduel": None,
        "bpp_files": [],
        "vegas": None,
        "batting_orders": None,
        "handedness": None,
        "recent_stats": None,
        "ownership_files": [],
        "projection_files": [],
        "lineup_paste": lineup_paste or "",
    }

    def _cache(uploaded) -> str:
        dest = UPLOAD_CACHE_DIR / uploaded.name
        dest.write_bytes(uploaded.getbuffer())
        return uploaded.name

    if fanduel_file:
        manifest["fanduel"] = _cache(fanduel_file)
    for f in (bpp_files or []):
        manifest["bpp_files"].append(_cache(f))
    if vegas_file:
        manifest["vegas"] = _cache(vegas_file)
    if batting_file:
        manifest["batting_orders"] = _cache(batting_file)
    if handedness_file:
        manifest["handedness"] = _cache(handedness_file)
    if recent_file:
        manifest["recent_stats"] = _cache(recent_file)
    for f in (ownership_files or []):
        manifest["ownership_files"].append(_cache(f))
    for f in (projection_files or []):
        manifest["projection_files"].append(_cache(f))

    UPLOAD_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


_DOWNLOADS_DIR = Path.home() / "Downloads"


def _detect_downloads_slate_files() -> Optional[Dict]:
    """Find the freshest complete slate-file set in the Downloads folder.

    Looks for the newest BallparkPal quartet, any 'Ballpark DFS' optimizer
    exports downloaded within 6 hours of it, and the newest FanDuel CSV —
    preferring the contest entries-upload-template (which enables direct
    upload export) over the plain players list when both are fresh.
    """
    if not _DOWNLOADS_DIR.exists():
        return None

    def newest(pattern: str) -> Optional[Path]:
        files = [p for p in _DOWNLOADS_DIR.glob(pattern) if p.is_file()]
        return max(files, key=lambda p: p.stat().st_mtime) if files else None

    quartet: Dict[str, Optional[Path]] = {}
    for stem in ("Batters", "Pitchers", "Games", "Teams"):
        quartet[stem] = newest(f"BallparkPal_{stem}*.xlsx")
    if not all(quartet.values()):
        return None
    anchor = max(p.stat().st_mtime for p in quartet.values())

    dfs_files = sorted(
        (
            p
            for p in _DOWNLOADS_DIR.glob("Ballpark DFS*.xlsx")
            if p.is_file() and abs(p.stat().st_mtime - anchor) <= 6 * 3600
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:2]

    players_list = newest("FanDuel-MLB-*players-list*.csv")
    template = newest("FanDuel-MLB-*entries-upload-template*.csv")

    def _fresh(p: Optional[Path]) -> bool:
        return p is not None and abs(p.stat().st_mtime - anchor) <= 12 * 3600

    if _fresh(template):
        fanduel = template
    elif _fresh(players_list):
        fanduel = players_list
    else:
        fanduel = players_list or template
    if fanduel is None:
        return None

    import time as _time

    age_hours = (_time.time() - anchor) / 3600.0
    return {
        "bpp": [quartet[s] for s in ("Batters", "Pitchers", "Games", "Teams")],
        "dfs": dfs_files,
        "fanduel": fanduel,
        "fanduel_is_template": fanduel == template and template is not None,
        "age_hours": age_hours,
    }


def _load_slate_cache() -> Optional[Dict]:
    """Return the cache manifest dict if it exists and all required files are present, else None."""
    import json
    if not UPLOAD_MANIFEST.exists():
        return None
    try:
        manifest = json.loads(UPLOAD_MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return None
    # Require at minimum the FanDuel file and at least one BPP file
    if not manifest.get("fanduel"):
        return None
    if not manifest.get("bpp_files"):
        return None
    # Verify all listed files actually exist on disk
    all_files = (
        [manifest["fanduel"]]
        + manifest["bpp_files"]
        + ([manifest["vegas"]] if manifest.get("vegas") else [])
        + ([manifest["batting_orders"]] if manifest.get("batting_orders") else [])
        + ([manifest["handedness"]] if manifest.get("handedness") else [])
        + ([manifest["recent_stats"]] if manifest.get("recent_stats") else [])
        + manifest.get("ownership_files", [])
        + manifest.get("projection_files", [])
    )
    for fname in all_files:
        if not (UPLOAD_CACHE_DIR / fname).exists():
            return None
    return manifest


def _proxies_from_manifest(manifest: Dict) -> Dict:
    """Convert a manifest dict into _FileProxy objects ready for _process_slate."""

    def _p(name: Optional[str]):
        return _FileProxy(UPLOAD_CACHE_DIR / name) if name else None

    return {
        "fanduel_file": _p(manifest.get("fanduel")),
        "bpp_files": [_FileProxy(UPLOAD_CACHE_DIR / n) for n in manifest.get("bpp_files", [])],
        "vegas_file": _p(manifest.get("vegas")),
        "batting_file": _p(manifest.get("batting_orders")),
        "handedness_file": _p(manifest.get("handedness")),
        "recent_file": _p(manifest.get("recent_stats")),
        "ownership_files": [_FileProxy(UPLOAD_CACHE_DIR / n) for n in manifest.get("ownership_files", [])],
        "projection_files": [_FileProxy(UPLOAD_CACHE_DIR / n) for n in manifest.get("projection_files", [])],
        "lineup_paste": manifest.get("lineup_paste", ""),
    }


def _combine_lineups(lineups) -> pd.DataFrame:
    rows = []
    for idx, lineup in enumerate(lineups, start=1):
        df = lineup.dataframe.copy()
        df.insert(0, "lineup_id", idx)
        df.insert(1, "lineup_salary", lineup.total_salary)
        df.insert(2, "lineup_projection", lineup.total_projection)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)



def _activate_sim_portfolio(workflow: Dict, selected_lineups) -> None:
    if not selected_lineups:
        st.warning("Simulation portfolio has no lineups to activate.")
        return
    workflow.setdefault("optimizer_lineups_backup", workflow.get("lineups"))
    workflow.setdefault("optimizer_lineups_df_backup", workflow.get("lineups_df"))
    workflow["lineups"] = selected_lineups
    workflow["lineups_df"] = _combine_lineups(selected_lineups)
    workflow["active_lineup_source"] = "simulation"
    st.success("Simulation portfolio is now active for Steps 5-6.")



def _restore_optimizer_lineups(workflow: Dict) -> None:
    backup = workflow.get("optimizer_lineups_backup")
    if not backup:
        st.info("Original optimizer lineups not available.")
        return
    workflow["lineups"] = backup
    workflow["lineups_df"] = workflow.get("optimizer_lineups_df_backup", _combine_lineups(backup))
    workflow["active_lineup_source"] = "optimizer"
    st.success("Restored optimizer-generated lineups.")


def _write_multiple(uploaded_files: Iterable, directory: Path) -> List[Path]:
    paths: List[Path] = []
    for file in uploaded_files:
        paths.append(_save_uploaded_file(file, directory))
    return paths


def _looks_like_ballpark_optimizer_projection(path: Path) -> bool:
    """Detect Ballpark DFS Optimizer workbooks that carry Bust/Median/Upside."""

    suffix = path.suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        return False
    name = path.name.lower()
    # BallparkPal renamed these exports mid-2026: old "Ballpark DFS Optimizer ..."
    # and new "Ballpark DFS  Ballpark Pal (N).xlsx" both carry Bust/Median/Upside.
    if "ballpark dfs optimizer" in name or ("ballpark dfs" in name and "ballpark pal" in name):
        return True
    try:
        preview = pd.read_excel(path, header=None, nrows=4)
    except Exception:
        return False
    text = " ".join(
        str(value).strip().lower()
        for value in preview.to_numpy().ravel()
        if pd.notna(value)
    )
    return (
        "ballpark dfs optimizer" in text
        or (
            "bust" in text
            and "median" in text
            and "upside" in text
            and ("points" in text or "players" in text)
        )
    )


def _auto_detect_projection_paths(bpp_dir: Path, existing_paths: Sequence[Path]) -> List[Path]:
    """Find sim/projection workbooks accidentally uploaded in the BallparkPal bucket."""

    existing_names = {Path(path).name.lower() for path in existing_paths}
    detected: List[Path] = []
    for candidate in sorted(bpp_dir.iterdir(), key=lambda p: p.name.lower()):
        if not candidate.is_file() or candidate.name.lower() in existing_names:
            continue
        if _looks_like_ballpark_optimizer_projection(candidate):
            detected.append(candidate)
    return detected


def _prepare_bpp_directory(uploaded_files: Sequence) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="bpp_"))
    if not uploaded_files:
        raise ValueError("At least one BallparkPal file is required.")
    for file in uploaded_files:
        _save_uploaded_file(file, temp_dir)
    return temp_dir


def _parse_weights(value: Optional[str], count: int) -> Optional[List[float]]:
    if not value:
        return None
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if len(tokens) != count:
        raise ValueError("Number of weights must match ownership sources")
    weights: List[float] = []
    for token in tokens:
        weights.append(float(token))
    return weights


def _parse_recency_blend(value: Optional[str]) -> Optional[Tuple[float, float]]:
    if not value:
        return None
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if len(tokens) != 2:
        raise ValueError("Recency blend must include exactly two comma-separated numbers")
    season_weight, recent_weight = float(tokens[0]), float(tokens[1])
    if season_weight + recent_weight <= 0:
        raise ValueError("Recency blend weights must sum to a positive value")
    return season_weight, recent_weight


def _weight_preset_values(preset: Optional[str], source_count: int) -> Optional[List[float]]:
    if not source_count:
        return None
    if preset == MANUAL_WEIGHT_PRESET:
        return None
    if not preset or preset not in WEIGHT_PRESETS:
        preset = DEFAULT_WEIGHT_PRESET
    preset_def = WEIGHT_PRESETS.get(preset)
    if not preset_def:
        return None
    generator = preset_def.get("generator")
    if generator is None:
        return None
    raw_weights = [float(value) for value in generator(source_count)]
    total = sum(raw_weights)
    if total <= 0:
        return None
    return [value / total for value in raw_weights]


def _weight_preset_preview(preset: Optional[str], sources: Sequence) -> str:
    weights = _weight_preset_values(preset, len(sources)) if sources else None
    if not weights:
        return ""
    labels = [getattr(upload, "name", f"Source {idx + 1}") for idx, upload in enumerate(sources)]
    preview = []
    for idx, weight in enumerate(weights):
        label = labels[idx] if idx < len(labels) else f"Source {idx + 1}"
        preview.append(f"{label}: {weight:.2f}")
    return ", ".join(preview)


def _merge_optional_sources(
    combined: pd.DataFrame,
    vegas_path: Optional[Path],
    batting_orders_path: Optional[Path],
    handedness_path: Optional[Path],
    recent_stats_path: Optional[Path],
) -> Tuple[pd.DataFrame, List[str]]:
    messages: List[str] = []
    alias_map = None

    if batting_orders_path:
        order_loader = BattingOrderLoader(batting_orders_path)
        orders = order_loader.load(alias_map=alias_map)
        combined = combined.merge(
            orders.entries.rename(columns={"batting_order_position": "_batting_order_position"}),
            on=["team_code", "canonical_name"],
            how="left",
        )
        combined["batting_order_position"] = combined["_batting_order_position"].combine_first(
            combined["batting_order_position"]
        )
        combined.drop(columns=["_batting_order_position"], inplace=True)
        combined["is_confirmed_lineup"] = combined["batting_order_position"].notna()
        combined["batting_order_position"] = pd.to_numeric(
            combined["batting_order_position"], errors="coerce"
        ).astype("Int64")
        messages.append(
            f"Loaded batting orders for {orders.summary()['teams']} teams from {batting_orders_path.name}"
        )

    if handedness_path:
        hand_loader = HandednessLoader(handedness_path)
        handed = hand_loader.load(alias_map=alias_map)
        combined = combined.merge(
            handed.entries.rename(
                columns={"batter_hand": "_batter_hand", "pitcher_hand": "_pitcher_hand"}
            ),
            on=["team_code", "canonical_name"],
            how="left",
        )
        combined["batter_hand"] = combined["batter_hand"].astype(str).str.upper().str.strip()
        combined["pitcher_hand"] = combined["pitcher_hand"].astype(str).str.upper().str.strip()
        combined["batter_hand"] = combined["batter_hand"].where(
            combined["batter_hand"].isin(["L", "R", "S"]), ""
        )
        combined["pitcher_hand"] = combined["pitcher_hand"].where(
            combined["pitcher_hand"].isin(["L", "R"]), ""
        )
        combined["batter_hand"] = combined["batter_hand"].where(
            combined["batter_hand"].ne(""), combined["_batter_hand"].fillna("")
        )
        combined["pitcher_hand"] = combined["pitcher_hand"].where(
            combined["pitcher_hand"].ne(""), combined["_pitcher_hand"].fillna("")
        )
        combined.drop(columns=["_batter_hand", "_pitcher_hand"], inplace=True)
        messages.append(
            f"Loaded handedness reference ({handed.summary()['total']} players) from {handedness_path.name}"
        )

    if vegas_path:
        vegas_loader = VegasLoader(vegas_path)
        vegas_lines = vegas_loader.load()
        vegas_totals = vegas_lines.team_totals.drop_duplicates(subset=["team_code", "opponent_code"])
        combined = combined.merge(vegas_totals, on=["team_code", "opponent_code"], how="left")
        messages.append(
            f"Merged Vegas lines for {vegas_lines.summary()['games']} games"
        )

    if recent_stats_path:
        stats_loader = RecentStatsLoader(recent_stats_path)
        stats = stats_loader.load(alias_map=alias_map)
        combined = combined.merge(
            stats.entries,
            on=["team_code", "canonical_name"],
            how="left",
        )
        for col in ("recent_last7_fppg", "recent_last14_fppg", "recent_season_fppg"):
            combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0.0)
        messages.append(
            f"Integrated recent stats for {stats.summary()['total']} players"
        )

    return combined, messages


def _format_diagnostics(diag) -> pd.DataFrame:
    data = asdict(diag)
    formatted = {
        "Hitters": f"{data['hitters_matched']}/{data['hitters_total']} matched",
        "Pitchers": f"{data['pitchers_matched']}/{data['pitchers_total']} matched",
    }
    return pd.DataFrame.from_dict(formatted, orient="index", columns=["Value"])


def _apply_ownership_edge(df: pd.DataFrame) -> pd.DataFrame:
    if "proj_fd_ceiling" in df.columns and "proj_fd_ownership" in df.columns:
        denom = df["proj_fd_ownership"].replace(0, np.nan)
        df["ownership_edge"] = df["proj_fd_ceiling"] / denom
        df["ownership_edge"] = df["ownership_edge"].replace([np.inf, -np.inf], np.nan).fillna(
            df["proj_fd_ceiling"]
        )
    else:
        df["ownership_edge"] = np.nan
    return df


def _process_slate(
    fanduel_file,
    bpp_files,
    vegas_file,
    batting_file,
    handed_file,
    recent_file,
    ownership_files,
    projection_files,
    lineup_paste_text: str = "",
    projection_preset: Optional[str] = None,
    projection_weights_input: Optional[str] = None,
    projection_baseline_weight: float = 0.5,
    ownership_preset: Optional[str] = None,
    ownership_weights_input: Optional[str] = None,
    ownership_model_settings: Optional[Dict] = None,
    recency_blend_input: Optional[str] = None,
    platoon_opposite_boost: float = 1.08,
    platoon_same_penalty: float = 0.95,
    platoon_switch_boost: float = 1.03,
) -> Dict:
    if not fanduel_file:
        raise ValueError("FanDuel CSV is required.")
    if not bpp_files:
        raise ValueError("BallparkPal Excel files are required.")

    with tempfile.TemporaryDirectory(prefix="slate_tmp_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        fd_path = _save_uploaded_file(fanduel_file, temp_dir)
        # Extract template entry metadata for FanDuel upload format
        template_entries = extract_template_entries(fd_path)
        bpp_dir = temp_dir / "bpp"
        bpp_dir.mkdir(exist_ok=True)
        _write_multiple(bpp_files, bpp_dir)

        vegas_path = _save_uploaded_file(vegas_file, temp_dir) if vegas_file else None
        # Handle batting orders: paste text takes priority over CSV upload
        batting_path = None
        confirmed_player_names = set()
        confirmed_last_team = set()  # fallback: (canonical_last_name, team_code)
        if lineup_paste_text and lineup_paste_text.strip():
            from slate_optimizer.ingestion.batting_orders import parse_lineup_paste
            paste_df = parse_lineup_paste(lineup_paste_text)
            if not paste_df.empty:
                batting_path = temp_dir / "pasted_batting_orders.csv"
                paste_df.to_csv(batting_path, index=False)
                # Build set of canonical names for direct filtering
                from slate_optimizer.ingestion.text_utils import canonicalize_series as _canon
                confirmed_player_names = set(
                    _canon(paste_df["player_name"]).tolist()
                )
                # Build fallback set of (last_name, team) for players that don't match on full name
                _paste_last_names = _canon(paste_df["player_name"].str.split().str[-1])
                _paste_teams = paste_df["team"].str.upper().str.strip()
                confirmed_last_team = set(zip(_paste_last_names, _paste_teams))
        if batting_path is None and batting_file:
            batting_path = _save_uploaded_file(batting_file, temp_dir)
        # If we have a batting orders CSV (from upload or API fetch) but no paste-text names yet,
        # read it to populate the confirmed-starters filter — so non-starters get removed.
        if batting_path is not None and not confirmed_player_names:
            try:
                _bo_df = pd.read_csv(batting_path)
                if "player_name" in _bo_df.columns and "team" in _bo_df.columns:
                    from slate_optimizer.ingestion.text_utils import canonicalize_series as _canon_bo
                    confirmed_player_names = set(_canon_bo(_bo_df["player_name"]).tolist())
                    _bo_last = _canon_bo(_bo_df["player_name"].str.split().str[-1])
                    _bo_teams = _bo_df["team"].str.upper().str.strip()
                    confirmed_last_team = set(zip(_bo_last, _bo_teams))
            except Exception:
                pass  # Non-fatal — batting orders used for positions only
        handed_path = _save_uploaded_file(handed_file, temp_dir) if handed_file else None
        recent_path = _save_uploaded_file(recent_file, temp_dir) if recent_file else None

        ownership_paths = []
        if ownership_files:
            for uploaded in ownership_files:
                ownership_paths.append(_save_uploaded_file(uploaded, temp_dir))

        projection_paths = []
        if projection_files:
            for uploaded in projection_files:
                projection_paths.append(_save_uploaded_file(uploaded, temp_dir))
        auto_projection_paths = _auto_detect_projection_paths(bpp_dir, projection_paths)
        projection_paths.extend(auto_projection_paths)
        upload_messages: List[str] = []
        if auto_projection_paths:
            names = ", ".join(path.name for path in auto_projection_paths)
            upload_messages.append(
                "Auto-detected Ballpark DFS Optimizer projection source(s) for "
                f"boom/bust data: {names}"
            )

        manual_weight_override = bool(ownership_weights_input and ownership_paths)
        manual_preset_selected = ownership_preset == MANUAL_WEIGHT_PRESET
        ownership_weights = None
        if ownership_paths:
            if manual_preset_selected and not manual_weight_override:
                raise ValueError("Enter ownership weights when using the manual preset.")
            if manual_weight_override:
                ownership_weights = _parse_weights(ownership_weights_input, len(ownership_paths))
            else:
                ownership_weights = _weight_preset_values(ownership_preset, len(ownership_paths))

        model_defaults = OwnershipModelConfig()
        ownership_model_settings = ownership_model_settings or {}
        model_values_raw = ownership_model_settings.get("values") or {}
        model_values: Dict[str, float] = {}
        for field, *_ in OWNERSHIP_MODEL_FIELDS:
            model_values[field] = float(model_values_raw.get(field, getattr(model_defaults, field)))
        min_pct = max(0.0, model_values.get("min_pct", model_defaults.min_pct))
        max_pct = max(model_values.get("max_pct", model_defaults.max_pct), min_pct + 1e-6)
        model_values["min_pct"] = min_pct
        model_values["max_pct"] = max_pct
        ownership_model_config = OwnershipModelConfig(
            salary_weight=model_values.get("salary_weight", model_defaults.salary_weight),
            projection_weight=model_values.get("projection_weight", model_defaults.projection_weight),
            value_weight=model_values.get("value_weight", model_defaults.value_weight),
            team_weight=model_values.get("team_weight", model_defaults.team_weight),
            name_weight=model_values.get("name_weight", model_defaults.name_weight),
            position_weight=model_values.get("position_weight", model_defaults.position_weight),
            min_pct=min_pct,
            max_pct=max_pct,
        )
        ownership_model_summary = {
            "preset": ownership_model_settings.get("preset") or DEFAULT_MODEL_PRESET,
            "manual_override": bool(ownership_model_settings.get("manual_override")),
            "values": model_values,
            "min_pct": min_pct,
            "max_pct": max_pct,
        }

        manual_projection_override = bool(projection_weights_input and projection_paths)
        manual_projection_preset = projection_preset == MANUAL_WEIGHT_PRESET
        projection_weights = None
        if projection_paths:
            if manual_projection_preset and not manual_projection_override:
                raise ValueError("Enter projection weights when using the manual preset.")
            if manual_projection_override:
                projection_weights = _parse_weights(projection_weights_input, len(projection_paths))
            else:
                projection_weights = _weight_preset_values(projection_preset, len(projection_paths))
        projection_baseline_weight = float(
            projection_baseline_weight if projection_baseline_weight is not None else 1.0
        )
        if projection_baseline_weight < 0:
            raise ValueError("Baseline weight must be non-negative.")

        recency_blend = _parse_recency_blend(recency_blend_input)

        bpp_loader = BallparkPalLoader(bpp_dir)
        bundle = bpp_loader.load_bundle()

        fd_loader = FanduelCSVLoader(fd_path)
        fd_players = fd_loader.load()

        combined, diagnostics = build_player_dataset(bundle, fd_players.players)
        defaults = pd.DataFrame({
            "is_confirmed_lineup": False,
            "batting_order_position": pd.array([pd.NA] * len(combined), dtype="Int64"),
            "batter_hand": "",
            "pitcher_hand": "",
            "recent_last7_fppg": 0.0,
            "recent_last14_fppg": 0.0,
            "recent_season_fppg": 0.0,
        }, index=combined.index)
        combined = pd.concat([combined, defaults], axis=1)

        combined, optional_messages = _merge_optional_sources(
            combined,
            vegas_path,
            batting_path,
            handed_path,
            recent_path,
        )
        optional_messages = upload_messages + optional_messages

        # Backfill handedness from the BallparkPal sim exports (BatterStand /
        # PitcherHand columns) for players not covered by a handedness upload.
        # Without this, platoon adjustments silently no-op when no handedness
        # file is provided even though BPP already ships the data.
        if "bpp_batter_stand" in combined.columns:
            _bpp_stand = (
                combined["bpp_batter_stand"].fillna("").astype(str).str.upper().str.strip()
            ).replace({"B": "S"})
            _empty_bh = combined["batter_hand"].fillna("").astype(str).str.strip() == ""
            _fill_bh = _empty_bh & _bpp_stand.isin(["L", "R", "S"])
            combined.loc[_fill_bh, "batter_hand"] = _bpp_stand.loc[_fill_bh]
        if "bpp_pitcher_hand" in combined.columns:
            _bpp_ph = combined["bpp_pitcher_hand"].fillna("").astype(str).str.upper().str.strip()
            _empty_ph = combined["pitcher_hand"].fillna("").astype(str).str.strip() == ""
            _fill_ph = _empty_ph & _bpp_ph.isin(["L", "R"])
            combined.loc[_fill_ph, "pitcher_hand"] = _bpp_ph.loc[_fill_ph]

        # Filter to confirmed starters only when lineup data was provided
        if confirmed_player_names:
            from slate_optimizer.ingestion.text_utils import canonicalize_series as _canon2
            combined["_canon_check"] = _canon2(combined["full_name"])
            name_match = combined["_canon_check"].isin(confirmed_player_names)
            # Fallback: match on last_name + team_code for players not matched by full name
            if confirmed_last_team:
                _last_col = "last_name" if "last_name" in combined.columns else "full_name"
                _canon_last = _canon2(combined[_last_col].str.split().str[-1] if _last_col == "full_name" else combined[_last_col])
                _team_col = combined.get("team_code", combined.get("team", pd.Series([""] * len(combined)))).str.upper().str.strip()
                _last_team_pairs = list(zip(_canon_last, _team_col))
                _last_team_match = pd.Series([pair in confirmed_last_team for pair in _last_team_pairs], index=combined.index)
                name_match = name_match | _last_team_match
            # Also include merge-based matches
            if "is_confirmed_lineup" in combined.columns:
                name_match = name_match | combined["is_confirmed_lineup"]
            # Never drop a FanDuel-probable pitcher because the paste names someone
            # else (BPP game pages sometimes list an opener while FanDuel lists the
            # bulk pitcher — e.g. Englert vs. Jax). Pitchers here already passed the
            # probable filter in build_player_dataset when FD populates the column.
            if "probable_pitcher" in combined.columns:
                _prob = (
                    combined["probable_pitcher"].fillna("").astype(str).str.strip().str.upper()
                )
                _fd_probable = (
                    (combined["player_type"].str.lower() == "pitcher")
                    & _prob.isin({"YES", "Y", "TRUE", "1"})
                )
                _rescued = _fd_probable & ~name_match
                if _rescued.any():
                    _names = ", ".join(combined.loc[_rescued, "full_name"].astype(str))
                    optional_messages.append(
                        f"NOTE: kept FanDuel probable pitcher(s) missing from your lineup "
                        f"paste: {_names}. Your paste may list an opener where FanDuel "
                        f"lists the bulk pitcher — the FanDuel probable is the one you "
                        f"can actually roster."
                    )
                name_match = name_match | _fd_probable
            confirmed = combined[name_match]
            n_confirmed_pitchers = (confirmed["player_type"].str.lower() == "pitcher").sum()
            n_confirmed_hitters = (confirmed["player_type"].str.lower() != "pitcher").sum()
            n_pasted = len(confirmed_player_names)

            # Always filter when lineup data is provided — non-starters skew results
            before_count = len(combined)
            full_pool_backup = combined.drop(columns=["_canon_check"], errors="ignore").copy()
            combined = confirmed.drop(columns=["_canon_check"]).reset_index(drop=True)
            filtered_count = before_count - len(combined)
            optional_messages.append(
                f"Filtered to confirmed starters: {len(combined)} players "
                f"({n_confirmed_pitchers} pitchers + {n_confirmed_hitters} hitters, "
                f"{filtered_count} bench/inactive removed)"
            )
            # Check if pool is viable for optimization
            n_matched = n_confirmed_pitchers + n_confirmed_hitters
            match_rate = n_matched / max(n_pasted, 1)

            # Detect slate mismatch: compare teams in paste vs FanDuel pool
            _paste_teams = set()
            if hasattr(confirmed_last_team, '__iter__'):
                _paste_teams = {t for _, t in confirmed_last_team}
            _fd_teams = set()
            _fd_team_col = "team_code" if "team_code" in full_pool_backup.columns else "team"
            if _fd_team_col in full_pool_backup.columns:
                _fd_teams = set(full_pool_backup[_fd_team_col].dropna().str.upper().str.strip().unique())
            _team_overlap = _paste_teams & _fd_teams
            _paste_only = _paste_teams - _fd_teams

            # Check position coverage
            _pos_col = "roster_position" if "roster_position" in combined.columns else "position"
            _positions_present = set()
            if _pos_col in combined.columns:
                for _val in combined[_pos_col].dropna().astype(str):
                    for _p in _val.upper().replace("-", "/").split("/"):
                        _p = _p.strip()
                        if _p and _p != "UTIL":
                            _positions_present.add(_p)
            _required = {"P", "C", "1B", "2B", "3B", "SS", "OF"}
            _missing = _required - _positions_present

            # Pool is non-viable if: too few players, missing positions, OR low match rate
            pool_too_small = (
                len(combined) < 40
                or bool(_missing)
                or match_rate < 0.5
            )

            if pool_too_small:
                reasons = []
                if bool(_missing):
                    reasons.append(f"missing positions: {', '.join(sorted(_missing))}")
                if len(combined) < 40:
                    reasons.append(f"only {len(combined)} players (need 40+)")
                if match_rate < 0.5:
                    reasons.append(f"low match rate: {n_matched}/{n_pasted} ({match_rate:.0%})")
                reason_str = "; ".join(reasons)

                # Add slate mismatch diagnostic if teams don't overlap
                if _paste_only:
                    optional_messages.append(
                        f"SLATE MISMATCH: Your lineup paste has teams not in the FanDuel CSV: "
                        f"{', '.join(sorted(_paste_only))}. "
                        f"Make sure you upload the FanDuel player list for TODAY's slate, "
                        f"not a previous day's file."
                    )

                optional_messages.append(
                    f"WARNING: Confirmed starters pool is not viable ({reason_str}). "
                    f"Falling back to full player pool ({before_count} players). "
                    f"Check that your lineup paste matches the FanDuel slate."
                )
                combined = full_pool_backup.reset_index(drop=True)

            elif match_rate < 0.75:
                # Moderate match rate — warn but still use filtered pool
                optional_messages.append(
                    f"NOTE: Matched {n_matched} of {n_pasted} pasted names ({match_rate:.0%}). "
                    f"Some players may not be in the FanDuel slate or have different name spellings."
                )

        projections = compute_baseline_projections(
            combined,
            recency_blend=recency_blend,
            platoon_opposite_boost=platoon_opposite_boost,
            platoon_same_penalty=platoon_same_penalty,
            platoon_switch_boost=platoon_switch_boost,
        )

        projection_paths_list = [Path(p) for p in projection_paths]
        try:
            projections, projection_blend_result = blend_projection_sources(
                combined,
                projections,
                source_paths=projection_paths_list,
                weights=projection_weights,
                baseline_weight=projection_baseline_weight,
            )
        except (ValueError, KeyError) as exc:
            file_names = ", ".join(p.name for p in projection_paths_list) or "(none)"
            raise ValueError(
                f"Could not read projection source file(s) [{file_names}]: {exc}. "
                "Expected the Ballpark DFS Optimizer export (Points/Bust/Median/Upside "
                "columns, header on the second row). Check that the right file is in "
                "the projections slot — a raw Batters/Pitchers/Games/Teams export or a "
                "FanDuel CSV will not work here."
            ) from exc

        ownership_paths_list = [Path(p) for p in ownership_paths]
        ownership_result = compute_ownership_series(
            combined,
            projections,
            source_paths=ownership_paths_list,
            weights=ownership_weights,
            model_config=ownership_model_config,
        )
        ownership_map = ownership_result.ownership.to_dict()
        projections["proj_fd_ownership"] = (
            projections["fd_player_id"].astype(str).map(ownership_map).fillna(0.0)
        )
        ownership_coverage_map = getattr(ownership_result, "external_coverage", pd.Series(dtype=bool)).to_dict()
        projections["ownership_source_covered"] = (
            projections["fd_player_id"].astype(str).map(ownership_coverage_map).fillna(False).astype(bool)
        )

        optimizer_df = build_optimizer_dataset(combined, projections)
        optimizer_df = _apply_ownership_edge(optimizer_df)

    summary_messages = [
        f"Players loaded: {len(combined)}",
        f"Projection sources blended: {projection_blend_result.source_count}",
        f"Ownership sources blended: {ownership_result.source_count}",
        f"Platoon multipliers -> opp:{platoon_opposite_boost:.2f} same:{platoon_same_penalty:.2f} switch:{platoon_switch_boost:.2f}",
    ]
    summary_messages.extend(optional_messages)

    blend_preview = []
    for detail in getattr(ownership_result, "sources", []) or []:
        blend_preview.append(f"{detail.name}:{detail.weight:.2f}")
    if blend_preview:
        summary_messages.append("Ownership weights -> " + ", ".join(blend_preview))
    summary_messages.append(
        "Ownership model -> "
        f"{ownership_model_summary['preset']} floor:{ownership_model_summary['min_pct'] * 100:.1f}%"
        f" ceiling:{ownership_model_summary['max_pct'] * 100:.1f}%"
    )

    projection_blend_message = []
    if projection_blend_result and getattr(projection_blend_result, "sources", None):
        projection_blend_message.append(f"BallparkPal:{projection_blend_result.baseline_share:.2f}")
        for detail in projection_blend_result.sources:
            projection_blend_message.append(f"{detail.name}:{detail.weight:.2f}")
    if projection_blend_message:
        summary_messages.append("Projection blend -> " + ", ".join(projection_blend_message))

    projection_blend_config = {
        "preset": projection_preset or DEFAULT_WEIGHT_PRESET,
        "manual_override": manual_projection_override,
        "source_files": [path.name for path in projection_paths_list],
        "baseline_weight_input": projection_baseline_weight,
        "baseline_share": getattr(projection_blend_result, "baseline_share", 1.0),
        "applied_weights": [
            {
                "name": detail.name,
                "weight": detail.weight,
                "matched_players": detail.matched_players,
                "has_floor": detail.has_floor,
                "has_ceiling": detail.has_ceiling,
                "has_bust": getattr(detail, "has_bust", False),
                "has_median": getattr(detail, "has_median", False),
                "has_upside": getattr(detail, "has_upside", False),
                "source_players": getattr(detail, "source_players", detail.matched_players),
                "unmatched_players": list(getattr(detail, "unmatched_players", []) or []),
            }
            for detail in (projection_blend_result.sources if projection_blend_result else [])
        ],
    }

    ownership_config = {
        "preset": ownership_preset or DEFAULT_WEIGHT_PRESET,
        "manual_override": manual_weight_override,
        "source_files": [path.name for path in ownership_paths_list],
        "requested_weights": ownership_weights,
        "applied_weights": [
            {
                "name": detail.name,
                "weight": detail.weight,
                "matched_players": detail.matched_players,
                "source_players": getattr(detail, "source_players", detail.matched_players),
                "unmatched_players": list(getattr(detail, "unmatched_players", []) or []),
            }
            for detail in getattr(ownership_result, "sources", []) or []
        ],
    }

    projection_config = {
        "recency_blend": list(recency_blend) if recency_blend else None,
        "platoon": {
            "opposite": platoon_opposite_boost,
            "same": platoon_same_penalty,
            "switch": platoon_switch_boost,
        },
        "ownership": ownership_config,
        "ownership_model": ownership_model_summary,
        "projection_blend": projection_blend_config,
    }

    workflow_payload = {
        "players": combined,
        "projections": projections,
        "optimizer": optimizer_df,
        "diagnostics": diagnostics,
        "ownership_summary": ownership_result,
        "projection_blend_summary": projection_blend_result,
        "messages": summary_messages,
        "recency_blend": recency_blend,
        "platoon_settings": {
            "opposite_boost": platoon_opposite_boost,
            "same_penalty": platoon_same_penalty,
            "switch_boost": platoon_switch_boost,
        },
        "projection_config": projection_config,
        "template_entries": template_entries,
    }
    return workflow_payload


def _serialize_strategy_config(workflow: Dict) -> str:
    sim_results = workflow.get(SIM_RESULTS_KEY) or {}
    sim_config = sim_results.get("sim_config")
    if isinstance(sim_config, SimulationConfig):
        sim_payload = sim_config.to_dict()
    else:
        sim_payload = dict(_get_sim_config_state())
    strategy_auto = workflow.get("strategy_auto") or {}
    payload = {
        "optimizer": dict(_get_config_state()),
        "simulation": sim_payload,
        "projection": workflow.get("projection_config"),
        "slate_run_preset": st.session_state.get("active_slate_run_preset"),
        "slate_profile": st.session_state.get("active_slate_profile") or strategy_auto.get("profile"),
        "slate_games": strategy_auto.get("games"),
        "slate_teams": strategy_auto.get("teams"),
        "active_strategy_build": sim_results.get("active_strategy_build"),
        "active_lineup_source": workflow.get("active_lineup_source", "optimizer"),
    }
    return json.dumps(payload)


def _extract_projection_config_from_lineups(df: Optional[pd.DataFrame]) -> Optional[Dict]:
    if df is None or df.empty or "strategy_config_json" not in df.columns:
        return None
    configs = df["strategy_config_json"].dropna()
    for raw in configs:
        if not isinstance(raw, str):
            continue
        try:
            payload = json.loads(raw)
        except Exception:  # pylint: disable=broad-except
            continue
        projection_cfg = payload.get("projection")
        if projection_cfg:
            return projection_cfg
    return None


def render_validation_summary(workflow: Dict) -> None:
    template_entries = workflow.get("template_entries")
    if template_entries is not None and not template_entries.empty:
        st.success(
            f"📤 Direct-upload export armed: detected {len(template_entries)} contest "
            "entries in your FanDuel file. Step 5 will produce a CSV you upload "
            "straight back to FanDuel — no copy-pasting."
        )
    elif workflow.get("optimizer") is not None:
        st.caption(
            "📋 No contest entries in the FanDuel file (players list uploaded). Step 5's CSV "
            "will need pasting into FanDuel's template — upload the entries-upload-template "
            "here instead to skip that step."
        )
    diagnostics = workflow.get("diagnostics")
    if diagnostics:
        st.subheader("Merge diagnostics")
        st.table(_format_diagnostics(diagnostics))
    projection_cfg = workflow.get("projection_config") or {}
    _render_projection_config_summary(config=projection_cfg)
    _render_projection_blend_summary(workflow.get("projection_blend_summary") or projection_cfg.get("projection_blend"))
    _render_ownership_model_summary(projection_cfg.get("ownership_model"))
    ownership_result = workflow.get("ownership_summary")
    if ownership_result:
        _render_ownership_summary(ownership_result)
    run_audit = _build_workflow_run_audit(workflow)
    if run_audit:
        _render_run_audit_report(run_audit)
    if workflow.get("messages"):
        st.subheader("Messages")
        for message in workflow["messages"]:
            st.write(message)


def _summary_source_count(summary) -> int:
    if not summary:
        return 0
    if hasattr(summary, "source_count"):
        return int(getattr(summary, "source_count") or 0)
    if isinstance(summary, dict):
        if "source_count" in summary:
            return int(summary.get("source_count") or 0)
        return len(summary.get("applied_weights") or summary.get("sources") or [])
    return 0


def _build_workflow_run_audit(
    workflow: Dict,
    lineup_df: Optional[pd.DataFrame] = None,
    expected_lineups: Optional[int] = None,
    config_state: Optional[Dict] = None,
    stack_plan: Optional[pd.DataFrame] = None,
):
    optimizer_df = workflow.get("optimizer")
    if not isinstance(optimizer_df, pd.DataFrame) or optimizer_df.empty:
        return None
    projection_cfg = workflow.get("projection_config") or {}
    projection_summary = workflow.get("projection_blend_summary") or projection_cfg.get("projection_blend")
    ownership_summary = workflow.get("ownership_summary") or (projection_cfg.get("ownership") or {})
    salary_cap = int((config_state or {}).get("salary_cap", 35000) or 35000)
    return build_run_audit_report(
        optimizer_df,
        lineup_df=lineup_df,
        expected_lineups=expected_lineups,
        stack_plan=stack_plan,
        projection_source_count=_summary_source_count(projection_summary),
        ownership_source_count=_summary_source_count(ownership_summary),
        salary_cap=salary_cap,
    )


def _render_run_audit_report(report, title: str = "Run health audit", expanded: Optional[bool] = None) -> None:
    if report is None or not getattr(report, "summary", None):
        return
    checks = report.checks if isinstance(report.checks, pd.DataFrame) else pd.DataFrame()
    issues = checks[checks["status"].isin(["Warn", "Fail"])] if not checks.empty and "status" in checks.columns else pd.DataFrame()
    has_fail = bool((issues.get("status") == "Fail").any()) if not issues.empty else False
    if expanded is None:
        expanded = not issues.empty

    st.subheader(title)
    summary = report.summary or {}
    metric_cols = st.columns(5)
    metric_cols[0].metric("Bust coverage", f"{summary.get('bust_coverage_pct', 0.0):.0%}")
    metric_cols[1].metric("Median coverage", f"{summary.get('median_coverage_pct', 0.0):.0%}")
    metric_cols[2].metric("Upside coverage", f"{summary.get('upside_coverage_pct', 0.0):.0%}")
    metric_cols[3].metric("Ownership", str(summary.get("ownership_mode", "Fallback")))
    metric_cols[4].metric("Issues", int(len(issues)))
    st.caption(
        f"Projection sources: {int(summary.get('projection_sources', 0) or 0)} | "
        f"Ownership sources: {int(summary.get('ownership_sources', 0) or 0)}"
    )

    if issues.empty:
        st.success("Run audit passed the core data-health checks.")
    elif has_fail:
        st.error("Run audit found a blocker to fix before trusting this output.")
        st.dataframe(issues, width="stretch", hide_index=True)
    else:
        st.warning("Run audit found warnings worth reviewing before upload.")
        st.dataframe(issues, width="stretch", hide_index=True)

    with st.expander("Run audit details", expanded=expanded):
        if not checks.empty:
            st.dataframe(checks, width="stretch", hide_index=True)
        if not report.lineup_checks.empty:
            lineup_view = report.lineup_checks.copy()
            for col in ["ownership", "avg_leverage"]:
                if col in lineup_view.columns:
                    lineup_view[col] = pd.to_numeric(lineup_view[col], errors="coerce").fillna(0.0).map(lambda x: f"{x:.3f}")
            st.markdown("**Lineup checks**")
            st.dataframe(lineup_view.head(120), width="stretch", hide_index=True)
        if not report.stack_mix.empty:
            stack_view = report.stack_mix.copy()
            if "lineup_pct" in stack_view.columns:
                stack_view["lineup_pct"] = pd.to_numeric(stack_view["lineup_pct"], errors="coerce").fillna(0.0).map(lambda x: f"{x:.1%}")
            st.markdown("**Stack template mix**")
            st.dataframe(stack_view, width="stretch", hide_index=True)
        if not getattr(report, "stack_alignment", pd.DataFrame()).empty:
            align_view = report.stack_alignment.copy()
            pct_cols = [
                "exposure_pct",
                "target_exposure",
                "recommended_max_exposure",
                "exposure_vs_target",
                "p_runs_ge_6",
                "stack_ownership",
            ]
            for col in pct_cols:
                if col in align_view.columns:
                    align_view[col] = pd.to_numeric(align_view[col], errors="coerce").fillna(0.0).map(lambda x: f"{x:.1%}")
            for col in ["sim_upside_score", "stack_leverage_score", "stack_score", "top5_upside"]:
                if col in align_view.columns:
                    align_view[col] = pd.to_numeric(align_view[col], errors="coerce").fillna(0.0).map(lambda x: f"{x:.2f}")
            keep_cols = [
                col
                for col in [
                    "team_code",
                    "stack_lineups",
                    "exposure_pct",
                    "target_exposure",
                    "recommended_max_exposure",
                    "stack_ownership",
                    "p_runs_ge_6",
                    "top5_upside",
                    "stack_leverage_score",
                    "positive_stack_leverage",
                    "chalk_without_leverage",
                    "over_recommended_max",
                ]
                if col in align_view.columns
            ]
            st.markdown("**Stack leverage alignment**")
            st.dataframe(align_view[keep_cols], width="stretch", hide_index=True)


def _render_ownership_summary(summary) -> None:
    if not summary:
        return
    sources = getattr(summary, "sources", []) or []
    total_players = len(summary.ownership) if hasattr(summary, "ownership") else None
    st.subheader("Ownership blend summary")
    if not sources:
        st.info("Using fallback ownership estimator (no external sources provided).")
        if total_players:
            st.caption(f"Coverage 0/{total_players} players.")
        return
    rows = [
        {
            "Source": detail.name,
            "Weight": f"{detail.weight:.2f}",
            "Matched players": detail.matched_players,
            "Source players": getattr(detail, "source_players", detail.matched_players),
            "Unmatched preview": ", ".join(list(getattr(detail, "unmatched_players", []) or [])[:3]),
        }
        for detail in sources
    ]
    table = pd.DataFrame(rows)
    st.table(table)
    if total_players is not None:
        st.caption(
            f"Coverage {summary.covered_players}/{total_players} players. External sources: {summary.source_count}"
        )
    unmatched_rows = []
    for detail in sources:
        for player in list(getattr(detail, "unmatched_players", []) or []):
            unmatched_rows.append({"Source": detail.name, "Unmatched player": player})
    if unmatched_rows:
        with st.expander("Ownership players that did not match the FanDuel slate", expanded=False):
            st.dataframe(pd.DataFrame(unmatched_rows), width="stretch", hide_index=True)


def _render_projection_blend_summary(summary) -> None:
    if not summary:
        return
    if hasattr(summary, "sources"):
        sources = summary.sources or []
        baseline_share = getattr(summary, "baseline_share", 1.0)
    else:
        summary_dict = summary if isinstance(summary, dict) else {}
        sources = summary_dict.get("applied_weights") or []
        baseline_share = float(summary_dict.get("baseline_share", 1.0))
    st.subheader("Projection blend summary")
    if not sources:
        st.info("Projection blend uses BallparkPal baseline only.")
        st.caption(f"BallparkPal share {baseline_share:.2f}")
        return
    rows = []
    unmatched_rows = []
    for detail in sources:
        if hasattr(detail, "name"):
            source_players = int(getattr(detail, "source_players", 0) or getattr(detail, "matched_players", 0) or 0)
            matched_players = int(getattr(detail, "matched_players", 0) or 0)
            unmatched_players = list(getattr(detail, "unmatched_players", []) or [])
            match_rate = matched_players / source_players if source_players else 0.0
            rows.append(
                {
                    "Source": detail.name,
                    "Weight": f"{detail.weight:.2f}",
                    "Matched players": matched_players,
                    "Source players": source_players,
                    "Match rate": f"{match_rate:.0%}" if source_players else "n/a",
                    "Floor": "Yes" if getattr(detail, "has_floor", False) else "No",
                    "Ceiling": "Yes" if getattr(detail, "has_ceiling", False) else "No",
                    "Bust": "Yes" if getattr(detail, "has_bust", False) else "No",
                    "Median": "Yes" if getattr(detail, "has_median", False) else "No",
                    "Upside": "Yes" if getattr(detail, "has_upside", False) else "No",
                    "Unmatched preview": ", ".join(unmatched_players[:3]),
                }
            )
            for player in unmatched_players:
                unmatched_rows.append({"Source": detail.name, "Unmatched player": player})
        else:
            source_players = int(detail.get("source_players") or detail.get("matched_players") or 0)
            matched_players = int(detail.get("matched_players") or 0)
            unmatched_players = list(detail.get("unmatched_players") or [])
            match_rate = matched_players / source_players if source_players else 0.0
            rows.append(
                {
                    "Source": detail.get("name", "source"),
                    "Weight": f"{float(detail.get('weight', 0.0)):.2f}",
                    "Matched players": matched_players,
                    "Source players": source_players,
                    "Match rate": f"{match_rate:.0%}" if source_players else "n/a",
                    "Floor": "Yes" if detail.get("has_floor") else "No",
                    "Ceiling": "Yes" if detail.get("has_ceiling") else "No",
                    "Bust": "Yes" if detail.get("has_bust") else "No",
                    "Median": "Yes" if detail.get("has_median") else "No",
                    "Upside": "Yes" if detail.get("has_upside") else "No",
                    "Unmatched preview": ", ".join(unmatched_players[:3]),
                }
            )
            for player in unmatched_players:
                unmatched_rows.append({"Source": detail.get("name", "source"), "Unmatched player": player})
    st.table(pd.DataFrame(rows))
    st.caption(f"BallparkPal share {baseline_share:.2f}")
    if unmatched_rows:
        with st.expander("Projection players that did not match the FanDuel slate", expanded=False):
            st.dataframe(pd.DataFrame(unmatched_rows), width="stretch", hide_index=True)


def _render_ownership_model_summary(model_cfg) -> None:
    if not model_cfg:
        return
    values = model_cfg.get("values") or model_cfg.get("parameters") or {}
    rows = []
    for field, label, *_ in OWNERSHIP_MODEL_FIELDS:
        if field in {"min_pct", "max_pct"}:
            continue
        rows.append({"Factor": label, "Weight": f"{values.get(field, 0.0):.2f}"})
    if rows:
        st.subheader("Ownership model summary")
        st.table(pd.DataFrame(rows))
    min_pct = model_cfg.get("min_pct")
    max_pct = model_cfg.get("max_pct")
    preset = model_cfg.get("preset")
    caption_parts = []
    if preset:
        caption_parts.append(f"Preset: {preset}")
    if min_pct is not None and max_pct is not None:
        caption_parts.append(f"Floor {float(min_pct) * 100:.1f}% / Ceiling {float(max_pct) * 100:.1f}%")
    if caption_parts:
        st.caption(" | ".join(caption_parts))


def _ownership_blend_preview_text(config: Dict) -> str:
    weights = config.get("applied_weights") or []
    entries = []
    for detail in weights[:4]:
        name = detail.get("name") or "source"
        weight = detail.get("weight")
        if weight is None:
            continue
        entries.append(f"{name}: {float(weight):.2f}")
    if len(weights) > 4:
        entries.append("...")
    return ", ".join(entries)


def _render_projection_config_summary(workflow: Optional[Dict] = None, config: Optional[Dict] = None) -> None:
    if config is None:
        config = (workflow or {}).get("projection_config") if workflow else None
    if not config:
        return
    st.subheader("Projection input settings")
    recency = config.get("recency_blend") or []
    if len(recency) == 2:
        recency_text = f"{float(recency[0]):.2f} season / {float(recency[1]):.2f} recent"
    else:
        recency_text = "Default"
    platoon = config.get("platoon") or {}
    opp = platoon.get("opposite")
    same = platoon.get("same")
    switch = platoon.get("switch")
    platoon_text = (
        f"opp {float(opp):.2f} / same {float(same):.2f} / switch {float(switch):.2f}"
        if all(value is not None for value in (opp, same, switch))
        else "Not specified"
    )
    ownership_cfg = config.get("ownership") or {}
    preset_label = ownership_cfg.get("preset") or DEFAULT_WEIGHT_PRESET
    if ownership_cfg.get("manual_override"):
        preset_label += " (manual override)"
    projection_blend_cfg = config.get("projection_blend") or {}
    projection_label = projection_blend_cfg.get("preset") or DEFAULT_WEIGHT_PRESET
    if projection_blend_cfg.get("manual_override"):
        projection_label += " (manual override)"
    col_recency, col_platoon, col_own, col_proj = st.columns(4)
    col_recency.write(f"Recency blend: **{recency_text}**")
    col_platoon.write(f"Platoon multipliers: **{platoon_text}**")
    col_own.write(f"Ownership preset: **{preset_label}**")
    preview = _ownership_blend_preview_text(ownership_cfg)
    if preview:
        col_own.caption(f"Weights -> {preview}")
    col_proj.write(f"Projection preset: **{projection_label}**")
    baseline_share = projection_blend_cfg.get("baseline_share")
    if baseline_share is not None:
        col_proj.caption(f"BallparkPal share {float(baseline_share):.2f}")
    proj_preview = _ownership_blend_preview_text(projection_blend_cfg)
    if proj_preview:
        col_proj.caption(f"Sources -> {proj_preview}")


def _game_status_dataframe(optimizer_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if optimizer_df is None or optimizer_df.empty:
        return pd.DataFrame()
    if "game_start_time" not in optimizer_df.columns:
        return pd.DataFrame()
    games = optimizer_df[["game_key", "game_start_time"]].drop_duplicates()
    games = games.dropna(subset=["game_start_time"])
    if games.empty:
        return pd.DataFrame()
    games["game_start_time"] = pd.to_datetime(games["game_start_time"], errors="coerce", utc=True)
    games = games.dropna(subset=["game_start_time"])
    if games.empty:
        return games
    now_utc = pd.Timestamp.now(tz="UTC")
    soon_threshold = now_utc + pd.Timedelta(minutes=30)
    def _status(ts: pd.Timestamp) -> str:
        if ts <= now_utc:
            return "Locked"
        if ts <= soon_threshold:
            return "Locking Soon"
        return "Open"
    games["status"] = games["game_start_time"].apply(_status)
    games["minutes_to_lock"] = (games["game_start_time"] - now_utc).dt.total_seconds() / 60.0
    games["local_start"] = games["game_start_time"].dt.tz_convert(ZoneInfo("US/Eastern")).dt.strftime("%I:%M %p")
    return games.sort_values("game_start_time")


def _render_game_status_panel(workflow: Dict) -> None:
    optimizer_df = workflow.get("optimizer")
    games = _game_status_dataframe(optimizer_df)
    if games.empty:
        return
    st.subheader("Game lock status")
    display_cols = games[["game_key", "local_start", "status", "minutes_to_lock"]]
    display_cols = display_cols.rename(
        columns={"game_key": "Game", "local_start": "Start (ET)", "minutes_to_lock": "Minutes"}
    )
    st.dataframe(display_cols, width="stretch")


def _next_lock_info(optimizer_df: Optional[pd.DataFrame]) -> Optional[Dict[str, str]]:
    games = _game_status_dataframe(optimizer_df)
    if games.empty:
        return None
    future = games[games["status"] != "Locked"].copy()
    if future.empty:
        return None
    soonest = future.sort_values("game_start_time").iloc[0]
    minutes = float(soonest.get("minutes_to_lock", float("nan")))
    local = soonest.get("local_start")
    return {
        "game": soonest.get("game_key"),
        "local": local,
        "minutes": minutes,
        "status": soonest.get("status"),
    }


def _render_lock_countdown(workflow: Dict, label: str = "Next lock") -> None:
    optimizer_df = workflow.get("optimizer")
    info = _next_lock_info(optimizer_df)
    if not info:
        return
    minutes = info.get("minutes")
    status = info.get("status")
    game = info.get("game")
    local = info.get("local")
    if minutes is None or pd.isna(minutes):
        return
    delta = pd.Timedelta(minutes=float(minutes))
    countdown = f"{int(delta.components.hours):02d}:{int(delta.components.minutes):02d}:{int(delta.components.seconds):02d}"
    st.info(
        f"{label}: {game} locks at {local} ({status}) in {minutes:.1f} min ({countdown}).",
        icon="⏳",
    )


def _bench_risk_flags(lineup_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    if "game_start_time" not in lineup_df.columns:
        return pd.Series(False, index=lineup_df.index), pd.Series(False, index=lineup_df.index)
    times = pd.to_datetime(lineup_df["game_start_time"], errors="coerce", utc=True)
    now = pd.Timestamp.now(tz="UTC")
    locked_mask = times.notna() & (times <= now)
    confirmed = lineup_df.get("is_confirmed_lineup")
    if confirmed is None:
        confirmed = pd.Series(False, index=lineup_df.index)
    confirmed = confirmed.astype(bool)
    risk_mask = (~confirmed) & (~locked_mask)
    return risk_mask, locked_mask


def _late_swap_candidates(lineup_df: pd.DataFrame, horizon_minutes: int = 90) -> pd.DataFrame:
    if "game_start_time" not in lineup_df.columns:
        return pd.DataFrame()
    horizon_minutes = max(15, int(horizon_minutes))
    times = pd.to_datetime(lineup_df["game_start_time"], errors="coerce", utc=True)
    if times.isna().all():
        return pd.DataFrame()
    confirmed = lineup_df.get("is_confirmed_lineup")
    if confirmed is None:
        confirmed = pd.Series(False, index=lineup_df.index)
    confirmed = confirmed.astype(bool)
    now = pd.Timestamp.now(tz="UTC")
    upper = now + pd.Timedelta(minutes=horizon_minutes)
    soon_mask = times.notna() & (times > now) & (times <= upper)
    risk_mask = soon_mask & (~confirmed)
    if not risk_mask.any():
        return pd.DataFrame()
    local_times = times.dt.tz_convert(ZoneInfo("US/Eastern"))
    result = lineup_df.loc[risk_mask, ["lineup_id", "full_name", "team_code", "proj_fd_mean", "salary"]].copy()
    result["game_start_time"] = times.loc[risk_mask]
    result["start_et"] = local_times.loc[risk_mask].dt.strftime("%I:%M %p")
    minutes_to_lock = (times.loc[risk_mask] - now).dt.total_seconds() / 60.0
    result["minutes_to_lock"] = minutes_to_lock.round(1)
    result = result.sort_values(["minutes_to_lock", "lineup_id", "full_name"])
    return result


def _render_late_swap_panel(workflow: Dict) -> None:
    lineup_df: Optional[pd.DataFrame] = workflow.get("lineups_df")
    if lineup_df is None or lineup_df.empty:
        return
    with st.expander("Late Swap Aide", expanded=False):
        refresh_cols = st.columns([1, 2])
        if refresh_cols[0].button("Refresh MLB lineups", key="late_news_refresh"):
            try:
                with st.spinner("Fetching current MLB lineups..."):
                    workflow["fresh_lineups_df"] = fetch_batting_orders()
                st.success("Fresh MLB lineups loaded for late-news checks.")
                st.rerun()
            except Exception as exc:  # pylint: disable=broad-except
                st.warning(f"Could not refresh MLB lineups: {exc}")
        fresh_orders = workflow.get("fresh_lineups_df")
        if isinstance(fresh_orders, pd.DataFrame) and not fresh_orders.empty:
            teams = fresh_orders["team"].nunique() if "team" in fresh_orders.columns else 0
            refresh_cols[1].caption(f"Fresh lineup check loaded for {teams} teams.")
        else:
            refresh_cols[1].caption("Use refresh near lock to compare against posted MLB lineups.")

        window = st.number_input(
            "Alert window (minutes)",
            min_value=15,
            max_value=180,
            value=int(st.session_state.get("late_swap_window", 90)),
            step=15,
            key="late_swap_window",
        )
        late_report = build_late_news_report(
            lineup_df,
            optimizer_df=workflow.get("optimizer"),
            fresh_orders=fresh_orders if isinstance(fresh_orders, pd.DataFrame) else None,
            scratched_players=st.session_state.get("scratched_players", []),
            lock_warning_minutes=int(window),
        )
        if not late_report.issues.empty:
            st.markdown("**Late-news issues**")
            st.dataframe(late_report.issues, width="stretch", hide_index=True)
            action_cols_report = st.columns(3)
            if action_cols_report[0].button("Focus affected lineups", key="late_news_focus"):
                st.session_state["late_swap_filter_ids"] = late_report.affected_lineup_ids
                st.rerun()
            if action_cols_report[1].button("Add issue players to scratches", key="late_news_scratch"):
                existing = set(st.session_state.get("scratched_players", []))
                existing.update(late_report.scratched_players)
                st.session_state["scratched_players"] = sorted(existing)
                st.rerun()
            if action_cols_report[2].button("Re-opt late-news issues", key="late_news_reopt"):
                try:
                    names = late_report.scratched_players or late_report.issues["player"].dropna().unique().tolist()
                    diffs = _reoptimize_scratches(names, workflow, _get_config_state())
                    for message in diffs:
                        st.caption(message)
                    st.success("Affected lineups re-optimized.")
                    st.rerun()
                except Exception as exc:  # pylint: disable=broad-except
                    st.error(f"Late-news re-optimization failed: {exc}")
        else:
            st.success("No late-news issues detected from current inputs.")

        candidates = _late_swap_candidates(lineup_df, window)
        if candidates.empty:
            st.success("No unconfirmed players approaching lock in the selected window.")
            return
        display_cols = ["lineup_id", "full_name", "team_code", "start_et", "minutes_to_lock"]
        st.dataframe(candidates[display_cols], width="stretch")
        st.caption("Players with unconfirmed lineups locking soon. Add them to your scratch list or focus view.")
        action_cols = st.columns(3)
        if action_cols[0].button("Focus on these lineups", key="late_swap_focus"):
            st.session_state["late_swap_filter_ids"] = sorted(candidates["lineup_id"].unique().tolist())
            st.rerun()
        if action_cols[1].button("Add to scratch list", key="late_swap_add_scratches"):
            existing = set(st.session_state.get("scratched_players", []))
            existing.update(candidates["full_name"].unique().tolist())
            st.session_state["scratched_players"] = sorted(existing)
            st.rerun()
        if action_cols[2].button("Re-opt flagged lineups", key="late_swap_reopt"):
            try:
                config_state = _get_config_state()
                diffs = _reoptimize_scratches(candidates["full_name"].unique().tolist(), workflow, config_state)
                if diffs:
                    for message in diffs:
                        st.caption(message)
                st.success("Affected lineups re-optimized.")
                st.rerun()
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Late swap re-optimization failed: {exc}")


def _reoptimize_scratches(
    scratched_players: List[str],
    workflow: Dict,
    config_state: Dict,
) -> List[str]:
    if not scratched_players:
        raise ValueError("No players selected for scratching.")
    lineup_df = workflow.get("lineups_df")
    lineups = workflow.get("lineups")
    optimizer_df = workflow.get("optimizer")
    if lineup_df is None or lineups is None or optimizer_df is None:
        raise ValueError("Missing lineup or optimizer data in session state.")
    affected_ids = lineup_df[lineup_df["full_name"].isin(scratched_players)]["lineup_id"].unique()
    if len(affected_ids) == 0:
        raise ValueError("No existing lineups contain the scratched players.")

    existing = workflow.get("lineups")
    diff_messages: List[str] = []
    scratch_set = {str(name).lower() for name in scratched_players}
    now = pd.Timestamp.now(tz="UTC")
    for lineup_id in affected_ids:
        idx = lineup_id - 1
        old_lineup = existing[idx]
        old_rows = lineup_df[lineup_df["lineup_id"] == lineup_id]
        locked_names = _locked_player_names_for_repair(old_rows, scratch_set, now)
        temp_config = dict(config_state)
        temp_config["num_lineups"] = 1
        try:
            new_lineups, _ = _run_solver(
                optimizer_df,
                temp_config,
                locked_players=locked_names,
                excluded_players=scratched_players,
            )
        except Exception:
            new_lineups, _ = _run_solver(
                optimizer_df,
                temp_config,
                excluded_players=scratched_players,
            )
            locked_names = []
        if not new_lineups:
            diff_messages.append(f"Lineup {lineup_id}: no valid repair found.")
            continue
        new_lineup = new_lineups[0]
        old_players = set(old_lineup.dataframe["full_name"].tolist())
        new_players = set(new_lineup.dataframe["full_name"].tolist())
        removed = old_players - new_players
        added = new_players - old_players
        locked_note = f"; preserved locked: {len(locked_names)}" if locked_names else ""
        message = (
            f"Lineup {lineup_id}: removed {', '.join(sorted(removed)) if removed else 'none'}, "
            f"added {', '.join(sorted(added)) if added else 'none'}{locked_note}"
        )
        diff_messages.append(message)
        existing[idx] = new_lineup
    workflow["lineups"] = existing
    workflow["lineups_df"] = _combine_lineups(existing)
    return diff_messages


def _locked_player_names_for_repair(
    lineup_rows: pd.DataFrame,
    scratch_set: Set[str],
    now: pd.Timestamp,
) -> List[str]:
    if lineup_rows is None or lineup_rows.empty:
        return []
    rows = lineup_rows.copy()
    if "game_start_time" not in rows.columns:
        return []
    starts = pd.to_datetime(rows["game_start_time"], errors="coerce", utc=True)
    locked = rows[starts.notna() & (starts <= now)]
    if locked.empty:
        return []
    names = []
    for name in locked["full_name"].dropna().astype(str).tolist():
        if name.lower() not in scratch_set:
            names.append(name)
    return names

def _parse_stack_templates(value: str) -> List[int]:
    """Legacy parser — kept for backward compatibility with scripts."""
    templates: List[int] = []
    if not value:
        return templates
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            templates.append(int(token))
        except ValueError:
            raise ValueError("Stack templates must be integers separated by commas.")
    return [t for t in templates if t > 0]


def _parse_player_overrides(value: str) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    for line in value.splitlines():
        if not line.strip():
            continue
        if ":" in line:
            name, cap = line.split(":", 1)
        elif "=" in line:
            name, cap = line.split("=", 1)
        else:
            raise ValueError("Overrides must be in 'Player Name:0.5' format")
        try:
            overrides[name.strip()] = float(cap.strip())
        except ValueError:
            raise ValueError(f"Invalid exposure value for {name.strip()}" )
    return overrides


def _apply_player_filters(df: pd.DataFrame, overrides: Dict[str, float]) -> Dict[str, float]:
    mapped: Dict[str, float] = {}
    if not overrides:
        return mapped
    name_to_id = df.set_index("full_name")["fd_player_id"].to_dict()
    for key, value in overrides.items():
        if key in name_to_id:
            mapped[str(name_to_id[key])] = value
        mapped[key] = value
    return mapped


def _run_solver(
    dataset: pd.DataFrame,
    config_settings: Dict,
    locked_players: Optional[List[str]] = None,
    excluded_players: Optional[List[str]] = None,
) -> Tuple[List, pd.DataFrame]:
    df = dataset.copy()
    locked_players = locked_players or []
    excluded_players = excluded_players or []

    if excluded_players:
        exclude_set = {name.lower() for name in excluded_players}
        df = df[~df["full_name"].str.lower().isin(exclude_set)]
        if df.empty:
            raise ValueError("All players excluded; cannot run optimizer.")

    locked_player_ids: List[str] = []
    if locked_players:
        locked_lower = {p.lower() for p in locked_players}
        locked_rows = df[df["full_name"].str.lower().isin(locked_lower)]
        missing_locks = sorted(locked_lower - set(locked_rows["full_name"].str.lower()))
        if missing_locks:
            raise ValueError(f"Locked players not available after exclusions: {', '.join(missing_locks)}")
        locked_player_ids = locked_rows["fd_player_id"].astype(str).dropna().unique().tolist()

    player_overrides = {}
    overrides_input = config_settings.get("player_overrides", "")
    if overrides_input.strip():
        overrides = _parse_player_overrides(overrides_input)
        player_overrides = _apply_player_filters(df, overrides)

    # Build stack rotation from multi-select + counts
    selected_templates = config_settings.get("stack_template_selections", ["4-3-1"])
    template_counts = config_settings.get("template_counts", {})
    stack_rotation: Optional[List] = None
    stack_template: Optional[Tuple] = None

    if len(selected_templates) == 1:
        # Single template mode
        stack_template = STACK_PRESETS.get(selected_templates[0])
    elif len(selected_templates) > 1:
        # Multi-template mode: build rotation list
        rotation: List = []
        for label in selected_templates:
            tmpl = STACK_PRESETS.get(label)
            count = int(template_counts.get(label, 1))
            rotation.extend([tmpl] * max(1, count))
        stack_rotation = rotation if rotation else None
    else:
        stack_template = None  # Auto
    batter_chalk_val = config_settings.get("batter_chalk_threshold", 0.0)
    chalk_threshold = batter_chalk_val / 100.0 if batter_chalk_val else None
    batter_cap_val = config_settings.get("batter_chalk_exposure_cap", 0.0)
    chalk_exposure_cap = batter_cap_val / 100.0 if batter_cap_val else None
    pitcher_chalk_val = config_settings.get("pitcher_chalk_threshold", 0.0)
    pitcher_chalk_threshold = pitcher_chalk_val / 100.0 if pitcher_chalk_val else None
    pitcher_cap_val = config_settings.get("pitcher_chalk_exposure_cap", 0.0)
    pitcher_chalk_exposure_cap = pitcher_cap_val / 100.0 if pitcher_cap_val else None
    max_lineup_ownership = config_settings.get("max_lineup_ownership", 0.0)
    if max_lineup_ownership:
        max_lineup_ownership = max_lineup_ownership / 100.0
    else:
        max_lineup_ownership = None
    bring_back_enabled = bool(config_settings.get("bring_back_enabled"))
    bring_back_count = int(config_settings.get("bring_back_count", 1) or 1)
    min_game_total = config_settings.get("min_game_total", 0.0) or 0.0
    min_game_total = float(min_game_total) if min_game_total else None
    leverage_weight_pct = config_settings.get("leverage_weight", 15.0) or 0.0
    leverage_weight = float(leverage_weight_pct) / 100.0
    randomness_pct = config_settings.get("randomness", 5.0) or 0.0
    randomness = float(randomness_pct) / 100.0
    # GPP objective controls (LEVERAGE_STRATEGY / BPP_UPSIDE):
    # ceiling_weight blends BPP Upside into the objective; ownership_penalty_weight
    # penalizes projected ownership directly in the objective (soft chalk fade).
    ceiling_weight_pct = config_settings.get("ceiling_weight", 35.0)
    ceiling_weight = float(ceiling_weight_pct or 0.0) / 100.0
    ownership_penalty_pct = config_settings.get("ownership_penalty_weight", 10.0)
    ownership_penalty_weight = float(ownership_penalty_pct or 0.0) / 100.0
    min_salary_value = int(config_settings.get("min_salary", 34000) or 0)
    min_salary = min_salary_value if min_salary_value > 0 else None
    min_uniques = max(1, int(config_settings.get("min_uniques", 1) or 1))

    optimizer_config = OptimizerConfig(
        salary_cap=int(config_settings.get("salary_cap", 35000) or 35000),
        min_stack_size=None,
        stack_templates=None,
        player_exposure_overrides=player_overrides,
        max_lineup_ownership=max_lineup_ownership,
        chalk_threshold=chalk_threshold,
        chalk_exposure_cap=chalk_exposure_cap,
        pitcher_chalk_threshold=pitcher_chalk_threshold,
        pitcher_chalk_exposure_cap=pitcher_chalk_exposure_cap,
        bring_back_enabled=bring_back_enabled,
        bring_back_count=bring_back_count,
        min_game_total_for_stacks=min_game_total,
        leverage_weight=leverage_weight,
        randomness=randomness,
    )

    df = optimizer_config.apply_exposure_overrides(df)
    df = optimizer_config.apply_ownership_strategy(df)

    num_lineups = int(config_settings.get("num_lineups", 20) or 20)
    salary_cap_value = optimizer_config.salary_cap or 35000

    extra_lineups = num_lineups + (len(locked_players) * 2)
    lineups = generate_lineups(
        df,
        num_lineups=extra_lineups,
        salary_cap=salary_cap_value,
        stack_template=stack_template,
        stack_rotation=stack_rotation,
        max_lineup_ownership=max_lineup_ownership,
        bring_back_enabled=bring_back_enabled,
        bring_back_count=bring_back_count,
        min_game_total_for_stacks=min_game_total,
        leverage_weight=optimizer_config.leverage_weight,
        randomness=optimizer_config.randomness,
        locked_player_ids=locked_player_ids,
        ceiling_weight=ceiling_weight,
        ownership_penalty_weight=ownership_penalty_weight,
        min_salary=min_salary,
        min_uniques=min_uniques,
    )

    if not lineups:
        n_players = len(df)
        n_pitchers = int((df["player_type"].str.lower() == "pitcher").sum()) if "player_type" in df.columns else 0
        n_batters = n_players - n_pitchers
        templates_str = ", ".join(selected_templates)
        raise ValueError(
            f"Optimizer could not generate any lineups. "
            f"Pool has {n_players} players ({n_pitchers} pitchers, {n_batters} batters). "
            f"Stack templates: {templates_str}. Salary cap: ${salary_cap_value:,}. "
            f"Try 'Auto' stack template or check your lineup filter."
        )

    if locked_players:
        locked_lower = {p.lower() for p in locked_players}

        def has_locked(lineup) -> bool:
            names = {name.lower() for name in lineup.dataframe["full_name"].tolist()}
            return locked_lower.issubset(names)

        filtered_lineups = [lineup for lineup in lineups if has_locked(lineup)]
        if len(filtered_lineups) < num_lineups:
            st.warning(
                f"Only {len(filtered_lineups)} lineups contain all locked players. Showing available lineups."
            )
            lineups = filtered_lineups or lineups
        else:
            lineups = filtered_lineups

    lineups = lineups[:num_lineups]

    if len(lineups) < num_lineups:
        shortage = num_lineups - len(lineups)
        st.warning(
            f"**Only {len(lineups)} unique lineups generated** (you requested {num_lineups} — "
            f"missing {shortage}). Your constraints are too tight for this pool size. "
            f"To fix this, try: loosening chalk exposure caps, selecting multiple stack templates, "
            f"reducing locked players, or requesting fewer lineups."
        )

    # Warn about lineups where stack constraints were silently dropped
    dropped_count = sum(1 for lu in lineups if lu.stacks_dropped)
    if dropped_count > 0:
        st.warning(
            f"**{dropped_count} of {len(lineups)} lineups were generated without stacking constraints** "
            f"because the requested stack template became infeasible. These lineups may have "
            f"scattered batters instead of correlated team stacks. Consider using a more "
            f"flexible template (e.g., 3-3-2 instead of 4-4) or selecting multiple templates."
        )

    lineup_df = _combine_lineups(lineups)
    return lineups, lineup_df


def _sidebar_navigation() -> str:
    all_steps = [
        "1. Slate Setup",
        "2. Review Projections",
        "3. Configure & Optimize",
        "4. Simulate & Select",
        "5. Review Lineups",
        "6. Post-Slate",
        "Backtest Dashboard",
    ]
    daily_steps = [
        "1. Slate Setup",
        "3. Configure & Optimize",
        "4. Simulate & Select",
        "5. Review Lineups",
    ]
    if UI_DAILY_MODE_KEY not in st.session_state:
        st.session_state[UI_DAILY_MODE_KEY] = True
    if UI_ADVANCED_KEY not in st.session_state:
        st.session_state[UI_ADVANCED_KEY] = False

    st.sidebar.markdown("**UI mode**")
    daily_mode = st.sidebar.checkbox(
        "Daily build mode",
        key=UI_DAILY_MODE_KEY,
        help="Shows the normal lineup-building path and keeps research/post-slate pages out of the way.",
    )
    st.sidebar.checkbox(
        "Show advanced controls",
        key=UI_ADVANCED_KEY,
        help="Keeps the daily path visible while expanding the deeper tuning and diagnostic controls.",
    )

    steps = daily_steps if daily_mode else all_steps
    default_step = st.session_state.get(NAV_KEY, steps[0])
    if default_step not in steps:
        default_step = steps[0]
    current_step = st.sidebar.radio("Workflow Step", steps, index=steps.index(default_step))
    st.session_state[NAV_KEY] = current_step
    st.sidebar.markdown("---")
    if daily_mode:
        st.sidebar.caption("Daily path: setup -> optimize -> simulate -> review/export.")
        _render_sidebar_daily_status(_get_session())
        st.sidebar.markdown("---")
    return current_step


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    teams = sorted(filtered.get("team_code", pd.Series([])).dropna().unique())
    positions_series = filtered.get("position", pd.Series([""]))
    positions = sorted(
        {
            token
            for val in positions_series.dropna()
            for token in str(val).upper().replace("-", "/").split("/")
            if token
        }
    )
    if teams:
        selected_teams = st.sidebar.multiselect("Teams", teams, default=teams)
        if selected_teams:
            filtered = filtered[filtered["team_code"].isin(selected_teams)]
    if positions:
        selected_positions = st.sidebar.multiselect("Positions", positions, default=positions)
        if selected_positions and "position" in filtered.columns:
            def match_position(value: str) -> bool:
                tokens = str(value).upper().replace("-", "/").split("/")
                return any(pos in tokens for pos in selected_positions)

            filtered = filtered[filtered["position"].apply(match_position)]

    if "salary" in filtered.columns and not filtered.empty:
        salary_values = pd.to_numeric(filtered["salary"], errors="coerce").fillna(0)
        sal_min = int(salary_values.min())
        sal_max = int(salary_values.max())
        selected_salary = st.sidebar.slider("Salary range", sal_min, sal_max, (sal_min, sal_max))
        filtered = filtered[(filtered["salary"] >= selected_salary[0]) & (filtered["salary"] <= selected_salary[1])]

    proj_values = pd.to_numeric(filtered.get("proj_fd_mean", pd.Series([0])), errors="coerce").fillna(0.0)
    if not proj_values.empty:
        proj_min = float(proj_values.min())
        proj_max = float(proj_values.max()) if proj_values.max() > proj_values.min() else proj_min + 1
        min_projection = st.sidebar.slider("Min projection", proj_min, proj_max, proj_min)
        filtered = filtered[filtered["proj_fd_mean"] >= min_projection]

    st.sidebar.write(f"Filtered players: {len(filtered)}")
    return filtered


def _section_stacks(df: pd.DataFrame, top: int = 5) -> None:
    if "bpp_runs" not in df.columns:
        st.info("No BallparkPal run data available in this dataset.")
        return
    hitters = df[df["player_type"].str.lower() == "batter"].dropna(subset=["bpp_runs"])
    if hitters.empty:
        st.info("No hitter run data available.")
        return
    summary = hitters.groupby("team_code")["bpp_runs"].mean().sort_values(ascending=False)
    st.table(summary.head(top).rename("Avg Runs"))


def _section_pitchers(df: pd.DataFrame, top: int = 5) -> None:
    pitchers = df[df["player_type"].str.lower() == "pitcher"]
    if pitchers.empty:
        st.info("No pitcher rows available.")
        return
    cols = [col for col in ["full_name", "team_code", "proj_fd_mean", "bpp_win_percent"] if col in pitchers.columns]
    board = pitchers.sort_values(by="proj_fd_mean", ascending=False)[cols]
    st.table(board.head(top))


def _section_leverage(df: pd.DataFrame) -> None:
    required = {"proj_fd_mean", "proj_fd_ownership"}
    if not required.issubset(df.columns):
        st.info("Leverage requires proj_fd_mean and proj_fd_ownership columns.")
        return
    temp = df.copy()
    temp["proj_fd_mean"] = pd.to_numeric(temp["proj_fd_mean"], errors="coerce").fillna(0.0)
    temp["proj_fd_ownership"] = pd.to_numeric(temp["proj_fd_ownership"], errors="coerce").fillna(0.0)
    temp["projection_rank"] = temp["proj_fd_mean"].rank(pct=True)
    temp["ownership_rank"] = temp["proj_fd_ownership"].rank(pct=True)
    temp["leverage_score"] = temp["projection_rank"] - temp["ownership_rank"]
    hitters = temp[temp["player_type"].str.lower() == "batter"].sort_values("leverage_score", ascending=False)
    pitchers = temp[temp["player_type"].str.lower() == "pitcher"].sort_values("leverage_score", ascending=False)
    st.subheader("Hitter leverage")
    st.dataframe(
        hitters[["full_name", "team_code", "proj_fd_mean", "proj_fd_ownership", "leverage_score"]].head(10),
        width="stretch",
    )
    st.subheader("Pitcher leverage")
    st.dataframe(
        pitchers[["full_name", "team_code", "proj_fd_mean", "proj_fd_ownership", "leverage_score"]].head(10),
        width="stretch",
    )




def _factor_leader_table(df: pd.DataFrame, column: str, label: str, ascending: bool = False) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame()
    data = df[["full_name", "team_code", column]].copy()
    data = data.rename(columns={column: label})
    data = data.sort_values(label, ascending=ascending).head(5)
    return data


def _section_chalk(df: pd.DataFrame, top: int = 5) -> None:
    if "proj_fd_ownership" not in df.columns:
        st.info("No ownership column available.")
        return
    chalk = df.sort_values("proj_fd_ownership", ascending=False)
    st.table(chalk[["full_name", "team_code", "proj_fd_ownership", "proj_fd_mean"]].head(top))


def _variance_scatter(df: pd.DataFrame) -> None:
    required = {"proj_fd_mean", "proj_fd_ceiling", "proj_fd_floor", "proj_fd_ownership"}
    if not required.issubset(df.columns):
        st.info("Variance explorer requires mean/floor/ceiling/ownership columns.")
        return
    temp = df.copy()
    temp["ceiling"] = pd.to_numeric(temp["proj_fd_ceiling"], errors="coerce")
    temp["floor"] = pd.to_numeric(temp["proj_fd_floor"], errors="coerce")
    temp["proj_fd_mean"] = pd.to_numeric(temp["proj_fd_mean"], errors="coerce")
    temp["proj_fd_ownership"] = pd.to_numeric(temp["proj_fd_ownership"], errors="coerce")
    temp["volatility"] = (temp["ceiling"] - temp["floor"]).clip(lower=0)
    temp = temp.dropna(subset=["volatility", "proj_fd_mean"])
    if temp.empty:
        st.info("Not enough data to show variance scatter.")
        return
    chart_df = temp[["full_name", "proj_fd_mean", "volatility", "proj_fd_ownership"]].rename(
        columns={
            "proj_fd_mean": "Projection",
            "volatility": "Volatility",
            "proj_fd_ownership": "Ownership",
            "full_name": "Player",
        }
    )
    st.scatter_chart(chart_df, x="Projection", y="Volatility", color="Ownership", size="Ownership")
    st.caption("Bubble size/color = ownership %. Targets in the upper-left are high-volatility leverage plays.")


def _variance_leaderboard(df: pd.DataFrame) -> None:
    if "volatility" not in df.columns:
        temp = df.copy()
        if {"proj_fd_ceiling", "proj_fd_floor"}.issubset(temp.columns):
            temp["volatility"] = pd.to_numeric(temp["proj_fd_ceiling"], errors="coerce") - pd.to_numeric(
                temp["proj_fd_floor"], errors="coerce"
            )
        else:
            st.info("Not enough columns to compute volatility leaderboard.")
            return
    else:
        temp = df.copy()
    temp["volatility"] = pd.to_numeric(temp["volatility"], errors="coerce")
    temp["proj_fd_ownership"] = pd.to_numeric(temp.get("proj_fd_ownership"), errors="coerce")
    sorted_df = temp.dropna(subset=["volatility"]).sort_values("volatility", ascending=False).head(10)
    if sorted_df.empty:
        st.info("No volatility data available.")
        return
    display = sorted_df[["full_name", "team_code", "volatility", "proj_fd_mean", "proj_fd_ownership"]]
    display = display.rename(
        columns={
            "volatility": "Volatility",
            "proj_fd_mean": "Proj",
            "proj_fd_ownership": "Own%",
        }
    )
    st.table(display)


def _ownership_accuracy_panel(date_str: str) -> None:
    try:
        metrics = backtest.calculate_ownership_accuracy(date_str)
    except Exception:  # pylint: disable=broad-except
        return
    if not metrics:
        return
    st.subheader("Ownership calibration by bucket")
    mae = metrics.get("mae")
    count = metrics.get("count")
    if mae is not None and count:
        st.caption(f"Global MAE {mae:.3f} across {count} players")
    bucket_mae = metrics.get("bucket_mae") or {}
    if bucket_mae:
        chart_df = pd.DataFrame(
            {
                "Projected % bucket": [f"{bucket}-{bucket + 5}" for bucket in bucket_mae.keys()],
                "MAE": list(bucket_mae.values()),
            }
        )
        st.bar_chart(chart_df.set_index("Projected % bucket"))
    else:
        st.info("No ownership calibration buckets recorded for this slate.")


def _render_readiness_checklist(workflow: Dict) -> None:
    st.subheader("Slate readiness checklist")
    optimizer_ready = workflow.get("optimizer") is not None
    projection_cfg = workflow.get("projection_config") or {}
    ownership_ready = bool(workflow.get("ownership_summary"))
    projection_blend_ready = bool(workflow.get("projection_blend_summary"))
    ownership_model_ready = bool(projection_cfg.get("ownership_model"))
    lineups_ready = bool(workflow.get("lineups"))
    checks = [
        ("Slate processed", optimizer_ready),
        ("Ownership sources blended", ownership_ready),
        ("Projection sources blended", projection_blend_ready),
        ("Ownership model tuned", ownership_model_ready),
        ("Lineups generated", lineups_ready),
    ]
    cols = st.columns(2)
    for idx, (label, ready) in enumerate(checks):
        icon = "✅" if ready else "⏳"
        cols[idx % 2].write(f"{icon} {label}")
    if not all(status for _, status in checks[:-1]):
        st.info("Complete the pending items before locking lineups.")


def _render_step_one() -> None:
    st.header("Step 1 · Slate Setup")
    st.write("Upload the required files and run the ingestion + projection pipeline.")
    _render_daily_step_note("setup")

    with st.expander("Which file goes where? (upload map)", expanded=False):
        st.markdown(
            """
| File | Upload slot | What it feeds | If missing / swapped |
|---|---|---|---|
| `FanDuel-MLB-...players-list.csv` | **FanDuel player CSV** | Salaries, positions, player IDs — the universe of legal players | **Required.** Wrong day's file = slate mismatch, players silently unmatched |
| `BallparkPal_Batters/Pitchers/Games/Teams .xlsx` (4 raw sim exports) | **BallparkPal Excel files** | Baseline projections (PointsFD), team run distributions, handedness, win% | **Required.** Missing one file = loader error; stale files = projections for the wrong slate |
| `Ballpark DFS  Ballpark Pal (N).xlsx` (DFS Optimizer export, header on row 2) | **Projection / sims files** (auto-detected if dropped in the BallparkPal slot) | **Upside / Bust / Median** — powers the ceiling-weighted GPP objective | Optimizer falls back to synthetic ceilings (weaker leverage signal) |
| Ownership CSVs (optional) | **Ownership projection CSVs** | Blends real ownership into leverage + chalk fades | Falls back to the internal salary/projection ownership model |
| Lineup paste (text) | **Paste confirmed lineups** | Confirmed starters filter + batting-order boosts + platoon data | Bench/inactive players stay in the pool and dilute projections |
| Vegas CSV (optional, advanced mode) | **Vegas lines** | Team-total multipliers on projections, stack game filters | Falls back to BallparkPal sim run totals |

**Do not** upload the FanDuel CSV into the BallparkPal slot or vice versa — the
pipeline will stop with an error naming the file it couldn't read.

**Tip:** upload your contest **entries-upload-template** (from FanDuel's "Enter
lineups via CSV") instead of the players list — it contains the same player data
*plus* your entry IDs, so Step 5 produces a file you upload straight back to
FanDuel. No copy-pasting into the template.
"""
        )

    # ── Quick load from Downloads ─────────────────────────────────────
    detected = _detect_downloads_slate_files()
    with st.container(border=True):
        st.markdown("**⚡ Quick load from Downloads**")
        if detected is None:
            st.caption(
                "No complete BallparkPal export set found in your Downloads folder. "
                "Download today's Batters/Pitchers/Games/Teams files (plus the "
                "Ballpark DFS export and a FanDuel CSV) and this panel lights up."
            )
        else:
            age = float(detected["age_hours"])
            fd_kind = (
                "entries template — direct-upload export enabled ✅"
                if detected["fanduel_is_template"]
                else "players list — Step 5 CSV will need pasting into FanDuel's template "
                "(download the entries template instead to skip that)"
            )
            file_lines = [
                f"- BallparkPal sims: {', '.join(p.name for p in detected['bpp'])}",
                f"- Upside/Bust/Median: {', '.join(p.name for p in detected['dfs']) or 'none found ⚠️ (ceiling data will be synthetic)'}",
                f"- FanDuel: {detected['fanduel'].name} ({fd_kind})",
            ]
            st.markdown("\n".join(file_lines))
            if age > 18:
                st.warning(
                    f"Newest BallparkPal export is {age:.0f} hours old — that's probably "
                    "yesterday's slate. Download fresh files before building."
                )
            else:
                st.caption(f"Newest export downloaded {age:.1f} h ago.")
            quick_paste = st.text_area(
                "Paste confirmed lineups (optional but recommended)",
                height=120,
                key="quick_lineup_paste",
                help="Paste the matchup/lineup block from BallparkPal's game pages. "
                "Filters the pool to confirmed starters and powers batting-order boosts.",
            )
            if st.button("⚡ Process detected files", type="primary", key="quick_process"):
                try:
                    config_state = _get_config_state()
                    model_defaults = _ownership_model_preset_values(DEFAULT_MODEL_PRESET)
                    fanduel_proxy = _FileProxy(detected["fanduel"])
                    bpp_proxies = [_FileProxy(p) for p in detected["bpp"]]
                    dfs_proxies = [_FileProxy(p) for p in detected["dfs"]]
                    with st.spinner("Running ingestion + projection pipeline on detected files..."):
                        workflow_payload = _process_slate(
                            fanduel_proxy,
                            bpp_proxies,
                            None,
                            None,
                            None,
                            None,
                            [],
                            dfs_proxies,
                            lineup_paste_text=quick_paste or "",
                            projection_preset=config_state.get("projection_preset", DEFAULT_WEIGHT_PRESET),
                            projection_weights_input=config_state.get("projection_weights_input", ""),
                            projection_baseline_weight=float(config_state.get("projection_baseline_weight", 1.0)),
                            ownership_preset=config_state.get("ownership_preset", DEFAULT_WEIGHT_PRESET),
                            ownership_weights_input=config_state.get("ownership_weights_input", ""),
                            ownership_model_settings={
                                "preset": DEFAULT_MODEL_PRESET,
                                "manual_override": False,
                                "values": model_defaults,
                            },
                            recency_blend_input=config_state.get("recency_blend_input", "0.7,0.3"),
                            platoon_opposite_boost=float(config_state.get("platoon_opp", 1.06)),
                            platoon_same_penalty=float(config_state.get("platoon_same", 0.95)),
                            platoon_switch_boost=float(config_state.get("platoon_switch", 1.03)),
                        )
                        session_state = _get_session()
                        session_state.update(workflow_payload)
                        _auto_configure_for_slate(session_state)
                        try:
                            _persist_slate_files(
                                fanduel_proxy,
                                bpp_proxies,
                                None,
                                None,
                                None,
                                None,
                                [],
                                dfs_proxies,
                                quick_paste or "",
                            )
                        except OSError as persist_exc:
                            st.warning(f"Slate processed, but caching files for re-use failed: {persist_exc}")
                    if _daily_mode_enabled():
                        st.session_state[NAV_KEY] = "3. Configure & Optimize"
                    st.success("Slate processed from Downloads. Strategy auto-applied — review it in Step 3.")
                    st.rerun()
                except Exception as exc:  # pylint: disable=broad-except
                    st.error(f"Quick load failed: {exc}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
    st.divider()

    # ── Saved slate panel (last processed slate, for re-use) ─────────
    cache = _load_slate_cache()
    if cache:
        saved_at = str(cache.get("saved_at", ""))
        saved_date = saved_at.split(" ")[0]
        today_str = pd.Timestamp.now(tz="US/Eastern").strftime("%Y-%m-%d")
        is_stale = bool(saved_date) and saved_date != today_str
        stale_tag = " ⚠️ previous slate" if is_stale else ""
        with st.expander(
            f"🗂️ Re-use last processed slate (saved {saved_at}){stale_tag}",
            expanded=False,
        ):
            if is_stale:
                st.warning(
                    f"These files were saved {saved_date} — that slate is over. "
                    "For today's build use ⚡ Quick load above (or upload fresh files below)."
                )
            bpp_names = ", ".join(cache.get("bpp_files", []))
            st.caption(
                f"FanDuel: `{cache['fanduel']}`  \n"
                f"BallparkPal: `{bpp_names}`"
                + (f"  \nVegas: `{cache['vegas']}`" if cache.get("vegas") else "")
                + (f"  \nBatting orders: `{cache['batting_orders']}`" if cache.get("batting_orders") else "")
                + (f"  \nOwnership files: `{', '.join(cache['ownership_files'])}`" if cache.get("ownership_files") else "")
                + (f"  \nProjection files: `{', '.join(cache['projection_files'])}`" if cache.get("projection_files") else "")
                + (f"  \nLineup paste: {len(cache['lineup_paste'])} characters saved" if cache.get("lineup_paste") else "")
            )
            btn_col, clear_col = st.columns([2, 1])
            if btn_col.button("Re-use saved slate", type="primary", key="reuse_slate"):
                try:
                    proxies = _proxies_from_manifest(cache)
                    config_state = _get_config_state()
                    with st.spinner("Re-processing saved slate..."):
                        workflow_payload = _process_slate(
                            proxies["fanduel_file"],
                            proxies["bpp_files"],
                            proxies["vegas_file"],
                            proxies["batting_file"],
                            proxies["handedness_file"],
                            proxies["recent_file"],
                            proxies["ownership_files"],
                            proxies["projection_files"],
                            lineup_paste_text=proxies["lineup_paste"],
                            projection_preset=config_state.get("projection_preset"),
                            projection_weights_input=config_state.get("projection_weights_input"),
                            projection_baseline_weight=float(config_state.get("projection_baseline_weight", 1.0)),
                            ownership_preset=config_state.get("ownership_preset"),
                            ownership_weights_input=config_state.get("ownership_weights_input"),
                            ownership_model_settings=config_state.get("ownership_model_settings"),
                            recency_blend_input=config_state.get("recency_blend_input", "0.7,0.3"),
                            platoon_opposite_boost=float(config_state.get("platoon_opp", 1.06)),
                            platoon_same_penalty=float(config_state.get("platoon_same", 0.95)),
                            platoon_switch_boost=float(config_state.get("platoon_switch", 1.03)),
                        )
                        session_state = _get_session()
                        session_state.update(workflow_payload)
                        _auto_configure_for_slate(session_state)
                    if _daily_mode_enabled():
                        st.session_state[NAV_KEY] = "3. Configure & Optimize"
                    next_step = "Step 3" if _daily_mode_enabled() else "Step 2"
                    st.success(f"Slate re-loaded from saved files. Proceed to {next_step}.")
                    st.rerun()
                except Exception as exc:  # pylint: disable=broad-except
                    st.error(f"Failed to reload saved slate: {exc}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
            if clear_col.button("Clear saved slate", type="secondary", key="clear_slate"):
                failed: List[str] = []
                for f in UPLOAD_CACHE_DIR.iterdir():
                    if not f.is_file():
                        continue  # never touch subdirectories
                    try:
                        f.unlink()
                    except OSError as exc:
                        failed.append(f"{f.name} ({exc})")
                if failed:
                    st.warning("Cleared most files, but these were locked: " + ", ".join(failed))
                else:
                    st.success("Saved slate cleared.")
                st.rerun()
        st.divider()
        st.caption("Or upload new files below to replace the saved slate:")

    col1, col2 = st.columns(2)
    fanduel_file = col1.file_uploader(
        "FanDuel CSV (players list — or your contest entries template)",
        type=["csv"],
        accept_multiple_files=False,
        help=(
            "Best option: the entries-upload-template from FanDuel's 'Enter lineups "
            "via CSV' page. It carries your entry IDs, so Step 5 exports a file you "
            "upload straight back — no copy-pasting. The plain players list works too."
        ),
    )
    bpp_files = col2.file_uploader(
        "BallparkPal Excel files (Batters, Pitchers, Games, Teams)",
        type=["xlsx"],
        accept_multiple_files=True,
        help="The Ballpark DFS (Upside/Bust/Median) export can be dropped here too — it's auto-detected and rerouted.",
    )

    if _daily_mode_enabled() and not _advanced_controls_enabled():
        st.markdown("**Optional daily inputs**")
        ownership_files = st.file_uploader(
            "Ownership projection CSVs",
            type=["csv"],
            accept_multiple_files=True,
            key="ownership_sources",
        )
        projection_files = st.file_uploader(
            "Projection / sims files",
            type=["csv", "xlsx"],
            accept_multiple_files=True,
            key="projection_sources",
            help="Upload alternate projection sets, including Ballpark DFS Optimizer workbooks with Bust/Median/Upside.",
        )
        lineup_paste = st.text_area(
            "Paste confirmed lineups",
            height=140,
            key="lineup_paste",
            help="Optional. Paste lineups from FantasyLabs, RotoGrinders, etc. when you have them.",
        )

        model_defaults = _ownership_model_preset_values(DEFAULT_MODEL_PRESET)
        ownership_model_settings = {
            "preset": DEFAULT_MODEL_PRESET,
            "manual_override": False,
            "values": model_defaults,
        }
        config_state = _get_config_state()
        if st.button("Process Slate", type="primary"):
            try:
                with st.spinner("Running ingestion + projection pipeline..."):
                    workflow_payload = _process_slate(
                        fanduel_file,
                        bpp_files or [],
                        None,
                        None,
                        None,
                        None,
                        ownership_files or [],
                        projection_files or [],
                        lineup_paste_text=lineup_paste or "",
                        projection_preset=config_state.get("projection_preset", DEFAULT_WEIGHT_PRESET),
                        projection_weights_input=config_state.get("projection_weights_input", ""),
                        projection_baseline_weight=float(config_state.get("projection_baseline_weight", 1.0)),
                        ownership_preset=config_state.get("ownership_preset", DEFAULT_WEIGHT_PRESET),
                        ownership_weights_input=config_state.get("ownership_weights_input", ""),
                        ownership_model_settings=ownership_model_settings,
                        recency_blend_input=config_state.get("recency_blend_input", "0.7,0.3"),
                        platoon_opposite_boost=float(config_state.get("platoon_opp", 1.06)),
                        platoon_same_penalty=float(config_state.get("platoon_same", 0.95)),
                        platoon_switch_boost=float(config_state.get("platoon_switch", 1.03)),
                    )
                    session_state = _get_session()
                    session_state.update(workflow_payload)
                    _auto_configure_for_slate(session_state)
                    if fanduel_file and bpp_files:
                        _persist_slate_files(
                            fanduel_file,
                            bpp_files or [],
                            None,
                            None,
                            None,
                            None,
                            ownership_files or [],
                            projection_files or [],
                            lineup_paste or "",
                        )
                if _daily_mode_enabled():
                    st.session_state[NAV_KEY] = "3. Configure & Optimize"
                st.success("Slate processed and saved. Go to Step 3 to build the candidate pool.")
                st.rerun()
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Failed to process slate: {exc}")
                import traceback
                st.code(traceback.format_exc(), language="python")

        current = _get_session()
        if current.get("optimizer") is not None:
            st.divider()
            render_validation_summary(current)
        return

    # ── Vegas lines: manual upload OR auto-fetch ──
    st.markdown("**Vegas lines**")
    vegas_col1, vegas_col2 = st.columns([3, 1])
    vegas_file = vegas_col1.file_uploader("Upload Vegas CSV (optional)", type=["csv"], key="vegas")
    if vegas_col2.button("Fetch from API", key="fetch_vegas", help="Pull today's odds from The Odds API (requires ODDS_API_KEY env var)"):
        try:
            with st.spinner("Fetching Vegas lines..."):
                vegas_lines_result = fetch_vegas_lines()
                # Store as CSV in session state so _process_slate can consume it
                csv_buf = vegas_lines_result.games[["game", "total", "home_ml", "away_ml"]].to_csv(index=False)
                st.session_state["fetched_vegas_csv"] = csv_buf
                n_games = len(vegas_lines_result.games)
            st.success(f"Fetched odds for {n_games} games.")
        except Exception as exc:
            st.error(f"Vegas fetch failed: {exc}")
    if st.session_state.get("fetched_vegas_csv"):
        st.caption("Auto-fetched Vegas lines loaded. Manual upload will override.")

    # ── Batting orders: manual upload, paste, OR auto-fetch ──
    st.markdown("**Batting orders**")
    orders_col1, orders_col2 = st.columns([3, 1])
    batting_orders_file = orders_col1.file_uploader("Upload batting orders CSV (optional)", type=["csv"], key="orders")
    if orders_col2.button("Fetch from MLB", key="fetch_lineups", help="Pull today's lineups from MLB Stats API (free, no key needed)"):
        try:
            with st.spinner("Fetching MLB lineups..."):
                status = lineup_fetch_status()
                orders_df = fetch_batting_orders()
                csv_buf = orders_df.to_csv(index=False)
                st.session_state["fetched_lineups_csv"] = csv_buf
                n_teams = orders_df["team"].nunique()
            st.success(
                f"Fetched lineups for {n_teams} teams "
                f"({status['games_with_lineups']}/{status['total_games']} games have lineups posted)."
            )
        except Exception as exc:
            st.error(f"Lineup fetch failed: {exc}")
    if st.session_state.get("fetched_lineups_csv"):
        st.caption("Auto-fetched MLB lineups loaded. Manual upload/paste will override.")

    lineup_paste = st.text_area(
        "Paste lineups from FantasyLabs / RotoGrinders (optional)",
        height=200,
        key="lineup_paste",
        help="Copy and paste the full lineup page from FantasyLabs, RotoGrinders, etc. "
             "This will be used as batting order data. Overrides CSV upload and API fetch if provided.",
    )
    handedness_file = st.file_uploader(
        "Handedness reference CSV (optional)",
        type=["csv"],
        key="handedness",
        help="This file changes infrequently; upload once and reuse.",
    )
    recent_stats_file = st.file_uploader(
        "Recent stats CSV (optional)",
        type=["csv"],
        key="recent_stats",
    )

    ownership_files = st.file_uploader(
        "Ownership projection CSVs (optional, multiple)",
        type=["csv"],
        accept_multiple_files=True,
        key="ownership_sources",
    )
    ownership_preset = st.selectbox(
        "Ownership weight preset",
        WEIGHT_PRESET_OPTIONS,
        index=WEIGHT_PRESET_OPTIONS.index(DEFAULT_WEIGHT_PRESET),
        help="Pick a blend strategy for multi-source ownership inputs. Source order follows the upload order.",
    )
    if ownership_files:
        preview_text = _weight_preset_preview(ownership_preset, ownership_files)
        if preview_text:
            st.caption(f"Preset weights -> {preview_text}")
    ownership_weights_input = st.text_input(
        "Ownership weights (comma-separated, optional)",
        placeholder="0.4,0.3,0.3",
        help="Leave blank to use the selected preset. Enter comma-separated weights to override (one per ownership file).",
    )

    model_preset_index = OWNERSHIP_MODEL_OPTIONS.index(DEFAULT_MODEL_PRESET)
    ownership_model_preset = st.selectbox(
        "Ownership model preset",
        OWNERSHIP_MODEL_OPTIONS,
        index=model_preset_index,
        help="Controls how the fallback ownership model weighs salary/projection/value/etc.",
    )
    active_model_key = "ownership_model_active_preset"
    if active_model_key not in st.session_state:
        _apply_ownership_model_preset(DEFAULT_MODEL_PRESET)
        st.session_state[active_model_key] = DEFAULT_MODEL_PRESET
    if ownership_model_preset != OWNERSHIP_MODEL_CUSTOM and st.session_state.get(active_model_key) != ownership_model_preset:
        _apply_ownership_model_preset(ownership_model_preset)
        st.session_state[active_model_key] = ownership_model_preset
    elif ownership_model_preset == OWNERSHIP_MODEL_CUSTOM and st.session_state.get(active_model_key) != OWNERSHIP_MODEL_CUSTOM:
        st.session_state[active_model_key] = OWNERSHIP_MODEL_CUSTOM

    ownership_model_values: Dict[str, float] = {}
    with st.expander("Ownership model tuning", expanded=False):
        model_cols = st.columns(3)
        preset_defaults = _ownership_model_preset_values(DEFAULT_MODEL_PRESET)
        for idx, (field, label, min_val, max_val, step_val, help_text) in enumerate(OWNERSHIP_MODEL_FIELDS):
            session_key = f"ownership_model_{field}"
            if session_key not in st.session_state:
                st.session_state[session_key] = preset_defaults.get(field, 0.0)
            column = model_cols[idx % len(model_cols)]
            ownership_model_values[field] = float(
                column.number_input(
                    label,
                    min_value=min_val,
                    max_value=max_val,
                    value=float(st.session_state.get(session_key, preset_defaults.get(field, 0.0))),
                    step=step_val,
                    key=session_key,
                    help=help_text,
                )
            )
    ownership_model_settings = {
        "preset": ownership_model_preset,
        "manual_override": ownership_model_preset == OWNERSHIP_MODEL_CUSTOM,
        "values": ownership_model_values,
    }

    projection_files = st.file_uploader(
        "Projection files (optional, multiple)",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key="projection_sources",
        help="Upload alternate projection sets, including Ballpark DFS Optimizer workbooks with Bust/Median/Upside.",
    )
    projection_preset = st.selectbox(
        "Projection blend preset",
        WEIGHT_PRESET_OPTIONS,
        index=WEIGHT_PRESET_OPTIONS.index(DEFAULT_WEIGHT_PRESET),
        help="Blend strategy for projection sources. Upload order determines anchor order.",
    )
    if projection_files:
        preview_text = _weight_preset_preview(projection_preset, projection_files)
        if preview_text:
            st.caption(f"Projection preset weights -> {preview_text}")
    projection_weights_input = st.text_input(
        "Projection weights (comma-separated, optional)",
        placeholder="0.6,0.4",
        help="Overrides the preset weights for projection sources (BallparkPal weight controlled separately).",
    )
    projection_baseline_weight = st.number_input(
        "BallparkPal baseline weight",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Controls how much the BallparkPal baseline contributes relative to uploaded sources before normalization.",
    )
    recency_blend_input = st.text_input(
        "Recency blend weights (season,recent)",
        value="0.7,0.3",
        help="Enter two comma-separated values like 0.7,0.3 (season first, then recent).",
    )
    platoon_opp_input = st.number_input(
        "Opposite-hand multiplier",
        min_value=0.5,
        max_value=1.5,
        value=1.06,
        step=0.01,
        help="Boost hitters vs. opposite-hand pitchers (default 1.06).",
    )
    platoon_same_input = st.number_input(
        "Same-hand multiplier",
        min_value=0.5,
        max_value=1.2,
        value=0.95,
        step=0.01,
        help="Penalty for same-hand matchups (default 0.95).",
    )
    platoon_switch_input = st.number_input(
        "Switch-hitter multiplier",
        min_value=0.5,
        max_value=1.5,
        value=1.03,
        step=0.01,
        help="Adjustment for switch hitters (default 1.03).",
    )

    st.caption("Recency weights mix season vs. short-term production, while the platoon multipliers control how much hitters gain or lose against pitcher handedness (defaults shown above).")

    # ── Resolve API-fetched data as fallbacks for manual uploads ──
    effective_vegas_file = vegas_file
    if not effective_vegas_file and st.session_state.get("fetched_vegas_csv"):
        from io import BytesIO
        buf = BytesIO(st.session_state["fetched_vegas_csv"].encode("utf-8"))
        buf.name = "vegas_api_fetched.csv"
        effective_vegas_file = buf

    effective_batting_file = batting_orders_file
    if not effective_batting_file and not (lineup_paste and lineup_paste.strip()) and st.session_state.get("fetched_lineups_csv"):
        from io import BytesIO
        buf = BytesIO(st.session_state["fetched_lineups_csv"].encode("utf-8"))
        buf.name = "mlb_lineups_api_fetched.csv"
        effective_batting_file = buf

    if st.button("Process Slate", type="primary"):
        try:
            with st.spinner("Running ingestion + projection pipeline..."):
                workflow_payload = _process_slate(
                    fanduel_file,
                    bpp_files or [],
                    effective_vegas_file,
                    effective_batting_file,
                    handedness_file,
                    recent_stats_file,
                    ownership_files or [],
                    projection_files or [],
                    lineup_paste_text=lineup_paste or "",
                    projection_preset=projection_preset,
                    projection_weights_input=projection_weights_input,
                    projection_baseline_weight=projection_baseline_weight,
                    ownership_preset=ownership_preset,
                    ownership_weights_input=ownership_weights_input,
                    ownership_model_settings=ownership_model_settings,
                    recency_blend_input=recency_blend_input,
                    platoon_opposite_boost=platoon_opp_input,
                    platoon_same_penalty=platoon_same_input,
                    platoon_switch_boost=platoon_switch_input,
                )
                session_state = _get_session()
                session_state.update(workflow_payload)
                _auto_configure_for_slate(session_state)
                # Auto-save all uploaded files for one-click reload next session
                if fanduel_file and bpp_files:
                    _persist_slate_files(
                        fanduel_file,
                        bpp_files or [],
                        vegas_file,
                        batting_orders_file,
                        handedness_file,
                        recent_stats_file,
                        ownership_files or [],
                        projection_files or [],
                        lineup_paste or "",
                    )
            st.success("Slate processed and saved. Next time, use **Re-use saved slate** to skip re-uploading.")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Failed to process slate: {exc}")
            import traceback
            st.code(traceback.format_exc(), language="python")

    current = _get_session()
    if current.get("optimizer") is not None:
        st.divider()
        render_validation_summary(current)


def _render_step_two() -> None:
    st.header("Step 2 · Review Projections")
    workflow = _get_session()
    _render_lock_countdown(workflow)
    optimizer_df: Optional[pd.DataFrame] = workflow.get("optimizer")
    if optimizer_df is None or optimizer_df.empty:
        st.info("Process a slate first (Step 1) to load projections.")
        return

    ownership_result = workflow.get("ownership_summary")
    if ownership_result:
        _render_ownership_summary(ownership_result)

    projection_cfg = workflow.get("projection_config") or {}
    _render_projection_config_summary(config=projection_cfg)
    _render_projection_blend_summary(workflow.get("projection_blend_summary") or projection_cfg.get("projection_blend"))
    _render_ownership_model_summary(projection_cfg.get("ownership_model"))

    with st.sidebar.expander("Projection Filters", expanded=True):
        filtered_df = _apply_filters(optimizer_df)

    base_columns = [
        "full_name",
        "team_code",
        "position",
        "salary",
        "proj_fd_mean",
        "proj_fd_ceiling",
        "proj_fd_ownership",
        "player_leverage_score",
        "ownership_edge",
    ]
    factor_options = {
        "Base projection": "base_projection",
        "Value score": "value_score",
        "Vegas multiplier": "vegas_multiplier",
        "Order factor": "order_factor",
        "Platoon factor": "platoon_factor",
        "Recency factor": "recency_factor",
        "Floor multiplier": "floor_multiplier",
        "Ceiling multiplier": "ceiling_multiplier",
        "Team total": "vegas_team_total",
    }
    with st.expander("Projection factors displayed", expanded=False):
        selected_labels = st.multiselect(
            "Select factor columns",
            list(factor_options.keys()),
            default=["Value score", "Vegas multiplier", "Order factor", "Recency factor"],
        )
    extra_columns = [factor_options[label] for label in selected_labels if factor_options[label] in filtered_df.columns]
    display_cols = [col for col in base_columns + extra_columns if col in filtered_df.columns]
    st.subheader("Projection table")
    st.dataframe(filtered_df[display_cols], width="stretch")
    st.download_button(
        label="Download filtered projections",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_projections.csv",
        mime="text/csv",
    )

    st.subheader("Factor leaders")
    col_val, col_floor, col_ceiling = st.columns(3)
    with col_val:
        top_value = _factor_leader_table(filtered_df, "value_score", "Value score")
        if top_value.empty:
            st.info("Value score unavailable.")
        else:
            st.table(top_value)
    with col_floor:
        top_floor = _factor_leader_table(filtered_df, "floor_multiplier", "Floor x", ascending=False)
        if top_floor.empty:
            st.info("Floor multipliers unavailable.")
        else:
            st.table(top_floor)
    with col_ceiling:
        top_ceiling = _factor_leader_table(filtered_df, "ceiling_multiplier", "Ceiling x", ascending=False)
        if top_ceiling.empty:
            st.info("Ceiling multipliers unavailable.")
        else:
            st.table(top_ceiling)

    st.subheader("Variance explorer")
    _variance_scatter(filtered_df)
    st.subheader("High-volatility targets")
    _variance_leaderboard(filtered_df)

    # Manual overrides
    override_cols = [col for col in ["proj_fd_mean", "proj_fd_ownership"] if col in filtered_df.columns]
    if override_cols:
        st.subheader("Manual overrides")
        editor_df = filtered_df[["fd_player_id", "full_name"] + override_cols].set_index("fd_player_id")
        edited_df = st.data_editor(
            editor_df,
            key="projection_editor",
            num_rows="dynamic",
        )
        if st.button("Save Overrides"):
            projections = workflow.get("projections")
            players = workflow.get("players")
            if projections is None or players is None:
                st.error("Missing projections in session state; reprocess the slate.")
            else:
                for fd_player_id, row in edited_df.iterrows():
                    mask = projections["fd_player_id"] == fd_player_id
                    if mask.any():
                        for col in override_cols:
                            projections.loc[mask, col] = row[col]
                workflow["projections"] = projections
                optimizer_df = build_optimizer_dataset(players, projections)
                optimizer_df = _apply_ownership_edge(optimizer_df)
                workflow["optimizer"] = optimizer_df
                st.success("Overrides applied. Projection table refreshed.")
                st.rerun()

    st.subheader("Stack overview")
    _section_stacks(filtered_df)
    st.subheader("Pitcher board")
    _section_pitchers(filtered_df)
    st.subheader("Leverage insights")
    _section_leverage(filtered_df)
    st.subheader("Ownership leaders")
    _section_chalk(filtered_df)


def _render_step_three() -> None:
    st.header("Step 3 · Configure & Optimize")
    workflow = _get_session()
    _render_lock_countdown(workflow)
    optimizer_df: Optional[pd.DataFrame] = workflow.get("optimizer")
    if optimizer_df is None or optimizer_df.empty:
        st.info("Process a slate first (Step 1) to configure and run the optimizer.")
        return

    if _advanced_controls_enabled():
        _render_optimizer_settings_guide()
    else:
        _render_advanced_caption()
    _render_daily_step_note("optimize")
    _render_strategy_banner(workflow)

    config_state = _get_config_state()
    st.subheader("One-click slate run preset")
    preset_options = list(SLATE_RUN_PRESETS.keys())
    active_preset = st.session_state.get("active_slate_run_preset", preset_options[0])
    preset_index = preset_options.index(active_preset) if active_preset in preset_options else 0
    selected_preset = st.selectbox(
        "Preset",
        preset_options,
        index=preset_index,
        help="Applies optimizer and simulation settings together for the slate type you are attacking.",
    )
    st.caption(SLATE_RUN_PRESETS[selected_preset].get("description", ""))
    stack_plan_for_preset = build_stack_exposure_plan(optimizer_df)
    recommendation = recommend_slate_preset(optimizer_df, stack_plan_for_preset)
    rec_cols = st.columns([2, 1])
    rec_cols[0].caption(
        f"Recommended today: {recommendation.preset} "
        f"({recommendation.confidence:.0%} confidence). "
        + " ".join(recommendation.reasons[:1])
    )
    if rec_cols[1].button("Apply Recommendation"):
        _apply_slate_run_preset(recommendation.preset)
        st.success(f"Applied {recommendation.preset}.")
        st.rerun()
    if st.button("Apply Slate Preset"):
        _apply_slate_run_preset(selected_preset)
        st.success(f"Applied {selected_preset}. Review settings, then run Step 3 and Step 4.")
        st.rerun()

    config_state["num_lineups"] = st.number_input(
        "Candidate lineup pool size",
        min_value=1,
        max_value=1000,
        value=int(config_state.get("num_lineups", 500) or 500),
        step=1,
        help="Generate more candidates than you plan to submit. For 300 entries, a 500-lineup pool gives Step 4 room to select the best 300.",
    )
    if _advanced_controls_enabled():
        config_state["salary_cap"] = st.number_input(
            "Salary cap",
            min_value=10000,
            max_value=40000,
            value=int(config_state.get("salary_cap", 35000) or 35000),
            step=100,
        )
    else:
        config_state["salary_cap"] = int(config_state.get("salary_cap", 35000) or 35000)
    preset_labels = list(STACK_PRESETS.keys())
    current_selected = config_state.get("stack_template_selections", ["4-3-1"])
    valid_selections = [s for s in current_selected if s in preset_labels]
    if not valid_selections:
        valid_selections = ["4-3-1"]
    config_state["stack_template_selections"] = st.multiselect(
        "Stack templates",
        options=preset_labels,
        default=valid_selections,
        help="Select one or more stacking strategies. Lineups are distributed across your selections. Each number is batters from one team (sum to 8). 'Auto' = no stacking constraints.",
    )
    selected_templates = config_state["stack_template_selections"]
    if not selected_templates:
        selected_templates = ["Auto (optimizer's choice)"]
        config_state["stack_template_selections"] = selected_templates

    # Show per-template lineup counts when multiple selected
    if len(selected_templates) > 1:
        st.caption("Lineups per template")
        num_lineups_total = int(config_state.get("num_lineups", 20) or 20)
        template_counts: Dict[str, int] = config_state.get("template_counts", {})
        even_share = max(1, num_lineups_total // len(selected_templates))
        remainder = num_lineups_total - even_share * len(selected_templates)
        count_cols = st.columns(min(len(selected_templates), 4))
        new_counts: Dict[str, int] = {}
        for i, label in enumerate(selected_templates):
            default_ct = template_counts.get(label, even_share + (1 if i < remainder else 0))
            with count_cols[i % len(count_cols)]:
                new_counts[label] = st.number_input(
                    label.split("(")[0].strip(),
                    min_value=0,
                    max_value=500,
                    value=int(default_ct),
                    step=1,
                    key=f"stack_ct_{label}",
                )
        config_state["template_counts"] = new_counts
    st.markdown("**Chalk controls**")
    chalk_col_left, chalk_col_right = st.columns(2)
    with chalk_col_left:
        st.caption("Batters")
        config_state["batter_chalk_threshold"] = st.number_input(
            "Batter chalk threshold (%)",
            min_value=0.0,
            max_value=50.0,
            value=float(config_state.get("batter_chalk_threshold", 25.0) or 0.0),
            step=1.0,
            help="Ownership % above which a batter is considered chalk",
        )
        config_state["batter_chalk_exposure_cap"] = st.number_input(
            "Batter chalk exposure cap (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(config_state.get("batter_chalk_exposure_cap", 30.0) or 0.0),
            step=1.0,
            help="Max % of lineups a chalky batter can appear in",
        )
    with chalk_col_right:
        st.caption("Pitchers")
        config_state["pitcher_chalk_threshold"] = st.number_input(
            "Pitcher chalk threshold (%)",
            min_value=0.0,
            max_value=50.0,
            value=float(config_state.get("pitcher_chalk_threshold", 35.0) or 0.0),
            step=1.0,
            help="Ownership % above which a pitcher is considered chalk",
        )
        config_state["pitcher_chalk_exposure_cap"] = st.number_input(
            "Pitcher chalk exposure cap (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(config_state.get("pitcher_chalk_exposure_cap", 50.0) or 0.0),
            step=1.0,
            help="Max % of lineups a chalky pitcher can appear in",
        )
    st.markdown("**Leverage strategy**")
    config_state["leverage_weight"] = st.number_input(
        "Leverage weight (%)",
        min_value=0.0,
        max_value=50.0,
        value=float(config_state.get("leverage_weight", 15.0) or 0.0),
        step=1.0,
        help=(
            "How much to favor high-leverage plays (high projection + low ownership) "
            "over chalk. At 15%, a player projected for 5.8 pts with strong leverage "
            "can beat a 6.0 pt chalk play. Set to 0 to disable (pure projection)."
        ),
    )
    config_state["randomness"] = st.number_input(
        "Solver randomness (%)",
        min_value=0.0,
        max_value=25.0,
        value=float(config_state.get("randomness", 5.0) or 0.0),
        step=1.0,
        help=(
            "Adds small random noise to projections so each lineup isn't identical. "
            "At 5%, a 6.0 pt projection gets jittered to ~5.7–6.3 per lineup solve. "
            "Higher values = more diverse pool. 0 = deterministic (same lineup repeated)."
        ),
    )
    gpp_col1, gpp_col2 = st.columns(2)
    config_state["ceiling_weight"] = gpp_col1.number_input(
        "Ceiling weight (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(config_state.get("ceiling_weight", 35.0) or 0.0),
        step=5.0,
        help=(
            "GPP objective blend between mean projection and BPP Upside. "
            "At 35%, objective = 0.65×mean + 0.35×upside. Requires the Ballpark "
            "DFS Optimizer file for real Upside data; falls back to synthetic "
            "ceilings otherwise. 0 = old cash-style mean-only objective."
        ),
    )
    config_state["ownership_penalty_weight"] = gpp_col2.number_input(
        "Ownership penalty (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(config_state.get("ownership_penalty_weight", 10.0) or 0.0),
        step=1.0,
        help=(
            "Soft chalk fade: subtracts penalty × ownership × (pool avg projection) "
            "from each player's objective score. Unlike the hard chalk caps above, "
            "this lets the solver keep chalk when the ceiling justifies it."
        ),
    )
    sal_col1, sal_col2 = st.columns(2)
    config_state["min_salary"] = sal_col1.number_input(
        "Min salary spend ($)",
        min_value=0,
        max_value=35000,
        value=int(config_state.get("min_salary", 34000) or 0),
        step=100,
        help="Floor on total lineup salary. Fixes salary underutilization; set 0 to disable.",
    )
    config_state["min_uniques"] = sal_col2.number_input(
        "Min unique players between lineups",
        min_value=1,
        max_value=5,
        value=int(config_state.get("min_uniques", 1) or 1),
        step=1,
        help="Each lineup must differ from every other by at least this many players. 2–3 recommended for large-field GPPs.",
    )
    if _advanced_controls_enabled():
        with st.expander("Advanced optimizer controls", expanded=True):
            config_state["max_lineup_ownership"] = st.number_input(
                "Max lineup ownership (sum %, optional)",
                min_value=0.0,
                max_value=500.0,
                value=float(config_state.get("max_lineup_ownership", 0.0) or 0.0),
                step=1.0,
            )
            config_state["player_overrides"] = st.text_area(
                "Player-specific exposure overrides",
                value=config_state.get("player_overrides", ""),
                help="One per line, e.g., Shohei Ohtani:0.5",
            )
            _render_exposure_auto_tuner(optimizer_df, config_state)
            config_state["bring_back_enabled"] = st.checkbox(
                "Enable bring-back requirement",
                value=bool(config_state.get("bring_back_enabled", False)),
            )
            if config_state["bring_back_enabled"]:
                config_state["bring_back_count"] = st.number_input(
                    "Bring-back hitter count",
                    min_value=1,
                    max_value=5,
                    value=int(config_state.get("bring_back_count", 1) or 1),
                    step=1,
                )
            config_state["min_game_total"] = st.number_input(
                "Min Vegas total for stacks (set 0 to disable)",
                min_value=0.0,
                max_value=20.0,
                value=float(config_state.get("min_game_total", 0.0) or 0.0),
                step=0.5,
            )
    else:
        config_state["max_lineup_ownership"] = float(config_state.get("max_lineup_ownership", 0.0) or 0.0)
        config_state["player_overrides"] = str(config_state.get("player_overrides", "") or "")
        config_state["bring_back_enabled"] = bool(config_state.get("bring_back_enabled", False))
        config_state["bring_back_count"] = int(config_state.get("bring_back_count", 1) or 1)
        config_state["min_game_total"] = float(config_state.get("min_game_total", 0.0) or 0.0)

    _render_readiness_checklist(workflow)

    if _advanced_controls_enabled():
        col_save, col_load = st.columns(2)
        with col_save:
            config_json = json.dumps(config_state, indent=2).encode("utf-8")
            st.download_button(
                "Save Config",
                data=config_json,
                file_name="optimizer_config.json",
                mime="application/json",
            )
        with col_load:
            uploaded_config = st.file_uploader("Load Config", type=["json"], key="config_loader")
            if uploaded_config is not None:
                try:
                    loaded_data = json.loads(uploaded_config.getvalue().decode("utf-8"))
                    config_state.update(loaded_data)
                    st.success("Configuration loaded.")
                    st.rerun()
                except Exception as exc:  # pylint: disable=broad-except
                    st.error(f"Failed to load config: {exc}")

        locks_info = workflow.get("lock_settings")
        if locks_info:
            st.caption(
                f"Last lock/exclude request: locks={locks_info.get('locks')}, excludes={locks_info.get('excludes')}"
            )

    if st.button("Run Optimizer", type="primary"):
        try:
            with st.spinner("Generating lineups..."):
                lineups, lineup_df = _run_solver(optimizer_df, config_state)
                workflow["lineups"] = lineups
                workflow["lineups_df"] = lineup_df
                workflow.pop("optimizer_lineups_backup", None)
                workflow.pop("optimizer_lineups_df_backup", None)
                workflow["active_lineup_source"] = "optimizer"
                _save_run_snapshot(lineup_df, config_state, int(config_state.get("num_lineups", 500) or 500))
            if _daily_mode_enabled():
                st.session_state[NAV_KEY] = "4. Simulate & Select"
            st.success(f"Generated {len(lineups)} lineups. Proceed to Step 4.")
            if _daily_mode_enabled():
                st.rerun()
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Optimizer failed: {exc}")

    if workflow.get("lineups_df") is not None:
        st.divider()
        _render_candidate_pool_quality(workflow["lineups_df"], config_state)


def _lock_controls(lineup_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    unique_players = sorted(lineup_df["full_name"].unique())
    locks = st.multiselect("Lock players", unique_players, key="lock_players")
    excludes = st.multiselect("Exclude players", unique_players, key="exclude_players")
    return locks, excludes


def _render_lineup_summary(
    lineups,
    lineup_df: pd.DataFrame,
    bench_mask: Optional[pd.Series] = None,
    locked_mask: Optional[pd.Series] = None,
    visible_ids: Optional[Set[int]] = None,
) -> None:
    if lineup_df.empty:
        st.info("No lineup data available.")
        return
    if bench_mask is None or locked_mask is None:
        bench_mask, locked_mask = _bench_risk_flags(lineup_df)
    summary_rows = []
    allowed_ids = set(visible_ids or [])
    for idx, lineup in enumerate(lineups, start=1):
        if allowed_ids and idx not in allowed_ids:
            continue
        df = lineup.dataframe
        total_ownership = pd.to_numeric(df.get("proj_fd_ownership"), errors="coerce").fillna(0).sum()
        avg_leverage = (
            pd.to_numeric(df.get("player_leverage_score"), errors="coerce").fillna(0).mean()
            if "player_leverage_score" in df.columns
            else np.nan
        )
        summary_rows.append(
            {
                "lineup_id": idx,
                "salary": lineup.total_salary,
                "projection": lineup.total_projection,
                "ownership": total_ownership,
                "avg_leverage": avg_leverage,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    st.subheader("Lineup summary")
    if summary_df.empty:
        st.info("No lineups available for this view.")
        return
    st.dataframe(summary_df, width="stretch")

    selected_lineup = st.selectbox(
        "View lineup",
        summary_df["lineup_id"].tolist(),
        key="lineup_view",
    )
    view_df = lineup_df[lineup_df["lineup_id"] == selected_lineup]
    lineup_bench = bench_mask[lineup_df["lineup_id"] == selected_lineup].values
    lineup_locked = locked_mask[lineup_df["lineup_id"] == selected_lineup].values
    view_df = view_df.copy()
    view_df["bench_risk"] = lineup_bench
    view_df["game_locked"] = lineup_locked
    view_cols = [
        col
        for col in [
            "full_name",
            "team_code",
            "position",
            "salary",
            "proj_fd_mean",
            "proj_fd_ownership",
            "player_leverage_score",
            "bench_risk",
            "game_locked",
        ]
        if col in view_df.columns
    ]
    st.table(view_df[view_cols])
    risk_players = view_df[view_df["bench_risk"]]["full_name"].tolist()
    if risk_players:
        st.warning(
            f"Lineup {selected_lineup} bench risk: {', '.join(risk_players)}"
        )


def _player_exposure_summary(lineup_df: pd.DataFrame) -> pd.DataFrame:
    lineup_count = lineup_df["lineup_id"].nunique()
    exposures = (
        lineup_df.groupby(["full_name", "team_code"])["lineup_id"].count().reset_index(name="lineups")
    )
    exposures["exposure_pct"] = exposures["lineups"] / lineup_count
    if "proj_fd_ownership" in lineup_df.columns:
        ownership_map = (
            lineup_df.groupby("full_name")["proj_fd_ownership"].mean().rename("avg_proj_ownership")
        )
        exposures = exposures.merge(ownership_map, on="full_name", how="left")
    return exposures.sort_values("lineups", ascending=False)


def _stack_exposure_summary(lineup_df: pd.DataFrame) -> pd.DataFrame:
    hitters = lineup_df[lineup_df["player_type"].str.lower() == "batter"]
    if hitters.empty:
        return pd.DataFrame()
    team_lineups = hitters.groupby(["team_code", "lineup_id"]).size().reset_index(name="count")
    exposure = team_lineups.groupby("team_code")["lineup_id"].nunique().reset_index(name="lineups")
    total_lineups = lineup_df["lineup_id"].nunique()
    exposure["exposure_pct"] = exposure["lineups"] / total_lineups
    return exposure.sort_values("lineups", ascending=False)


def _render_stack_exposure_plan(plan: pd.DataFrame) -> None:
    if plan is None or plan.empty:
        st.info("No stack exposure recommendations available yet.")
        return
    st.subheader("Team stack exposure plan")
    display = plan.copy()
    pct_cols = [
        "target_exposure",
        "recommended_min_exposure",
        "recommended_max_exposure",
        "p_runs_ge_5",
        "p_runs_ge_6",
        "p_runs_ge_8",
        "p_hr_ge_2",
        "stack_ownership",
    ]
    for col in pct_cols:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").fillna(0.0).map(lambda x: f"{x:.1%}")
    for col in ["team_runs", "top5_projection", "top5_upside", "stack_score", "stack_leverage_score"]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").fillna(0.0).round(2)
    keep_cols = [
        "team_code",
        "tier",
        "target_exposure",
        "recommended_min_exposure",
        "recommended_max_exposure",
        "team_runs",
        "p_runs_ge_6",
        "p_runs_ge_8",
        "p_hr_ge_2",
        "top5_projection",
        "top5_upside",
        "stack_ownership",
        "stack_leverage_score",
        "recommendation",
    ]
    keep_cols = [col for col in keep_cols if col in display.columns]
    st.dataframe(display[keep_cols].head(20), width="stretch", hide_index=True)


def _render_exposure_auto_tuner(optimizer_df: pd.DataFrame, config_state: Dict) -> None:
    if optimizer_df is None or optimizer_df.empty:
        return
    with st.expander("Exposure auto-tuner", expanded=False):
        stack_plan = build_stack_exposure_plan(optimizer_df)
        recs = build_exposure_recommendations(optimizer_df, stack_plan)
        if recs.empty:
            st.info("No exposure recommendations available.")
            return
        view = recs.copy()
        for col in ["proj_fd_ownership", "player_leverage_score", "recommended_max_exposure", "team_target_exposure"]:
            if col in view.columns:
                if col in {"proj_fd_ownership", "recommended_max_exposure", "team_target_exposure"}:
                    view[col] = pd.to_numeric(view[col], errors="coerce").fillna(0.0).map(lambda x: f"{x:.1%}")
                else:
                    view[col] = pd.to_numeric(view[col], errors="coerce").fillna(0.0).round(3)
        st.dataframe(
            view[[
                "full_name",
                "team_code",
                "position",
                "player_type",
                "proj_fd_mean",
                "proj_fd_ownership",
                "player_leverage_score",
                "recommended_max_exposure",
                "reason",
            ]].head(80),
            width="stretch",
            hide_index=True,
        )
        top_n = st.number_input(
            "Apply top N player caps",
            min_value=10,
            max_value=200,
            value=80,
            step=10,
            key="auto_tune_top_n",
        )
        if st.button("Apply auto-tuned player caps", key="apply_auto_tuned_caps"):
            config_state["player_overrides"] = player_overrides_text(recs, int(top_n))
            st.success("Player-specific exposure overrides updated from auto-tuner.")
            st.rerun()


def _render_final_audit_report(
    display_df: pd.DataFrame,
    workflow: Dict,
    sim_results: Optional[Dict],
    config_state: Dict,
    audit_label: str = "Final upload",
    expected_lineups: Optional[int] = None,
) -> None:
    if display_df is None or display_df.empty:
        return
    stack_plan = None
    if sim_results:
        stack_plan = sim_results.get("stack_plan")
    if stack_plan is None or not isinstance(stack_plan, pd.DataFrame) or stack_plan.empty:
        optimizer_df = workflow.get("optimizer")
        if isinstance(optimizer_df, pd.DataFrame) and not optimizer_df.empty:
            stack_plan = build_stack_exposure_plan(optimizer_df)
    lineup_target = expected_lineups
    if lineup_target is None and "lineup_id" in display_df.columns:
        lineup_target = int(display_df["lineup_id"].nunique())
    run_audit = _build_workflow_run_audit(
        workflow,
        lineup_df=display_df,
        expected_lineups=lineup_target,
        config_state=config_state,
        stack_plan=stack_plan if isinstance(stack_plan, pd.DataFrame) else None,
    )
    if run_audit:
        _render_run_audit_report(run_audit, title=f"{audit_label} run health audit", expanded=False)

    selected_ids = sim_results.get("selected_ids") if sim_results else None
    contest_df = sim_results.get("contest_df") if sim_results else None
    report = build_portfolio_audit(
        display_df,
        optimizer_df=workflow.get("optimizer"),
        stack_plan=stack_plan,
        contest_df=contest_df if isinstance(contest_df, pd.DataFrame) else None,
        selected_ids=selected_ids,
        batter_chalk_threshold=float(config_state.get("batter_chalk_threshold", 25.0) or 25.0) / 100.0,
        pitcher_chalk_threshold=float(config_state.get("pitcher_chalk_threshold", 35.0) or 35.0) / 100.0,
        salary_cap=int(config_state.get("salary_cap", 35000) or 35000),
    )
    st.subheader(f"{audit_label} portfolio audit")
    summary = report.summary or {}
    metric_cols = st.columns(5)
    metric_cols[0].metric("Lineups", int(summary.get("lineups", 0)))
    metric_cols[1].metric("Avg salary", f"${summary.get('avg_salary', 0.0):,.0f}")
    metric_cols[2].metric("Avg ownership", f"{summary.get('avg_lineup_ownership', 0.0):.0%}")
    metric_cols[3].metric("Avg leverage", f"{summary.get('avg_leverage', 0.0):+.3f}")
    metric_cols[4].metric("Issues", int(len(report.issues)))

    if report.issues.empty:
        st.success("No final audit issues detected.")
    else:
        st.dataframe(report.issues, width="stretch", hide_index=True)

    if not report.stack_exposure.empty:
        st.markdown("**Stack exposure vs recommendation**")
        stack_view = report.stack_exposure.copy()
        for col in ["exposure_pct", "target_exposure", "recommended_min_exposure", "recommended_max_exposure", "target_delta", "p_runs_ge_6", "stack_ownership"]:
            if col in stack_view.columns:
                stack_view[col] = pd.to_numeric(stack_view[col], errors="coerce").fillna(0.0).map(lambda x: f"{x:.1%}")
        st.dataframe(stack_view, width="stretch", hide_index=True)

    with st.expander("Player exposure audit", expanded=False):
        player_view = report.player_exposure.copy()
        if not player_view.empty:
            for col in ["exposure_pct", "ownership"]:
                player_view[col] = pd.to_numeric(player_view[col], errors="coerce").fillna(0.0).map(lambda x: f"{x:.1%}")
            st.dataframe(player_view.head(80), width="stretch", hide_index=True)
    with st.expander("Lineup-level checks", expanded=False):
        st.dataframe(report.lineup_checks, width="stretch", hide_index=True)

    st.download_button(
        "Download final audit report",
        data=audit_report_to_csv(report).encode("utf-8"),
        file_name="final_portfolio_audit.csv",
        mime="text/csv",
    )


def _render_lineup_scorecard(
    display_df: pd.DataFrame,
    sim_results: Optional[Dict],
    stack_plan: Optional[pd.DataFrame] = None,
) -> None:
    if display_df is None or display_df.empty:
        return
    portfolio_df = sim_results.get("portfolio_df") if sim_results else None
    if stack_plan is None and sim_results:
        stack_plan = sim_results.get("stack_plan")
    scorecard = build_lineup_scorecard(
        display_df,
        portfolio_df=portfolio_df if isinstance(portfolio_df, pd.DataFrame) else None,
        stack_plan=stack_plan if isinstance(stack_plan, pd.DataFrame) else None,
    )
    if scorecard.empty:
        return
    with st.expander("Why these lineups?", expanded=False):
        view = scorecard.copy()
        pct_cols = [
            "projection_component",
            "leverage_component",
            "stack_component",
            "duplication_component",
            "ownership_scenario_component",
            "uniqueness_component",
            "why_selected_score",
            "top_1pct_rate",
        ]
        for col in pct_cols:
            if col in view.columns:
                view[col] = pd.to_numeric(view[col], errors="coerce").fillna(0.0).map(lambda x: f"{x:.1%}")
        keep = [
            col
            for col in [
                "lineup_id",
                "why_selected_score",
                "why_selected",
                "projection",
                "ownership",
                "leverage",
                "primary_stack",
                "projection_component",
                "leverage_component",
                "stack_component",
                "duplication_component",
                "ownership_scenario_component",
                "uniqueness_component",
                "payout_ev",
                "estimated_dupes",
            ]
            if col in view.columns
        ]
        st.dataframe(view[keep].head(300), width="stretch", hide_index=True)


def _render_candidate_pool_quality(lineup_df: pd.DataFrame, config_state: Dict, sim_state: Optional[Dict] = None) -> None:
    if lineup_df is None or lineup_df.empty:
        return
    sim_state = sim_state or _get_sim_config_state()
    target = int(sim_state.get("num_candidates", 300) or 300)
    report = build_candidate_pool_quality_report(
        lineup_df,
        target_final_lineups=target,
        max_player_exposure=float(config_state.get("batter_chalk_exposure_cap", 55.0) or 55.0) / 100.0 if float(config_state.get("batter_chalk_exposure_cap", 55.0) or 55.0) > 1 else float(config_state.get("batter_chalk_exposure_cap", 0.55) or 0.55),
        max_pitcher_exposure=float(config_state.get("pitcher_chalk_exposure_cap", 65.0) or 65.0) / 100.0 if float(config_state.get("pitcher_chalk_exposure_cap", 65.0) or 65.0) > 1 else float(config_state.get("pitcher_chalk_exposure_cap", 0.65) or 0.65),
    )
    if not report.summary:
        return
    st.subheader("Candidate pool quality gate")
    summary = report.summary
    cols = st.columns(5)
    cols[0].metric("Candidates", int(summary.get("lineups", 0)))
    cols[1].metric("Unique", int(summary.get("unique_lineups", 0)))
    cols[2].metric("Max player", f"{summary.get('max_player_exposure', 0.0):.0%}")
    cols[3].metric("Max pitcher", f"{summary.get('max_pitcher_exposure', 0.0):.0%}")
    cols[4].metric("Max stack", f"{summary.get('max_stack_exposure', 0.0):.0%}")
    issues = report.checks[report.checks["status"].isin(["Warn", "Fail"])] if not report.checks.empty else pd.DataFrame()
    if issues.empty:
        st.success("Candidate pool passed the core quality gate.")
    else:
        st.warning("Candidate pool has quality warnings before Step 4 selection.")
        st.dataframe(issues, width="stretch", hide_index=True)
    with st.expander("Candidate pool details", expanded=False):
        st.dataframe(report.checks, width="stretch", hide_index=True)
        if not report.stack_template_mix.empty:
            st.markdown("**Stack template mix**")
            view = report.stack_template_mix.copy()
            view["exposure_pct"] = pd.to_numeric(view["exposure_pct"], errors="coerce").fillna(0.0).map(lambda x: f"{x:.1%}")
            st.dataframe(view, width="stretch", hide_index=True)
        if not report.stack_exposure.empty:
            st.markdown("**Top stack exposure**")
            view = report.stack_exposure.head(20).copy()
            view["exposure_pct"] = pd.to_numeric(view["exposure_pct"], errors="coerce").fillna(0.0).map(lambda x: f"{x:.1%}")
            st.dataframe(view, width="stretch", hide_index=True)
        if not report.player_exposure.empty:
            st.markdown("**Top player exposure**")
            view = report.player_exposure.head(40).copy()
            view["exposure_pct"] = pd.to_numeric(view["exposure_pct"], errors="coerce").fillna(0.0).map(lambda x: f"{x:.1%}")
            st.dataframe(view, width="stretch", hide_index=True)


def _stack_composition_breakdown(lineup_df: pd.DataFrame) -> tuple:
    """Return (signature_df, team_roles_df) for the stack composition panel.

    signature_df: one row per stack type (e.g. '4-3-1'), with lineup count and %.
    team_roles_df: for each stack size (4, 3, 2 …), which teams appear and how often.
    """
    hitters = lineup_df[lineup_df["player_type"].str.lower() == "batter"]
    if hitters.empty:
        return pd.DataFrame(), pd.DataFrame()

    total_lineups = lineup_df["lineup_id"].nunique()

    # Per-lineup batter count per team
    team_counts = (
        hitters.groupby(["lineup_id", "team_code"])
        .size()
        .reset_index(name="batters")
    )

    # Build stack signature for each lineup
    signatures = []
    team_role_rows = []
    for lineup_id, grp in team_counts.groupby("lineup_id"):
        counts = sorted(grp["batters"].tolist(), reverse=True)
        sig = "-".join(str(c) for c in counts if c >= 2)  # only groups ≥2 are "stacks"
        if not sig:
            sig = "No stacks"
        signatures.append({"lineup_id": lineup_id, "stack_type": sig})
        # Record which team fills which stack size
        for _, row in grp.iterrows():
            team_role_rows.append({
                "stack_size": row["batters"],
                "team_code": row["team_code"],
                "lineup_id": lineup_id,
            })

    sig_df = pd.DataFrame(signatures)
    composition = (
        sig_df.groupby("stack_type")["lineup_id"]
        .nunique()
        .reset_index(name="lineups")
        .sort_values("lineups", ascending=False)
    )
    composition["pct_of_lineups"] = (composition["lineups"] / total_lineups * 100).round(1)
    composition = composition.rename(columns={"stack_type": "Stack Type", "lineups": "Lineups", "pct_of_lineups": "% of Lineups"})

    roles_df = pd.DataFrame(team_role_rows)
    if roles_df.empty or roles_df["stack_size"].max() < 2:
        team_roles = pd.DataFrame()
    else:
        team_roles = (
            roles_df[roles_df["stack_size"] >= 2]
            .groupby(["stack_size", "team_code"])["lineup_id"]
            .nunique()
            .reset_index(name="lineups")
            .sort_values(["stack_size", "lineups"], ascending=[False, False])
        )
        team_roles["pct"] = (team_roles["lineups"] / total_lineups * 100).round(1)
        team_roles = team_roles.rename(columns={
            "stack_size": "Stack Size",
            "team_code": "Team",
            "lineups": "Lineups",
            "pct": "% of Lineups",
        })

    return composition, team_roles


def _render_stack_breakdown(lineup_df: pd.DataFrame) -> None:
    """Render the full stack composition breakdown panel in Step 5."""
    st.subheader("Stack composition breakdown")
    composition, team_roles = _stack_composition_breakdown(lineup_df)

    if composition.empty:
        st.info("No batter data available to build stack breakdown.")
        return

    total = composition["Lineups"].sum()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("**Stack types across all lineups**")
        st.dataframe(composition, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("**Stack type distribution**")
        chart_data = composition.set_index("Stack Type")["Lineups"]
        st.bar_chart(chart_data)

    if not team_roles.empty:
        st.markdown("**Which teams are in each stack size**")
        st.caption(
            "Shows how many lineups each team appears in as a 4-man, 3-man, or 2-man stack. "
            "A team can appear in multiple stack sizes across different lineups."
        )
        for size in sorted(team_roles["Stack Size"].unique(), reverse=True):
            size_rows = team_roles[team_roles["Stack Size"] == size].drop(columns="Stack Size")
            label = f"{size}-man stacks — {len(size_rows)} team(s) used"
            with st.expander(label, expanded=(size >= 4)):
                st.dataframe(size_rows, use_container_width=True, hide_index=True)


# ── Run History ───────────────────────────────────────────────────────────────

_SETTINGS_LABELS = {
    "stack_template_selections": "Stack Templates",
    "num_lineups": "Lineups Requested",
    "batter_chalk_threshold": "Batter Chalk Threshold (%)",
    "batter_chalk_exposure_cap": "Batter Chalk Cap (%)",
    "pitcher_chalk_threshold": "Pitcher Chalk Threshold (%)",
    "pitcher_chalk_exposure_cap": "Pitcher Chalk Cap (%)",
    "leverage_weight": "Leverage Weight (%)",
    "randomness": "Solver Randomness (%)",
    "max_lineup_ownership": "Max Lineup Ownership (%)",
    "bring_back_enabled": "Bring-Back",
    "bring_back_count": "Bring-Back Count",
    "min_game_total": "Min Game Total",
}


def _save_run_snapshot(lineup_df: pd.DataFrame, config_settings: Dict, num_requested: int) -> None:
    """Save a lightweight snapshot of this optimizer run to session state history."""
    import datetime

    history = st.session_state.setdefault(RUN_HISTORY_KEY, [])
    run_id = len(history) + 1

    composition, _ = _stack_composition_breakdown(lineup_df)
    stack_summary = composition.to_dict("records") if not composition.empty else []

    exposures = _player_exposure_summary(lineup_df)
    top_players = (
        exposures[["full_name", "team_code", "exposure_pct"]]
        .head(10)
        .to_dict("records")
        if not exposures.empty
        else []
    )

    avg_salary = 0.0
    avg_proj = 0.0
    if not lineup_df.empty and "lineup_id" in lineup_df.columns:
        per_lineup = lineup_df.groupby("lineup_id").agg(
            salary=("salary", "sum"),
            proj=("proj_fd_mean", "sum"),
        )
        avg_salary = float(per_lineup["salary"].mean())
        avg_proj = float(per_lineup["proj"].mean())

    settings_snapshot = {k: config_settings.get(k) for k in _SETTINGS_LABELS}

    history.append({
        "run_id": run_id,
        "label": f"Run {run_id}",
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "requested": num_requested,
        "generated": lineup_df["lineup_id"].nunique() if not lineup_df.empty else 0,
        "settings": settings_snapshot,
        "stack_summary": stack_summary,
        "top_players": top_players,
        "avg_salary": avg_salary,
        "avg_projection": avg_proj,
    })


def _render_optimizer_settings_guide() -> None:
    """Collapsible plain-English guide for every Step 3 optimizer setting."""
    with st.expander("? What do these settings do?", expanded=False):
        st.markdown("""
**Number of lineups**
How many different lineups you want to build. Match this to the number of contest entries you plan to submit. If you have 150 entries, set this to 150. The optimizer will try to make every lineup unique.

---

**Salary cap**
FanDuel gives you $35,000 to spend. Leave this at 35,000 unless you want to force the optimizer to spend less (rarely useful — spending up is almost always better).

---

**Stack templates**
A "stack" means putting multiple batters from the same team in a lineup. When hitters in your lineup are teammates, their scores correlate — if the team has a big inning, multiple players benefit at once.

| Template | What it means |
|---|---|
| **4-3-1** | 4 batters from one team, 3 from another, 1 from a third. The most popular GPP structure. |
| **4-4** | 4 batters from two different teams. Maximizes correlation but is a riskier, boom-or-bust structure. |
| **3-3-2** | Two medium stacks plus a pair. More diversity across teams, slightly safer. |
| **4-2-2** | One big stack plus two smaller pairs. Common balance play. |
| **Auto** | No stacking requirement — optimizer picks freely. Good for cash games, not ideal for GPP. |

You can select **multiple templates** and set how many lineups use each. Example: 50 lineups on 4-3-1 and 50 on 3-3-2 gives you a mixed portfolio.

---

**Batter chalk threshold / Batter chalk exposure cap**
"Chalk" = high-ownership players (everyone is playing them). The threshold sets the ownership % that defines chalk (e.g., 25% means any batter owned by 25%+ of the field is chalk). The cap limits how many of your lineups can include that player (e.g., 30% cap means chalk batters appear in at most 30% of your lineups). This keeps you from being too correlated with the field.

*Lower the cap → more contrarian, more differentiated lineups. Higher the cap → safer, more chalk-heavy lineups.*

---

**Pitcher chalk threshold / Pitcher chalk exposure cap**
Same concept but for pitchers. Pitchers are usually higher ownership, so the default threshold is higher (35%). Pitchers also anchor lineups more, so the default cap is looser (50%).

---

**Leverage weight**
This is the core of the leverage strategy. Instead of picking players purely by projection, the optimizer adjusts each player's value based on their leverage score (high projection + low ownership = positive leverage).

At 15%, a player projected for 5.8 pts with strong leverage (+0.5) gets treated like a 6.24 pt player, while a 6.0 pt chalk player with negative leverage (-0.3) gets treated like 5.73 pts. This makes the optimizer prefer the contrarian play when projections are close — which is exactly the edge in GPP tournaments.

*Higher weight → more contrarian. Lower weight → closer to pure projection. 0% disables leverage entirely.*

---

**Solver randomness**
Adds small random noise to each player's projection on every lineup solve. Without this, the optimizer would build the same (or very similar) lineups repeatedly. At 5%, a 6.0 pt projection gets jittered to somewhere in the ~5.7–6.3 range each time. This creates natural diversity in your lineup pool — different lineups "discover" different player combinations.

*Higher = more diverse pool but projections matter less. Lower = tighter around optimal but less variety. 0% = fully deterministic.*

---

**Max lineup ownership**
Caps the total combined ownership of all players in a single lineup. For example, if set to 200%, no lineup can have players whose ownerships sum to more than 200%. This forces the optimizer to avoid "super-chalky" lineups where every player is heavily owned. Leave at 0 to disable.

---

**Player-specific exposure overrides**
Type a player name and a decimal to set their exact exposure cap. Example:
```
Aaron Judge:0.3
Shohei Ohtani:0.5
```
This caps Aaron Judge to 30% of lineups and Ohtani to 50%, regardless of other settings. Useful when you want to fade or limit a specific player.

---

**Bring-back requirement**
When enabled, every lineup that uses a pitcher must also include at least N batters from the team that pitcher is facing (the opposing team). Example: if you use Gerrit Cole (NYY) as a pitcher, you must bring back at least 1 Boston batter. This leverages the fact that when a pitcher dominates, the opposing team often still scores a couple runs.

*Bring-back count* = minimum number of opposing batters required (1 is standard, 2 is aggressive).

---

**Min Vegas total for stacks**
Excludes teams from being used in stacks if their game's over/under is below this number. Example: setting 8.0 means you won't stack batters from a game projected to score fewer than 8 total runs. Filters out low-ceiling games from your stacking pool. Leave at 0 to include all games.
""")


def _render_simulation_settings_guide() -> None:
    """Collapsible plain-English guide for every Step 4 simulation setting."""
    with st.expander("? What do these settings do?", expanded=False):
        st.markdown("""
**What Step 4 actually does**
Step 3 builds the lineups using an optimizer (it finds the highest-projected combinations). Step 4 then *tests* those lineups by simulating thousands of contest scenarios — asking "if this slate were played 10,000 times with realistic randomness, which of my lineups perform best?" The simulation picks the subset of lineups most likely to win.

---

**Number of simulations**
How many times to simulate the full slate. More simulations = more accurate results but slower. 10,000 is a good default. Go up to 25,000–50,000 for important contests. Don't go below 5,000 or the results get noisy.

---

**Volatility scale**
Multiplies the randomness (variance) in player scores. 1.0 = realistic historical variance. Higher values (1.5–2.0) model more chaotic, boom-or-bust slates. Lower (0.5–0.8) model more predictable slates. For large GPPs, consider bumping this up slightly since upsets are more common in big fields.

---

**Copula ν (nu)**
Controls how often multiple correlated players have a big game at the same time. Lower values (3–5) mean "fat tails" — big correlated outcomes happen more often, so stacks can go nuclear or go bust together. Higher values (10–30) means scores are more independent. For GPP, keep this low (3–7) to reward good stacks.

---

**Teammate correlation**
How much a batter's score is linked to his teammates' scores. 0.25 = mild correlation (realistic). 0.40–0.50 = strong correlation (if the team has a big inning, everyone benefits a lot). Higher values reward team stacking more. Don't go above 0.6 or the simulation becomes unrealistic.

---

**Pitcher vs. opposing correlation**
How a pitcher's score moves relative to the batters he's facing. This is always negative — when a pitcher throws a shutout, the opposing batters score nothing. Default is -0.15. More negative (e.g., -0.30) makes pitcher/opposing batter correlation stronger. This affects how the simulation values "bring-back" strategies.

---

**Field size**
How many opponent lineups to simulate against. 1,000 is a standard GPP approximation. If you're entering a 100-person tournament, 1,000 still works well. Larger values (5,000+) give more accurate win rates but run much slower.

---

**Selection metric**
How the simulation ranks and selects your best lineups:

| Metric | Best for |
|---|---|
| **top_1pct_rate** | Large GPPs — picks lineups most likely to land in the top 1% (max score, winner mentality) |
| **win_rate** | Head-to-head / very small contests — which lineup wins outright most often |
| **cash_rate** | Cash games / 50-50s — which lineup finishes in the top half most often |
| **expected_roi** | Balanced — maximizes expected return across all outcome scenarios |
| **p99_score** | Pure ceiling — picks lineups with the highest 99th percentile possible score |

For large GPPs, use **top_1pct_rate**. For cash games, use **cash_rate**.

---

**Diversity weight**
Tradeoff between "pick the highest-scoring lineup every time" vs. "spread across different constructions." 0 = pure score maximization (all your selected lineups will look similar). 1 = pure diversity (lineups spread as differently as possible). 0.3 is a good GPP default — mostly score-focused with some variety.

---

**Max batter / pitcher exposure**
After simulation selects your best lineups, these caps limit how often any single player appears across your final portfolio. Max batter exposure of 0.4 means no batter appears in more than 40% of your selected lineups. This forces differentiation — you're not putting all your eggs in one basket. *Too low → lineups feel random. Too high → over-concentrated on a few players.*

---

**Min batter / pitcher exposure**
Floor — ensures a player appears in at least X% of your final lineups. Use this to guarantee a player you really like shows up meaningfully across your entries. Leave at 0 unless you have a strong conviction play.

---

**Min / Max stack exposure per team**
Controls how spread out your team stacks are across the portfolio. Min stack exposure of 0.05 means every team used in a stack must appear in at least 5% of lineups (no one-lineup wonders). Max of 0.35 means no single team can anchor more than 35% of your lineups. Use these to avoid over-concentrating on one game.

---

**Advanced: stratified sampling**
Normally, the simulation draws random numbers like throwing darts at a board — purely random, so sometimes they cluster together by chance. Stratified sampling divides the probability space into equal sections and forces one draw per section, so the samples are more evenly spread across all possible outcomes. Think of it like making sure you test every range of scenarios rather than accidentally oversampling the same range.

**When to turn it ON:** If you're running a small number of simulations (under 5,000) and you notice the win rates or ROI numbers change a lot between runs, stratified sampling will make your results more consistent.

**When to leave it OFF (the default):** With 10,000+ simulations, regular random sampling already converges to stable results and stratified sampling adds no meaningful benefit. For typical use — 10,000 simulations on a full slate — leave this unchecked.

**Bottom line:** Leave it off. Only turn it on if you're running quick/small sim counts and want tighter, more repeatable numbers.
""")


def _render_run_history() -> None:
    """Render the run history table and side-by-side comparison UI."""
    history = st.session_state.get(RUN_HISTORY_KEY, [])
    if not history:
        return

    st.subheader("Run history")

    # ── Summary table of all runs ──────────────────────────────────────
    summary_rows = []
    for r in history:
        top_stack = r["stack_summary"][0]["Stack Type"] if r["stack_summary"] else "—"
        summary_rows.append({
            "Run": r["label"],
            "Time": r["timestamp"],
            "Requested": r["requested"],
            "Generated": r["generated"],
            "Duplicates": max(0, r["requested"] - r["generated"]),
            "Top Stack": top_stack,
            "Avg Salary": f"${r['avg_salary']:,.0f}",
            "Avg Projection": f"{r['avg_projection']:.1f}",
            "Stack Templates": ", ".join(r["settings"].get("stack_template_selections") or []),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    if len(history) < 2:
        st.caption("Run the optimizer again with different settings to enable side-by-side comparison.")
        return

    # ── Side-by-side comparison ────────────────────────────────────────
    st.markdown("**Compare two runs**")
    run_labels = [r["label"] for r in history]
    col1, col2 = st.columns(2)
    label_a = col1.selectbox("Run A", run_labels, index=len(run_labels) - 2, key="compare_a")
    label_b = col2.selectbox("Run B", run_labels, index=len(run_labels) - 1, key="compare_b")

    run_a = next(r for r in history if r["label"] == label_a)
    run_b = next(r for r in history if r["label"] == label_b)

    # Settings diff
    diff_rows = []
    for key, display_name in _SETTINGS_LABELS.items():
        val_a = run_a["settings"].get(key)
        val_b = run_b["settings"].get(key)
        if val_a != val_b:
            diff_rows.append({
                "Setting": display_name,
                label_a: str(val_a),
                label_b: str(val_b),
            })

    with st.expander("Settings that changed between runs", expanded=True):
        if diff_rows:
            st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No settings changed between these two runs.")

    # Lineup count comparison
    with st.expander("Lineup counts", expanded=True):
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric(f"{label_a} Generated", run_a["generated"],
                   delta=run_a["generated"] - run_a["requested"],
                   delta_color="normal")
        mc2.metric(f"{label_b} Generated", run_b["generated"],
                   delta=run_b["generated"] - run_b["requested"],
                   delta_color="normal")
        mc3.metric("Change in Unique Lineups", "",
                   delta=run_b["generated"] - run_a["generated"],
                   delta_color="normal")
        mc4.metric("Change in Duplicates",
                   "",
                   delta=-(run_b["generated"] - run_a["generated"]),
                   delta_color="inverse")

    # Stack composition comparison
    with st.expander("Stack composition", expanded=True):
        stacks_a = {r["Stack Type"]: r["Lineups"] for r in run_a["stack_summary"]}
        stacks_b = {r["Stack Type"]: r["Lineups"] for r in run_b["stack_summary"]}
        all_types = sorted(set(stacks_a) | set(stacks_b))
        if all_types:
            stack_rows = []
            for stype in all_types:
                cnt_a = stacks_a.get(stype, 0)
                cnt_b = stacks_b.get(stype, 0)
                stack_rows.append({
                    "Stack Type": stype,
                    f"{label_a} Lineups": cnt_a,
                    f"{label_b} Lineups": cnt_b,
                    "Change": cnt_b - cnt_a,
                })
            st.dataframe(pd.DataFrame(stack_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No stack data available for comparison.")

    # Top player exposure comparison
    with st.expander("Top player exposure shift", expanded=False):
        exp_a = {r["full_name"]: r["exposure_pct"] for r in run_a["top_players"]}
        exp_b = {r["full_name"]: r["exposure_pct"] for r in run_b["top_players"]}
        all_players = sorted(set(exp_a) | set(exp_b))
        if all_players:
            exp_rows = []
            for player in all_players:
                pct_a = exp_a.get(player, 0.0)
                pct_b = exp_b.get(player, 0.0)
                exp_rows.append({
                    "Player": player,
                    f"{label_a} Exposure": f"{pct_a:.1%}",
                    f"{label_b} Exposure": f"{pct_b:.1%}",
                    "Shift": f"{pct_b - pct_a:+.1%}",
                })
            st.dataframe(pd.DataFrame(exp_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No exposure data available for comparison.")


def _match_actual_scores(
    actual_scores: pd.DataFrame,
    ownership_df: Optional[pd.DataFrame],
    optimizer_df: pd.DataFrame,
) -> pd.DataFrame:
    scores = actual_scores.copy()
    scores.columns = [col.strip() for col in scores.columns]
    if "fd_player_id" not in scores.columns:
        if "player_name" not in scores.columns:
            raise ValueError("Actual scores CSV must include fd_player_id or player_name column.")
        name_map = optimizer_df[["full_name", "fd_player_id"]].drop_duplicates()
        scores = scores.merge(name_map, left_on="player_name", right_on="full_name", how="left")
        if scores["fd_player_id"].isna().any():
            raise ValueError("Could not match some player names to FanDuel IDs.")
        scores.drop(columns=["full_name"], inplace=True, errors="ignore")
    if ownership_df is not None and not ownership_df.empty:
        ownership_df.columns = [col.strip() for col in ownership_df.columns]
        if "fd_player_id" not in ownership_df.columns and "player_name" in ownership_df.columns:
            name_map = optimizer_df[["full_name", "fd_player_id"]].drop_duplicates()
            ownership_df = ownership_df.merge(name_map, left_on="player_name", right_on="full_name", how="left")
        scores = scores.merge(
            ownership_df[["fd_player_id", "actual_ownership_pct"]],
            on="fd_player_id",
            how="left",
        )
    else:
        scores["actual_ownership_pct"] = np.nan
    if "player_name" not in scores.columns:
        name_map = optimizer_df[["fd_player_id", "full_name"]].drop_duplicates()
        scores = scores.merge(name_map, on="fd_player_id", how="left")
        scores.rename(columns={"full_name": "player_name"}, inplace=True)
    return scores


def _calculate_lineup_actuals(lineup_df: pd.DataFrame, actual_map: pd.Series) -> pd.DataFrame:
    df = lineup_df.copy()
    df["actual_points"] = df["fd_player_id"].map(actual_map).fillna(0.0)
    aggregations = {
        "total_actual_points": ("actual_points", "sum"),
        "total_projection": ("proj_fd_mean", "sum"),
    }
    # Construction metrics used by the slate-profile tuner.
    for name, source in (
        ("total_ownership", "proj_fd_ownership"),
        ("total_upside", "proj_fd_upside"),
        ("total_salary", "salary"),
    ):
        if source in df.columns:
            df[source] = pd.to_numeric(df[source], errors="coerce").fillna(0.0)
            aggregations[name] = (source, "sum")
    grouped = df.groupby("lineup_id").agg(**aggregations)
    grouped.reset_index(inplace=True)
    return grouped



def _record_simulation_accuracy(
    db: SlateDatabase,
    date_str: str,
    lineup_points: pd.DataFrame,
    workflow: Dict,
    contest_meta: Dict[str, float],
    matched_actuals: Optional[pd.DataFrame] = None,
) -> None:
    sim_results = workflow.get(SIM_RESULTS_KEY)
    if not sim_results:
        return
    contest_df = sim_results.get("contest_df")
    if not isinstance(contest_df, pd.DataFrame) or contest_df.empty:
        return
    contest_for_merge = contest_df.copy()
    if "lineup_id" in contest_for_merge.columns and "lineup_id" in lineup_points.columns:
        sim_ids = pd.to_numeric(contest_for_merge["lineup_id"], errors="coerce")
        actual_ids = pd.to_numeric(lineup_points["lineup_id"], errors="coerce")
        if sim_ids.notna().any() and actual_ids.notna().any() and sim_ids.min() == 0 and actual_ids.min() >= 1:
            contest_for_merge["lineup_id"] = sim_ids.astype(int) + 1
    merged = contest_for_merge.merge(
        lineup_points[["lineup_id", "total_actual_points", "payout", "roi"]],
        on="lineup_id",
        how="inner",
    ).dropna(subset=["total_actual_points"])
    if merged.empty:
        return

    threshold = 150.0
    mean_scores = merged["mean_score"].astype(float)
    std_scores = merged["std_score"].astype(float).clip(lower=1e-3)
    z_scores = (threshold - mean_scores) / std_scores
    predicted_prob = 1 - norm.cdf(z_scores)
    actual_event = (merged["total_actual_points"].astype(float) >= threshold).astype(float)
    metrics: Dict[str, float] = {}
    metrics["brier_score_15"] = float(((predicted_prob - actual_event) ** 2).mean())

    for label, column in [("p10", "p10_score"), ("p25", "p25_score"), ("p50", "median_score"), ("p75", "p75_score"), ("p90", "p90_score")]:
        if column in merged.columns:
            calibration = (
                merged["total_actual_points"].astype(float) <= merged[column].astype(float)
            ).mean()
            metrics[f"dist_calibration_{label}"] = float(calibration)

    sim_config = sim_results.get("sim_config")
    if sim_config:
        metrics["teammate_corr_predicted"] = float(
            getattr(sim_config.correlation, "teammate_base", float("nan"))
        )

    metrics["field_winning_score_predicted"] = float(
        contest_df["max_score"].astype(float).max()
    )
    if "duplication_score" in contest_df.columns:
        metrics["avg_duplication_score_predicted"] = float(pd.to_numeric(contest_df["duplication_score"], errors="coerce").mean())
    if "estimated_dupes" in contest_df.columns:
        metrics["avg_estimated_dupes_predicted"] = float(pd.to_numeric(contest_df["estimated_dupes"], errors="coerce").mean())
    if contest_meta and contest_meta.get("winning_score"):
        metrics["field_winning_score_actual"] = float(contest_meta["winning_score"])
    else:
        metrics["field_winning_score_actual"] = float(
            lineup_points["total_actual_points"].max()
        )

    correlation_metrics = _teammate_correlation_metrics(
        matched_actuals,
        workflow.get("optimizer"),
        sim_results.get("correlation_model") if sim_results else None,
    )
    metrics.update(correlation_metrics)

    lineup_corr = _lineup_score_correlation(merged)
    if lineup_corr is not None:
        metrics["lineup_score_corr"] = lineup_corr

    if metrics:
        db.insert_simulation_accuracy(date_str, metrics, len(merged))



def _teammate_correlation_metrics(
    matched_actuals: Optional[pd.DataFrame],
    optimizer_df: Optional[pd.DataFrame],
    corr_model,
) -> Dict[str, float]:
    if matched_actuals is None or optimizer_df is None or optimizer_df.empty or corr_model is None:
        return {}
    meta = optimizer_df[["fd_player_id", "team_code", "player_type"]].drop_duplicates()
    merged = matched_actuals.merge(meta, on="fd_player_id", how="left")
    merged = merged.dropna(subset=["team_code", "actual_fd_points"])
    merged["player_type"] = merged["player_type"].astype(str).str.lower()
    batters = merged[merged["player_type"] == "batter"]
    if batters.empty:
        return {}
    index_map = {pid: idx for idx, pid in enumerate(corr_model.player_ids)}
    score_a: list[float] = []
    score_b: list[float] = []
    predicted: list[float] = []
    for _, group in batters.groupby("team_code"):
        players = group[["fd_player_id", "actual_fd_points"]].values.tolist()
        if len(players) < 2:
            continue
        for i in range(len(players)):
            pid_i, score_i = players[i]
            idx_i = index_map.get(str(pid_i))
            if idx_i is None:
                continue
            for j in range(i + 1, len(players)):
                pid_j, score_j = players[j]
                idx_j = index_map.get(str(pid_j))
                if idx_j is None:
                    continue
                score_a.append(float(score_i))
                score_b.append(float(score_j))
                predicted.append(float(corr_model.matrix[idx_i, idx_j]))
    metrics: Dict[str, float] = {}
    if predicted:
        metrics["teammate_corr_predicted"] = float(np.mean(predicted))
    if len(score_a) >= 2:
        actual_corr = np.corrcoef(score_a, score_b)[0, 1]
        if np.isfinite(actual_corr):
            metrics["teammate_corr_actual"] = float(actual_corr)
            if predicted:
                metrics["teammate_corr_error"] = float(actual_corr) - float(np.mean(predicted))
    return metrics



def _lineup_score_correlation(merged: pd.DataFrame) -> Optional[float]:
    if merged.empty or "total_actual_points" not in merged.columns:
        return None
    if "mean_score" not in merged.columns:
        return None
    actual_scores = merged["total_actual_points"].astype(float)
    predicted_scores = merged["mean_score"].astype(float)
    if actual_scores.nunique() <= 1 or predicted_scores.nunique() <= 1:
        return None
    corr = np.corrcoef(predicted_scores, actual_scores)[0, 1]
    return float(corr) if np.isfinite(corr) else None



def _render_simulation_calibration(date_str: str) -> None:
    db: Optional[SlateDatabase] = None
    try:
        db = SlateDatabase(DEFAULT_DB_PATH)
        metrics_df = db.fetch_simulation_accuracy(date_str)
    except Exception as exc:  # pylint: disable=broad-except
        st.warning(f"Unable to load simulation calibration metrics: {exc}")
        return
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:  # pylint: disable=broad-except
                pass

    if metrics_df.empty:
        st.caption("No simulation calibration metrics recorded for this date.")
        return

    latest = metrics_df.sort_values("created_at").drop_duplicates("metric_name", keep="last")
    metrics_indexed = latest.set_index("metric_name")
    st.subheader("Simulation calibration")
    metric_cols = st.columns(3)
    if "brier_score_15" in metrics_indexed.index:
        metric_cols[0].metric(
            "Brier score @150",
            f"{metrics_indexed.loc['brier_score_15', 'metric_value']:.4f}",
        )
    if {"field_winning_score_predicted", "field_winning_score_actual"}.issubset(metrics_indexed.index):
        predicted = metrics_indexed.loc["field_winning_score_predicted", "metric_value"]
        actual = metrics_indexed.loc["field_winning_score_actual", "metric_value"]
        metric_cols[1].metric(
            "Winning score",
            f"{actual:.1f}",
            f"Δ {actual - predicted:+.1f}",
        )
    if "teammate_corr_predicted" in metrics_indexed.index:
        metric_cols[2].metric(
            "Teammate corr",
            f"{metrics_indexed.loc['teammate_corr_predicted', 'metric_value']:.2f}",
        )
    st.dataframe(
        latest[["metric_name", "metric_value", "num_players", "created_at"]],
        width="stretch",
    )


def _render_post_slate_learning(report) -> None:
    if not report:
        return
    st.subheader("Post-slate learning loop")
    summary = getattr(report, "summary", {}) or {}
    if summary:
        cols = st.columns(4)
        cols[0].metric("Avg actual", f"{summary.get('avg_actual', 0.0):.1f}")
        cols[1].metric("Max actual", f"{summary.get('max_actual', 0.0):.1f}")
        if "selected_vs_raw_avg_delta" in summary:
            cols[2].metric("Selected vs raw", f"{summary.get('selected_vs_raw_avg_delta', 0.0):+.1f}")
        else:
            cols[2].metric("Lineups", int(summary.get("lineups", 0)))
        cols[3].metric("Best stack", str(summary.get("best_stack_team", "")))
    lessons = getattr(report, "lessons", pd.DataFrame())
    if isinstance(lessons, pd.DataFrame) and not lessons.empty:
        st.markdown("**Lessons**")
        st.dataframe(lessons, width="stretch", hide_index=True)
    stack_results = getattr(report, "stack_results", pd.DataFrame())
    if isinstance(stack_results, pd.DataFrame) and not stack_results.empty:
        st.markdown("**Stack plan results**")
        st.dataframe(stack_results, width="stretch", hide_index=True)
    leverage_results = getattr(report, "leverage_results", pd.DataFrame())
    if isinstance(leverage_results, pd.DataFrame) and not leverage_results.empty:
        st.markdown("**Leverage/chalk results**")
        st.dataframe(leverage_results, width="stretch", hide_index=True)
    comparison = getattr(report, "portfolio_comparison", pd.DataFrame())
    if isinstance(comparison, pd.DataFrame) and not comparison.empty:
        st.markdown("**Selected portfolio vs raw optimizer pool**")
        st.dataframe(comparison, width="stretch", hide_index=True)


def _render_failure_attribution(report) -> None:
    if report is None:
        return
    st.subheader("Failure attribution")
    summary = getattr(report, "summary", {}) or {}
    if summary:
        cols = st.columns(4)
        cols[0].metric("Lineups", summary.get("lineups", 0))
        cols[1].metric("Avg actual", f"{float(summary.get('avg_actual', 0.0) or 0.0):.1f}")
        cols[2].metric("Best actual", f"{float(summary.get('best_actual', 0.0) or 0.0):.1f}")
        cols[3].metric("High-severity factors", summary.get("high_severity_factors", 0))
    for note in getattr(report, "lessons", []) or []:
        st.caption(note)
    factors = getattr(report, "factors", pd.DataFrame())
    if isinstance(factors, pd.DataFrame) and not factors.empty:
        st.dataframe(factors, width="stretch", hide_index=True)
    stack_breakdown = getattr(report, "stack_breakdown", pd.DataFrame())
    if isinstance(stack_breakdown, pd.DataFrame) and not stack_breakdown.empty:
        with st.expander("Stack attribution"):
            st.dataframe(stack_breakdown, width="stretch", hide_index=True)
    player_breakdown = getattr(report, "player_breakdown", pd.DataFrame())
    if isinstance(player_breakdown, pd.DataFrame) and not player_breakdown.empty:
        with st.expander("Player attribution"):
            st.dataframe(player_breakdown.head(100), width="stretch", hide_index=True)


def _render_boom_bust_calibration(report) -> None:
    if report is None or not getattr(report, "summary", None):
        return
    st.subheader("Boom/bust calibration")
    summary = report.summary
    cols = st.columns(4)
    cols[0].metric("Players matched", int(summary.get("matched_players", 0)))
    cols[1].metric("Proj bust", f"{summary.get('avg_projected_bust', 0.0):.1%}")
    cols[2].metric("Actual under median", f"{summary.get('actual_under_median_rate', 0.0):.1%}")
    cols[3].metric("Upside hit", f"{summary.get('upside_hit_rate', 0.0):.1%}")
    lessons = getattr(report, "lessons", pd.DataFrame())
    if isinstance(lessons, pd.DataFrame) and not lessons.empty:
        st.dataframe(lessons, width="stretch", hide_index=True)
    by_bucket = getattr(report, "by_bust_bucket", pd.DataFrame())
    if isinstance(by_bucket, pd.DataFrame) and not by_bucket.empty:
        with st.expander("Boom/bust bucket calibration", expanded=False):
            view = by_bucket.copy()
            for col in ["avg_projected_bust", "actual_under_median_rate", "upside_hit_rate", "bust_calibration_error"]:
                if col in view.columns:
                    view[col] = pd.to_numeric(view[col], errors="coerce").fillna(0.0).map(lambda x: f"{x:.1%}")
            st.dataframe(view, width="stretch", hide_index=True)


def _render_field_duplication_learning(optimizer_df: Optional[pd.DataFrame]) -> None:
    with st.expander("Field duplication learning", expanded=False):
        profile = load_field_duplication_profile(FIELD_DUPLICATION_PROFILE_PATH)
        if profile:
            cols = st.columns(4)
            cols[0].metric("Field samples", f"{int(profile.get('samples', 0) or 0):,}")
            cols[1].metric("Duplicate lineup %", f"{float(profile.get('duplicate_lineup_pct', 0.0) or 0.0):.1%}")
            cols[2].metric("Avg dup count", f"{float(profile.get('avg_duplicate_count', 1.0) or 1.0):.2f}")
            cols[3].metric("Full-salary %", f"{float(profile.get('full_salary_pct', 0.0) or 0.0):.1%}")
            st.caption("This profile is used by the duplication estimator during Step 4 simulation.")
        else:
            st.caption("No learned field profile yet. Upload a contest export with lineup players to tune duplicate estimates.")
        field_file = st.file_uploader(
            "Learn from FanDuel contest lineup CSV",
            type=["csv"],
            key="field_duplication_learning_file",
        )
        if field_file is not None and st.button("Update Duplication Profile"):
            try:
                contest_df = pd.read_csv(field_file)
                profile_obj = learn_field_duplication_profile(
                    contest_df,
                    optimizer_df=optimizer_df,
                    source=field_file.name,
                )
                save_field_duplication_profile(profile_obj, FIELD_DUPLICATION_PROFILE_PATH)
                st.success(f"Learned duplication profile from {profile_obj.samples:,} field lineups.")
                st.rerun()
            except Exception as exc:  # pylint: disable=broad-except
                st.warning(f"Could not learn duplication profile: {exc}")


def _render_portfolio_stress_report(
    selected_players: Optional[pd.DataFrame],
    optimizer_df: Optional[pd.DataFrame],
    stack_plan: Optional[pd.DataFrame],
) -> None:
    if selected_players is None or selected_players.empty or optimizer_df is None or optimizer_df.empty:
        return
    report = build_portfolio_stress_report(selected_players, optimizer_df, stack_plan=stack_plan)
    if report.scenarios.empty:
        return
    st.subheader("Portfolio stress tests")
    cols = st.columns(4)
    cols[0].metric("Top chalk stack", str(report.summary.get("top_chalk_stack", "")))
    cols[1].metric("Top pitcher", str(report.summary.get("top_pitcher", "")))
    cols[2].metric("Top batter", str(report.summary.get("top_batter", "")))
    cols[3].metric("Max stack dependency", f"{float(report.summary.get('max_stack_dependency', 0.0) or 0.0):.0%}")
    view = report.scenarios.copy()
    for col in ["portfolio_alive_pct", "fragile_pct"]:
        if col in view.columns:
            view[col] = pd.to_numeric(view[col], errors="coerce").fillna(0.0).map(lambda x: f"{x:.1%}")
    st.dataframe(view, width="stretch", hide_index=True)
    for note in report.recommendations:
        st.caption(note)


def _render_portfolio_frontier(sim_results: Dict, lineup_df: Optional[pd.DataFrame], target_lineups: int) -> None:
    if not sim_results:
        return
    contest_df = sim_results.get("contest_df")
    if not isinstance(contest_df, pd.DataFrame) or contest_df.empty:
        return
    report = build_portfolio_frontier_report(contest_df, lineup_df=lineup_df, target_lineups=target_lineups)
    if report.summary.empty:
        return
    with st.expander("Portfolio frontier", expanded=False):
        if report.recommendation:
            st.caption(report.recommendation)
        view = report.summary.copy()
        for col in ["avg_top_1pct", "avg_ownership", "avg_duplication", "avg_uniqueness"]:
            if col in view.columns:
                view[col] = pd.to_numeric(view[col], errors="coerce").fillna(0.0).map(lambda x: f"{x:.1%}")
        for col in ["avg_payout_ev", "avg_roi", "avg_leverage"]:
            if col in view.columns:
                view[col] = pd.to_numeric(view[col], errors="coerce").fillna(0.0).map(lambda x: f"{x:.3f}")
        st.dataframe(view, width="stretch", hide_index=True)


def _render_calibration_feedback_controls(sim_state: Dict) -> None:
    with st.expander("Calibration feedback adjustments", expanded=False):
        st.caption("Uses saved post-slate calibration to lightly shade Step 4 settings when the model has drifted.")
        strength = st.slider(
            "Adjustment strength",
            min_value=0.0,
            max_value=1.0,
            value=float(sim_state.get("calibration_adjustment_strength", 0.0) or 0.0),
            step=0.1,
        )
        sim_state["calibration_adjustment_strength"] = strength
        lookback_days = st.number_input("Lookback days", min_value=7, max_value=120, value=30, step=7)
        end = pd.Timestamp.today()
        start = end - pd.Timedelta(days=int(lookback_days))
        if st.button("Preview Calibration Feedback"):
            feedback = build_calibration_feedback(
                DEFAULT_DB_PATH,
                start.strftime("%Y%m%d"),
                end.strftime("%Y%m%d"),
                sim_state,
                strength=max(strength, 0.1),
            )
            st.session_state["calibration_feedback_preview"] = feedback
        feedback = st.session_state.get("calibration_feedback_preview")
        if feedback:
            for note in feedback.notes:
                st.caption(note)
            if not feedback.adjustments.empty:
                st.dataframe(feedback.adjustments, width="stretch", hide_index=True)
                if st.button("Apply Calibration Feedback"):
                    sim_state.update(feedback.adjusted_state)
                    st.success("Applied calibration feedback to Step 4 settings.")
                    st.rerun()


def _render_contest_type_brain(sim_state: Dict, payout_bands) -> None:
    with st.expander("Contest-type brain", expanded=False):
        st.caption("Tunes portfolio construction to the contest shape: field size, number of entries, and payout top-heaviness.")
        options = ["Auto"] + list(CONTEST_TYPE_PROFILES.keys())
        current = str(sim_state.get("contest_type_profile", "Auto"))
        current_index = options.index(current) if current in options else 0
        selected = st.selectbox("Contest profile", options, index=current_index)
        auto_profile = apply_contest_type_to_state(dict(sim_state), "Auto", payout_bands=payout_bands)
        st.caption(f"Auto read: {auto_profile.name} - {auto_profile.description}")
        if st.button("Apply Contest Brain"):
            profile = apply_contest_type_to_state(sim_state, selected, payout_bands=payout_bands)
            st.success(f"Applied {profile.name}.")
            st.rerun()


def _render_field_realism_controls(sim_state: Dict) -> None:
    with st.expander("Opponent field realism", expanded=False):
        st.caption("Controls how the simulated field chases value chalk, optimizer trains, full salary, and stack-heavy builds.")
        sim_state["use_field_realism"] = st.checkbox(
            "Use realistic field ownership adjustments",
            value=bool(sim_state.get("use_field_realism", True)),
        )
        cols = st.columns(3)
        sim_state["field_realism_value_bias"] = cols[0].slider(
            "Value chalk bias",
            min_value=0.0,
            max_value=0.60,
            value=float(sim_state.get("field_realism_value_bias", 0.22) or 0.0),
            step=0.02,
        )
        sim_state["field_realism_train_bias"] = cols[1].slider(
            "Optimizer train bias",
            min_value=0.0,
            max_value=0.60,
            value=float(sim_state.get("field_realism_train_bias", 0.18) or 0.0),
            step=0.02,
        )
        sim_state["field_realism_upside_bias"] = cols[2].slider(
            "Upside chalk bias",
            min_value=0.0,
            max_value=0.40,
            value=float(sim_state.get("field_realism_upside_bias", 0.10) or 0.0),
            step=0.02,
        )
        salary_cols = st.columns(2)
        sim_state["field_salary_target_shark"] = salary_cols[0].slider(
            "Sharp salary target",
            min_value=0.90,
            max_value=1.00,
            value=float(sim_state.get("field_salary_target_shark", 0.970) or 0.970),
            step=0.005,
        )
        sim_state["field_salary_target_rec"] = salary_cols[1].slider(
            "Field salary target",
            min_value=0.86,
            max_value=0.99,
            value=float(sim_state.get("field_salary_target_rec", 0.930) or 0.930),
            step=0.005,
        )





def _render_lineup_rankings(contest_df: pd.DataFrame, selected_ids: List[int]) -> None:
    if contest_df is None or contest_df.empty:
        return
    st.subheader("Candidate lineup rankings")
    display = contest_df.copy()
    display["lineup_number"] = display["lineup_id"].astype(int) + 1
    selection = set(selected_ids or [])
    display["selected"] = display["lineup_number"].isin(selection)
    keep_cols = [
        "lineup_number",
        "mean_score",
        "p90_score",
        "p99_score",
        "win_rate",
        "top_1pct_rate",
        "cash_rate",
        "expected_roi",
        "expected_payout",
        "payout_ev",
        "payout_cash_rate",
        "total_ownership",
        "duplication_score",
        "estimated_dupes",
        "uniqueness_score",
        "ownership_scenario_score",
        "worst_case_duplication_score",
        "worst_case_ownership",
        "selected",
    ]
    for col in keep_cols:
        if col not in display.columns:
            display[col] = np.nan
    renamed = display[keep_cols].rename(
        columns={
            "lineup_number": "Lineup",
            "mean_score": "Mean",
            "p90_score": "P90",
            "p99_score": "P99",
            "win_rate": "Win %",
            "top_1pct_rate": "Top 1%",
            "cash_rate": "Cash %",
            "expected_roi": "Exp ROI",
            "expected_payout": "Exp Payout",
            "payout_ev": "Payout EV",
            "payout_cash_rate": "Payout Cash %",
            "total_ownership": "Own %",
            "duplication_score": "Dup Score",
            "estimated_dupes": "Est Dupes",
            "uniqueness_score": "Unique Score",
            "ownership_scenario_score": "Scenario Score",
            "worst_case_duplication_score": "Worst Dup",
            "worst_case_ownership": "Worst Own",
            "selected": "Selected",
        }
    )
    st.dataframe(renamed, width="stretch")



def _player_distribution_viewer(sim_results: Dict, optimizer_df: pd.DataFrame) -> None:
    slate_sim = sim_results.get("slate_sim") if sim_results else None
    if slate_sim is None or optimizer_df is None or optimizer_df.empty:
        return
    st.subheader("Player distribution viewer")
    players = (
        optimizer_df[["fd_player_id", "full_name", "proj_fd_mean"]]
        .drop_duplicates()
        .sort_values("full_name")
    )
    if players.empty:
        st.info("No player metadata available for simulation view.")
        return
    selected_name = st.selectbox(
        "Select player",
        players["full_name"].tolist(),
    )
    player_row = players[players["full_name"] == selected_name].iloc[0]
    player_id = player_row["fd_player_id"]
    projection = float(player_row.get("proj_fd_mean", 0.0))
    try:
        scores = slate_sim.player_scores(player_id)
    except KeyError:
        st.info("Player missing from simulation output.")
        return
    counts, bins = np.histogram(scores, bins=30)
    midpoints = (bins[1:] + bins[:-1]) / 2
    hist_df = pd.DataFrame({"bin": midpoints, "frequency": counts})
    st.bar_chart(hist_df.set_index("bin"))
    st.caption(
        f"Mean {scores.mean():.2f} | Std {scores.std():.2f} | P10 {np.percentile(scores,10):.2f} | P90 {np.percentile(scores,90):.2f} | Projection {projection:.2f}"
    )



def _render_correlation_heatmap(sim_results: Dict, optimizer_df: pd.DataFrame, top_n: int = 15) -> None:
    corr_model = sim_results.get("correlation_model") if sim_results else None
    if corr_model is None:
        return
    st.subheader("Correlation heatmap (top hitters)")
    if optimizer_df is None or optimizer_df.empty:
        st.info("Optimizer dataset unavailable for correlation view.")
        return
    top_players = (
        optimizer_df[optimizer_df["player_type"].str.lower() == "batter"]
        .sort_values("proj_fd_mean", ascending=False)
        .head(top_n)
    )
    if top_players.empty:
        st.info("No hitters found for correlation heatmap.")
        return
    ids = [pid for pid in top_players["fd_player_id"].astype(str) if pid in corr_model.player_ids]
    if len(ids) < 2:
        st.info("Insufficient overlap between hitters and correlation matrix.")
        return
    index_map = {pid: idx for idx, pid in enumerate(corr_model.player_ids)}
    indices = [index_map[pid] for pid in ids]
    matrix = corr_model.matrix[np.ix_(indices, indices)]
    labels = top_players.set_index("fd_player_id").loc[ids, "full_name"].tolist()
    heatmap_df = pd.DataFrame(matrix, index=labels, columns=labels)
    styled = heatmap_df.style.background_gradient(cmap="RdBu_r", axis=None)
    st.dataframe(styled, width="stretch")



def _render_convergence_chart(sim_results: Dict, lineups) -> None:
    slate_sim = sim_results.get("slate_sim") if sim_results else None
    if slate_sim is None or not lineups:
        return
    st.subheader("Simulation convergence")
    first_lineup = lineups[0]
    player_ids = first_lineup.dataframe["fd_player_id"].astype(str).tolist()
    scores = slate_sim.lineup_scores(player_ids)
    running_mean = np.cumsum(scores) / np.arange(1, len(scores) + 1)
    convergence_df = pd.DataFrame({
        "simulation": np.arange(1, len(running_mean) + 1),
        "running_mean": running_mean,
    })
    st.line_chart(convergence_df.set_index("simulation"))
    st.caption("Running average of the first lineup's simulated score as iterations increase.")


def _build_ownership_history_rows(workflow: Dict, matched_actuals: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Per-player training snapshot: model features + predicted + actual ownership.

    Built from the full pre-optimizer player frame (workflow["players"]) because
    the optimizer dataset drops several feature columns (HR probability, K
    projection, FPPG) that the fit script needs.
    """
    players = workflow.get("players")
    projections = workflow.get("projections")
    optimizer_df = workflow.get("optimizer")
    if players is None or projections is None or optimizer_df is None or optimizer_df.empty:
        return None
    try:
        features = build_ownership_features(players, projections)
    except Exception:  # pylint: disable=broad-except
        return None
    if features.empty:
        return None

    predicted_map = (
        optimizer_df.assign(_fd=optimizer_df["fd_player_id"].astype(str))
        .set_index("_fd")["proj_fd_ownership"]
        .to_dict()
    )
    actual_map = {}
    if matched_actuals is not None and not matched_actuals.empty and "actual_ownership_pct" in matched_actuals.columns:
        actual_map = (
            matched_actuals.assign(_fd=matched_actuals["fd_player_id"].astype(str))
            .set_index("_fd")["actual_ownership_pct"]
            .to_dict()
        )

    meta_cols = {"fd_player_id", "full_name", "player_type", "team_code"}
    feature_cols = [col for col in features.columns if col not in meta_cols]
    fd_ids = features["fd_player_id"].astype(str)
    return pd.DataFrame(
        {
            "fd_player_id": fd_ids,
            "player_name": features["full_name"],
            "player_type": features["player_type"],
            "predicted_own": fd_ids.map(predicted_map),
            "actual_own": fd_ids.map(actual_map),
            "features": features[feature_cols].apply(lambda row: json.dumps(row.to_dict()), axis=1),
        }
    )


def _process_results_submission(
    date_str: str,
    actual_scores_file,
    ownership_file,
    contest_results_file,
    contest_meta: Dict[str, float],
    workflow: Dict,
) -> Dict:
    optimizer_df = workflow.get("optimizer")
    if optimizer_df is None or optimizer_df.empty:
        raise ValueError("Optimizer dataset missing. Process slate first.")
    lineup_df = workflow.get("lineups_df")
    if lineup_df is None or lineup_df.empty:
        raise ValueError("No lineups available. Run optimizer first.")
    if actual_scores_file is None:
        raise ValueError("Actual player scores CSV is required.")
    actual_scores = pd.read_csv(actual_scores_file)
    ownership_df = pd.read_csv(ownership_file) if ownership_file else None
    contest_results = pd.read_csv(contest_results_file) if contest_results_file else pd.DataFrame()
    field_profile = None
    if not contest_results.empty:
        field_profile = learn_field_duplication_profile(
            contest_results,
            optimizer_df=optimizer_df,
            source=getattr(contest_results_file, "name", "contest_results.csv"),
        )
        if field_profile.samples:
            save_field_duplication_profile(field_profile, FIELD_DUPLICATION_PROFILE_PATH)
        # No separate ownership upload needed: the contest export contains every
        # entrant's lineup, so actual ownership = appearances / entries.
        if ownership_df is None or ownership_df.empty:
            derived_ownership = compute_actual_ownership(contest_results, optimizer_df=optimizer_df)
            derived_ownership = derived_ownership.dropna(subset=["fd_player_id"])
            if not derived_ownership.empty:
                ownership_df = derived_ownership[["fd_player_id", "actual_ownership_pct"]].copy()
                ownership_df["fd_player_id"] = ownership_df["fd_player_id"].astype(str)
    matched_actuals = _match_actual_scores(actual_scores, ownership_df, optimizer_df)
    actual_map = matched_actuals.set_index("fd_player_id")["actual_fd_points"].astype(float)
    lineup_points = _calculate_lineup_actuals(lineup_df, actual_map)
    if not contest_results.empty and "lineup_id" in contest_results.columns:
        lineup_points = lineup_points.merge(contest_results, on="lineup_id", how="left")
    elif not contest_results.empty:
        # Find our entries in the raw FanDuel contest export by 9-player
        # signature — pulls rank/payout without any manual mapping.
        entry_matches = match_lineups_to_contest_entries(contest_results, lineup_df)
        if not entry_matches.empty:
            lineup_points = lineup_points.merge(entry_matches, on="lineup_id", how="left")
    entry_fee = contest_meta.get("entry_fee")
    if entry_fee and "payout" in lineup_points.columns:
        lineup_points["roi"] = ((lineup_points.get("payout", 0).fillna(0) - entry_fee) / entry_fee).astype(float)
    else:
        lineup_points["roi"] = np.nan
    lineup_points["strategy_config_json"] = _serialize_strategy_config(workflow)
    sim_results = workflow.get(SIM_RESULTS_KEY) or {}
    stack_plan = sim_results.get("stack_plan")
    if not isinstance(stack_plan, pd.DataFrame) or stack_plan.empty:
        stack_plan = build_stack_exposure_plan(optimizer_df)
    learning_report = build_post_slate_learning_report(
        lineup_df,
        matched_actuals,
        optimizer_df=optimizer_df,
        stack_plan=stack_plan,
        raw_optimizer_lineups=workflow.get("optimizer_lineups_df_backup"),
    )
    failure_report = build_failure_attribution_report(
        lineup_df,
        matched_actuals,
        optimizer_df=optimizer_df,
        stack_plan=stack_plan,
        raw_optimizer_lineups=workflow.get("optimizer_lineups_df_backup"),
        sim_portfolio=sim_results.get("portfolio_df") if isinstance(sim_results, dict) else None,
    )
    boom_bust_report = build_boom_bust_calibration_report(optimizer_df, matched_actuals)

    db = SlateDatabase(DEFAULT_DB_PATH)
    db.insert_actual_scores(
        date_str,
        matched_actuals[["fd_player_id", "player_name", "actual_fd_points", "actual_ownership_pct"]],
    )
    db.insert_lineup_results(date_str, lineup_points)
    db.insert_slate_result(
        slate_tag=date_str,
        date=date_str,
        contest_type=contest_meta.get("contest_type"),
        entry_fee=entry_fee,
        num_entries=contest_meta.get("num_entries"),
        winning_score=contest_meta.get("winning_score"),
        cash_line=contest_meta.get("cash_line"),
    )
    _record_simulation_accuracy(db, date_str, lineup_points, workflow, contest_meta, matched_actuals)
    ownership_history = _build_ownership_history_rows(workflow, matched_actuals)
    if ownership_history is not None and not ownership_history.empty:
        db.insert_ownership_history(date_str, ownership_history)
    db.close()
    return {
        "lineup_points": lineup_points,
        "actuals": matched_actuals,
        "date": date_str,
        "learning_report": learning_report,
        "failure_report": failure_report,
        "boom_bust_report": boom_bust_report,
        "field_profile": field_profile,
    }




def _render_step_four() -> None:
    st.header("Step 4 – Simulate & Select")
    workflow = _get_session()
    _render_lock_countdown(workflow)
    optimizer_df: Optional[pd.DataFrame] = workflow.get("optimizer")
    lineups = workflow.get("lineups")
    lineup_df: Optional[pd.DataFrame] = workflow.get("lineups_df")
    if optimizer_df is None or optimizer_df.empty:
        st.info("Process a slate first (Step 1) to load projections.")
        return
    if not lineups or lineup_df is None or lineup_df.empty:
        st.info("Run the optimizer in Step 3 to generate candidate lineups.")
        return

    projection_cfg = workflow.get("projection_config") or {}
    if _advanced_controls_enabled():
        _render_projection_config_summary(config=projection_cfg)
        _render_projection_blend_summary(workflow.get("projection_blend_summary") or projection_cfg.get("projection_blend"))
        _render_ownership_model_summary(projection_cfg.get("ownership_model"))
    _render_candidate_pool_quality(lineup_df, _get_config_state(), _get_sim_config_state())

    if _advanced_controls_enabled():
        _render_simulation_settings_guide()
    else:
        _render_advanced_caption()
    _render_daily_step_note("simulate")

    sim_state = _get_sim_config_state()
    st.markdown("**Portfolio risk budget**")
    risk_options = list(RISK_BUDGET_PROFILES.keys())
    current_risk = str(sim_state.get("risk_profile", "Balanced leverage"))
    risk_index = risk_options.index(current_risk) if current_risk in risk_options else 0
    selected_risk = st.selectbox(
        "Risk profile",
        risk_options,
        index=risk_index,
        help="Tunes exposure, leverage, duplication, ownership scenario, marginal value, and stack-plan weights together.",
    )
    st.caption(RISK_BUDGET_PROFILES[selected_risk].description)
    if st.button("Apply Risk Budget"):
        apply_risk_budget_to_state(sim_state, selected_risk)
        st.success(f"Applied {selected_risk} risk budget.")
        st.rerun()
    if _advanced_controls_enabled():
        _render_calibration_feedback_controls(sim_state)

    st.markdown("**How many lineups to submit?**")
    sim_state["num_candidates"] = st.number_input(
        "Lineups to submit (best N from your pool)",
        min_value=1,
        max_value=int(sim_state.get("num_candidates", 300)) + 500,
        value=int(sim_state.get("num_candidates", 300) or 300),
        step=1,
        help=(
            "The simulation will rank all your generated lineups and return the best N. "
            "Example: generate 500 in Step 3, set this to 300, and you'll get your "
            "top 300 lineups to upload to FanDuel."
        ),
    )
    if int(sim_state["num_candidates"]) >= len(lineups):
        st.warning(
            f"You have {len(lineups)} candidate lineups and are asking Step 4 to submit "
            f"{int(sim_state['num_candidates'])}. Generate a larger pool in Step 3 "
            "if you want the simulation to cut down to the best 300."
        )

    if _daily_mode_enabled() and not _advanced_controls_enabled():
        stack_plan_preview = build_stack_exposure_plan(optimizer_df)
        _render_stack_exposure_plan(stack_plan_preview)

        sim_config = _build_simulation_config(sim_state)
        config_state = _get_config_state()
        if st.button("Run Simulation & Select Lineups", type="primary"):
            try:
                with st.spinner("Running Monte Carlo simulations..."):
                    (
                        contest_df,
                        portfolio_df,
                        summary,
                        selected_players,
                        selected_ids,
                        slate_sim,
                        correlation_model,
                        selected_lineup_objects,
                        stack_plan,
                        contest_result,
                        scenario_df,
                    ) = _run_simulation_stack(
                        optimizer_df,
                        lineups,
                        lineup_df,
                        sim_config,
                        salary_cap=int(config_state.get("salary_cap", 35000) or 35000),
                        use_cache=bool(sim_state.get("use_simulation_cache", True)),
                    )
                    workflow[SIM_RESULTS_KEY] = {
                        "contest_df": contest_df,
                        "portfolio_df": portfolio_df,
                        "summary": summary,
                        "sim_config": sim_config,
                        "selected_players": selected_players,
                        "selected_ids": selected_ids,
                        "slate_sim": slate_sim,
                        "correlation_model": correlation_model,
                        "selected_lineup_objects": selected_lineup_objects,
                        "stack_plan": stack_plan,
                        "contest_result": contest_result,
                        "scenario_df": scenario_df,
                        "candidate_lineups": lineups,
                        "candidate_lineup_df": lineup_df,
                    }
                st.success("Simulation complete. Review the selected portfolio below, then go to Step 5.")
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Simulation failed: {exc}")

        sim_results = workflow.get(SIM_RESULTS_KEY)
        if not sim_results:
            return

        summary = sim_results.get("summary") or {}
        if summary:
            metric_cols = st.columns(4)
            metric_cols[0].metric("Portfolio win rate", f"{summary.get('win_rate', 0.0):.2%}")
            metric_cols[1].metric("Portfolio top 1%", f"{summary.get('top1', 0.0):.2%}")
            metric_cols[2].metric("Portfolio cash rate", f"{summary.get('cash', 0.0):.2%}")
            metric_cols[3].metric("Expected ROI (sum)", f"{summary.get('roi', 0.0):.2f}x")

        portfolio_df = sim_results.get("portfolio_df")
        selected_objects = sim_results.get("selected_lineup_objects") or []
        if isinstance(portfolio_df, pd.DataFrame) and not portfolio_df.empty:
            st.subheader("Selected portfolio")
            st.dataframe(portfolio_df, width="stretch")
            st.download_button(
                "Download selected lineups",
                data=portfolio_df.to_csv(index=False).encode("utf-8"),
                file_name="portfolio_lineups.csv",
                mime="text/csv",
            )
            stack_exp = summary.get("stack_exposure") or {}
            if stack_exp:
                st.subheader("Portfolio stack exposure")
                stack_rows = [{"Team": team, "Exposure": f"{pct:.0%}"} for team, pct in sorted(stack_exp.items(), key=lambda x: x[1], reverse=True)]
                st.dataframe(pd.DataFrame(stack_rows), width="stretch")
        if selected_objects:
            col_use, col_restore = st.columns(2)
            if col_use.button("Use Simulation Portfolio for Step 5", type="primary"):
                _activate_sim_portfolio(workflow, selected_objects)
                if _daily_mode_enabled():
                    st.session_state[NAV_KEY] = "5. Review Lineups"
                st.rerun()
            if workflow.get("optimizer_lineups_backup"):
                if col_restore.button("Restore Optimizer Lineups", type="secondary"):
                    _restore_optimizer_lineups(workflow)
                    st.rerun()
        elif workflow.get("optimizer_lineups_backup"):
            if st.button("Restore Optimizer Lineups", type="secondary"):
                _restore_optimizer_lineups(workflow)
                st.rerun()
        return

    st.markdown("**Contest payout EV**")
    payout_col1, payout_col2 = st.columns(2)
    with payout_col1:
        sim_state["entry_fee"] = st.number_input(
            "Entry fee",
            min_value=0.0,
            max_value=10000.0,
            value=float(sim_state.get("entry_fee", 20.0) or 20.0),
            step=1.0,
            help="Used to convert payout dollars into expected profit and ROI.",
        )
    with payout_col2:
        sim_state["contest_entries"] = st.number_input(
            "Contest entries",
            min_value=1,
            max_value=500000,
            value=int(sim_state.get("contest_entries", 50000) or 50000),
            step=1000,
            help="Actual contest size. The simulator maps lineup percentiles into this rank field.",
        )
    payout_file = st.file_uploader(
        "Import FanDuel payout file",
        type=["csv", "xlsx", "xls"],
        key="payout_ladder_import",
        help="Upload the actual FanDuel payout table when available. Columns like Place/Prize or Rank/Payout are supported.",
    )
    if payout_file is not None:
        try:
            if payout_file.name.lower().endswith((".xlsx", ".xls")):
                payout_df = pd.read_excel(payout_file)
            else:
                payout_df = pd.read_csv(payout_file)
            imported_payouts = parse_payout_ladder_dataframe(payout_df)
            if imported_payouts:
                sim_state["payout_ladder_text"] = payout_ladder_to_text(imported_payouts)
                st.success(f"Imported {len(imported_payouts)} payout bands from {payout_file.name}.")
            else:
                st.warning("No payout bands were found in that file. Paste the ladder manually below.")
        except Exception as exc:  # pylint: disable=broad-except
            st.warning(f"Could not import payout file: {exc}")
    sim_state["payout_ladder_text"] = st.text_area(
        "Payout ladder (rank or rank range, payout dollars)",
        value=str(sim_state.get("payout_ladder_text", DEFAULT_PAYOUT_LADDER_TEXT) or DEFAULT_PAYOUT_LADDER_TEXT),
        height=150,
        help="Examples: 1,100000 or 4-10,2500. Paste the actual FanDuel payout table when available.",
    )
    parsed_payouts = parse_payout_ladder_text(sim_state["payout_ladder_text"])
    if parsed_payouts:
        total_prizes = sum((band.max_rank - band.min_rank + 1) * band.payout for band in parsed_payouts)
        paid_entries = max(band.max_rank for band in parsed_payouts)
        st.caption(
            f"Parsed {len(parsed_payouts)} payout bands, {paid_entries:,} paid spots, "
            f"${total_prizes:,.0f} total listed prizes."
        )
    else:
        st.warning("No payout ladder parsed. Step 4 will use the legacy percentile payout model.")
    _render_contest_type_brain(sim_state, parsed_payouts)
    _render_field_duplication_learning(optimizer_df)
    _render_field_realism_controls(sim_state)
    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        sim_state["num_simulations"] = st.number_input(
            "Number of simulations",
            min_value=1000,
            max_value=50000,
            value=int(sim_state.get("num_simulations", 10000) or 10000),
            step=1000,
        )
        sim_state["volatility_scale"] = st.number_input(
            "Volatility scale",
            min_value=0.5,
            max_value=2.0,
            value=float(sim_state.get("volatility_scale", 1.0) or 1.0),
            step=0.1,
        )
        sim_state["copula_nu"] = st.number_input(
            "Copula \u03bd",
            min_value=3,
            max_value=20,
            value=int(sim_state.get("copula_nu", 5) or 5),
            step=1,
        )
        sim_state["teammate_corr"] = st.number_input(
            "Teammate correlation",
            min_value=0.05,
            max_value=0.50,
            value=float(sim_state.get("teammate_corr", 0.25) or 0.25),
            step=0.01,
        )
    with col_right:
        sim_state["pitcher_vs_opposing"] = st.number_input(
            "Pitcher vs opposing hitters correlation",
            min_value=-0.30,
            max_value=0.0,
            value=float(sim_state.get("pitcher_vs_opposing", -0.15) or -0.15),
            step=0.01,
        )
        sim_state["field_size"] = st.number_input(
            "Field size (opponent lineups)",
            min_value=500,
            max_value=5000,
            value=int(sim_state.get("field_size", 1000) or 1000),
            step=500,
        )
        metric_options = ["payout_ev", "top_1pct_rate", "win_rate", "cash_rate", "expected_roi", "p99_score"]
        current_metric = sim_state.get("selection_metric", "top_1pct_rate")
        metric_index = metric_options.index(current_metric) if current_metric in metric_options else 0
        sim_state["selection_metric"] = st.selectbox(
            "Selection metric",
            metric_options,
            index=metric_index,
        )
        sim_state["diversity_weight"] = st.number_input(
            "Diversity weight",
            min_value=0.0,
            max_value=1.0,
            value=float(sim_state.get("diversity_weight", 0.3) or 0.3),
            step=0.1,
        )
        sim_state["selection_leverage_weight"] = st.number_input(
            "Selection leverage weight",
            min_value=0.0,
            max_value=1.0,
            value=float(sim_state.get("selection_leverage_weight", 0.15) or 0.0),
            step=0.05,
            help="Adds preference for high-leverage lineups during final portfolio selection.",
        )
        sim_state["selection_ownership_weight"] = st.number_input(
            "Low-ownership weight",
            min_value=0.0,
            max_value=1.0,
            value=float(sim_state.get("selection_ownership_weight", 0.10) or 0.0),
            step=0.05,
            help="Adds preference for lower total ownership lineups when simulation scores are close.",
        )
        sim_state["selection_duplication_weight"] = st.number_input(
            "Low-duplication weight",
            min_value=0.0,
            max_value=1.0,
            value=float(sim_state.get("selection_duplication_weight", 0.15) or 0.0),
            step=0.05,
            help="Adds preference for lineups less similar to the simulated field.",
        )
        sim_state["use_ownership_scenarios"] = st.checkbox(
            "Use ownership scenario engine",
            value=bool(sim_state.get("use_ownership_scenarios", True)),
            help="Tests final lineups against multiple ownership worlds, including chalk steam, value steam, and stack steam.",
        )
        sim_state["selection_scenario_weight"] = st.number_input(
            "Ownership scenario weight",
            min_value=0.0,
            max_value=1.0,
            value=float(sim_state.get("selection_scenario_weight", 0.10) or 0.0),
            step=0.05,
            help="Adds preference for lineups whose ownership and duplication risk stay acceptable across ownership scenarios.",
        )
        sim_state["selection_marginal_value_weight"] = st.number_input(
            "Portfolio marginal value weight",
            min_value=0.0,
            max_value=1.0,
            value=float(sim_state.get("selection_marginal_value_weight", 0.20) or 0.0),
            step=0.05,
            help="Rewards lineups that add new win paths to the portfolio: lower overlap, new stacks, new players, and lower duplication.",
        )
        sim_state["use_stack_exposure_engine"] = st.checkbox(
            "Use team stack exposure engine",
            value=bool(sim_state.get("use_stack_exposure_engine", True)),
            help="Uses BallparkPal team sims, stack ownership, and leverage to nudge final selection toward the best stack portfolio.",
        )
        if sim_state["use_stack_exposure_engine"]:
            sim_state["stack_target_weight"] = st.number_input(
                "Stack plan weight",
                min_value=0.0,
                max_value=1.0,
                value=float(sim_state.get("stack_target_weight", 0.20) or 0.0),
                step=0.05,
                help="How strongly Step 4 should prefer lineups using teams rated well by the stack exposure engine.",
            )
            sim_state["use_stack_auto_caps"] = st.checkbox(
                "Auto-apply team stack caps",
                value=bool(sim_state.get("use_stack_auto_caps", True)),
                help="Uses each team's recommended max stack exposure from the stack plan during final selection.",
            )
            sim_state["use_stack_auto_mins"] = st.checkbox(
                "Auto-apply core stack minimums",
                value=bool(sim_state.get("use_stack_auto_mins", False)),
                help="Adds minimum stack exposure for strong targets. Leave off unless your candidate pool is deep.",
            )
        sim_state["max_batter_exposure"] = st.number_input(
            "Max batter exposure",
            min_value=0.1,
            max_value=1.0,
            value=float(sim_state.get("max_batter_exposure", 0.4) or 0.4),
            step=0.05,
        )
        sim_state["max_pitcher_exposure"] = st.number_input(
            "Max pitcher exposure",
            min_value=0.1,
            max_value=1.0,
            value=float(sim_state.get("max_pitcher_exposure", 0.6) or 0.6),
            step=0.05,
        )
    st.markdown("**Minimum Exposure Floors**")
    min_col_left, min_col_right = st.columns(2)
    with min_col_left:
        sim_state["min_batter_exposure"] = st.number_input(
            "Min batter exposure",
            min_value=0.0,
            max_value=0.5,
            value=float(sim_state.get("min_batter_exposure", 0.0) or 0.0),
            step=0.05,
            help="Minimum % of lineups each batter should appear in. 0 = no minimum.",
        )
        sim_state["min_pitcher_exposure"] = st.number_input(
            "Min pitcher exposure",
            min_value=0.0,
            max_value=0.5,
            value=float(sim_state.get("min_pitcher_exposure", 0.0) or 0.0),
            step=0.05,
            help="Minimum % of lineups each pitcher should appear in. 0 = no minimum.",
        )
    with min_col_right:
        sim_state["min_stack_exposure"] = st.number_input(
            "Min stack exposure (per team)",
            min_value=0.0,
            max_value=0.5,
            value=float(sim_state.get("min_stack_exposure", 0.0) or 0.0),
            step=0.05,
            help="Minimum % of lineups that must include a 3+ batter stack from each team. 0 = no minimum.",
        )
        sim_state["max_stack_exposure"] = st.number_input(
            "Max stack exposure (per team)",
            min_value=0.1,
            max_value=1.0,
            value=float(sim_state.get("max_stack_exposure", 1.0) or 1.0),
            step=0.05,
            help="Maximum % of lineups that can include a stack from any single team.",
        )
    sim_state["use_stratified"] = st.checkbox(
        "Advanced: stratified sampling",
        value=bool(sim_state.get("use_stratified", False)),
        help=(
            "Makes random draws more evenly spread across all possible outcomes, "
            "reducing noise in the results. Useful when running fewer simulations "
            "(under 5,000). With 10,000+ simulations the benefit is minimal — "
            "leave this OFF unless you're running a small sim count and getting "
            "inconsistent results between runs."
        ),
    )
    sim_state["use_portfolio_optimizer"] = st.checkbox(
        "Advanced: optimize final portfolio all at once",
        value=bool(sim_state.get("use_portfolio_optimizer", True)),
        help="Uses an ILP solver to choose the final lineup set as one portfolio under exposure, stack, and overlap constraints. Falls back automatically if infeasible.",
    )
    if sim_state["use_portfolio_optimizer"]:
        sim_state["portfolio_time_limit_seconds"] = st.number_input(
            "Portfolio optimizer time limit (seconds)",
            min_value=5,
            max_value=180,
            value=int(sim_state.get("portfolio_time_limit_seconds", 45) or 45),
            step=5,
        )
    sim_state["use_simulation_cache"] = st.checkbox(
        "Advanced: reuse unchanged slate simulations",
        value=bool(sim_state.get("use_simulation_cache", True)),
        help="Caches the expensive slate and field simulation when player inputs and sim-shape settings have not changed.",
    )

    stack_plan_preview = build_stack_exposure_plan(optimizer_df)
    _render_stack_exposure_plan(stack_plan_preview)

    st.markdown("**Presets**")
    preset_cols = st.columns(3)
    if preset_cols[0].button("GPP"):
        _apply_sim_preset(SimulationConfig.gpp_preset(), sim_state)
        st.rerun()
    if preset_cols[1].button("Cash"):
        _apply_sim_preset(SimulationConfig.cash_preset(), sim_state)
        st.rerun()
    if preset_cols[2].button("Single Entry"):
        _apply_sim_preset(SimulationConfig.single_entry_preset(), sim_state)
        st.rerun()

    sim_config = _build_simulation_config(sim_state)
    config_state = _get_config_state()
    if st.button("Run Simulation & Select Lineups", type="primary"):
        try:
            with st.spinner("Running Monte Carlo simulations..."):
                sim_config = _build_simulation_config(sim_state)
                (
                    contest_df,
                    portfolio_df,
                    summary,
                    selected_players,
                    selected_ids,
                    slate_sim,
                    correlation_model,
                    selected_lineup_objects,
                    stack_plan,
                    contest_result,
                    scenario_df,
                ) = _run_simulation_stack(
                    optimizer_df,
                    lineups,
                    lineup_df,
                    sim_config,
                    salary_cap=int(config_state.get("salary_cap", 35000) or 35000),
                    use_cache=bool(sim_state.get("use_simulation_cache", True)),
                )
                workflow[SIM_RESULTS_KEY] = {
                    "contest_df": contest_df,
                    "portfolio_df": portfolio_df,
                    "summary": summary,
                    "sim_config": sim_config,
                    "selected_players": selected_players,
                    "selected_ids": selected_ids,
                    "slate_sim": slate_sim,
                    "correlation_model": correlation_model,
                    "selected_lineup_objects": selected_lineup_objects,
                    "stack_plan": stack_plan,
                    "contest_result": contest_result,
                    "scenario_df": scenario_df,
                    "candidate_lineups": lineups,
                    "candidate_lineup_df": lineup_df,
                }
            st.success("Simulation complete. Proceed to Step 5 to review final lineups.")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Simulation failed: {exc}")

    sim_results = workflow.get(SIM_RESULTS_KEY)
    if not sim_results:
        return

    selected_ids = sim_results.get("selected_ids") or []
    contest_df = sim_results.get("contest_df")
    if isinstance(contest_df, pd.DataFrame) and not contest_df.empty:
        _render_lineup_rankings(contest_df, selected_ids)
        st.download_button(
            "Download simulation results",
            data=contest_df.to_csv(index=False).encode("utf-8"),
            file_name="simulation_lineups.csv",
            mime="text/csv",
        )

    summary = sim_results.get("summary") or {}
    if summary:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Portfolio win rate", f"{summary.get('win_rate', 0.0):.2%}")
        metric_cols[1].metric("Portfolio top 1%", f"{summary.get('top1', 0.0):.2%}")
        metric_cols[2].metric("Portfolio cash rate", f"{summary.get('cash', 0.0):.2%}")
        metric_cols[3].metric("Expected ROI (sum)", f"{summary.get('roi', 0.0):.2f}x")
        if summary.get("avg_duplication_score") is not None:
            st.caption(
                f"Avg duplication score {summary.get('avg_duplication_score', 0.0):.2f} "
                f"with estimated {summary.get('avg_estimated_dupes', 0.0):.1f} duplicates per lineup. "
                f"Scenario score {summary.get('avg_scenario_score', 0.0):.2f}; "
                f"worst-case dup {summary.get('worst_case_duplication', 0.0):.2f}."
            )

    if _advanced_controls_enabled():
        _render_strategy_comparison(workflow, sim_results, optimizer_df)
        _render_portfolio_frontier(
            sim_results,
            sim_results.get("candidate_lineup_df") if isinstance(sim_results.get("candidate_lineup_df"), pd.DataFrame) else workflow.get("lineups_df"),
            int(sim_config.num_candidates),
        )

    portfolio_df = sim_results.get("portfolio_df")
    selected_objects = sim_results.get("selected_lineup_objects") or []
    if isinstance(portfolio_df, pd.DataFrame) and not portfolio_df.empty:
        st.subheader("Selected portfolio")
        st.dataframe(portfolio_df, width="stretch")
        st.download_button(
            "Download selected lineups",
            data=portfolio_df.to_csv(index=False).encode("utf-8"),
            file_name="portfolio_lineups.csv",
            mime="text/csv",
        )
        selected_players = sim_results.get("selected_players")
        if isinstance(selected_players, pd.DataFrame) and not selected_players.empty:
            exposures = _player_exposure_summary(selected_players)
            st.subheader("Portfolio exposure summary")
            st.dataframe(exposures, width="stretch")
            _render_portfolio_stress_report(selected_players, optimizer_df, sim_results.get("stack_plan"))

        stack_exp = summary.get("stack_exposure") or {}
        if stack_exp:
            st.subheader("Portfolio stack exposure")
            stack_rows = [{"Team": team, "Exposure": f"{pct:.0%}"} for team, pct in sorted(stack_exp.items(), key=lambda x: x[1], reverse=True)]
            st.dataframe(pd.DataFrame(stack_rows), width="stretch")
        scenario_df = sim_results.get("scenario_df")
        if isinstance(scenario_df, pd.DataFrame) and not scenario_df.empty:
            with st.expander("Ownership scenario details"):
                st.dataframe(scenario_df.head(1000), width="stretch", hide_index=True)
    if selected_objects:
        col_use, col_restore = st.columns(2)
        if col_use.button("Use Simulation Portfolio for Steps 5-6", type="primary"):
            _activate_sim_portfolio(workflow, selected_objects)
            st.rerun()
        if workflow.get("optimizer_lineups_backup"):
            if col_restore.button("Restore Optimizer Lineups", type="secondary"):
                _restore_optimizer_lineups(workflow)
                st.rerun()
    elif workflow.get("optimizer_lineups_backup"):
        if st.button("Restore Optimizer Lineups", type="secondary"):
            _restore_optimizer_lineups(workflow)
            st.rerun()

    if _advanced_controls_enabled():
        _player_distribution_viewer(sim_results, optimizer_df)
        _render_correlation_heatmap(sim_results, optimizer_df)
        _render_convergence_chart(sim_results, lineups)

def _render_portfolio_overview(display_df: pd.DataFrame) -> None:
    """Top-of-page portfolio summary: what did you build at a glance."""
    if display_df.empty:
        return

    n_lineups = display_df["lineup_id"].nunique()
    hitters = display_df[display_df["player_type"].str.lower() == "batter"]

    # ── Key metrics row ──
    avg_salary = display_df.groupby("lineup_id")["salary"].sum().mean()
    avg_proj = display_df.groupby("lineup_id")["proj_fd_mean"].sum().mean()
    has_ownership = "proj_fd_ownership" in display_df.columns
    avg_own = display_df.groupby("lineup_id")["proj_fd_ownership"].sum().mean() if has_ownership else None
    unique_players = display_df["full_name"].nunique()

    st.subheader("Portfolio overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lineups", n_lineups)
    m2.metric("Avg salary", f"${avg_salary:,.0f}")
    m3.metric("Avg projection", f"{avg_proj:.1f} pts")
    if avg_own is not None:
        m4.metric("Avg lineup ownership", f"{avg_own:.0%}")
    else:
        m4.metric("Unique players", unique_players)

    if hitters.empty:
        return

    # ── Team stack distribution ──
    # Count batters per team per lineup, then summarize
    team_per_lineup = (
        hitters.groupby(["lineup_id", "team_code"])
        .size()
        .reset_index(name="batters")
    )

    # Primary stacks (3+ batters from a team)
    stacks = team_per_lineup[team_per_lineup["batters"] >= 3]
    if not stacks.empty:
        stack_summary = (
            stacks.groupby("team_code")["lineup_id"]
            .nunique()
            .reset_index(name="lineups")
            .sort_values("lineups", ascending=False)
        )
        stack_summary["pct"] = (stack_summary["lineups"] / n_lineups * 100).round(1)
        stack_summary = stack_summary.rename(columns={
            "team_code": "Team",
            "lineups": "Lineups",
            "pct": "% of portfolio",
        })

        st.markdown("**Team stack distribution** (3+ batters from same team)")

        # Show as a compact horizontal bar-style summary
        team_col, chart_col = st.columns([1, 2])
        with team_col:
            st.dataframe(stack_summary, hide_index=True, use_container_width=True)
        with chart_col:
            chart_data = stack_summary.set_index("Team")["% of portfolio"]
            st.bar_chart(chart_data)

    # ── Leverage profile ──
    has_leverage = "player_leverage_score" in display_df.columns
    if has_leverage and has_ownership:
        # Per-lineup average leverage score
        lineup_lev = (
            hitters.groupby("lineup_id")["player_leverage_score"].mean()
        )
        lineup_own = (
            display_df.groupby("lineup_id")["proj_fd_ownership"].sum()
        )

        lev_pos = (lineup_lev > 0.05).sum()
        lev_neutral = ((lineup_lev >= -0.05) & (lineup_lev <= 0.05)).sum()
        lev_neg = (lineup_lev < -0.05).sum()

        st.markdown("**Leverage profile**")
        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.metric("Leverage-heavy lineups", f"{lev_pos} ({lev_pos/n_lineups:.0%})")
        lc2.metric("Neutral lineups", f"{lev_neutral} ({lev_neutral/n_lineups:.0%})")
        lc3.metric("Chalk-heavy lineups", f"{lev_neg} ({lev_neg/n_lineups:.0%})")
        lc4.metric("Avg leverage score", f"{lineup_lev.mean():+.3f}")

    # ── Pitcher distribution ──
    pitchers = display_df[display_df["player_type"].str.lower() == "pitcher"]
    if not pitchers.empty:
        pitcher_exp = (
            pitchers.groupby("full_name")["lineup_id"]
            .nunique()
            .reset_index(name="lineups")
            .sort_values("lineups", ascending=False)
        )
        pitcher_exp["pct"] = (pitcher_exp["lineups"] / n_lineups * 100).round(1)
        pitcher_exp = pitcher_exp.rename(columns={
            "full_name": "Pitcher",
            "lineups": "Lineups",
            "pct": "% of portfolio",
        })
        st.markdown("**Pitcher distribution**")
        st.dataframe(pitcher_exp, hide_index=True, use_container_width=True)

    st.markdown("---")


def _render_step_five() -> None:
    st.header("Step 5 – Review Lineups")
    workflow = _get_session()
    _render_lock_countdown(workflow)
    _render_strategy_banner(workflow)
    lineups = workflow.get("lineups")
    lineup_df = workflow.get("lineups_df")
    active_source = workflow.get("active_lineup_source", "optimizer")
    if active_source == "simulation":
        st.caption("Active lineup source: Simulation portfolio")
    else:
        st.caption("Active lineup source: Optimizer output")
    if not lineups or lineup_df is None or lineup_df.empty:
        st.info("Run the optimizer in Step 3 to review lineups.")
        return

    projection_cfg = workflow.get("projection_config") or {}
    if _advanced_controls_enabled():
        _render_projection_config_summary(config=projection_cfg)
        _render_projection_blend_summary(workflow.get("projection_blend_summary") or projection_cfg.get("projection_blend"))
        _render_ownership_model_summary(projection_cfg.get("ownership_model"))
    else:
        _render_advanced_caption()
    _render_daily_step_note("review")

    sim_results = workflow.get(SIM_RESULTS_KEY)
    selected_ids: List[int] = []
    if sim_results:
        selected_ids = sim_results.get("selected_ids") or []

    late_swap_filter_ids = st.session_state.get("late_swap_filter_ids")

    view_options = ["All lineups"]
    if selected_ids:
        view_options.append("Selected portfolio")
    if late_swap_filter_ids:
        view_options.append("Late swap focus")
    view_choice = st.radio(
        "View mode",
        view_options,
        horizontal=True,
    )

    display_df = lineup_df
    if view_choice == "Selected portfolio" and selected_ids:
        if active_source == "simulation":
            display_df = lineup_df
            st.caption(
                f"Showing {display_df['lineup_id'].nunique()} active simulation portfolio lineups."
            )
        else:
            display_df = lineup_df[lineup_df["lineup_id"].isin(selected_ids)]
        if display_df.empty:
            st.info("Selected portfolio lineups have not been generated yet.")
            display_df = lineup_df
        elif active_source != "simulation":
            st.caption(
                f"Showing {display_df['lineup_id'].nunique()} lineups from the simulated portfolio."
            )
    elif view_choice == "Selected portfolio" and not selected_ids:
        st.info("Run Step 4 to generate a simulated portfolio before filtering.")
    elif view_choice == "Late swap focus" and late_swap_filter_ids:
        display_df = lineup_df[lineup_df["lineup_id"].isin(late_swap_filter_ids)]
        if display_df.empty:
            st.info("No lineups remaining in the late swap filter. Clearing filter.")
            st.session_state.pop("late_swap_filter_ids", None)
            st.rerun()
        else:
            st.caption(
                f"Focusing on {display_df['lineup_id'].nunique()} lineups flagged in the late swap aide."
            )

    config_state = _get_config_state()
    sim_state = _get_sim_config_state()
    final_upload_view = active_source == "simulation" or view_choice == "Selected portfolio"
    audit_label = "Final upload" if final_upload_view else "Candidate pool"
    expected_lineups = (
        int(sim_state.get("num_candidates", 300) or 300)
        if final_upload_view
        else int(config_state.get("num_lineups", display_df["lineup_id"].nunique()) or display_df["lineup_id"].nunique())
    )

    _render_portfolio_overview(display_df)
    _render_final_audit_report(
        display_df,
        workflow,
        sim_results,
        config_state,
        audit_label=audit_label,
        expected_lineups=expected_lineups,
    )
    _render_lineup_scorecard(display_df, sim_results)

    if _advanced_controls_enabled():
        _render_run_history()

    _render_game_status_panel(workflow)
    if st.button("Check Lineups Now", type="secondary"):
        st.rerun()
    if late_swap_filter_ids and st.button("Clear late swap filter", key="clear_late_swap_filter"):
        st.session_state.pop("late_swap_filter_ids", None)
        st.rerun()

    visible_ids = set(display_df["lineup_id"].unique().tolist())
    if active_source == "simulation" and not visible_ids:
        st.info("Simulation portfolio is empty. Restore optimizer lineups or run Step 4 again.")

    bench_risk_mask, locked_mask = _bench_risk_flags(lineup_df)
    if visible_ids:
        view_mask = lineup_df["lineup_id"].isin(visible_ids)
    else:
        view_mask = pd.Series(True, index=lineup_df.index)
    risk_players = lineup_df[bench_risk_mask & view_mask]
    if not risk_players.empty:
        st.warning(
            f"Bench risk: {len(risk_players)} players across {risk_players['lineup_id'].nunique()} lineups without confirmed orders."
        )
    else:
        st.success("No bench-risk players detected in unlocked games.")

    _render_late_swap_panel(workflow)

    if _advanced_controls_enabled():
        with st.expander("Advanced lineup review", expanded=True):
            _render_lineup_summary(
                lineups,
                lineup_df,
                bench_risk_mask,
                locked_mask,
                visible_ids if visible_ids else None,
            )

            st.subheader("Player exposure summary")
            exposures = _player_exposure_summary(display_df)
            if exposures.empty:
                st.info("No lineups available for the selected view.")
            else:
                st.dataframe(exposures, width="stretch")

            st.subheader("Lineup variance summary")
            _variance_leaderboard(display_df)

            stack_exposure = _stack_exposure_summary(display_df)
            if not stack_exposure.empty:
                st.subheader("Stack exposure by team")
                st.dataframe(stack_exposure, width="stretch")

            if not display_df.empty:
                _render_stack_breakdown(display_df)

            if not display_df.empty:
                st.subheader("Exposure heatmap (team x position)")
                pivot = (
                    display_df.pivot_table(
                        index="team_code",
                        columns="position",
                        values="fd_player_id",
                        aggfunc=lambda x: len(x.unique()),
                        fill_value=0,
                    )
                    if {"team_code", "position", "fd_player_id"}.issubset(display_df.columns)
                    else pd.DataFrame()
                )
                if not pivot.empty:
                    st.dataframe(pivot, width="stretch")
                else:
                    st.info("Unable to build exposure heatmap (missing team/position data).")

    with st.expander("Late swap and rerun tools", expanded=_advanced_controls_enabled()):
        locks, excludes = _lock_controls(lineup_df)
        if st.button("Re-Run with Locks", type="secondary", key="rerun_locks"):
            try:
                config_state = _get_config_state()
                optimizer_df: Optional[pd.DataFrame] = workflow.get("optimizer")
                if optimizer_df is None:
                    st.error("Missing optimizer dataset in session state.")
                else:
                    with st.spinner("Re-running optimizer with locks/exclusions..."):
                        lineups, lineup_df = _run_solver(optimizer_df, config_state, locks, excludes)
                        workflow["lineups"] = lineups
                        workflow["lineups_df"] = lineup_df
                        workflow["lock_settings"] = {"locks": locks, "excludes": excludes}
                    st.success("Lineups regenerated with updated constraints.")
                    st.rerun()
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Failed to re-run optimizer: {exc}")

        scratched_players = st.multiselect(
            "Mark players scratched",
            sorted(lineup_df["full_name"].unique()),
            key="scratched_players",
        )
        if st.button("Re-Optimize Affected Lineups", type="primary"):
            try:
                config_state = _get_config_state()
                diffs = _reoptimize_scratches(scratched_players, workflow, config_state)
                if diffs:
                    for message in diffs:
                        st.write(message)
                st.success("Affected lineups re-optimized. Download files to update your entries.")
                st.rerun()
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Late swap optimization failed: {exc}")

    template_entries = workflow.get("template_entries")

    # Shuffle control
    shuffle_col1, shuffle_col2 = st.columns([2, 1])
    shuffle_enabled = shuffle_col1.checkbox(
        "Shuffle lineup order before assigning to entries",
        value=True,
        help="Randomizes which lineup goes to which entry, so your best lineups are spread evenly across all contest buy-ins.",
    )
    if shuffle_enabled:
        if shuffle_col2.button("Re-shuffle", help="Generate a new random order"):
            st.session_state["shuffle_seed"] = st.session_state.get("shuffle_seed", 0) + 1
            st.rerun()
        import random
        seed = st.session_state.get("shuffle_seed", 42)
        shuffled = list(lineups)
        random.Random(seed).shuffle(shuffled)
        export_lineups = shuffled
    else:
        export_lineups = lineups

    # ── Pre-submit lineup health check ──────────────────────────────────
    st.subheader("Pre-Submit Lineup Check")

    n_unique_lineups = len(export_lineups)
    export_errors = validate_fanduel_lineups(export_lineups)
    invalid_export_numbers = {lineup_number for lineup_number, _ in export_errors}
    exportable_lineups = [
        lineup
        for lineup_number, lineup in enumerate(export_lineups, start=1)
        if lineup_number not in invalid_export_numbers
    ]
    n_exportable_lineups = len(exportable_lineups)
    n_entries = (
        len(template_entries)
        if template_entries is not None and not template_entries.empty
        else None
    )

    if n_entries is not None:
        # Contest-aware fill preview: lineups are dealt round-robin across
        # contests and never repeated within one, so the fill count per
        # contest is what actually matters — not the global entry total.
        assignment = assign_lineups_to_contests(n_exportable_lineups, template_entries)
        contest_meta = (
            template_entries.assign(_cid=template_entries["contest_id"].astype(str))
            .groupby("_cid", sort=False)
            .agg(
                contest_name=("contest_name", "first"),
                entry_fee=("entry_fee", "first"),
                entries=("entry_id", "count"),
            )
            .reset_index()
        )
        contest_meta["filled"] = contest_meta["_cid"].map(
            lambda cid: sum(1 for idx in assignment.get(cid, []) if idx is not None)
        )
        contest_meta["unfilled"] = contest_meta["entries"] - contest_meta["filled"]
        export_row_count = int(contest_meta["filled"].sum())

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Generated Lineups", n_unique_lineups)
        col_b.metric("Exportable", n_exportable_lineups)
        col_c.metric("Contest Entries", n_entries)
        col_d.metric("Export Rows", export_row_count)

        if len(contest_meta) > 1:
            st.markdown("**Entries by contest** — shared-exposure split, no duplicate lineups within a contest")
            st.dataframe(
                contest_meta.rename(
                    columns={
                        "_cid": "Contest ID",
                        "contest_name": "Contest",
                        "entry_fee": "Entry fee",
                        "entries": "Entries",
                        "filled": "Lineups assigned",
                        "unfilled": "Unfilled",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

        if export_errors:
            first_lineup, first_error = export_errors[0]
            st.error(
                f"**{len(export_errors)} of your {n_unique_lineups} lineups cannot fit FanDuel roster slots** "
                f"and will be omitted from the upload. First skipped lineup #{first_lineup}: {first_error}. "
                f"Re-run the optimizer before submitting if you need those entries filled."
            )
        shorted = contest_meta[contest_meta["unfilled"] > 0]
        if not shorted.empty:
            detail = "; ".join(
                f"{row['contest_name'] or row['_cid']}: {int(row['unfilled'])} of {int(row['entries'])} unfilled"
                for _, row in shorted.iterrows()
            )
            st.error(
                f"**Not enough unique lineups to fill every contest entry** ({detail}). "
                f"Those entries are omitted from the upload so no contest gets a duplicate lineup. "
                f"Upload what's here, then loosen Step 3 constraints (raise chalk caps or add "
                f"stack templates) and re-run to generate the missing lineups."
            )
        elif n_exportable_lineups > export_row_count:
            st.warning(
                f"Your FanDuel template has {n_entries} contest entries, so the upload will contain "
                f"{export_row_count} rows even though {n_exportable_lineups} lineups are exportable."
            )
        else:
            st.success(f"All {n_entries} contest entries are filled. Ready to submit.")
    else:
        export_row_count = n_exportable_lineups
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Generated Lineups", n_unique_lineups)
        col_b.metric("Exportable", n_exportable_lineups)
        col_c.metric("Invalid", len(export_errors))
        if export_errors:
            first_lineup, first_error = export_errors[0]
            st.error(
                f"**{len(export_errors)} of your {n_unique_lineups} lineups cannot fit FanDuel roster slots** "
                f"and will be omitted from the upload. First skipped lineup #{first_lineup}: {first_error}. "
                f"Re-run the optimizer before submitting if you need those entries filled."
            )
        else:
            st.success(f"All {n_unique_lineups} lineups are exportable. Ready to submit.")

    # Build export — invalid lineups are filtered above; the contest-aware
    # assignment inside lineups_to_fanduel_template handles entry mapping.
    fan_duel_df = lineups_to_fanduel_template(exportable_lineups, template_entries)

    # Auto-save to local outputs folder as a fallback (always runs)
    from pathlib import Path as _Path
    import datetime as _dt
    _output_dir = _Path("outputs")
    _output_dir.mkdir(exist_ok=True)
    _stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _fd_path = _output_dir / f"fanduel_upload_{_stamp}.csv"
    _full_path = _output_dir / f"lineups_full_{_stamp}.csv"
    fan_duel_df.to_csv(_fd_path, index=False)
    lineup_df.to_csv(_full_path, index=False)
    st.caption(f"✅ Auto-saved to: `{_fd_path.resolve()}` and `{_full_path.resolve()}`")

    st.download_button(
        "Download FanDuel Upload CSV",
        data=fan_duel_df.to_csv(index=False).encode("utf-8"),
        file_name="fanduel_upload.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download Full Lineups CSV",
        data=lineup_df.to_csv(index=False).encode("utf-8"),
        file_name="lineups_full.csv",
        mime="text/csv",
    )


def _render_step_six() -> None:
    st.header("Step 6 – Post-Slate Review")
    workflow = _get_session()
    if workflow.get("optimizer") is None:
        st.info("Process a slate first to review post-slate metrics.")
        return
    if workflow.get("lineups_df") is None:
        st.info("Run the optimizer to generate lineups before posting results.")
        return

    selected_date = st.date_input("Slate date", value=pd.Timestamp.today())
    date_str = pd.Timestamp(selected_date).strftime("%Y%m%d")
    contest_type = st.text_input("Contest type", value="GPP")
    entry_fee = st.number_input("Entry fee", min_value=0.0, value=0.0, step=1.0)
    num_entries = st.number_input("Number of entries", min_value=0, value=0, step=1)
    winning_score = st.number_input("Winning score", min_value=0.0, value=0.0, step=1.0)
    cash_line = st.number_input("Cash line", min_value=0.0, value=0.0, step=1.0)

    col_scores, col_ownership, col_contest = st.columns(3)
    actual_scores_file = col_scores.file_uploader("Actual player scores CSV", type=["csv"], key="actual_scores")
    actual_ownership_file = col_ownership.file_uploader(
        "Actual ownership CSV (optional)",
        type=["csv"],
        key="actual_ownership",
        help="Leave empty if you upload a contest results CSV — actual ownership is computed automatically from every entrant's lineup in that file.",
    )
    contest_results_file = col_contest.file_uploader("Contest results CSV (optional)", type=["csv"], key="contest_results")

    contest_meta = {
        "contest_type": contest_type,
        "entry_fee": entry_fee if entry_fee > 0 else None,
        "num_entries": num_entries if num_entries > 0 else None,
        "winning_score": winning_score if winning_score > 0 else None,
        "cash_line": cash_line if cash_line > 0 else None,
    }

    if st.button("Process Results", type="primary"):
        try:
            with st.spinner("Processing actual results..."):
                result_payload = _process_results_submission(
                    date_str,
                    actual_scores_file,
                    actual_ownership_file,
                    contest_results_file,
                    contest_meta,
                    workflow,
                )
                workflow["post_slate"] = result_payload
            st.success("Results recorded. Backtest metrics updated.")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Failed to process results: {exc}")

    payload = workflow.get("post_slate")
    if payload:
        lineup_points = payload.get("lineup_points")
        actuals = payload.get("actuals")
        field_profile = payload.get("field_profile")
        _render_post_slate_learning(payload.get("learning_report"))
        _render_failure_attribution(payload.get("failure_report"))
        _render_boom_bust_calibration(payload.get("boom_bust_report"))
        if field_profile is not None and getattr(field_profile, "samples", 0):
            st.subheader("Learned field duplication profile")
            fp_cols = st.columns(4)
            fp_cols[0].metric("Field lineups", f"{field_profile.samples:,}")
            fp_cols[1].metric("Duplicate lineup %", f"{field_profile.duplicate_lineup_pct:.1%}")
            fp_cols[2].metric("Avg duplicate count", f"{field_profile.avg_duplicate_count:.2f}")
            fp_cols[3].metric("Full-salary %", f"{field_profile.full_salary_pct:.1%}")
        if lineup_points is not None:
            stored_config = _extract_projection_config_from_lineups(lineup_points)
            if stored_config:
                _render_projection_config_summary(config=stored_config)
                _render_projection_blend_summary(stored_config.get("projection_blend"))
                _render_ownership_model_summary(stored_config.get("ownership_model"))
            else:
                projection_cfg = workflow.get("projection_config") or {}
                _render_projection_config_summary(config=projection_cfg)
                _render_projection_blend_summary(workflow.get("projection_blend_summary") or projection_cfg.get("projection_blend"))
                _render_ownership_model_summary(projection_cfg.get("ownership_model"))
            st.subheader("Lineup actual performance")
            st.dataframe(lineup_points, width="stretch")
        if actuals is not None:
            st.subheader("Ownership accuracy (top 20)")
            comparison_df = actuals.merge(
                workflow["optimizer"][
                    ["fd_player_id", "full_name", "player_type", "proj_fd_ownership", "proj_fd_mean"]
                ],
                on="fd_player_id",
                how="left",
            )
            comparison_df["ownership_diff"] = (
                pd.to_numeric(comparison_df["proj_fd_ownership"], errors="coerce")
                - pd.to_numeric(comparison_df["actual_ownership_pct"], errors="coerce")
            )
            scored = comparison_df.dropna(subset=["ownership_diff"])
            if not scored.empty:
                is_pitcher = scored["player_type"].astype(str).str.lower() == "pitcher"
                mae_cols = st.columns(3)
                mae_cols[0].metric("Ownership MAE", f"{scored['ownership_diff'].abs().mean():.1%}")
                if is_pitcher.any():
                    mae_cols[1].metric("Pitcher MAE", f"{scored.loc[is_pitcher, 'ownership_diff'].abs().mean():.1%}")
                if (~is_pitcher).any():
                    mae_cols[2].metric("Hitter MAE", f"{scored.loc[~is_pitcher, 'ownership_diff'].abs().mean():.1%}")
                st.caption(
                    f"{len(scored)} players with actual ownership recorded — snapshot saved for model training "
                    "(fit with scripts/fit_ownership_model.py once 10+ slates accumulate)."
                )
            st.dataframe(
                comparison_df.sort_values("actual_ownership_pct", ascending=False).head(20)[
                    ["full_name", "fd_player_id", "proj_fd_ownership", "actual_ownership_pct", "ownership_diff"]
                ],
                width="stretch",
            )
        _ownership_accuracy_panel(date_str)
        try:
            brier = backtest.calculate_brier_score(date_str)
            if brier.get("num_players"):
                pct = brier.get("pct_above_threshold")
                st.metric(
                    f"Projection Brier (> {brier.get('threshold', 15):.0f})",
                    f"{brier.get('brier_score', float('nan')):.4f}",
                    help=f"Pct above threshold: {pct:.1%}" if pct == pct else None,
                )
        except Exception as exc:  # pylint: disable=broad-except
            st.caption(f"Brier score unavailable: {exc}")

    _render_simulation_calibration(date_str)
    _render_profile_tuning()


def _render_profile_tuning() -> None:
    """Step 6 panel: tune slate-size profile numbers from recorded finishes."""
    st.subheader("Slate-profile tuning")
    st.caption(
        "Compares how chalk-heavy and Upside-heavy lineups actually finished per slate-size "
        "profile, then recommends bounded adjustments to the auto-applied profile numbers. "
        "Requires 8+ recorded slates per profile."
    )
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=60)
    try:
        report = build_profile_tuning_report(
            DEFAULT_DB_PATH,
            start.strftime("%Y%m%d"),
            end.strftime("%Y%m%d"),
            SLATE_SIZE_PROFILES,
        )
    except Exception as exc:  # pylint: disable=broad-except
        st.caption(f"Profile tuning unavailable: {exc}")
        return

    for note in report.notes:
        st.caption(note)
    if report.per_profile is not None and not report.per_profile.empty:
        display = report.per_profile.copy()
        display["avg_roi"] = display["avg_roi"].map(lambda v: f"{v:+.0%}" if pd.notna(v) else "—")
        for col in ("own_corr", "upside_corr"):
            display[col] = display[col].map(lambda v: f"{v:+.2f}" if pd.notna(v) else "—")
        st.dataframe(display, width="stretch", hide_index=True)
    if report.recommendations is not None and not report.recommendations.empty:
        st.markdown("**Recommended profile adjustments**")
        st.dataframe(report.recommendations, width="stretch", hide_index=True)
        if st.button("Apply recommendations", key="apply_profile_tuning", type="primary"):
            _save_slate_profile_overrides(report.adjustments)
            st.success(
                f"Saved to {SLATE_PROFILES_CONFIG_PATH.name}. The next slate you process "
                "picks up the tuned profile numbers automatically."
            )


def _render_placeholder(step_label: str) -> None:
    st.header(step_label)
    st.info("This step will be implemented in a future iteration of the workflow.")


def _load_optimizer_csv(date_str: str) -> pd.DataFrame:
    matches = list(DATA_DIR.glob(f"*{date_str}_optimizer_dataset.csv"))
    if matches:
        return pd.read_csv(matches[0])
    raise FileNotFoundError(f"Optimizer dataset for {date_str} not found in {DATA_DIR}")


def _render_backtest_dashboard() -> None:
    st.header("Backtest Dashboard")
    db = SlateDatabase(DEFAULT_DB_PATH)
    default_end = pd.Timestamp.today()
    default_start = default_end - pd.Timedelta(days=30)
    start_date, end_date = st.date_input(
        "Date range",
        value=(default_start, default_end),
    )
    start_str = pd.Timestamp(start_date).strftime("%Y%m%d")
    end_str = pd.Timestamp(end_date).strftime("%Y%m%d")
    slate_df = db.fetch_slate_results_range(start_str, end_str)
    if slate_df.empty:
        st.info("No slates recorded in this range.")
        db.close()
        return

    metrics_rows = []
    for date in slate_df["date"]:
        try:
            perf = backtest.calculate_lineup_performance(date)
            own = backtest.calculate_ownership_accuracy(date)
            proj = backtest.calculate_projection_accuracy(date)
            brier = backtest.calculate_brier_score(date)
            variance = backtest.calculate_variance_metrics(date)
        except Exception:
            continue
        metrics_rows.append(
            {
                "date": date,
                "roi": perf.get("avg_roi"),
                "cash_rate": perf.get("cash_rate"),
                "ownership_mae": own.get("mae"),
                "projection_mae": proj.get("mae"),
                "brier_score": brier.get("brier_score"),
                "avg_volatility": variance.get("avg_volatility"),
                "high_leverage_count": variance.get("high_leverage_count"),
            }
        )
    metrics_df = pd.DataFrame(metrics_rows).sort_values("date")
    if metrics_df.empty:
        st.info("No metrics available for the selected range.")
    else:
        st.subheader("ROI over time")
        st.line_chart(metrics_df.set_index("date")["roi"])
        st.subheader("Cash rate over time")
        st.line_chart(metrics_df.set_index("date")["cash_rate"])
        st.subheader("Ownership MAE")
        st.line_chart(metrics_df.set_index("date")["ownership_mae"])
        st.subheader("Projection MAE")
        st.line_chart(metrics_df.set_index("date")["projection_mae"])
        if "brier_score" in metrics_df.columns:
            st.subheader("Projection Brier (>15 pts)")
            st.line_chart(metrics_df.set_index("date")["brier_score"])
        if "avg_volatility" in metrics_df.columns:
            st.subheader("Average player volatility")
            st.line_chart(metrics_df.set_index("date")["avg_volatility"])
        if "high_leverage_count" in metrics_df.columns:
            st.subheader("High-leverage targets per slate")
            st.bar_chart(metrics_df.set_index("date")["high_leverage_count"])

    brier_avg = (
        float(metrics_df["brier_score"].dropna().mean())
        if (not metrics_df.empty and "brier_score" in metrics_df.columns and metrics_df["brier_score"].notna().any())
        else float("nan")
    )

    try:
        calibration_report = build_model_calibration_report(DEFAULT_DB_PATH, start_str, end_str)
        if not calibration_report.daily.empty:
            st.subheader("Model calibration dashboard")
            cal_cols = st.columns(4)
            daily_cal = calibration_report.daily
            if "avg_roi" in daily_cal.columns:
                cal_cols[0].metric("Avg ROI", f"{pd.to_numeric(daily_cal['avg_roi'], errors='coerce').mean():.2f}x")
            if "brier_score_15" in daily_cal.columns:
                cal_cols[1].metric("Lineup Brier @150", f"{pd.to_numeric(daily_cal['brier_score_15'], errors='coerce').mean():.4f}")
            if "distribution_calibration_error" in daily_cal.columns:
                cal_cols[2].metric("Dist calibration error", f"{pd.to_numeric(daily_cal['distribution_calibration_error'], errors='coerce').mean():.3f}")
            if "winning_score_error" in daily_cal.columns:
                cal_cols[3].metric("Win score error", f"{pd.to_numeric(daily_cal['winning_score_error'], errors='coerce').mean():+.1f}")
            for note in calibration_report.lessons:
                st.caption(note)
            with st.expander("Calibration details"):
                if not calibration_report.summary.empty:
                    st.dataframe(calibration_report.summary, width="stretch", hide_index=True)
                st.dataframe(calibration_report.daily, width="stretch", hide_index=True)
    except Exception as exc:  # pylint: disable=broad-except
        st.caption(f"Model calibration dashboard unavailable: {exc}")

    try:
        strategy_report = build_strategy_backtest_report(DEFAULT_DB_PATH, start_str, end_str)
        if not strategy_report.by_strategy.empty:
            st.subheader("Historical strategy backtest")
            for note in strategy_report.lessons:
                st.caption(note)
            st.dataframe(strategy_report.by_strategy, width="stretch", hide_index=True)
            with st.expander("Daily strategy results"):
                st.dataframe(strategy_report.daily, width="stretch", hide_index=True)
    except Exception as exc:  # pylint: disable=broad-except
        st.caption(f"Strategy backtest unavailable: {exc}")

    try:
        replay_report = build_slate_replay_report(DEFAULT_DB_PATH, start_str, end_str, data_dir=DATA_DIR)
        if not replay_report.summary.empty:
            st.subheader("Historical slate replay lab")
            for note in replay_report.lessons:
                st.caption(note)
            st.dataframe(replay_report.summary, width="stretch", hide_index=True)
            with st.expander("Replay slate details"):
                st.dataframe(replay_report.daily, width="stretch", hide_index=True)
                if not replay_report.strategy_scores.empty:
                    st.dataframe(replay_report.strategy_scores, width="stretch", hide_index=True)
    except Exception as exc:  # pylint: disable=broad-except
        st.caption(f"Slate replay lab unavailable: {exc}")

    try:
        tuning_report = build_optimizer_weight_tuning_report(DEFAULT_DB_PATH, start_str, end_str, data_dir=DATA_DIR)
        if not tuning_report.candidates.empty:
            st.subheader("Optimizer weight auto-tuner")
            for note in tuning_report.lessons:
                st.caption(note)
            st.dataframe(tuning_report.candidates, width="stretch", hide_index=True)
            if st.button("Apply Tuned Weights to Step 4"):
                _get_sim_config_state().update(tuning_report.recommended_state)
                st.success(f"Applied {tuning_report.recommended_profile} tuning to Step 4.")
                st.rerun()
    except Exception as exc:  # pylint: disable=broad-except
        st.caption(f"Optimizer auto-tuner unavailable: {exc}")

    scatter_date = st.selectbox("Leverage scatter date", slate_df["date"].tolist())
    scatter_lineups = db.fetch_lineup_results(scatter_date)
    config_preview = _extract_projection_config_from_lineups(scatter_lineups)
    if config_preview:
        _render_projection_config_summary(config=config_preview)
        _render_projection_blend_summary(config_preview.get("projection_blend"))
        _render_ownership_model_summary(config_preview.get("ownership_model"))
    try:
        projections = _load_optimizer_csv(scatter_date)
        team_options = ["All"] + sorted(projections.get("team_code", pd.Series(dtype=str)).dropna().unique().tolist())
        pos_options = ["All"] + sorted(projections.get("position", pd.Series(dtype=str)).dropna().unique().tolist())
        col_team, col_pos = st.columns(2)
        team_filter = col_team.selectbox("Team filter", team_options)
        pos_filter = col_pos.selectbox("Position filter", pos_options)
        filtered_proj = projections.copy()
        if team_filter != "All":
            filtered_proj = filtered_proj[filtered_proj["team_code"] == team_filter]
        if pos_filter != "All":
            filtered_proj = filtered_proj[filtered_proj["position"] == pos_filter]
        actual = db.fetch_actual_scores(scatter_date)
        merged = filtered_proj.merge(actual[["fd_player_id", "actual_fd_points"]], on="fd_player_id", how="inner")
        merged["outperformance"] = merged["actual_fd_points"].astype(float) - merged["proj_fd_mean"].astype(float)
        scatter_data = merged[["player_leverage_score", "outperformance"]]
        st.subheader("Leverage score effectiveness")
        st.scatter_chart(
            scatter_data.rename(columns={"player_leverage_score": "Leverage", "outperformance": "Outperformance"})
        )
        leverage_stats = backtest.calculate_leverage_roi(scatter_date)
        if leverage_stats:
            st.metric(
                "Positive leverage avg outperformance",
                f"{leverage_stats.get('positive_leverage_outperformance', float('nan')):.2f}",
            )
            st.metric(
                "Negative leverage avg outperformance",
                f"{leverage_stats.get('negative_leverage_outperformance', float('nan')):.2f}",
            )
    except Exception as exc:  # pylint: disable=broad-except
        st.warning(f"Unable to load leverage scatter data: {exc}")

    _ownership_accuracy_panel(scatter_date)

    summary = backtest.get_cumulative_metrics(start_str, end_str)
    st.subheader("Summary")
    st.metric("Total slates", summary.get("total_slates"))
    st.metric("Overall ROI", summary.get("overall_roi"))
    st.metric("Overall cash rate", summary.get("overall_cash_rate"))
    st.metric("Avg Brier (>15)", f"{brier_avg:.4f}" if brier_avg == brier_avg else "N/A")
    db.close()


def main() -> None:
    st.set_page_config(page_title="MLB Daily Workflow", layout="wide")
    current_step = _sidebar_navigation()

    if current_step.startswith("1"):
        _render_step_one()
    elif current_step.startswith("2"):
        _render_step_two()
    elif current_step.startswith("3"):
        _render_step_three()
    elif current_step.startswith("4"):
        _render_step_four()
    elif current_step.startswith("5"):
        _render_step_five()
    elif current_step.startswith("6"):
        _render_step_six()
    else:
        _render_backtest_dashboard()


if __name__ == "__main__":
    main()
