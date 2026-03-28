"""Requirements: streamlit>=1.30, pandas.

Unified Streamlit workflow for the MLB slate optimizer (Steps 1-4).
"""
from __future__ import annotations

import io
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from zoneinfo import ZoneInfo

from slate_optimizer.analysis import backtest
from slate_optimizer.data.storage import SlateDatabase

from slate_optimizer.ingestion.ballparkpal import BallparkPalLoader
from slate_optimizer.ingestion.batting_orders import BattingOrderLoader
from slate_optimizer.ingestion.fanduel import FanduelCSVLoader
from slate_optimizer.ingestion.handedness import HandednessLoader
from slate_optimizer.ingestion.recent_stats import RecentStatsLoader
from slate_optimizer.ingestion.slate_builder import build_player_dataset
from slate_optimizer.ingestion.vegas import VegasLoader
from slate_optimizer.optimizer import build_optimizer_dataset, generate_lineups
from slate_optimizer.optimizer.solver import STACK_PRESETS
from slate_optimizer.optimizer.config import OptimizerConfig
from slate_optimizer.optimizer.dataset import OPTIMIZER_COLUMNS
from slate_optimizer.optimizer.export import (
    FANDUEL_UPLOAD_COLUMNS,
    extract_template_entries,
    lineups_to_fanduel_template,
    lineups_to_fanduel_upload,
)
from slate_optimizer.projection import (
    OwnershipModelConfig,
    blend_projection_sources,
    compute_baseline_projections,
    compute_ownership_series,
)
from slate_optimizer.simulation import (
    FieldQualityMix,
    SimulationConfig,
    build_correlation_matrix,
    fit_player_distributions,
    select_portfolio,
    simulate_contest,
    simulate_field,
    simulate_slate,
)
from scipy.stats import norm

WORKFLOW_KEY = "workflow_state"
NAV_KEY = "workflow_nav"
CONFIG_KEY = "optimizer_config"
LINEUPS_KEY = "lineup_results"
SIM_CONFIG_KEY = "simulation_config"
SIM_RESULTS_KEY = "simulation_results"
DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "slates.db"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

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
        "num_lineups": 20,
        "salary_cap": 35000,
        "stack_template_selections": ["4-3-1"],
        "batter_chalk_threshold": 25.0,
        "pitcher_chalk_threshold": 35.0,
        "batter_chalk_exposure_cap": 30.0,
        "pitcher_chalk_exposure_cap": 50.0,
        "max_lineup_ownership": 0.0,
        "player_overrides": "",
        "bring_back_enabled": False,
        "bring_back_count": 1,
        "min_game_total": 0.0,
    }
    return st.session_state.setdefault(CONFIG_KEY, default_config)


def _get_sim_config_state() -> Dict:
    default_config = {
        "num_simulations": 10000,
        "volatility_scale": 1.0,
        "copula_nu": 5,
        "teammate_corr": 0.25,
        "pitcher_vs_opposing": -0.15,
        "field_size": 1000,
        "selection_metric": "top_1pct_rate",
        "diversity_weight": 0.3,
        "max_batter_exposure": 0.4,
        "max_pitcher_exposure": 0.6,
        "min_batter_exposure": 0.0,
        "min_pitcher_exposure": 0.0,
        "min_stack_exposure": 0.0,
        "max_stack_exposure": 1.0,
        "use_stratified": False,
    }
    return st.session_state.setdefault(SIM_CONFIG_KEY, default_config)


def _apply_sim_preset(preset: SimulationConfig, state: Dict) -> None:
    state["num_simulations"] = preset.num_simulations
    state["volatility_scale"] = preset.volatility_scale
    state["field_size"] = preset.num_field_lineups
    state["selection_metric"] = preset.selection_metric
    state["diversity_weight"] = preset.diversity_weight
    state["max_batter_exposure"] = preset.max_batter_exposure
    state["max_pitcher_exposure"] = preset.max_pitcher_exposure
    state["min_batter_exposure"] = preset.min_batter_exposure
    state["min_pitcher_exposure"] = preset.min_pitcher_exposure
    state["min_stack_exposure"] = preset.min_stack_exposure
    state["max_stack_exposure"] = preset.max_stack_exposure


def _build_simulation_config(state: Dict) -> SimulationConfig:
    config = SimulationConfig(
        num_simulations=int(state.get("num_simulations", 10000)),
        volatility_scale=float(state.get("volatility_scale", 1.0)),
        num_field_lineups=int(state.get("field_size", 1000)),
        selection_metric=str(state.get("selection_metric", "top_1pct_rate")),
        diversity_weight=float(state.get("diversity_weight", 0.3)),
        max_batter_exposure=float(state.get("max_batter_exposure", 0.4)),
        max_pitcher_exposure=float(state.get("max_pitcher_exposure", 0.6)),
        min_batter_exposure=float(state.get("min_batter_exposure", 0.0)),
        min_pitcher_exposure=float(state.get("min_pitcher_exposure", 0.0)),
        min_stack_exposure=float(state.get("min_stack_exposure", 0.0)),
        max_stack_exposure=float(state.get("max_stack_exposure", 1.0)),
    )
    config.correlation.teammate_base = float(state.get("teammate_corr", config.correlation.teammate_base))
    config.correlation.pitcher_vs_opposing = float(state.get("pitcher_vs_opposing", config.correlation.pitcher_vs_opposing))
    config.correlation.copula_nu = int(state.get("copula_nu", config.correlation.copula_nu))
    config.use_stratified = bool(state.get("use_stratified", False))
    return config


def _run_simulation_stack(
    optimizer_df: pd.DataFrame,
    lineups,
    lineup_df: Optional[pd.DataFrame],
    sim_config: SimulationConfig,
    salary_cap: int,
):
    distributions = fit_player_distributions(optimizer_df, sim_config.volatility_scale)
    correlation_model = build_correlation_matrix(optimizer_df, sim_config.correlation)
    slate_sim = simulate_slate(
        distributions,
        correlation_model,
        num_simulations=sim_config.num_simulations,
        seed=sim_config.seed,
        use_antithetic=sim_config.use_antithetic,
    )
    quality_mix = FieldQualityMix(
        shark_pct=sim_config.field_quality_shark_pct,
        rec_pct=sim_config.field_quality_rec_pct,
        random_pct=sim_config.field_quality_random_pct,
    )
    field_sim = simulate_field(
        optimizer_df,
        num_opponent_lineups=sim_config.num_field_lineups,
        salary_cap=salary_cap,
        seed=sim_config.seed,
        position_constraints=True,
        quality_mix=quality_mix,
    )
    contest_result = simulate_contest(
        lineups,
        slate_sim,
        field_sim,
        entry_fee=sim_config.entry_fee,
        payout_structure=sim_config.payout_structure,
    )
    contest_df = contest_result.to_dataframe().sort_values(
        sim_config.selection_metric, ascending=False
    )
    # Build pitcher ID set for position-aware exposure limits
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
    )
    portfolio_df = portfolio.to_dataframe() if portfolio.selected else pd.DataFrame()
    raw_ids: List[int] = []
    if not portfolio_df.empty and "lineup_id" in portfolio_df.columns:
        raw_ids = [int(value) for value in portfolio_df["lineup_id"].tolist()]
    selected_ids = [idx + 1 for idx in raw_ids]
    selected_players = pd.DataFrame()
    if selected_ids and lineup_df is not None and not lineup_df.empty:
        selected_players = lineup_df[lineup_df["lineup_id"].isin(selected_ids)].copy()
    summary = {
        "win_rate": portfolio.portfolio_win_rate,
        "top1": portfolio.portfolio_top1pct_rate,
        "cash": portfolio.portfolio_cash_rate,
        "roi": portfolio.portfolio_expected_roi,
        "stack_exposure": portfolio.stack_exposure,
    }
    selected_objects = []
    if selected_ids:
        zero_based = [max(0, idx - 1) for idx in selected_ids]
        for idx in zero_based:
            if 0 <= idx < len(lineups):
                selected_objects.append(lineups[idx])
    return (
        contest_df,
        portfolio_df,
        summary,
        selected_players,
        selected_ids,
        slate_sim,
        correlation_model,
        selected_objects,
    )


def _save_uploaded_file(uploaded, directory: Path) -> Path:
    path = directory / uploaded.name
    with open(path, "wb") as temp_file:
        temp_file.write(uploaded.getbuffer())
    return path


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
        if batting_path is None and batting_file:
            batting_path = _save_uploaded_file(batting_file, temp_dir)
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

        # Filter to confirmed starters only when lineup data was provided
        if confirmed_player_names:
            from slate_optimizer.ingestion.text_utils import canonicalize_series as _canon2
            combined["_canon_check"] = _canon2(combined["full_name"])
            name_match = combined["_canon_check"].isin(confirmed_player_names)
            # Also include merge-based matches
            if "is_confirmed_lineup" in combined.columns:
                name_match = name_match | combined["is_confirmed_lineup"]
            confirmed = combined[name_match]
            n_confirmed_pitchers = (confirmed["player_type"].str.lower() == "pitcher").sum()
            n_confirmed_hitters = (confirmed["player_type"].str.lower() != "pitcher").sum()
            n_pasted = len(confirmed_player_names)

            # Always filter when lineup data is provided — non-starters skew results
            before_count = len(combined)
            combined = confirmed.drop(columns=["_canon_check"]).reset_index(drop=True)
            filtered_count = before_count - len(combined)
            optional_messages.append(
                f"Filtered to confirmed starters: {len(combined)} players "
                f"({n_confirmed_pitchers} pitchers + {n_confirmed_hitters} hitters, "
                f"{filtered_count} bench/inactive removed)"
            )
            if n_confirmed_pitchers + n_confirmed_hitters < n_pasted * 0.5:
                # Build list of unmatched names for debugging
                matched_canon = set(confirmed["_canon_check"].dropna()) if "_canon_check" in confirmed.columns else set()
                unmatched = sorted(confirmed_player_names - matched_canon)[:10]
                optional_messages.append(
                    f"LOW MATCH RATE: pasted {n_pasted} names but only matched "
                    f"{n_confirmed_pitchers + n_confirmed_hitters} in FanDuel pool. "
                    f"Unmatched sample: {', '.join(unmatched)}"
                )

        projections = compute_baseline_projections(
            combined,
            recency_blend=recency_blend,
            platoon_opposite_boost=platoon_opposite_boost,
            platoon_same_penalty=platoon_same_penalty,
            platoon_switch_boost=platoon_switch_boost,
        )

        projection_paths_list = [Path(p) for p in projection_paths]
        projections, projection_blend_result = blend_projection_sources(
            combined,
            projections,
            source_paths=projection_paths_list,
            weights=projection_weights,
            baseline_weight=projection_baseline_weight,
        )

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

        optimizer_df = build_optimizer_dataset(combined, projections)
        optimizer_df = _apply_ownership_edge(optimizer_df)

    summary_messages = [
        f"Players loaded: {len(combined)}",
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
    payload = {
        "optimizer": dict(_get_config_state()),
        "projection": workflow.get("projection_config"),
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
    if workflow.get("messages"):
        st.subheader("Messages")
        for message in workflow["messages"]:
            st.write(message)


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
        }
        for detail in sources
    ]
    table = pd.DataFrame(rows)
    st.table(table)
    if total_players is not None:
        st.caption(
            f"Coverage {summary.covered_players}/{total_players} players. External sources: {summary.source_count}"
        )


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
    for detail in sources:
        if hasattr(detail, "name"):
            rows.append(
                {
                    "Source": detail.name,
                    "Weight": f"{detail.weight:.2f}",
                    "Matched players": detail.matched_players,
                    "Floor": "Yes" if getattr(detail, "has_floor", False) else "No",
                    "Ceiling": "Yes" if getattr(detail, "has_ceiling", False) else "No",
                }
            )
        else:
            rows.append(
                {
                    "Source": detail.get("name", "source"),
                    "Weight": f"{float(detail.get('weight', 0.0)):.2f}",
                    "Matched players": detail.get("matched_players"),
                    "Floor": "Yes" if detail.get("has_floor") else "No",
                    "Ceiling": "Yes" if detail.get("has_ceiling") else "No",
                }
            )
    st.table(pd.DataFrame(rows))
    st.caption(f"BallparkPal share {baseline_share:.2f}")


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
        window = st.number_input(
            "Alert window (minutes)",
            min_value=15,
            max_value=180,
            value=int(st.session_state.get("late_swap_window", 90)),
            step=15,
            key="late_swap_window",
        )
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

    temp_config = dict(config_state)
    temp_config["num_lineups"] = len(affected_ids)
    new_lineups, _ = _run_solver(
        optimizer_df,
        temp_config,
        excluded_players=scratched_players,
    )
    existing = workflow.get("lineups")
    diff_messages: List[str] = []
    for lineup_id, new_lineup in zip(affected_ids, new_lineups):
        idx = lineup_id - 1
        old_lineup = existing[idx]
        old_players = set(old_lineup.dataframe["full_name"].tolist())
        new_players = set(new_lineup.dataframe["full_name"].tolist())
        removed = old_players - new_players
        added = new_players - old_players
        message = f"Lineup {lineup_id}: removed {', '.join(removed) if removed else 'none'}, added {', '.join(added) if added else 'none'}"
        diff_messages.append(message)
        existing[idx] = new_lineup
    workflow["lineups"] = existing
    workflow["lineups_df"] = _combine_lineups(existing)
    return diff_messages

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
    lineup_df = _combine_lineups(lineups)
    return lineups, lineup_df


def _sidebar_navigation() -> str:
    steps = [
        "1. Slate Setup",
        "2. Review Projections",
        "3. Configure & Optimize",
        "4. Simulate & Select",
        "5. Review Lineups",
        "6. Post-Slate",
        "Backtest Dashboard",
    ]
    default_step = st.session_state.get(NAV_KEY, steps[0])
    current_step = st.sidebar.radio("Workflow Step", steps, index=steps.index(default_step))
    st.session_state[NAV_KEY] = current_step
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

    col1, col2 = st.columns(2)
    fanduel_file = col1.file_uploader("FanDuel player CSV", type=["csv"], accept_multiple_files=False)
    bpp_files = col2.file_uploader(
        "BallparkPal Excel files (Batters, Pitchers, Games, Teams)",
        type=["xlsx"],
        accept_multiple_files=True,
    )

    vegas_file = st.file_uploader("Vegas lines CSV (optional)", type=["csv"], key="vegas")
    batting_orders_file = st.file_uploader("Batting orders CSV (optional)", type=["csv"], key="orders")
    lineup_paste = st.text_area(
        "Paste lineups from FantasyLabs / RotoGrinders (optional)",
        height=200,
        key="lineup_paste",
        help="Copy and paste the full lineup page from FantasyLabs, RotoGrinders, etc. "
             "This will be used as batting order data. Overrides the CSV upload above if both are provided.",
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
        "Projection CSVs (optional, multiple)",
        type=["csv"],
        accept_multiple_files=True,
        key="projection_sources",
        help="Upload alternate projection sets (e.g., SaberSim, RG).",
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

    if st.button("Process Slate", type="primary"):
        try:
            with st.spinner("Running ingestion + projection pipeline..."):
                workflow_payload = _process_slate(
                    fanduel_file,
                    bpp_files or [],
                    vegas_file,
                    batting_orders_file,
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
            st.success("Slate processed successfully.")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Failed to process slate: {exc}")

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

    config_state = _get_config_state()
    config_state["num_lineups"] = st.number_input(
        "Number of lineups",
        min_value=1,
        max_value=500,
        value=int(config_state.get("num_lineups", 20) or 20),
        step=1,
        help="For multi-contest play, generate enough unique lineups to cover all your entries across contests.",
    )
    config_state["salary_cap"] = st.number_input(
        "Salary cap",
        min_value=10000,
        max_value=40000,
        value=int(config_state.get("salary_cap", 35000) or 35000),
        step=100,
    )
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

    _render_readiness_checklist(workflow)

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
            st.success(f"Generated {len(lineups)} lineups. Proceed to Step 4.")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Optimizer failed: {exc}")


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
    grouped = df.groupby("lineup_id").agg(
        total_actual_points=("actual_points", "sum"),
        total_projection=("proj_fd_mean", "sum"),
    )
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
    merged = contest_df.merge(
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
        "total_ownership",
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
            "total_ownership": "Own %",
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
    matched_actuals = _match_actual_scores(actual_scores, ownership_df, optimizer_df)
    actual_map = matched_actuals.set_index("fd_player_id")["actual_fd_points"].astype(float)
    lineup_points = _calculate_lineup_actuals(lineup_df, actual_map)
    if not contest_results.empty and "lineup_id" in contest_results.columns:
        lineup_points = lineup_points.merge(contest_results, on="lineup_id", how="left")
    entry_fee = contest_meta.get("entry_fee")
    if entry_fee and "payout" in lineup_points.columns:
        lineup_points["roi"] = ((lineup_points.get("payout", 0).fillna(0) - entry_fee) / entry_fee).astype(float)
    else:
        lineup_points["roi"] = np.nan
    lineup_points["strategy_config_json"] = _serialize_strategy_config(workflow)

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
    db.close()
    return {"lineup_points": lineup_points, "actuals": matched_actuals, "date": date_str}




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
    _render_projection_config_summary(config=projection_cfg)
    _render_projection_blend_summary(workflow.get("projection_blend_summary") or projection_cfg.get("projection_blend"))
    _render_ownership_model_summary(projection_cfg.get("ownership_model"))

    sim_state = _get_sim_config_state()
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
        metric_options = ["top_1pct_rate", "win_rate", "cash_rate", "expected_roi", "p99_score"]
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
    )

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
                ) = _run_simulation_stack(
                    optimizer_df,
                    lineups,
                    lineup_df,
                    sim_config,
                    salary_cap=int(config_state.get("salary_cap", 35000) or 35000),
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

    portfolio_df = sim_results.get("portfolio_df")
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

        stack_exp = summary.get("stack_exposure") or {}
        if stack_exp:
            st.subheader("Portfolio stack exposure")
            stack_rows = [{"Team": team, "Exposure": f"{pct:.0%}"} for team, pct in sorted(stack_exp.items(), key=lambda x: x[1], reverse=True)]
            st.dataframe(pd.DataFrame(stack_rows), width="stretch")

        selected_objects = sim_results.get("selected_lineup_objects") or []
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

    _player_distribution_viewer(sim_results, optimizer_df)
    _render_correlation_heatmap(sim_results, optimizer_df)
    _render_convergence_chart(sim_results, lineups)

def _render_step_five() -> None:
    st.header("Step 5 – Review Lineups")
    workflow = _get_session()
    _render_lock_countdown(workflow)
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
    _render_projection_config_summary(config=projection_cfg)
    _render_projection_blend_summary(workflow.get("projection_blend_summary") or projection_cfg.get("projection_blend"))
    _render_ownership_model_summary(projection_cfg.get("ownership_model"))

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
        display_df = lineup_df[lineup_df["lineup_id"].isin(selected_ids)]
        if display_df.empty:
            st.info("Selected portfolio lineups have not been generated yet.")
            display_df = lineup_df
        else:
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

    if template_entries is not None and not template_entries.empty:
        n_entries = len(template_entries)
        n_unique = len(export_lineups)
        if n_unique < n_entries:
            st.warning(
                f"You have {n_entries} contest entries but only {n_unique} unique lineups. "
                f"Each lineup will be repeated up to {-(-n_entries // n_unique)} times. "
                f"Increase **Number of lineups** in Step 3 to at least {n_entries} to avoid duplicates."
            )

    fan_duel_df = lineups_to_fanduel_template(export_lineups, template_entries)

    # Flag any duplicate lineups in the export
    if not fan_duel_df.empty:
        upload_cols = [c for c in FANDUEL_UPLOAD_COLUMNS if c in fan_duel_df.columns]
        if upload_cols:
            lineup_strs = fan_duel_df[upload_cols].apply(
                lambda row: "|".join(sorted(str(v) for v in row)), axis=1
            )
            n_dupes = lineup_strs.duplicated().sum()
            if n_dupes > 0:
                st.warning(f"{n_dupes} duplicate lineup(s) detected in the export file.")

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
    actual_ownership_file = col_ownership.file_uploader("Actual ownership CSV (optional)", type=["csv"], key="actual_ownership")
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
                    ["fd_player_id", "full_name", "proj_fd_ownership", "proj_fd_mean"]
                ],
                on="fd_player_id",
                how="left",
            )
            comparison_df["ownership_diff"] = (
                comparison_df["proj_fd_ownership"].astype(float) - comparison_df["actual_ownership_pct"].astype(float)
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
