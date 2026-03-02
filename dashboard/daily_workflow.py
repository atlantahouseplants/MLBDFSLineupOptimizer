"""Requirements: streamlit>=1.30, pandas.

Unified Streamlit workflow for the MLB slate optimizer (Steps 1-4).
"""
from __future__ import annotations

import io
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
from slate_optimizer.optimizer.config import OptimizerConfig
from slate_optimizer.optimizer.dataset import OPTIMIZER_COLUMNS
from slate_optimizer.optimizer.export import lineups_to_fanduel_upload
from slate_optimizer.projection import compute_baseline_projections, compute_ownership_series


WORKFLOW_KEY = "workflow_state"
NAV_KEY = "workflow_nav"
CONFIG_KEY = "optimizer_config"
LINEUPS_KEY = "lineup_results"
DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "slates.db"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def _get_session() -> Dict:
    if WORKFLOW_KEY not in st.session_state:
        st.session_state[WORKFLOW_KEY] = {}
    return st.session_state[WORKFLOW_KEY]


def _get_config_state() -> Dict:
    default_config = {
        "num_lineups": 20,
        "salary_cap": 35000,
        "stack_templates": "4,3",
        "chalk_threshold": 35.0,
        "chalk_exposure_cap": 40.0,
        "max_lineup_ownership": 0.0,
        "player_overrides": "",
        "bring_back_enabled": False,
        "bring_back_count": 1,
        "min_game_total": 0.0,
    }
    return st.session_state.setdefault(CONFIG_KEY, default_config)


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
    ownership_weights_input: Optional[str],
    recency_blend_input: Optional[str],
) -> Dict:
    if not fanduel_file:
        raise ValueError("FanDuel CSV is required.")
    if not bpp_files:
        raise ValueError("BallparkPal Excel files are required.")

    with tempfile.TemporaryDirectory(prefix="slate_tmp_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        fd_path = _save_uploaded_file(fanduel_file, temp_dir)
        bpp_dir = temp_dir / "bpp"
        bpp_dir.mkdir(exist_ok=True)
        _write_multiple(bpp_files, bpp_dir)

        vegas_path = _save_uploaded_file(vegas_file, temp_dir) if vegas_file else None
        batting_path = _save_uploaded_file(batting_file, temp_dir) if batting_file else None
        handed_path = _save_uploaded_file(handed_file, temp_dir) if handed_file else None
        recent_path = _save_uploaded_file(recent_file, temp_dir) if recent_file else None

        ownership_paths = []
        if ownership_files:
            for uploaded in ownership_files:
                ownership_paths.append(_save_uploaded_file(uploaded, temp_dir))

        ownership_weights = None
        if ownership_paths:
            ownership_weights = _parse_weights(ownership_weights_input, len(ownership_paths))

        recency_blend = _parse_recency_blend(recency_blend_input)

        bpp_loader = BallparkPalLoader(bpp_dir)
        bundle = bpp_loader.load_bundle()

        fd_loader = FanduelCSVLoader(fd_path)
        fd_players = fd_loader.load()

        combined, diagnostics = build_player_dataset(bundle, fd_players.players)
        combined["is_confirmed_lineup"] = False
        combined["batting_order_position"] = pd.Series(pd.NA, index=combined.index, dtype="Int64")
        combined["batter_hand"] = ""
        combined["pitcher_hand"] = ""
        combined["recent_last7_fppg"] = 0.0
        combined["recent_last14_fppg"] = 0.0
        combined["recent_season_fppg"] = 0.0

        combined, optional_messages = _merge_optional_sources(
            combined,
            vegas_path,
            batting_path,
            handed_path,
            recent_path,
        )

        projections = compute_baseline_projections(combined, recency_blend=recency_blend)

        ownership_paths_list = [Path(p) for p in ownership_paths]
        ownership_result = compute_ownership_series(
            combined,
            projections,
            source_paths=ownership_paths_list,
            weights=ownership_weights,
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
    ]
    summary_messages.extend(optional_messages)

    workflow_payload = {
        "players": combined,
        "projections": projections,
        "optimizer": optimizer_df,
        "diagnostics": diagnostics,
        "ownership_summary": ownership_result,
        "messages": summary_messages,
        "recency_blend": recency_blend,
    }
    return workflow_payload


def render_validation_summary(workflow: Dict) -> None:
    diagnostics = workflow.get("diagnostics")
    if diagnostics:
        st.subheader("Merge diagnostics")
        st.table(_format_diagnostics(diagnostics))
    ownership_result = workflow.get("ownership_summary")
    if ownership_result:
        if ownership_result.source_count:
            st.success(
                f"Ownership blended from {ownership_result.source_count} source(s); "
                f"coverage {ownership_result.covered_players}/{len(ownership_result.ownership)} players"
            )
        else:
            st.info("Using fallback ownership estimator (no external sources provided).")
    if workflow.get("messages"):
        for message in workflow["messages"]:
            st.write(f"- {message}")


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
    games["local_start"] = games["game_start_time"].dt.tz_convert(ZoneInfo("US/Eastern")).dt.strftime("%I:%M %p")
    return games.sort_values("game_start_time")


def _render_game_status_panel(workflow: Dict) -> None:
    optimizer_df = workflow.get("optimizer")
    games = _game_status_dataframe(optimizer_df)
    if games.empty:
        return
    st.subheader("Game lock status")
    display_cols = games[["game_key", "local_start", "status"]]
    st.dataframe(display_cols.rename(columns={"game_key": "Game", "local_start": "Start (ET)"}), use_container_width=True)


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

    stack_templates = _parse_stack_templates(config_settings.get("stack_templates", ""))
    chalk_threshold_value = config_settings.get("chalk_threshold", 0.0)
    chalk_threshold = chalk_threshold_value / 100.0 if chalk_threshold_value else None
    chalk_cap_value = config_settings.get("chalk_exposure_cap", 0.0)
    chalk_exposure_cap = chalk_cap_value / 100.0 if chalk_cap_value else None
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
        stack_templates=stack_templates,
        player_exposure_overrides=player_overrides,
        max_lineup_ownership=max_lineup_ownership,
        chalk_threshold=chalk_threshold,
        chalk_exposure_cap=chalk_exposure_cap,
        bring_back_enabled=bring_back_enabled,
        bring_back_count=bring_back_count,
        min_game_total_for_stacks=min_game_total,
    )

    df = optimizer_config.apply_exposure_overrides(df)
    df = optimizer_config.apply_ownership_strategy(df)

    num_lineups = int(config_settings.get("num_lineups", 20) or 20)
    template_tuple = tuple(stack_templates) if stack_templates else None
    stack_player_types = ("batter",)
    salary_cap_value = optimizer_config.salary_cap or 35000

    extra_lineups = num_lineups + (len(locked_players) * 2)
    lineups = generate_lineups(
        df,
        num_lineups=extra_lineups,
        salary_cap=salary_cap_value,
        min_stack_size=config_settings.get("min_stack_size", 0) or 0,
        stack_player_types=stack_player_types,
        stack_templates=template_tuple,
        max_lineup_ownership=max_lineup_ownership,
        bring_back_enabled=bring_back_enabled,
        bring_back_count=bring_back_count,
        min_game_total_for_stacks=min_game_total,
    )

    if not lineups:
        raise ValueError("Optimizer could not generate any lineups. Check constraints.")

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
        "4. Review Lineups",
        "5. Post-Slate",
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
        use_container_width=True,
    )
    st.subheader("Pitcher leverage")
    st.dataframe(
        pitchers[["full_name", "team_code", "proj_fd_mean", "proj_fd_ownership", "leverage_score"]].head(10),
        use_container_width=True,
    )


def _section_chalk(df: pd.DataFrame, top: int = 5) -> None:
    if "proj_fd_ownership" not in df.columns:
        st.info("No ownership column available.")
        return
    chalk = df.sort_values("proj_fd_ownership", ascending=False)
    st.table(chalk[["full_name", "team_code", "proj_fd_ownership", "proj_fd_mean"]].head(top))


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
    ownership_weights_input = st.text_input(
        "Ownership weights (comma-separated, optional)",
        placeholder="0.4,0.3,0.3",
    )
    recency_blend_input = st.text_input(
        "Recency blend weights (season,recent)",
        value="0.7,0.3",
    )

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
                    ownership_weights_input,
                    recency_blend_input,
                )
                session_state = _get_session()
                session_state.update(workflow_payload)
            st.success("Slate processed successfully.")
            render_validation_summary(workflow_payload)
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Failed to process slate: {exc}")
    else:
        current = _get_session()
        if current.get("optimizer") is not None:
            st.info("Session already contains a processed slate. You can re-run if inputs change.")


def _render_step_two() -> None:
    st.header("Step 2 · Review Projections")
    workflow = _get_session()
    optimizer_df: Optional[pd.DataFrame] = workflow.get("optimizer")
    if optimizer_df is None or optimizer_df.empty:
        st.info("Process a slate first (Step 1) to load projections.")
        return

    with st.sidebar.expander("Projection Filters", expanded=True):
        filtered_df = _apply_filters(optimizer_df)

    display_cols = [
        "full_name",
        "team_code",
        "position",
        "salary",
        "proj_fd_mean",
        "proj_fd_ceiling",
        "proj_fd_ownership",
        "player_leverage_score",
        "ownership_edge",
        "vegas_team_total",
        "order_factor",
        "platoon_factor",
        "recency_factor",
    ]
    available_cols = [col for col in display_cols if col in filtered_df.columns]
    st.subheader("Projection table")
    st.dataframe(filtered_df[available_cols], use_container_width=True)
    st.download_button(
        label="Download filtered projections",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_projections.csv",
        mime="text/csv",
    )

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
                st.experimental_rerun()

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
    optimizer_df: Optional[pd.DataFrame] = workflow.get("optimizer")
    if optimizer_df is None or optimizer_df.empty:
        st.info("Process a slate first (Step 1) to configure and run the optimizer.")
        return

    config_state = _get_config_state()
    config_state["num_lineups"] = st.slider(
        "Number of lineups",
        min_value=1,
        max_value=150,
        value=int(config_state.get("num_lineups", 20) or 20),
        step=1,
    )
    config_state["salary_cap"] = st.number_input(
        "Salary cap",
        min_value=10000,
        max_value=40000,
        value=int(config_state.get("salary_cap", 35000) or 35000),
        step=100,
    )
    config_state["stack_templates"] = st.text_input(
        "Stack templates (comma-separated)",
        value=config_state.get("stack_templates", "4,3"),
        help="Examples: 4,3 or 5,2. Use comma-separated integers.",
    )
    config_state["chalk_threshold"] = st.slider(
        "Chalk threshold (%)",
        min_value=0.0,
        max_value=50.0,
        value=float(config_state.get("chalk_threshold", 35.0) or 0.0),
        step=1.0,
    )
    config_state["chalk_exposure_cap"] = st.slider(
        "Chalk exposure cap (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(config_state.get("chalk_exposure_cap", 40.0) or 0.0),
        step=1.0,
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
                st.experimental_rerun()
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
) -> None:
    if lineup_df.empty:
        st.info("No lineup data available.")
        return
    if bench_mask is None or locked_mask is None:
        bench_mask, locked_mask = _bench_risk_flags(lineup_df)
    summary_rows = []
    for idx, lineup in enumerate(lineups, start=1):
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
    st.dataframe(summary_df, use_container_width=True)

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
    lineup_points["strategy_config_json"] = json.dumps(_get_config_state())

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
    db.close()
    return {"lineup_points": lineup_points, "actuals": matched_actuals}


def _render_step_four() -> None:
    st.header("Step 4 · Review Lineups")
    workflow = _get_session()
    lineups = workflow.get("lineups")
    lineup_df = workflow.get("lineups_df")
    if not lineups or lineup_df is None or lineup_df.empty:
        st.info("Run the optimizer in Step 3 to review lineups.")
        return

    _render_game_status_panel(workflow)
    if st.button("Check Lineups Now", type="secondary"):
        st.experimental_rerun()

    bench_risk_mask, locked_mask = _bench_risk_flags(lineup_df)
    risk_players = lineup_df[bench_risk_mask]
    if not risk_players.empty:
        st.warning(
            f"Bench risk: {len(risk_players)} players across {risk_players['lineup_id'].nunique()} lineups without confirmed orders."
        )
    else:
        st.success("No bench-risk players detected in unlocked games.")

    _render_lineup_summary(lineups, lineup_df, bench_risk_mask, locked_mask)

    exposures = _player_exposure_summary(lineup_df)
    st.subheader("Player exposure summary")
    st.dataframe(exposures, use_container_width=True)

    stack_exposure = _stack_exposure_summary(lineup_df)
    if not stack_exposure.empty:
        st.subheader("Stack exposure by team")
        st.dataframe(stack_exposure, use_container_width=True)

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
                st.experimental_rerun()
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
            st.experimental_rerun()
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Late swap optimization failed: {exc}")

    fan_duel_df = lineups_to_fanduel_upload(lineups)
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


def _render_step_five() -> None:
    st.header("Step 5 · Post-Slate Review")
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
            st.subheader("Lineup actual performance")
            st.dataframe(lineup_points, use_container_width=True)
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
                use_container_width=True,
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
        except Exception:
            continue
        metrics_rows.append(
            {
                "date": date,
                "roi": perf.get("avg_roi"),
                "cash_rate": perf.get("cash_rate"),
                "ownership_mae": own.get("mae"),
                "projection_mae": proj.get("mae"),
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

    scatter_date = st.selectbox("Leverage scatter date", slate_df["date"].tolist())
    try:
        projections = _load_optimizer_csv(scatter_date)
        actual = db.fetch_actual_scores(scatter_date)
        merged = projections.merge(actual[["fd_player_id", "actual_fd_points"]], on="fd_player_id", how="inner")
        merged["outperformance"] = merged["actual_fd_points"].astype(float) - merged["proj_fd_mean"].astype(float)
        scatter_data = merged[["player_leverage_score", "outperformance"]]
        st.subheader("Leverage score effectiveness")
        st.scatter_chart(scatter_data.rename(columns={"player_leverage_score": "Leverage", "outperformance": "Outperformance"}))
    except Exception as exc:  # pylint: disable=broad-except
        st.warning(f"Unable to load leverage scatter data: {exc}")

    summary = backtest.get_cumulative_metrics(start_str, end_str)
    st.subheader("Summary")
    st.metric("Total slates", summary.get("total_slates"))
    st.metric("Overall ROI", summary.get("overall_roi"))
    st.metric("Overall cash rate", summary.get("overall_cash_rate"))
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
    else:
        _render_backtest_dashboard()


if __name__ == "__main__":
    main()
