from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = REPO_ROOT / "data" / "live"
OUTPUT_DIR = REPO_ROOT / "data" / "output"
FD_UPLOAD_PATH = LIVE_DIR / "fanduel_today.csv"

STACK_STYLE_MAP = {
    "4-3 (recommended)": "4,3",
    "4-4 (two big stacks)": "4,4",
    "5-3 (power stack)": "5,3",
    "3-3-2 (spread)": "3,3,2",
}

PIPELINE_MESSAGE_MAP = {
    "Loaded Vegas": "✅ Vegas lines loaded",
    "Loaded batting orders": "✅ Batting orders loaded",
    "Computing ownership": "⚙️ Estimating ownership...",
    "Estimated ownership": "✅ Ownership estimated",
    "No feasible": "❌ No lineups generated — try removing the ownership cap",
    "Saved FanDuel upload": "✅ FanDuel upload file saved",
    "fit_player_distributions": "⚙️ Fitting player score distributions...",
    "Fitting distributions": "⚙️ Fitting player score distributions...",
    "simulate_slate": "⚙️ Running Monte Carlo simulation...",
    "simulate_field": "⚙️ Simulating field...",
    "simulate_contest": "⚙️ Simulating contest performance...",
    "Wrote simulation metrics": "✅ Simulation complete",
    "Wrote portfolio summary": "✅ Portfolio selected",
    "Wrote simulated FanDuel": "✅ Simulated upload file saved",
    "Portfolio win rate": "📊 Portfolio stats calculated",
    "select_portfolio": "⚙️ Selecting best portfolio...",
}

TAB_OPTIONS = ["Today's Slate", "Run Optimizer", "Review Lineups"]


st.set_page_config(page_title="MLB DFS Dashboard", page_icon="⚾", layout="wide")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"]  {
            font-family: sans-serif;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .dashboard-card {
            background: #1e2329;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            border: 1px solid rgba(255, 255, 255, 0.06);
        }
        .status-card {
            background: #1e2329;
            border-radius: 8px;
            padding: 14px 16px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            min-height: 88px;
        }
        .status-label {
            color: #9aa4af;
            font-size: 0.84rem;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .status-value {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .status-help {
            color: #c7d0d9;
            font-size: 0.9rem;
        }
        .summary-card {
            background: #1e2329;
            border-radius: 8px;
            padding: 14px 16px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            min-height: 94px;
        }
        .summary-title {
            color: #9aa4af;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 6px;
        }
        .summary-value {
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .summary-sub {
            color: #c7d0d9;
            font-size: 0.9rem;
            margin-top: 6px;
        }
        .stack-card {
            background: #1e2329;
            border-radius: 8px;
            padding: 16px;
            border-left: 4px solid #6c757d;
            margin-bottom: 12px;
        }
        .stack-card.high {
            border-left-color: #00c851;
        }
        .stack-card.medium {
            border-left-color: #f4c542;
        }
        .stack-team {
            font-size: 1.25rem;
            font-weight: 800;
            margin-bottom: 6px;
        }
        .stack-line {
            color: #dbe2ea;
            margin: 2px 0;
            font-size: 0.95rem;
        }
        .big-download button {
            width: 100%;
            min-height: 3.2rem;
            font-weight: 700;
        }
        .positive-text {
            color: #00c851;
        }
        .warning-text {
            color: #ff6b6b;
        }
        div[data-testid="stFileUploader"] section {
            background: #1e2329;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    st.session_state.setdefault("uploaded_fd_path", str(FD_UPLOAD_PATH) if FD_UPLOAD_PATH.exists() else "")
    st.session_state.setdefault("last_run_tag", "")
    st.session_state.setdefault("last_lineups_path", "")
    st.session_state.setdefault("last_upload_output_path", "")
    st.session_state.setdefault("last_run_lineup_count", 0)
    st.session_state.setdefault("last_run_message", "")
    st.session_state.setdefault("active_tab", TAB_OPTIONS[0])


def latest_file(pattern: str, directory: Path) -> Path | None:
    matches = sorted(directory.glob(pattern), reverse=True)
    return matches[0] if matches else None


def get_live_files() -> dict[str, Path | None]:
    return {
        "batter": latest_file("bpp_batters_*.csv", LIVE_DIR),
        "pitcher": latest_file("bpp_pitchers_*.csv", LIVE_DIR),
        "projection": latest_file("bpp_dfs_projections_*.csv", LIVE_DIR),
        "vegas": latest_file("vegas_lines_*.csv", LIVE_DIR),
        "batting": latest_file("batting_orders_*.csv", LIVE_DIR),
        "park": latest_file("bpp_park_factors_*.csv", LIVE_DIR),
        "handedness": latest_file("handedness_*.csv", LIVE_DIR),
    }


def pick_output_file(suffix: str, preferred_tag: str | None = None) -> Path | None:
    matches = sorted(OUTPUT_DIR.glob(f"*{suffix}"), reverse=True)
    if preferred_tag:
        tagged = OUTPUT_DIR / f"{preferred_tag}{suffix}"
        if tagged.exists():
            return tagged
    return matches[0] if matches else None


@st.cache_data(ttl=300)
def load_csv(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def extract_data_date(files: dict[str, Path | None]) -> str:
    for file_path in (files["batter"], files["pitcher"], files["projection"], files["vegas"]):
        if not file_path:
            continue
        match = re.search(r"(\d{4}-\d{2}-\d{2})", file_path.name)
        if match:
            return match.group(1)
    return "No dated slate files found"


def render_status_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">{label}</div>
            <div class="status-value">{value}</div>
            <div class="status-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_card(title: str, value: str, subtext: str = "") -> None:
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-title">{title}</div>
            <div class="summary-value">{value}</div>
            <div class="summary-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_pct(value: float | int | None, digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}%}"


def build_stack_targets(vegas_df: pd.DataFrame, batter_df: pd.DataFrame) -> pd.DataFrame:
    if vegas_df.empty or batter_df.empty:
        return pd.DataFrame()
    games = vegas_df.copy()
    teams_in_batters = set(batter_df["Team"].dropna().astype(str).str.upper())
    rows: list[dict[str, object]] = []
    for _, row in games.iterrows():
        game = str(row.get("game", ""))
        total = pd.to_numeric(row.get("total"), errors="coerce")
        home_ml = pd.to_numeric(row.get("home_ml"), errors="coerce")
        away_ml = pd.to_numeric(row.get("away_ml"), errors="coerce")
        if "@" not in game or pd.isna(total) or pd.isna(home_ml) or pd.isna(away_ml):
            continue
        away_team, home_team = [part.strip().upper() for part in game.split("@", 1)]
        away_prob_raw = 100 / (away_ml + 100) if away_ml > 0 else (-away_ml) / (-away_ml + 100)
        home_prob_raw = 100 / (home_ml + 100) if home_ml > 0 else (-home_ml) / (-home_ml + 100)
        prob_total = away_prob_raw + home_prob_raw
        if prob_total == 0:
            continue
        away_prob = away_prob_raw / prob_total
        home_prob = home_prob_raw / prob_total
        away_total = total * away_prob
        home_total = total * home_prob
        rows.extend(
            [
                {
                    "team": away_team,
                    "opponent": home_team,
                    "game_total": total,
                    "implied_total": away_total,
                },
                {
                    "team": home_team,
                    "opponent": away_team,
                    "game_total": total,
                    "implied_total": home_total,
                },
            ]
        )
    stacks = pd.DataFrame(rows)
    if stacks.empty:
        return stacks
    stacks = stacks[stacks["team"].isin(teams_in_batters)].copy()
    stacks = stacks.sort_values("implied_total", ascending=False).head(5).reset_index(drop=True)
    return stacks


def stack_tier(implied_total: float) -> tuple[str, str]:
    if implied_total > 5.0:
        return "🔥 HIGH", "high"
    if implied_total >= 4.0:
        return "🟡 MED", "medium"
    return "⚪ LOW", "low"


def render_stack_cards(stacks: pd.DataFrame) -> None:
    if stacks.empty:
        st.info("Top stacks will appear once both BPP batter data and Vegas lines are available.")
        return
    for start in range(0, len(stacks), 3):
        cols = st.columns(3)
        for col, (_, row) in zip(cols, stacks.iloc[start : start + 3].iterrows()):
            tag, tier_class = stack_tier(float(row["implied_total"]))
            with col:
                st.markdown(
                    f"""
                    <div class="stack-card {tier_class}">
                        <div class="stack-team">{row['team']}</div>
                        <div class="stack-line">Opp: {row['opponent']}</div>
                        <div class="stack-line">Game Total: {row['game_total']:.1f}</div>
                        <div class="stack-line">Implied: {row['implied_total']:.2f}</div>
                        <div class="stack-line">{tag}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def build_pitcher_board(pitcher_df: pd.DataFrame) -> pd.DataFrame:
    if pitcher_df.empty:
        return pd.DataFrame()
    board = pitcher_df.copy()
    for col in ["PointsFD", "WinPct", "Strikeouts", "Innings"]:
        if col in board.columns:
            board[col] = safe_numeric(board[col])
    board = board.sort_values("PointsFD", ascending=False).head(5).copy()
    if board.empty:
        return board
    board["Pitcher"] = board["FullName"]
    board.loc[board.index[0], "Pitcher"] = "🥇 " + str(board.iloc[0]["Pitcher"])
    board["Win%"] = board["WinPct"].map(lambda x: format_pct(x, 1))
    board["Proj FD Pts"] = board["PointsFD"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    board["K's"] = board["Strikeouts"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    board["Inn"] = board["Innings"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    return board[["Pitcher", "Team", "Opponent", "Proj FD Pts", "Win%", "K's", "Inn"]]


def build_leverage_board(projection_df: pd.DataFrame) -> pd.DataFrame:
    if projection_df.empty:
        return pd.DataFrame()
    bats = projection_df.copy()
    bats["dfs_avg"] = safe_numeric(bats["dfs_avg"])
    bats["bust_pct"] = safe_numeric(bats["bust_pct"])
    bats = bats.dropna(subset=["dfs_avg", "bust_pct"])
    if bats.empty:
        return bats
    bats["pts_rank_pct"] = bats["dfs_avg"].rank(pct=True)
    bats["own_proxy"] = (1 - bats["bust_pct"]).clip(lower=0, upper=1)
    bats["ownership_rank_pct"] = bats["own_proxy"].rank(pct=True)
    bats["leverage_score"] = bats["pts_rank_pct"] - bats["ownership_rank_pct"]
    bats = bats.sort_values(["leverage_score", "dfs_avg"], ascending=[False, False])
    bats = bats[bats["leverage_score"] > 0].head(8).copy()
    if bats.empty:
        return bats
    bats["Proj"] = bats["dfs_avg"].map(lambda x: f"{x:.2f}")
    bats["Own%"] = bats["own_proxy"].map(lambda x: f"{x:.0%}")
    bats["Leverage ↑"] = bats["leverage_score"]
    return bats.rename(
        columns={
            "full_name": "Player",
            "team": "Team",
            "position": "Pos",
        }
    )[["Player", "Team", "Pos", "Proj", "Own%", "Leverage ↑"]]


def style_leverage_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    return df.style.format({"Leverage ↑": "{:.2f}"}).map(
        lambda value: "color: #00C851; font-weight: 700;" if float(value) > 0.1 else "color: #b7bec7;",
        subset=["Leverage ↑"],
    )


def build_chalk_board(projection_df: pd.DataFrame) -> pd.DataFrame:
    if projection_df.empty:
        return pd.DataFrame()
    chalk = projection_df.copy()
    chalk["dfs_avg"] = safe_numeric(chalk["dfs_avg"])
    chalk["bust_pct"] = safe_numeric(chalk["bust_pct"])
    chalk = chalk.dropna(subset=["dfs_avg", "bust_pct"])
    if chalk.empty:
        return chalk
    chalk["Own%"] = (1 - chalk["bust_pct"]).clip(lower=0, upper=1)
    chalk = chalk.sort_values(["bust_pct", "dfs_avg"], ascending=[True, False]).head(5).copy()
    chalk["Proj"] = chalk["dfs_avg"].map(lambda x: f"{x:.2f}")
    chalk["Own%"] = chalk["Own%"].map(lambda x: f"{x:.0%}")
    chalk["Tag"] = chalk["Own%"].apply(
        lambda x: "⚠️" if float(x.strip("%")) > 30 else ""
    )
    return chalk.rename(
        columns={
            "full_name": "Player",
            "team": "Team",
            "position": "Pos",
        }
    )[["Player", "Team", "Pos", "Proj", "Own%", "Tag"]]


def filter_player_pool(df: pd.DataFrame, query: str) -> pd.DataFrame:
    pool = df.copy()
    pool["dfs_avg"] = safe_numeric(pool["dfs_avg"])
    pool["upside"] = safe_numeric(pool["upside"])
    pool["bust_pct"] = safe_numeric(pool["bust_pct"])
    pool["pts_per_k"] = safe_numeric(pool["pts_per_k"])
    if query:
        needle = query.strip().lower()
        mask = pool["full_name"].str.lower().str.contains(needle, na=False) | pool["team"].str.lower().str.contains(needle, na=False)
        pool = pool[mask]
    pool = pool.sort_values("dfs_avg", ascending=False).copy()
    pool["Proj FD"] = pool["dfs_avg"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    pool["Upside"] = pool["upside"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    pool["Bust%"] = pool["bust_pct"].map(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
    pool["Pts/$"] = pool["pts_per_k"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    return pool.rename(
        columns={
            "full_name": "Player",
            "team": "Team",
            "position": "Pos",
        }
    )[["Player", "Team", "Pos", "Proj FD", "Upside", "Bust%", "Pts/$"]]


def save_uploaded_fanduel_file(upload) -> tuple[Path, int]:
    content = upload.getvalue()
    FD_UPLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    FD_UPLOAD_PATH.write_bytes(content)
    player_count = len(pd.read_csv(FD_UPLOAD_PATH))
    st.session_state["uploaded_fd_path"] = str(FD_UPLOAD_PATH)
    return FD_UPLOAD_PATH, player_count


def checklist_badge(ok: bool, label: str, warning: bool = False) -> str:
    icon = "✅" if ok else ("⚠️" if warning else "❌")
    return f"{icon} {label}"


def map_pipeline_line(raw_line: str) -> tuple[str | None, int | None]:
    cleaned = raw_line.strip()
    if not cleaned:
        return None, None
    for needle, friendly in PIPELINE_MESSAGE_MAP.items():
        if needle in cleaned:
            return friendly, extract_generated_count(cleaned)
    if cleaned.startswith("Generated "):
        count = extract_generated_count(cleaned)
        if count is not None:
            return f"✅ Generated {count} lineups!", count
    return None, extract_generated_count(cleaned)


def extract_generated_count(text: str) -> int | None:
    match = re.search(r"Generated\s+(\d+)\s+lineups", text)
    if match:
        return int(match.group(1))
    return None


def run_pipeline(
    pool_size: int,
    submit_count: int,
    stack_style: str,
    bring_back: bool,
    max_own: float,
    files: dict[str, Path | None],
) -> None:
    python_exe = sys.executable
    stack_template_str = STACK_STYLE_MAP[stack_style]
    tag = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    # Use the simulated pipeline: generate pool → MC sim → select best portfolio
    cmd = [python_exe, str(REPO_ROOT / "scripts" / "run_simulated_pipeline.py")]
    cmd += ["--bpp-source", str(REPO_ROOT / "data" / "live")]
    cmd += ["--fanduel-csv", str(Path(st.session_state["uploaded_fd_path"]))]
    cmd += ["--output-dir", str(REPO_ROOT / "data" / "output")]
    cmd += ["--write-intermediate"]
    cmd += ["--num-candidates", str(pool_size)]   # how many to generate
    cmd += ["--num-lineups", str(submit_count)]   # how many to select for submission
    cmd += ["--stack-templates", stack_template_str]
    cmd += ["--tag", tag]
    if bring_back:
        cmd += ["--bring-back"]
    if max_own > 0:
        cmd += ["--max-lineup-ownership", str(max_own)]
    if files["vegas"]:
        cmd += ["--vegas-csv", str(files["vegas"])]
    if files["batting"]:
        cmd += ["--batting-orders-csv", str(files["batting"])]
    if files["handedness"]:
        cmd += ["--handedness-csv", str(files["handedness"])]

    log_placeholder = st.empty()
    status_lines = ["🚀 Starting optimizer..."]
    output_lines: list[str] = []
    generated_count = 0
    log_placeholder.markdown("\n".join(status_lines))

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert process.stdout is not None
    for raw_line in iter(process.stdout.readline, ""):
        output_lines.append(raw_line.rstrip())
        friendly, count = map_pipeline_line(raw_line)
        if count is not None:
            generated_count = count
        if friendly and friendly not in status_lines:
            status_lines.append(friendly)
        log_placeholder.markdown("\n".join(status_lines))
    return_code = process.wait()

    lineups_path = OUTPUT_DIR / f"{tag}_lineups.csv"
    # Simulated pipeline produces _simulated_upload.csv; fall back to basic upload
    upload_path = OUTPUT_DIR / f"{tag}_simulated_upload.csv"
    if not upload_path.exists():
        upload_path = OUTPUT_DIR / f"{tag}_fanduel_upload.csv"
    sim_results_path = OUTPUT_DIR / f"{tag}_sim_results.csv"
    success = return_code == 0 and upload_path.exists()

    if success:
        st.session_state["last_run_tag"] = tag
        st.session_state["last_lineups_path"] = str(lineups_path) if lineups_path.exists() else str(upload_path)
        st.session_state["last_upload_output_path"] = str(upload_path)
        st.session_state["last_run_lineup_count"] = generated_count

        # Show simulation portfolio stats if available
        portfolio_path = OUTPUT_DIR / f"{tag}_sim_portfolio.csv"
        sim_stats = ""
        if portfolio_path.exists():
            try:
                import pandas as _pd
                pf = _pd.read_csv(portfolio_path)
                if not pf.empty and "win_rate" in pf.columns:
                    wr = pf["win_rate"].mean()
                    t1 = pf["top_1pct_rate"].mean()
                    roi = pf["expected_roi"].mean()
                    sim_stats = f"  \n📊 **Sim stats:** Win rate {wr:.2%} | Top-1% rate {t1:.2%} | ROI {roi:.2f}x"
            except Exception:
                pass

        st.session_state["last_run_message"] = (
            f"✅ Done! Selected {submit_count} lineups from {pool_size} simulated.\n\n"
            f"📁 FanDuel upload: {upload_path.relative_to(REPO_ROOT)}{sim_stats}"
        )
        st.success(f"✅ Done! Selected best {submit_count} lineups from a pool of {pool_size}.")
        if sim_stats:
            st.markdown(sim_stats)
        st.write(f"📁 Upload file: `{upload_path.relative_to(REPO_ROOT)}`")
        with upload_path.open("rb") as handle:
            st.download_button(
                "Download FanDuel Upload CSV",
                data=handle.read(),
                file_name=upload_path.name,
                mime="text/csv",
            )
        st.session_state["pending_tab"] = "Review Lineups"
        st.rerun()

    st.session_state["last_run_message"] = "\n".join(status_lines[-3:] if status_lines else output_lines[-3:])
    if return_code != 0:
        st.error("Optimizer run failed. Check the progress log and try again.")
    elif generated_count == 0:
        st.error("No lineups generated — try removing the ownership cap.")
    else:
        st.error("Pipeline finished, but expected output files were not found.")

    if output_lines:
        with st.expander("Pipeline log"):
            st.code("\n".join(output_lines), language="text")


def load_review_files() -> tuple[Path | None, Path | None]:
    preferred_tag = st.session_state.get("last_run_tag") or None
    lineups_path = None
    upload_path = None

    saved_lineups = st.session_state.get("last_lineups_path")
    if saved_lineups and Path(saved_lineups).exists():
        lineups_path = Path(saved_lineups)
    else:
        lineups_path = pick_output_file("_lineups.csv", preferred_tag)

    saved_upload = st.session_state.get("last_upload_output_path")
    if saved_upload and Path(saved_upload).exists():
        upload_path = Path(saved_upload)
    else:
        # Prefer simulated upload over basic upload
        upload_path = pick_output_file("_simulated_upload.csv", preferred_tag)
        if not upload_path:
            upload_path = pick_output_file("_fanduel_upload.csv", preferred_tag)

    return lineups_path, upload_path


def build_review_summary(lineups_df: pd.DataFrame) -> dict[str, str]:
    total_lineups = int(lineups_df["lineup_id"].nunique())
    salary_summary = lineups_df.groupby("lineup_id")["salary"].sum()
    proj_summary = safe_numeric(lineups_df["proj_fd_mean"]).groupby(lineups_df["lineup_id"]).sum()
    own_summary = safe_numeric(lineups_df["proj_fd_ownership"]).groupby(lineups_df["lineup_id"]).sum()

    hitters = lineups_df[lineups_df["player_type"].astype(str).str.lower() == "batter"].copy()
    team_lineups = (
        hitters.groupby("team_code")["lineup_id"].nunique().sort_values(ascending=False)
        if not hitters.empty
        else pd.Series(dtype="int64")
    )
    top_stack = "—"
    if not team_lineups.empty:
        top_stack = f"{team_lineups.index[0]} ({team_lineups.iloc[0]} lineups)"

    pitchers = lineups_df[lineups_df["roster_position"].astype(str) == "P"].copy()
    main_pitcher = "—"
    if not pitchers.empty:
        pitcher_counts = pitchers.groupby("full_name")["lineup_id"].nunique().sort_values(ascending=False)
        top_pitcher_name = pitcher_counts.index[0]
        main_pitcher = f"{top_pitcher_name} ({pitcher_counts.iloc[0] / total_lineups:.0%})"

    return {
        "n_lineups": str(total_lineups),
        "avg_proj": f"{proj_summary.mean():.1f} pts" if not proj_summary.empty else "—",
        "avg_own": f"{own_summary.mean():.0%}" if not own_summary.empty else "—",
        "top_stack": top_stack,
        "main_pitcher": main_pitcher,
        "avg_salary": f"${salary_summary.mean():,.0f}" if not salary_summary.empty else "—",
    }


def build_exposure_df(lineups_df: pd.DataFrame) -> pd.DataFrame:
    total_lineups = int(lineups_df["lineup_id"].nunique())
    exposures = (
        lineups_df.groupby(["full_name", "team_code"])["lineup_id"]
        .nunique()
        .reset_index(name="lineups")
        .sort_values(["lineups", "full_name"], ascending=[False, True])
    )
    exposures["exposure"] = exposures["lineups"] / total_lineups
    exposures["label"] = exposures["full_name"] + " (" + exposures["team_code"] + ")"
    return exposures.head(20)


def build_exposure_chart(exposure_df: pd.DataFrame) -> alt.Chart:
    chart_df = exposure_df.copy()
    chart_df["tier"] = chart_df["exposure"].apply(
        lambda x: "red" if x > 0.65 else ("yellow" if x >= 0.5 else "green")
    )
    color_scale = alt.Scale(
        domain=["green", "yellow", "red"],
        range=["#00C851", "#F4C542", "#FF6B6B"],
    )
    return (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("exposure:Q", axis=alt.Axis(format="%"), title="Exposure"),
            y=alt.Y("label:N", sort="-x", title=""),
            color=alt.Color("tier:N", scale=color_scale, legend=None),
            tooltip=[
                alt.Tooltip("full_name:N", title="Player"),
                alt.Tooltip("team_code:N", title="Team"),
                alt.Tooltip("lineups:Q", title="Lineups"),
                alt.Tooltip("exposure:Q", title="Exposure", format=".0%"),
            ],
        )
        .properties(height=520)
    )


def build_stack_breakdown(lineups_df: pd.DataFrame) -> pd.DataFrame:
    hitters = lineups_df[lineups_df["player_type"].astype(str).str.lower() == "batter"].copy()
    total_lineups = int(lineups_df["lineup_id"].nunique())
    if hitters.empty or total_lineups == 0:
        return pd.DataFrame()
    lineups_per_team = hitters.groupby("team_code")["lineup_id"].nunique()
    hitter_spots = hitters.groupby("team_code").size()
    breakdown = pd.DataFrame(
        {
            "Team": lineups_per_team.index,
            "Lineups": [f"{count}/{total_lineups}" for count in lineups_per_team.values],
            "Hitter Spots": [int(hitter_spots.loc[team]) for team in lineups_per_team.index],
            "Stack %": [f"{count / total_lineups:.0%}" for count in lineups_per_team.values],
        }
    )
    return breakdown.sort_values("Hitter Spots", ascending=False).reset_index(drop=True)


def build_lineup_table(lineups_df: pd.DataFrame) -> pd.DataFrame:
    if lineups_df.empty:
        return pd.DataFrame()
    ordered_positions = ["P", "C/1B", "2B", "3B", "SS", "OF", "OF", "OF", "UTIL"]
    rows: list[dict[str, object]] = []
    for lineup_id, lineup in lineups_df.groupby("lineup_id"):
        row: dict[str, object] = {
            "Lineup #": int(lineup_id),
            "Salary": int(safe_numeric(lineup["salary"]).sum()),
            "Proj FD": round(float(safe_numeric(lineup["proj_fd_mean"]).sum()), 2),
        }
        position_counts: dict[str, int] = {}
        for _, player in lineup.iterrows():
            roster_position = str(player.get("roster_position", ""))
            if roster_position == "OF":
                position_counts["OF"] = position_counts.get("OF", 0) + 1
                key = f"OF{position_counts['OF']}"
            else:
                key = roster_position
            row[key] = player.get("full_name", "—")
        row["P"] = row.get("P", "—")
        row["C/1B"] = row.get("C/1B", "—")
        row["2B"] = row.get("2B", "—")
        row["3B"] = row.get("3B", "—")
        row["SS"] = row.get("SS", "—")
        row["OF"] = row.get("OF1", "—")
        row["OF "] = row.get("OF2", "—")
        row["OF  "] = row.get("OF3", "—")
        row["UTIL"] = row.get("UTIL", "—")
        rows.append(row)
    columns = ["Lineup #", "Salary", "Proj FD", "P", "C/1B", "2B", "3B", "SS", "OF", "OF ", "OF  ", "UTIL"]
    return pd.DataFrame(rows)[columns].sort_values("Lineup #").reset_index(drop=True)


def render_today_tab(files: dict[str, Path | None]) -> None:
    st.subheader("Today's Slate")
    st.caption(f"Data date: {extract_data_date(files)}")

    status_cols = st.columns(3)
    with status_cols[0]:
        render_status_card("BPP Data", "✅ Ready" if files["batter"] else "❌ Missing")
    with status_cols[1]:
        render_status_card(
            "Vegas Lines",
            "✅ Ready" if files["vegas"] else "❌ Missing",
            "" if files["vegas"] else "Run fetch_live_data.py",
        )
    with status_cols[2]:
        render_status_card(
            "Batting Orders",
            "✅ Ready" if files["batting"] else "⚠️ Missing",
            "" if files["batting"] else "Lineups post 2hrs before gametime",
        )

    # Refresh live data button
    st.markdown("---")
    refresh_col, info_col = st.columns([1, 3])
    with refresh_col:
        if st.button("🔄 Refresh Live Data", use_container_width=True,
                     help="Fetches fresh BPP simulations, Vegas lines, and batting orders"):
            with st.spinner("Fetching live data..."):
                fetch_script = REPO_ROOT / "scripts" / "fetch_live_data.py"
                result = subprocess.run(
                    [sys.executable, str(fetch_script)],
                    capture_output=True, text=True, cwd=str(REPO_ROOT)
                )
                if result.returncode == 0:
                    st.cache_data.clear()
                    st.success("Live data refreshed!")
                    st.rerun()
                else:
                    st.error("Refresh failed — check your BPP_SESSION and ODDS_API_KEY in .env")
                    with st.expander("Error details"):
                        st.code(result.stdout + result.stderr, language="text")
    with info_col:
        if files.get("batter"):
            date_str = extract_data_date(files)
            st.caption(f"ℹ️ Data loaded for **{date_str}**. Click Refresh to pull today's latest simulations.")
        else:
            st.caption("ℹ️ No live data found. Click **Refresh Live Data** to fetch today's slate.")
    st.markdown("---")

    batter_df = load_csv(str(files["batter"])) if files["batter"] else pd.DataFrame()
    pitcher_df = load_csv(str(files["pitcher"])) if files["pitcher"] else pd.DataFrame()
    projection_df = load_csv(str(files["projection"])) if files["projection"] else pd.DataFrame()
    vegas_df = load_csv(str(files["vegas"])) if files["vegas"] else pd.DataFrame()

    st.markdown("### Top Stacks")
    render_stack_cards(build_stack_targets(vegas_df, batter_df))

    st.markdown("### Best Pitchers")
    pitcher_board = build_pitcher_board(pitcher_df)
    if pitcher_board.empty:
        st.info("Pitcher projections will appear once the BPP pitcher file is available.")
    else:
        st.dataframe(pitcher_board, use_container_width=True, hide_index=True)

    st.markdown("### Leverage Bats")
    leverage_board = build_leverage_board(projection_df)
    if leverage_board.empty:
        st.info("Leverage bats need the BPP DFS projections file.")
    else:
        st.dataframe(style_leverage_table(leverage_board), use_container_width=True, hide_index=True)

    st.markdown("### Chalk Alert")
    chalk_board = build_chalk_board(projection_df)
    if chalk_board.empty:
        st.info("Chalk alert needs the BPP DFS projections file.")
    else:
        st.warning("These players will be heavily owned — consider fading at least one.")
        st.dataframe(chalk_board, use_container_width=True, hide_index=True)

    with st.expander(f"Full Player Pool ({len(projection_df) if not projection_df.empty else 0} players)"):
        query = st.text_input("Search by player or team", key="player_pool_search")
        if projection_df.empty:
            st.info("Player pool will populate when the BPP DFS projections file is present.")
        else:
            st.dataframe(filter_player_pool(projection_df, query), use_container_width=True, hide_index=True)


def render_run_tab(files: dict[str, Path | None]) -> None:
    st.subheader("Run Optimizer")

    upload = st.file_uploader("Upload your FanDuel salary CSV", type=["csv"])
    if upload is not None:
        try:
            saved_path, player_count = save_uploaded_fanduel_file(upload)
            st.success(f"Uploaded FanDuel salary CSV with {player_count} players to `{saved_path.relative_to(REPO_ROOT)}`.")
        except Exception as exc:
            st.error(f"Could not save the FanDuel salary CSV: {exc}")

    uploaded_path = Path(st.session_state["uploaded_fd_path"]) if st.session_state.get("uploaded_fd_path") else None
    fd_ready = bool(uploaded_path and uploaded_path.exists())
    bpp_ready = all(files[key] is not None for key in ("batter", "pitcher", "projection"))
    vegas_ready = files["vegas"] is not None
    batting_ready = files["batting"] is not None

    checklist_cols = st.columns(4)
    checklist_cols[0].markdown(checklist_badge(fd_ready, "FD Salary CSV"))
    checklist_cols[1].markdown(checklist_badge(bpp_ready, "BPP Data"))
    checklist_cols[2].markdown(checklist_badge(vegas_ready, "Vegas Lines"))
    checklist_cols[3].markdown(checklist_badge(batting_ready, "Batting Orders", warning=not batting_ready))

    st.markdown("**Lineup Settings**")
    left_col, right_col = st.columns(2)
    with left_col:
        pool_size = st.number_input(
            "Candidate Pool Size",
            min_value=50, max_value=2000, value=500, step=50,
            help="Generate this many lineups first, then simulation picks the best ones to submit."
        )
        submit_count = st.number_input(
            "Lineups to Submit",
            min_value=1, max_value=350, value=20, step=10,
            help="How many lineups you actually want to enter. Must be less than Pool Size."
        )
        stack_style = st.selectbox(
            "Stack Style",
            ["4-3 (recommended)", "4-4 (two big stacks)", "5-3 (power stack)", "3-3-2 (spread)"],
            index=0,
        )
    with right_col:
        bring_back = st.toggle("Bring-Back", value=True, help="Require 1+ hitter from opposing team in stacks")
        max_own = st.number_input(
            "Max Lineup Ownership (optional)",
            min_value=0.0, max_value=5.0, value=0.0, step=0.1,
            help="Cap total projected ownership per lineup. 0 = no cap.",
        )
        st.markdown("")
        st.markdown("")
        st.info(f"**How it works:** Generates {pool_size:,} candidate lineups → Monte Carlo simulation → selects best {submit_count} for your portfolio", icon="🎯")

    blocked_reasons = []
    if not fd_ready:
        blocked_reasons.append("Upload a FanDuel salary CSV first.")
    if not bpp_ready:
        blocked_reasons.append("BPP batter, pitcher, and projection files are required.")

    button_cols = st.columns([1, 2, 1])
    with button_cols[1]:
        run_clicked = st.button("🚀 Run Optimizer", use_container_width=True, disabled=bool(blocked_reasons))
    if blocked_reasons:
        st.caption(" ".join(blocked_reasons))

    if st.session_state.get("last_run_message"):
        st.info(st.session_state["last_run_message"])

    if run_clicked:
        run_pipeline(
            pool_size=int(pool_size),
            submit_count=int(submit_count),
            stack_style=stack_style,
            bring_back=bring_back,
            max_own=float(max_own),
            files=files,
        )


def render_review_tab() -> None:
    st.subheader("Review Lineups")
    lineups_path, upload_path = load_review_files()

    if not lineups_path or not lineups_path.exists():
        st.info("No lineup set found yet. Run the optimizer to populate this tab.")
        return

    lineups_df = load_csv(str(lineups_path))
    if lineups_df.empty:
        st.info("The latest lineup file is empty.")
        return

    summary = build_review_summary(lineups_df)
    summary_cols = st.columns(5)
    with summary_cols[0]:
        render_summary_card("🎯 N Lineups", summary["n_lineups"], f"Avg Salary {summary['avg_salary']}")
    with summary_cols[1]:
        render_summary_card("📈 Avg Proj", summary["avg_proj"])
    with summary_cols[2]:
        render_summary_card("👥 Avg Ownership", summary["avg_own"])
    with summary_cols[3]:
        render_summary_card("🔥 Top Stack", summary["top_stack"])
    with summary_cols[4]:
        render_summary_card("⚾ Main Pitcher", summary["main_pitcher"])

    # Show simulation metrics if available
    sim_results_files = sorted(OUTPUT_DIR.glob("*_sim_results.csv"), reverse=True)
    if sim_results_files:
        sim_path = sim_results_files[0]
        # prefer the one matching our tag
        tag_pref = st.session_state.get("last_run_tag")
        if tag_pref:
            tagged = OUTPUT_DIR / f"{tag_pref}_sim_results.csv"
            if tagged.exists():
                sim_path = tagged
        try:
            sim_df = load_csv(str(sim_path))
            if not sim_df.empty and "top_1pct_rate" in sim_df.columns:
                st.markdown("### Simulation Results")
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Avg Win Rate", f"{sim_df['win_rate'].mean():.2%}")
                sc2.metric("Avg Top-1% Rate", f"{sim_df['top_1pct_rate'].mean():.2%}")
                sc3.metric("Avg Cash Rate", f"{sim_df['cash_rate'].mean():.2%}")
                sc4.metric("Avg Expected ROI", f"{sim_df['expected_roi'].mean():.2f}x")

                with st.expander("📊 Per-lineup simulation scores (top 20)"):
                    show_cols = [c for c in ["lineup_id","mean_score","top_1pct_rate","win_rate","cash_rate","expected_roi","total_ownership","field_duplication_rate"] if c in sim_df.columns]
                    display = sim_df.sort_values("top_1pct_rate", ascending=False).head(20)[show_cols].copy()
                    for col in ["top_1pct_rate","win_rate","cash_rate"]:
                        if col in display.columns:
                            display[col] = display[col].map(lambda x: f"{x:.2%}")
                    if "expected_roi" in display.columns:
                        display["expected_roi"] = display["expected_roi"].map(lambda x: f"{x:.2f}x")
                    st.dataframe(display, use_container_width=True, hide_index=True)
        except Exception:
            pass

    st.markdown("### Player Exposure Board")
    exposure_df = build_exposure_df(lineups_df)
    if exposure_df.empty:
        st.info("No exposures available in the current lineup file.")
    else:
        st.altair_chart(build_exposure_chart(exposure_df), use_container_width=True)

    st.markdown("### Stack Breakdown")
    stack_breakdown = build_stack_breakdown(lineups_df)
    if stack_breakdown.empty:
        st.info("No hitter stacks found in the current lineup file.")
    else:
        st.dataframe(stack_breakdown, use_container_width=True, hide_index=True)

    st.markdown("### Lineup Table")
    lineup_table = build_lineup_table(lineups_df)
    st.dataframe(lineup_table, use_container_width=True, hide_index=True)

    st.markdown("### Downloads")
    download_cols = st.columns(2)
    with download_cols[0]:
        st.markdown('<div class="big-download">', unsafe_allow_html=True)
        if upload_path and upload_path.exists():
            with upload_path.open("rb") as handle:
                st.download_button(
                    "⬇️ Download FanDuel Upload CSV",
                    data=handle.read(),
                    file_name=upload_path.name,
                    mime="text/csv",
                    use_container_width=True,
                )
        else:
            st.button("⬇️ Download FanDuel Upload CSV", disabled=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with download_cols[1]:
        st.markdown('<div class="big-download">', unsafe_allow_html=True)
        with lineups_path.open("rb") as handle:
            st.download_button(
                "⬇️ Download Full Lineup Breakdown",
                data=handle.read(),
                file_name=lineups_path.name,
                mime="text/csv",
                use_container_width=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


def _load_env() -> None:
    """Load API keys from .env in repo root into environment."""
    import os
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def _data_is_fresh() -> bool:
    """Return True if live data files exist and are from today."""
    from datetime import date
    today = date.today().strftime("%Y-%m-%d")
    batter_file = LIVE_DIR / f"bpp_batters_{today}.csv"
    vegas_file = LIVE_DIR / f"vegas_lines_{today}.csv"
    return batter_file.exists() and vegas_file.exists()


def _auto_fetch_if_stale() -> None:
    """Silently fetch live data in the background if today's data is missing."""
    if _data_is_fresh():
        return
    if st.session_state.get("auto_fetch_done"):
        return
    fetch_script = REPO_ROOT / "scripts" / "fetch_live_data.py"
    if not fetch_script.exists():
        return
    with st.spinner("⏳ Fetching today's live data (BPP + Vegas + lineups)..."):
        result = subprocess.run(
            [sys.executable, str(fetch_script)],
            capture_output=True, text=True, cwd=str(REPO_ROOT)
        )
        st.session_state["auto_fetch_done"] = True
        if result.returncode == 0:
            st.cache_data.clear()
        else:
            # Don't crash — just note it failed silently
            st.session_state["auto_fetch_error"] = result.stdout[-500:] + result.stderr[-200:]


def main() -> None:
    inject_css()
    _load_env()
    init_session_state()
    _auto_fetch_if_stale()
    files = get_live_files()

    # Apply any pending tab switch (must happen before radio widget)
    if st.session_state.get("pending_tab"):
        st.session_state["active_tab"] = st.session_state.pop("pending_tab")

    st.title("MLB DFS Daily Workflow")

    # Show auto-fetch error if it occurred
    if st.session_state.get("auto_fetch_error"):
        with st.expander("⚠️ Live data auto-fetch failed — click to see why"):
            st.code(st.session_state["auto_fetch_error"], language="text")
            st.markdown("**Fix:** Update `BPP_SESSION` in your `.env` file — your BallparkPal session expired.")
            if st.button("Clear error"):
                del st.session_state["auto_fetch_error"]
                del st.session_state["auto_fetch_done"]
                st.rerun()
    selected_tab = st.radio(
        "Workflow",
        TAB_OPTIONS,
        horizontal=True,
        label_visibility="collapsed",
        key="active_tab",
    )

    if selected_tab == "Today's Slate":
        render_today_tab(files)
    elif selected_tab == "Run Optimizer":
        render_run_tab(files)
    else:
        render_review_tab()


if __name__ == "__main__":
    main()
