import io
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

@st.cache_data
def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("proj_fd_mean", "proj_fd_ownership", "bpp_runs", "bpp_win_percent"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data
def load_lineups(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def list_files(pattern: str) -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob(pattern), reverse=True)

def section_stacks(df: pd.DataFrame, top: int = 5):
    if "bpp_runs" not in df.columns:
        st.info("No BallparkPal run data available in this dataset.")
        return
    hitters = df[df["player_type"].str.lower() == "batter"].copy()
    hitters = hitters.dropna(subset=["bpp_runs"])
    if hitters.empty:
        st.info("No hitter run data available.")
        return
    summary = hitters.groupby("team_code")["bpp_runs"].mean().sort_values(ascending=False)
    st.table(summary.head(top).rename("Avg Runs"))

def section_pitchers(df: pd.DataFrame, top: int = 5):
    pitchers = df[df["player_type"].str.lower() == "pitcher"].copy()
    if pitchers.empty:
        st.info("No pitcher rows available.")
        return
    cols = [c for c in ["full_name", "team_code", "proj_fd_mean", "bpp_win_percent"] if c in pitchers.columns]
    board = pitchers.sort_values(by="proj_fd_mean", ascending=False)[cols]
    st.table(board.head(top))

def compute_leverage(df: pd.DataFrame) -> pd.DataFrame:
    required = {"proj_fd_mean", "proj_fd_ownership"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    lev = df.copy()
    lev["proj_fd_mean"] = pd.to_numeric(lev["proj_fd_mean"], errors="coerce").fillna(0.0)
    lev["proj_fd_ownership"] = pd.to_numeric(lev["proj_fd_ownership"], errors="coerce").fillna(0.0)
    lev["projection_rank"] = lev["proj_fd_mean"].rank(pct=True)
    lev["ownership_rank"] = lev["proj_fd_ownership"].rank(pct=True)
    lev["leverage_score"] = lev["projection_rank"] - lev["ownership_rank"]
    return lev

def section_leverage(df: pd.DataFrame, top: int = 10):
    lev = compute_leverage(df)
    if lev.empty:
        st.info("Leverage requires proj_fd_mean and proj_fd_ownership columns.")
        return lev
    hitters = lev[lev["player_type"].str.lower() == "batter"].sort_values("leverage_score", ascending=False)
    pitchers = lev[lev["player_type"].str.lower() == "pitcher"].sort_values("leverage_score", ascending=False)
    st.subheader("Hitter leverage")
    st.dataframe(hitters[["full_name", "team_code", "proj_fd_mean", "proj_fd_ownership", "leverage_score"]].head(top))
    st.subheader("Pitcher leverage")
    st.dataframe(pitchers[["full_name", "team_code", "proj_fd_mean", "proj_fd_ownership", "leverage_score"]].head(top))
    return lev

def section_chalk(df: pd.DataFrame, top: int = 5):
    if "proj_fd_ownership" not in df.columns:
        st.info("No ownership column available.")
        return
    chalk = df.sort_values("proj_fd_ownership", ascending=False)
    st.table(chalk[["full_name", "team_code", "proj_fd_ownership", "proj_fd_mean"]].head(top))

def section_exposures(lineup_df: pd.DataFrame):
    if lineup_df.empty:
        st.info("No lineups data available.")
        return
    lineup_count = lineup_df["lineup_id"].nunique()
    exposures = (
        lineup_df.groupby(["full_name", "team"])["lineup_id"].count().reset_index(name="lineups")
    )
    exposures["exposure_pct"] = exposures["lineups"] / lineup_count
    top_exposures = exposures.sort_values("lineups", ascending=False).head(15)
    st.bar_chart(top_exposures.set_index("full_name")["exposure_pct"])
    st.dataframe(exposures.sort_values("lineups", ascending=False))

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    teams = sorted(filtered["team_code"].dropna().unique())
    positions_raw = filtered.get("position", pd.Series([""] * len(filtered)))
    positions = sorted({pos for val in positions_raw.dropna() for pos in str(val).upper().replace('-', '/').split('/') if pos})
    selected_teams = st.sidebar.multiselect("Teams", teams, default=teams)
    if selected_teams:
        filtered = filtered[filtered["team_code"].isin(selected_teams)]
    selected_positions = st.sidebar.multiselect("Positions", positions, default=positions)
    if selected_positions and "position" in filtered.columns:
        def has_position(val: str) -> bool:
            tokens = str(val).upper().replace('-', '/').split('/')
            return any(pos in tokens for pos in selected_positions)
        filtered = filtered[filtered["position"].apply(has_position)]
    proj_series = pd.to_numeric(filtered.get("proj_fd_mean", pd.Series([0])), errors="coerce").fillna(0.0)
    proj_min = float(proj_series.min()) if not proj_series.empty else 0.0
    proj_max = float(proj_series.max()) if proj_series.max() > proj_min else proj_min + 1
    proj_threshold = st.sidebar.slider("Min projection", proj_min, proj_max, proj_min)
    filtered = filtered[filtered["proj_fd_mean"] >= proj_threshold]
    ownership_series = pd.to_numeric(filtered.get("proj_fd_ownership", pd.Series([0])), errors="coerce").fillna(0.0)
    own_max = float(ownership_series.max()) if not ownership_series.empty else 0.0
    ownership_upper = own_max if own_max > 0 else 0.5
    own_range = st.sidebar.slider("Ownership range", 0.0, max(ownership_upper, 0.1), (0.0, max(ownership_upper, 0.1)))
    if "proj_fd_ownership" in filtered.columns:
        filtered = filtered[(filtered["proj_fd_ownership"] >= own_range[0]) & (filtered["proj_fd_ownership"] <= own_range[1])]
    return filtered

def main():
    st.title("MLB Slate Dashboard")
    optimizer_files = list_files("*_optimizer_dataset.csv")
    if not optimizer_files:
        st.warning("No optimizer datasets found in data/processed.")
        st.stop()
    selection = st.selectbox("Optimizer dataset", [f.name for f in optimizer_files])
    dataset_path = optimizer_files[[f.name for f in optimizer_files].index(selection)]
    df = load_dataset(dataset_path)
    st.caption(f"Loaded {dataset_path}")

    filtered_df = apply_filters(df)
    st.sidebar.write(f"Filtered players: {len(filtered_df)}")

    tabs = st.tabs(["Overview", "Pitchers", "Leverage", "Ownership", "Lineup Exposure"])

    with tabs[0]:
        st.subheader("Stack overview")
        section_stacks(filtered_df)
        st.subheader("Filtered players preview")
        st.dataframe(filtered_df.head(50))
        st.download_button(
            label="Download filtered players",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{dataset_path.stem}_filtered.csv",
            mime="text/csv",
        )

    with tabs[1]:
        st.subheader("Top pitchers")
        section_pitchers(filtered_df)

    with tabs[2]:
        lev_df = section_leverage(filtered_df)
        if lev_df is not None and not lev_df.empty:
            csv = lev_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download leverage CSV",
                data=csv,
                file_name=f"{dataset_path.stem}_leverage.csv",
                mime="text/csv",
            )

    with tabs[3]:
        st.subheader("Highest projected ownership")
        section_chalk(filtered_df)

    with tabs[4]:
        lineups_files = list_files("*_lineups.csv")
        if not lineups_files:
            st.info("No lineups found. Run the optimizer first.")
        else:
            lineup_options = [f.name for f in lineups_files]
            lineup_selection = st.selectbox("Lineups CSV", lineup_options)
            lineup_df = load_lineups(lineups_files[lineup_options.index(lineup_selection)])
            section_exposures(lineup_df)

if __name__ == "__main__":
    main()
