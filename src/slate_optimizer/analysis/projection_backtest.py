"""Backtest projection variants against real box-score outcomes.

Rebuilds each saved slate from the BallparkPal / Vegas / batting-order files in
``data/live``, runs :func:`compute_baseline_projections` under different
:class:`ProjectionConfig` settings, and scores every variant against actual
FanDuel points (``slate_optimizer.analysis.outcomes``).

The question it answers: does each multiplier layered on top of BallparkPal's
simulated points make the projection better or worse than leaving it alone?
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from slate_optimizer.analysis.outcomes import load_actuals_range
from slate_optimizer.ingestion.vegas import VegasLoader, implied_team_totals
from slate_optimizer.projection.baseline import (
    ProjectionConfig,
    compute_baseline_projections,
    load_projection_config,
)

TEAM_ALIASES = {
    "TBR": "TB",
    "CHW": "CWS",
    "AZ": "ARI",
    "WAS": "WSH",
    "SFG": "SF",
    "SDP": "SD",
    "KCR": "KC",
    "OAK": "ATH",
}

_DATE_RE = re.compile(r"bpp_batters_(\d{4}-\d{2}-\d{2})\.csv$")
_NAME_RE = re.compile(r"[^a-z]")


def normalize_team(value: object) -> str:
    code = str(value or "").strip().upper()
    return TEAM_ALIASES.get(code, code)


def normalize_name(value: object) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    for suffix in (" jr.", " sr.", " jr", " sr", " iii", " ii"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return _NAME_RE.sub("", text)


def game_start_minutes(game_time: object) -> float:
    """Minutes after midnight ET for a BallparkPal ``GameTime`` like ``7:05``.

    BallparkPal omits AM/PM.  MLB first pitches fall between 11:35 AM and
    about 10:15 PM ET, so hours 1-10 are PM and 11/12 are late morning/noon.
    """
    text = str(game_time or "").strip()
    match = re.match(r"^(\d{1,2}):(\d{2})", text)
    if not match:
        return float("nan")
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour < 11:
        hour += 12
    return float(hour * 60 + minute)


def live_dates(live_dir: Path | str) -> List[str]:
    found = []
    for path in sorted(Path(live_dir).glob("bpp_batters_*.csv")):
        match = _DATE_RE.search(path.name)
        if match:
            found.append(match.group(1))
    return found


def _snapshot_minutes(path: Path, date: str) -> float:
    """When the file was written, as minutes after midnight on the slate date.

    Files saved on a later calendar day return +inf (definitely post-game);
    the caller treats NaN as unknown.
    """
    try:
        stamp = datetime.fromtimestamp(os.path.getmtime(path))
    except OSError:
        return float("nan")
    if stamp.strftime("%Y-%m-%d") != date:
        return float("inf") if stamp.strftime("%Y-%m-%d") > date else float("nan")
    return float(stamp.hour * 60 + stamp.minute)


def build_slate_frame(
    date: str,
    live_dir: Path | str,
    snapshot_minutes: Optional[float] = None,
) -> pd.DataFrame:
    """One row per projected player, in the column contract baseline.py expects.

    Column semantics deliberately mirror the production slate merge - including
    its quirk that ``bpp_runs`` is the batter's own projected runs on batter
    rows and the team total on pitcher rows - so that the ``legacy`` variants
    reproduce what actually ran in 2026.
    """
    live = Path(live_dir)
    batters_path = live / f"bpp_batters_{date}.csv"
    pitchers_path = live / f"bpp_pitchers_{date}.csv"
    if not batters_path.exists() or not pitchers_path.exists():
        return pd.DataFrame()
    batters = pd.read_csv(batters_path)
    pitchers = pd.read_csv(pitchers_path)
    games_path = live / f"bpp_games_{date}.csv"
    teams_path = live / f"bpp_teams_{date}.csv"
    games = pd.read_csv(games_path) if games_path.exists() else pd.DataFrame()
    teams = pd.read_csv(teams_path) if teams_path.exists() else pd.DataFrame()

    if snapshot_minutes is None:
        snapshot_minutes = _snapshot_minutes(batters_path, date)

    def _common(src: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=src.index)
        out["date"] = date
        out["game_pk"] = pd.to_numeric(src["GamePk"], errors="coerce").astype("Int64")
        out["mlb_id"] = pd.to_numeric(src["PlayerId"], errors="coerce").astype("Int64")
        out["full_name"] = src["FullName"].astype(str)
        out["team_code"] = src["Team"].map(normalize_team)
        out["opponent_code"] = src["Opponent"].map(normalize_team)
        out["team"] = out["team_code"]
        out["opponent"] = out["opponent_code"]
        out["bpp_side"] = src["Side"].astype(str).str.upper().str[:1]
        out["bpp_points_fd"] = pd.to_numeric(src["PointsFD"], errors="coerce")
        out["game_start_minutes"] = src["GameTime"].map(game_start_minutes)
        return out

    bat = _common(batters)
    bat["player_type"] = "batter"
    bat["position"] = "UTIL"
    bat["bpp_runs"] = pd.to_numeric(batters["Runs"], errors="coerce")
    bat["bpp_batting_position"] = pd.to_numeric(batters["BattingPosition"], errors="coerce")
    bat["bpp_plate_appearances"] = pd.to_numeric(batters.get("PlateAppearances"), errors="coerce")
    bat["batter_hand"] = batters.get("BatterStand", pd.Series("", index=batters.index)).fillna("").astype(str).str.upper()
    bat["pitcher_hand"] = ""
    for src_col, dst_col in (
        ("HomeRunProbability", "bpp_home_run_probability"),
        ("HitProbability", "bpp_hit_probability"),
        ("StolenBaseProbability", "bpp_stolen_base_probability"),
    ):
        bat[dst_col] = pd.to_numeric(batters.get(src_col), errors="coerce")

    pit = _common(pitchers)
    pit["player_type"] = "pitcher"
    pit["position"] = "P"
    pit["batter_hand"] = ""
    pit["pitcher_hand"] = pitchers.get("PitcherHand", pd.Series("", index=pitchers.index)).fillna("").astype(str).str.upper()
    for src_col, dst_col in (
        ("WinPct", "bpp_win_pct"),
        ("QualityStart", "bpp_quality_start"),
        ("RunsAllowed", "bpp_runs_allowed"),
        ("Innings", "bpp_innings"),
        ("Strikeouts", "bpp_strikeouts"),
    ):
        pit[dst_col] = pd.to_numeric(pitchers.get(src_col), errors="coerce")

    frame = pd.concat([bat, pit], ignore_index=True, sort=False)

    if not games.empty:
        meta = pd.DataFrame(
            {
                "game_pk": pd.to_numeric(games["GamePk"], errors="coerce").astype("Int64"),
                "bpp_runs_away": pd.to_numeric(games["RunsAway"], errors="coerce"),
                "bpp_runs_home": pd.to_numeric(games["RunsHome"], errors="coerce"),
            }
        ).drop_duplicates("game_pk")
        frame = frame.merge(meta, on="game_pk", how="left")
    else:
        frame["bpp_runs_away"] = np.nan
        frame["bpp_runs_home"] = np.nan
    frame["bpp_team_runs"] = frame["bpp_runs_home"].where(frame["bpp_side"] == "H", frame["bpp_runs_away"])
    frame["bpp_opponent_runs"] = frame["bpp_runs_away"].where(frame["bpp_side"] == "H", frame["bpp_runs_home"])

    if not teams.empty:
        team_meta = pd.DataFrame(
            {
                "game_pk": pd.to_numeric(teams["GamePk"], errors="coerce").astype("Int64"),
                "team_code": teams["Team"].map(normalize_team),
                "bpp_win_percent": pd.to_numeric(teams["WinPercent"], errors="coerce"),
            }
        ).drop_duplicates(["game_pk", "team_code"])
        frame = frame.merge(team_meta, on=["game_pk", "team_code"], how="left")
    else:
        frame["bpp_win_percent"] = np.nan
    pitcher_rows = frame["player_type"] == "pitcher"
    frame.loc[pitcher_rows, "bpp_runs"] = frame.loc[pitcher_rows, "bpp_team_runs"]

    # Confirmed batting orders (MLB Stats API snapshot written by fetch_live_data).
    frame["batting_order_position"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    orders_path = live / f"batting_orders_{date}.csv"
    if orders_path.exists():
        orders = pd.read_csv(orders_path)
        if {"team", "order_position", "player_name"}.issubset(orders.columns) and not orders.empty:
            orders = orders.assign(
                team_code=orders["team"].map(normalize_team),
                _name=orders["player_name"].map(normalize_name),
                _order=pd.to_numeric(orders["order_position"], errors="coerce"),
            ).drop_duplicates(["team_code", "_name"])
            keyed = frame.assign(_name=frame["full_name"].map(normalize_name)).merge(
                orders[["team_code", "_name", "_order"]], on=["team_code", "_name"], how="left"
            )
            hitters = (frame["player_type"] == "batter").to_numpy()
            values = keyed["_order"].where(hitters)
            frame["batting_order_position"] = values.astype("Int64")
    frame["is_confirmed_lineup"] = frame["batting_order_position"].notna()

    # Vegas lines.  Team totals use the corrected (Pythagorean) conversion;
    # the legacy total-times-win-probability number is kept for comparison.
    vegas_cols = [
        "vegas_game_total",
        "vegas_team_total",
        "vegas_opponent_total",
        "vegas_moneyline",
        "vegas_implied_win_prob",
        "vegas_team_total_legacy",
    ]
    vegas_path = live / f"vegas_lines_{date}.csv"
    merged_vegas = False
    if vegas_path.exists():
        try:
            lines = VegasLoader(vegas_path).load()
            totals = lines.team_totals.copy()
        except Exception:
            totals = pd.DataFrame()
        if not totals.empty:
            totals["team_code"] = totals["team_code"].map(normalize_team)
            totals["opponent_code"] = totals["opponent_code"].map(normalize_team)
            totals = totals.drop_duplicates(["team_code", "opponent_code"])
            game_total = pd.to_numeric(totals["vegas_game_total"], errors="coerce")
            win_prob = pd.to_numeric(totals["vegas_implied_win_prob"], errors="coerce")
            totals["vegas_team_total_legacy"] = game_total * win_prob
            team_total, opp_total = implied_team_totals(game_total, win_prob)
            totals["vegas_team_total"] = team_total
            totals["vegas_opponent_total"] = opp_total
            # A pre-game MLB total below 5.5 or above 15 is an in-play or broken line.
            sane = game_total.between(5.5, 15.0)
            totals = totals[sane]
            frame = frame.merge(totals[["team_code", "opponent_code"] + vegas_cols], on=["team_code", "opponent_code"], how="left")
            merged_vegas = True
    if not merged_vegas:
        for column in vegas_cols:
            frame[column] = np.nan

    # A line fetched after first pitch is an in-play number that already knows
    # part of the result - using it would flatter every Vegas variant.
    frame["snapshot_minutes"] = snapshot_minutes
    if snapshot_minutes is None or not np.isfinite(snapshot_minutes):
        frame["pregame_snapshot"] = False if snapshot_minutes == float("inf") else np.nan
    else:
        frame["pregame_snapshot"] = frame["game_start_minutes"] > float(snapshot_minutes)

    frame["fd_player_id"] = (
        frame["mlb_id"].astype(str) + "-" + frame["game_pk"].astype(str) + "-" + frame["player_type"].str[0]
    )
    frame["salary"] = 0
    return frame


def join_actuals(frame: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Attach real outcomes on exact (game, player, role) ids."""
    if frame.empty or actuals.empty:
        return pd.DataFrame()
    act = actuals.copy()
    act["game_pk"] = pd.to_numeric(act["game_pk"], errors="coerce").astype("Int64")
    act["mlb_id"] = pd.to_numeric(act["mlb_id"], errors="coerce").astype("Int64")
    act = act.rename(
        columns={
            "role": "player_type",
            "fd_points": "actual_fd_points",
            "started": "actual_started",
            "batting_order": "actual_batting_order",
            "plate_appearances": "actual_pa",
            "team_runs": "actual_team_runs",
        }
    )
    keep = [
        "game_pk",
        "mlb_id",
        "player_type",
        "actual_fd_points",
        "actual_started",
        "actual_batting_order",
        "actual_pa",
        "actual_team_runs",
    ]
    act = act[keep].drop_duplicates(["game_pk", "mlb_id", "player_type"])
    joined = frame.merge(act, on=["game_pk", "mlb_id", "player_type"], how="left")
    joined["played"] = joined["actual_fd_points"].notna()
    joined["actual_started"] = joined["actual_started"].fillna(False).astype(bool)
    return joined


@dataclass
class Variant:
    name: str
    config: ProjectionConfig
    description: str = ""
    drop_vegas: bool = False
    drop_context: bool = False  # blank confirmed orders + handedness (the cron's inputs)
    legacy_vegas_totals: bool = False


def default_variants() -> List[Variant]:
    raw = ProjectionConfig.raw_bpp

    def only(**overrides) -> ProjectionConfig:
        payload = raw().to_dict()
        payload.update(overrides)
        return ProjectionConfig.from_dict(payload)

    variants = [
        Variant("raw_bpp", raw(), "BallparkPal PointsFD untouched"),
        Variant(
            "legacy_dashboard",
            ProjectionConfig(),
            "2026 dashboard path: every multiplier, legacy Vegas totals (total x win prob)",
            legacy_vegas_totals=True,
        ),
        Variant(
            "legacy_cron",
            ProjectionConfig(),
            "2026 cron path: no Vegas file, no confirmed orders or handedness passed",
            drop_vegas=True,
            drop_context=True,
        ),
        Variant("only_batting_order", only(apply_batting_order=True), "raw + batting-order multiplier"),
        Variant("only_platoon", only(apply_platoon=True), "raw + platoon multiplier"),
        Variant("only_park_weather", only(apply_park_weather=True), "raw + park/weather additive"),
        Variant("only_pitcher_adj", only(apply_pitcher_adjustments=True), "raw + pitcher win multiplier and adjustment"),
        Variant(
            "only_vegas_legacy",
            only(team_total_mode="legacy"),
            "raw + legacy Vegas level multiplier (legacy totals)",
            legacy_vegas_totals=True,
        ),
        Variant(
            "only_vegas_level_fixed",
            only(team_total_mode="legacy"),
            "raw + Vegas level multiplier with corrected implied totals",
        ),
    ]
    for alpha in (0.25, 0.5, 0.75, 1.0):
        variants.append(
            Variant(
                f"market_ratio_{alpha:.2f}",
                only(team_total_mode="market_ratio", market_blend_alpha=alpha, market_ratio_clip=0.20),
                f"raw x (Vegas team total / BPP team runs)^{alpha:g}",
            )
        )
    variants.append(
        Variant(
            "calibrated_config",
            load_projection_config(),
            "whatever config/projection_model.json currently says (what production runs)",
        )
    )
    return variants


def project_variant(frame: pd.DataFrame, variant: Variant) -> pd.Series:
    """Projection (``proj_fd_mean``) for every row of ``frame`` under ``variant``."""
    work = frame.copy()
    if variant.legacy_vegas_totals and "vegas_team_total_legacy" in work.columns:
        work["vegas_team_total"] = work["vegas_team_total_legacy"]
    if variant.drop_vegas:
        work = work.drop(columns=[col for col in work.columns if col.startswith("vegas_")])
    if variant.drop_context:
        work["batting_order_position"] = pd.Series(pd.NA, index=work.index, dtype="Int64")
        work["is_confirmed_lineup"] = False
        work["batter_hand"] = ""
        work["pitcher_hand"] = ""
    projections = compute_baseline_projections(work, config=variant.config)
    return pd.Series(projections["proj_fd_mean"].to_numpy(dtype=float), index=frame.index)


def score_projection(projected: pd.Series, actual: pd.Series, dates: Optional[pd.Series] = None) -> Dict[str, float]:
    """Accuracy metrics for one projection column."""
    proj = pd.to_numeric(projected, errors="coerce")
    act = pd.to_numeric(actual, errors="coerce")
    mask = proj.notna() & act.notna()
    proj, act = proj[mask], act[mask]
    n = int(mask.sum())
    if n < 3:
        return {"n": n}
    err = proj - act
    proj_var = float(proj.var(ddof=0))
    slope = float(np.cov(proj, act, ddof=0)[0, 1] / proj_var) if proj_var > 1e-12 else float("nan")
    out = {
        "n": n,
        "mean_proj": float(proj.mean()),
        "mean_actual": float(act.mean()),
        "bias": float(err.mean()),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "pearson": float(proj.corr(act)) if proj_var > 1e-12 else float("nan"),
        "spearman": float(proj.rank().corr(act.rank())) if proj_var > 1e-12 else float("nan"),
        "slope": slope,
    }
    if dates is not None:
        groups = pd.DataFrame({"proj": proj, "act": act, "date": dates[mask]})
        tops = []
        for _, group in groups.groupby("date"):
            if len(group) < 10:
                continue
            cutoff = group["proj"].quantile(0.90)
            tops.append(float(group.loc[group["proj"] >= cutoff, "act"].mean()))
        out["top_decile_actual"] = float(np.mean(tops)) if tops else float("nan")
    return out


def _date_stats(proj: pd.Series, act: pd.Series, dates: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"x": proj, "y": act, "date": dates}).dropna()
    frame["xx"] = frame["x"] ** 2
    frame["yy"] = frame["y"] ** 2
    frame["xy"] = frame["x"] * frame["y"]
    frame["se"] = (frame["x"] - frame["y"]) ** 2
    frame["n"] = 1.0
    return frame.groupby("date")[["n", "x", "y", "xx", "yy", "xy", "se"]].sum()


def _pooled(stats: np.ndarray) -> tuple[float, float]:
    n, sx, sy, sxx, syy, sxy, sse = stats
    if n < 3:
        return float("nan"), float("nan")
    rmse = float(np.sqrt(sse / n))
    var_x = sxx / n - (sx / n) ** 2
    var_y = syy / n - (sy / n) ** 2
    cov = sxy / n - (sx / n) * (sy / n)
    corr = float(cov / np.sqrt(var_x * var_y)) if var_x > 1e-12 and var_y > 1e-12 else float("nan")
    return rmse, corr


def paired_bootstrap(
    proj: pd.Series,
    reference: pd.Series,
    actual: pd.Series,
    dates: pd.Series,
    reps: int = 2000,
    seed: int = 7,
) -> Dict[str, float]:
    """Slate-clustered bootstrap of (variant - reference) for RMSE and correlation.

    Slates are resampled whole because players inside a slate share weather,
    umpires and run environment - treating rows as independent would overstate
    confidence by a wide margin.
    """
    mask = proj.notna() & reference.notna() & actual.notna()
    a = _date_stats(proj[mask], actual[mask], dates[mask])
    b = _date_stats(reference[mask], actual[mask], dates[mask])
    b = b.reindex(a.index)
    if len(a) < 4:
        return {}
    rng = np.random.default_rng(seed)
    a_values, b_values = a.to_numpy(), b.to_numpy()
    d_rmse, d_corr = [], []
    for _ in range(reps):
        pick = rng.integers(0, len(a_values), size=len(a_values))
        rmse_a, corr_a = _pooled(a_values[pick].sum(axis=0))
        rmse_b, corr_b = _pooled(b_values[pick].sum(axis=0))
        d_rmse.append(rmse_a - rmse_b)
        d_corr.append(corr_a - corr_b)
    d_rmse_arr, d_corr_arr = np.array(d_rmse), np.array(d_corr)
    return {
        "d_rmse_lo": float(np.nanpercentile(d_rmse_arr, 2.5)),
        "d_rmse_hi": float(np.nanpercentile(d_rmse_arr, 97.5)),
        "d_corr_lo": float(np.nanpercentile(d_corr_arr, 2.5)),
        "d_corr_hi": float(np.nanpercentile(d_corr_arr, 97.5)),
        "p_better_rmse": float(np.mean(d_rmse_arr < 0)),
        "p_better_corr": float(np.mean(d_corr_arr > 0)),
    }


@dataclass
class BacktestResult:
    player_games: pd.DataFrame
    summary: pd.DataFrame
    team_totals: pd.DataFrame
    dates: List[str] = field(default_factory=list)


def build_player_games(
    dates: Sequence[str],
    live_dir: Path | str,
    actuals_dir: Path | str | None = None,
    variants: Optional[Sequence[Variant]] = None,
) -> pd.DataFrame:
    """Projection inputs, every variant's projection and the real outcome."""
    variants = list(variants) if variants is not None else default_variants()
    actuals = load_actuals_range(dates, actuals_dir)
    frames = []
    for date in dates:
        frame = build_slate_frame(date, live_dir)
        if frame.empty:
            continue
        day_actuals = actuals[actuals["date"].astype(str) == date] if not actuals.empty else actuals
        joined = join_actuals(frame, day_actuals)
        if joined.empty:
            continue
        for variant in variants:
            joined[f"proj_{variant.name}"] = project_variant(frame, variant).to_numpy()
        frames.append(joined)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def summarize(
    player_games: pd.DataFrame,
    variants: Optional[Sequence[Variant]] = None,
    reference: str = "raw_bpp",
    bootstrap_reps: int = 2000,
) -> pd.DataFrame:
    """Per-variant, per-role accuracy table with clustered-bootstrap deltas."""
    variants = list(variants) if variants is not None else default_variants()
    if player_games.empty:
        return pd.DataFrame()
    rows = []
    started = player_games[player_games["actual_started"] & player_games["bpp_points_fd"].notna()]
    for role in ("batter", "pitcher"):
        role_df = started[started["player_type"] == role]
        for variant in variants:
            column = f"proj_{variant.name}"
            if column not in role_df.columns:
                continue
            uses_vegas = (
                variant.config.team_total_mode != "off" or variant.config.apply_pitcher_adjustments
            ) and not variant.drop_vegas
            subsets = [("all", role_df)]
            pregame = role_df[role_df["pregame_snapshot"] == True]  # noqa: E712
            pregame = pregame[pregame["vegas_team_total"].notna()]
            subsets.append(("pregame_vegas", pregame))
            for subset_name, subset in subsets:
                if subset.empty:
                    continue
                metrics = score_projection(subset[column], subset["actual_fd_points"], subset["date"])
                if metrics.get("n", 0) < 3:
                    continue
                row = {
                    "role": role,
                    "subset": subset_name,
                    "variant": variant.name,
                    "description": variant.description,
                    "uses_vegas": bool(uses_vegas),
                    **metrics,
                }
                if variant.name != reference and f"proj_{reference}" in subset.columns:
                    row.update(
                        paired_bootstrap(
                            subset[column],
                            subset[f"proj_{reference}"],
                            subset["actual_fd_points"],
                            subset["date"],
                            reps=bootstrap_reps,
                        )
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def team_total_accuracy(player_games: pd.DataFrame) -> pd.DataFrame:
    """How well each source predicts actual team runs (pre-game lines only)."""
    if player_games.empty:
        return pd.DataFrame()
    teams = (
        player_games[player_games["player_type"] == "batter"]
        .dropna(subset=["actual_team_runs"])
        .drop_duplicates(["date", "game_pk", "team_code"])
    )
    teams = teams[(teams["pregame_snapshot"] == True) & teams["vegas_team_total"].notna()]  # noqa: E712
    if teams.empty:
        return pd.DataFrame()
    rows = []
    blend = 0.5 * teams["bpp_team_runs"] + 0.5 * teams["vegas_team_total"]
    sources = {
        "bpp_team_runs": teams["bpp_team_runs"],
        "vegas_pythagorean": teams["vegas_team_total"],
        "vegas_legacy_total_x_winprob": teams["vegas_team_total_legacy"],
        "blend_50_50": blend,
    }
    for name, values in sources.items():
        metrics = score_projection(values, teams["actual_team_runs"])
        metrics.pop("top_decile_actual", None)
        rows.append({"source": name, **metrics, "spread_sd": float(pd.to_numeric(values, errors="coerce").std())})
    return pd.DataFrame(rows)


def run_backtest(
    live_dir: Path | str,
    actuals_dir: Path | str | None = None,
    dates: Optional[Iterable[str]] = None,
    variants: Optional[Sequence[Variant]] = None,
    bootstrap_reps: int = 2000,
) -> BacktestResult:
    date_list = sorted(dates) if dates is not None else live_dates(live_dir)
    variants = list(variants) if variants is not None else default_variants()
    player_games = build_player_games(date_list, live_dir, actuals_dir, variants)
    summary = summarize(player_games, variants, bootstrap_reps=bootstrap_reps)
    totals = team_total_accuracy(player_games)
    used = sorted(player_games["date"].unique().tolist()) if not player_games.empty else []
    return BacktestResult(player_games=player_games, summary=summary, team_totals=totals, dates=used)


__all__ = [
    "BacktestResult",
    "Variant",
    "build_player_games",
    "build_slate_frame",
    "default_variants",
    "game_start_minutes",
    "join_actuals",
    "live_dates",
    "normalize_name",
    "normalize_team",
    "paired_bootstrap",
    "project_variant",
    "run_backtest",
    "score_projection",
    "summarize",
    "team_total_accuracy",
]
