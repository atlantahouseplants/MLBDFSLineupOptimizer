"""Measure real score distributions and correlations, and fit the simulator to them.

Input is the joined player-game table written by ``scripts/backtest_projections.py``
(projection inputs, projections and actual FanDuel points for every start).

Three things are measured and then reproduced in the simulator:

1. Hitter marginals - how often a starter scores zero, and how wide the
   non-zero scores are, as a function of the projection.
2. Pitcher marginals - spread of actual minus projected points.
3. Correlations - teammates, opposing hitters, pitcher vs the lineup he
   faces, pitcher vs his own hitters.

The copula parameters are *latent* correlations, not correlations of fantasy
points, so they are solved for by simulation: adjust each parameter until the
simulated fantasy-point correlation equals the measured one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from slate_optimizer.projection.baseline import ProjectionConfig, compute_baseline_projections
from slate_optimizer.simulation.calibration import SimulationCalibration
from slate_optimizer.simulation.correlation import CorrelationConfig, build_correlation_matrix
from slate_optimizer.simulation.distributions import fit_player_distributions
from slate_optimizer.simulation.slate_simulator import simulate_slate

PAIR_CLASSES = ("teammates", "opposing_hitters", "pitcher_vs_opposing", "pitcher_own_hitters")
DEFAULT_PROJECTION_COLUMN = "proj_calibrated_config"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def starters(player_games: pd.DataFrame, proj_col: str = DEFAULT_PROJECTION_COLUMN) -> pd.DataFrame:
    """Rows for players who actually started and had a projection."""
    if proj_col not in player_games.columns:
        proj_col = "proj_raw_bpp"
    frame = player_games[player_games["actual_started"].astype(bool)].copy()
    frame["projection"] = pd.to_numeric(frame[proj_col], errors="coerce")
    frame["actual"] = pd.to_numeric(frame["actual_fd_points"], errors="coerce")
    frame = frame[frame["projection"].notna() & frame["actual"].notna() & (frame["projection"] > 0)]
    frame["resid"] = frame["actual"] - frame["projection"]
    frame["slot"] = pd.to_numeric(frame.get("actual_batting_order"), errors="coerce").fillna(0).astype(int)
    return frame


# ---------------------------------------------------------------------------
# Marginals
# ---------------------------------------------------------------------------

def fit_hitter_model(batters: pd.DataFrame, shift: float = 3.0) -> Dict[str, float]:
    """Zero-rate scale (maximum likelihood) and non-zero spread for hitters."""
    proj = batters["projection"].to_numpy(dtype=float)
    actual = batters["actual"].to_numpy(dtype=float)
    is_zero = actual <= 0.0
    best_scale, best_ll = 6.0, -np.inf
    for scale in np.arange(3.0, 14.01, 0.05):
        p_zero = np.clip(np.exp(-proj / scale), 0.02, 0.90)
        ll = float(np.sum(np.where(is_zero, np.log(p_zero), np.log1p(-p_zero))))
        if ll > best_ll:
            best_scale, best_ll = float(scale), ll
    p_zero = np.clip(np.exp(-proj / best_scale), 0.02, 0.90)
    positive = ~is_zero
    mean_positive = proj[positive] / (1.0 - p_zero[positive])
    # SD of non-zero scores as a fraction of their expected value.
    cv = float(np.sqrt(np.mean(((actual[positive] - mean_positive) / mean_positive) ** 2)))
    return {
        "hitter_zero_scale": round(best_scale, 2),
        "hitter_positive_cv": round(cv, 3),
        "hitter_positive_shift": float(shift),
        "observed_zero_rate": float(is_zero.mean()),
        "observed_sd": float(np.std(actual - proj)),
    }


def fit_pitcher_model(pitchers: pd.DataFrame) -> Dict[str, float]:
    """Spread of pitcher outcomes around the projection, allowing it to grow with the projection."""
    proj = pitchers["projection"].to_numpy(dtype=float)
    resid = pitchers["resid"].to_numpy(dtype=float)
    overall = float(np.sqrt(np.mean(resid**2)))
    slope, intercept = 0.0, overall
    if len(proj) >= 60 and np.std(proj) > 1e-6:
        # E|resid| = sd * sqrt(2/pi) for a normal, so regress |resid| on the projection.
        fitted_slope, fitted_intercept = np.polyfit(proj, np.abs(resid) * np.sqrt(np.pi / 2.0), 1)
        if fitted_slope > 0 and fitted_intercept > 0:
            slope, intercept = float(fitted_slope), float(fitted_intercept)
            # Rescale so the pooled SD is reproduced exactly.
            implied = np.sqrt(np.mean((intercept + slope * proj) ** 2))
            factor = overall / implied if implied > 0 else 1.0
            slope, intercept = slope * factor, intercept * factor
    return {
        "pitcher_sd_intercept": round(intercept, 3),
        "pitcher_sd_slope": round(slope, 4),
        "observed_sd": overall,
        "observed_negative_rate": float((pitchers["actual"] < 0).mean()),
    }


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

def _pairs_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Every within-game pair of starters with its pair class and both residuals."""
    rows: List[tuple] = []
    rng = np.random.default_rng(0)
    for (date, game_pk), game in frame.groupby(["date", "game_pk"]):
        bats = game[game["player_type"] == "batter"]
        arms = game[game["player_type"] == "pitcher"]
        by_team = {team: grp for team, grp in bats.groupby("team_code")}
        for team, grp in by_team.items():
            values = grp[["slot", "resid"]].to_numpy(dtype=float)
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    gap = abs(values[i, 0] - values[j, 0])
                    gap = min(gap, 9 - gap)
                    a, b = (values[i, 1], values[j, 1]) if rng.random() < 0.5 else (values[j, 1], values[i, 1])
                    rows.append((date, "teammates", int(gap), a, b))
        teams = list(by_team)
        if len(teams) == 2:
            left = by_team[teams[0]]["resid"].to_numpy(dtype=float)
            right = by_team[teams[1]]["resid"].to_numpy(dtype=float)
            for a in left:
                for b in right:
                    rows.append((date, "opposing_hitters", -1, a, b))
        for _, arm in arms.iterrows():
            for team, grp in by_team.items():
                if team == arm["opponent_code"]:
                    label = "pitcher_vs_opposing"
                elif team == arm["team_code"]:
                    label = "pitcher_own_hitters"
                else:
                    continue
                for b in grp["resid"].to_numpy(dtype=float):
                    rows.append((date, label, -1, float(arm["resid"]), b))
    return pd.DataFrame(rows, columns=["date", "pair_class", "order_gap", "a", "b"])


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 20 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def empirical_correlations(frame: pd.DataFrame, bootstrap_reps: int = 300, seed: int = 11) -> pd.DataFrame:
    """Measured fantasy-point correlation per pair class, with slate-clustered intervals."""
    pairs = _pairs_frame(frame)
    if pairs.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    dates = pairs["date"].unique()
    rows = []
    groups = [(label, grp) for label, grp in pairs.groupby("pair_class")]
    team = pairs[pairs["pair_class"] == "teammates"]
    for gap, grp in team.groupby("order_gap"):
        if len(grp) >= 200:
            groups.append((f"teammates_gap_{int(gap)}", grp))
    for label, grp in groups:
        estimate = _corr(grp["a"].to_numpy(), grp["b"].to_numpy())
        by_date = {date: sub[["a", "b"]].to_numpy() for date, sub in grp.groupby("date")}
        samples = []
        for _ in range(bootstrap_reps):
            pick = rng.choice(dates, size=len(dates), replace=True)
            stacked = [by_date[d] for d in pick if d in by_date]
            if not stacked:
                continue
            block = np.vstack(stacked)
            samples.append(_corr(block[:, 0], block[:, 1]))
        samples_arr = np.array([s for s in samples if np.isfinite(s)])
        rows.append(
            {
                "pair_class": label,
                "pairs": int(len(grp)),
                "correlation": estimate,
                "ci_low": float(np.percentile(samples_arr, 2.5)) if len(samples_arr) else float("nan"),
                "ci_high": float(np.percentile(samples_arr, 97.5)) if len(samples_arr) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _synthetic_slate(slot_means: Dict[int, float], pitcher_mean: float, games: int = 2) -> pd.DataFrame:
    rows = []
    for game in range(games):
        home, away = f"H{game}", f"A{game}"
        for team, opp in ((home, away), (away, home)):
            for slot in range(1, 10):
                rows.append(
                    {
                        "fd_player_id": f"{team}-b{slot}",
                        "full_name": f"{team} {slot}",
                        "player_type": "batter",
                        "team_code": team,
                        "opponent_code": opp,
                        "stack_key": team,
                        "game_key": "_vs_".join(sorted([team, opp])),
                        "batting_order_position": slot,
                        "stack_priority": "mid",
                        "vegas_game_total": 0.0,
                        "proj_fd_mean": float(slot_means.get(slot, 9.0)),
                        "salary": 3000,
                    }
                )
            rows.append(
                {
                    "fd_player_id": f"{team}-p",
                    "full_name": f"{team} P",
                    "player_type": "pitcher",
                    "team_code": team,
                    "opponent_code": opp,
                    "stack_key": team,
                    "game_key": "_vs_".join(sorted([team, opp])),
                    "batting_order_position": 0,
                    "stack_priority": "mid",
                    "vegas_game_total": 0.0,
                    "proj_fd_mean": float(pitcher_mean),
                    "salary": 8500,
                }
            )
    return pd.DataFrame(rows)


def simulated_correlations(
    calibration: SimulationCalibration,
    slot_means: Optional[Dict[int, float]] = None,
    pitcher_mean: float = 26.0,
    num_simulations: int = 40_000,
    seed: int = 5,
    correlation_config: Optional[CorrelationConfig] = None,
) -> Dict[str, float]:
    """Fantasy-point correlations the simulator produces for each pair class."""
    slot_means = slot_means or {slot: 9.0 for slot in range(1, 10)}
    slate = _synthetic_slate(slot_means, pitcher_mean)
    config = correlation_config or calibration.correlation_config()
    dists = fit_player_distributions(slate, calibration=calibration)
    model = build_correlation_matrix(slate, config)
    sim = simulate_slate(dists, model, num_simulations=num_simulations, seed=seed, use_antithetic=False)
    scores = pd.DataFrame(sim.scores, columns=sim.player_ids)
    corr = scores.corr().to_numpy()
    meta = slate.set_index("fd_player_id").loc[sim.player_ids]
    kind = meta["player_type"].to_numpy()
    team = meta["team_code"].to_numpy()
    opp = meta["opponent_code"].to_numpy()
    buckets: Dict[str, List[float]] = {label: [] for label in PAIR_CLASSES}
    buckets["cross_game_hitters"] = []
    count = len(sim.player_ids)
    for i in range(count):
        for j in range(i + 1, count):
            if kind[i] == "batter" and kind[j] == "batter":
                if team[i] == team[j]:
                    buckets["teammates"].append(corr[i, j])
                elif opp[i] == team[j]:
                    buckets["opposing_hitters"].append(corr[i, j])
                else:
                    buckets["cross_game_hitters"].append(corr[i, j])
            elif kind[i] != kind[j]:
                pitcher, hitter = (i, j) if kind[i] == "pitcher" else (j, i)
                if opp[pitcher] == team[hitter]:
                    buckets["pitcher_vs_opposing"].append(corr[i, j])
                elif team[pitcher] == team[hitter]:
                    buckets["pitcher_own_hitters"].append(corr[i, j])
    return {label: float(np.mean(values)) if values else float("nan") for label, values in buckets.items()}


_PARAMETER_FOR_CLASS = {
    "teammates": "teammate_base",
    "opposing_hitters": "same_game_opponent",
    "pitcher_vs_opposing": "pitcher_vs_opposing",
    "pitcher_own_hitters": "pitcher_own_hitters",
}


def fit_copula(
    calibration: SimulationCalibration,
    targets: Dict[str, float],
    slot_means: Optional[Dict[int, float]] = None,
    pitcher_mean: float = 26.0,
    iterations: int = 6,
    num_simulations: int = 40_000,
) -> SimulationCalibration:
    """Solve for latent correlations whose simulated output matches ``targets``."""
    fitted = SimulationCalibration.from_dict(calibration.to_dict())
    for label, parameter in _PARAMETER_FOR_CLASS.items():
        target = targets.get(label)
        if target is not None and np.isfinite(target):
            setattr(fitted, parameter, float(np.clip(target * 1.2, -0.6, 0.6)))
    for step in range(iterations):
        simulated = simulated_correlations(
            fitted, slot_means, pitcher_mean, num_simulations=num_simulations, seed=100 + step
        )
        for label, parameter in _PARAMETER_FOR_CLASS.items():
            target = targets.get(label)
            got = simulated.get(label)
            if target is None or not np.isfinite(target) or got is None or not np.isfinite(got):
                continue
            current = float(getattr(fitted, parameter))
            # Additive correction: stable near zero, where a ratio would blow up.
            updated = current + (float(target) - float(got))
            setattr(fitted, parameter, float(np.clip(updated, -0.6, 0.6)))
    for parameter in _PARAMETER_FOR_CLASS.values():
        setattr(fitted, parameter, round(float(getattr(fitted, parameter)), 3))
    return fitted


# ---------------------------------------------------------------------------
# Validation: does the simulator reproduce real stacks?
# ---------------------------------------------------------------------------

def _quantile_row(label: str, values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "source": label,
        "n": int(len(values)),
        "mean": float(values.mean()),
        "sd": float(values.std()),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
    }


def _simulate_games(
    frame: pd.DataFrame,
    calibration: Optional[SimulationCalibration],
    draws: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Simulate every real game; return pooled hitter, pitcher, team-sum and 4-stack scores."""
    work = frame.copy()
    work["proj_fd_mean"] = work["projection"]
    work["salary"] = np.where(work["player_type"] == "pitcher", 8500, 3000)
    work["stack_key"] = work["team_code"]
    work["game_key"] = work["game_pk"].astype(str)
    work["batting_order_position"] = work["slot"]
    work["stack_priority"] = "mid"
    if calibration is None:
        # Legacy model needs floor/ceiling: take the multipliers the 2026 projection code produced.
        legacy = compute_baseline_projections(work.drop(columns=["proj_fd_mean"]), config=ProjectionConfig())
        ratio_floor = (legacy["proj_fd_floor"] / legacy["proj_fd_mean"].replace(0, np.nan)).to_numpy()
        ratio_ceiling = (legacy["proj_fd_ceiling"] / legacy["proj_fd_mean"].replace(0, np.nan)).to_numpy()
        work["proj_fd_floor"] = work["proj_fd_mean"].to_numpy() * np.nan_to_num(ratio_floor, nan=0.8)
        work["proj_fd_ceiling"] = work["proj_fd_mean"].to_numpy() * np.nan_to_num(ratio_ceiling, nan=1.3)
        corr_config = CorrelationConfig()
    else:
        corr_config = calibration.correlation_config()
    pooled: Dict[str, List[np.ndarray]] = {"hitters": [], "pitchers": [], "team_sum": [], "stack4": []}
    for idx, (_, game) in enumerate(work.groupby(["date", "game_pk"])):
        game = game.drop_duplicates("fd_player_id").reset_index(drop=True)
        if (game["player_type"] == "batter").sum() < 8:
            continue
        dists = fit_player_distributions(game, calibration=calibration)
        model = build_correlation_matrix(game, corr_config)
        sim = simulate_slate(dists, model, num_simulations=draws, seed=seed + idx, use_antithetic=False)
        scores = sim.scores
        is_bat = (game["player_type"] == "batter").to_numpy()
        pooled["hitters"].append(scores[:, is_bat].ravel())
        if (~is_bat).any():
            pooled["pitchers"].append(scores[:, ~is_bat].ravel())
        for _, team_rows in game[is_bat].groupby("team_code"):
            if len(team_rows) < 9:
                continue
            cols = team_rows.index.to_numpy()
            pooled["team_sum"].append(scores[:, cols].sum(axis=1))
            top4 = team_rows.sort_values("slot").head(4).index.to_numpy()
            pooled["stack4"].append(scores[:, top4].sum(axis=1))
    return {key: np.concatenate(values) if values else np.array([]) for key, values in pooled.items()}


def validation_tables(
    frame: pd.DataFrame,
    calibration: SimulationCalibration,
    draws: int = 200,
    seed: int = 21,
) -> pd.DataFrame:
    """Real vs legacy-simulated vs calibrated-simulated score distributions."""
    bats = frame[frame["player_type"] == "batter"]
    arms = frame[frame["player_type"] == "pitcher"]
    real: Dict[str, np.ndarray] = {
        "hitters": bats["actual"].to_numpy(dtype=float),
        "pitchers": arms["actual"].to_numpy(dtype=float),
    }
    team_sums, stack_sums = [], []
    for _, team_rows in bats.groupby(["date", "game_pk", "team_code"]):
        if len(team_rows) < 9:
            continue
        team_sums.append(float(team_rows["actual"].sum()))
        stack_sums.append(float(team_rows.sort_values("slot").head(4)["actual"].sum()))
    real["team_sum"] = np.array(team_sums)
    real["stack4"] = np.array(stack_sums)

    legacy = _simulate_games(frame, None, draws, seed)
    fitted = _simulate_games(frame, calibration, draws, seed)
    rows = []
    for quantity in ("hitters", "pitchers", "team_sum", "stack4"):
        for label, source in (("actual", real), ("legacy_sim", legacy), ("calibrated_sim", fitted)):
            values = source.get(quantity, np.array([]))
            if len(values) == 0:
                continue
            row = {"quantity": quantity, **_quantile_row(label, values)}
            if quantity == "hitters":
                row["zero_rate"] = float((values <= 0).mean())
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class CalibrationReport:
    calibration: SimulationCalibration
    correlations: pd.DataFrame
    simulated: Dict[str, float]
    validation: pd.DataFrame
    notes: Dict[str, float] = field(default_factory=dict)


def calibrate(
    player_games: pd.DataFrame,
    proj_col: str = DEFAULT_PROJECTION_COLUMN,
    validate: bool = True,
    num_simulations: int = 40_000,
    validation_draws: int = 200,
    copula_nu: int = 30,
) -> CalibrationReport:
    """Fit every simulator setting from real outcomes.

    ``copula_nu`` is fixed rather than fitted.  The hand-set value of 5 makes
    every player on the slate share one "wild night" factor, which produced a
    0.04 correlation between hitters in *different games* (measured: about
    0.01) and team-stack 99th percentiles about 10% above reality.  30 keeps
    a little joint-tail behaviour and matches real stack totals better.
    """
    frame = starters(player_games, proj_col)
    bats = frame[frame["player_type"] == "batter"]
    arms = frame[frame["player_type"] == "pitcher"]
    if len(bats) < 500 or len(arms) < 60:
        raise ValueError(
            f"Not enough outcomes to calibrate ({len(bats)} hitter starts, {len(arms)} pitcher starts). "
            "Run scripts/fetch_actuals.py and scripts/backtest_projections.py over more slates first."
        )
    hitter = fit_hitter_model(bats)
    pitcher = fit_pitcher_model(arms)
    correlations = empirical_correlations(frame)
    targets = {
        row["pair_class"]: float(row["correlation"])
        for _, row in correlations.iterrows()
        if row["pair_class"] in PAIR_CLASSES
    }
    slot_means = bats[bats["slot"].between(1, 9)].groupby("slot")["projection"].mean().to_dict()
    pitcher_mean = float(arms["projection"].mean())

    base = SimulationCalibration(
        hitter_zero_scale=hitter["hitter_zero_scale"],
        hitter_positive_cv=hitter["hitter_positive_cv"],
        hitter_positive_shift=hitter["hitter_positive_shift"],
        pitcher_sd_intercept=pitcher["pitcher_sd_intercept"],
        pitcher_sd_slope=pitcher["pitcher_sd_slope"],
        copula_nu=int(copula_nu),
        source=f"fitted to real box scores, projection column {proj_col}",
        slates_used=int(frame["date"].nunique()),
        hitter_starts=int(len(bats)),
        pitcher_starts=int(len(arms)),
        fitted_at=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )
    fitted = fit_copula(base, targets, slot_means, pitcher_mean, num_simulations=num_simulations)
    simulated = simulated_correlations(fitted, slot_means, pitcher_mean, num_simulations=num_simulations, seed=999)
    validation = validation_tables(frame, fitted, draws=validation_draws) if validate else pd.DataFrame()
    notes = {
        "observed_hitter_zero_rate": hitter["observed_zero_rate"],
        "observed_hitter_sd": hitter["observed_sd"],
        "observed_pitcher_sd": pitcher["observed_sd"],
        "observed_pitcher_negative_rate": pitcher["observed_negative_rate"],
    }
    return CalibrationReport(fitted, correlations, simulated, validation, notes)


__all__ = [
    "CalibrationReport",
    "PAIR_CLASSES",
    "calibrate",
    "empirical_correlations",
    "fit_copula",
    "fit_hitter_model",
    "fit_pitcher_model",
    "simulated_correlations",
    "starters",
    "validation_tables",
]
