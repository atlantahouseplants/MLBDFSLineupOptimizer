"""Run-level health checks for slate data, boom/bust inputs, and lineups."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class RunAuditReport:
    """Compact report used by the app to make a slate run auditable."""

    summary: Dict[str, object]
    checks: pd.DataFrame
    lineup_checks: pd.DataFrame
    stack_mix: pd.DataFrame
    stack_alignment: pd.DataFrame


def build_run_audit_report(
    optimizer_df: Optional[pd.DataFrame],
    lineup_df: Optional[pd.DataFrame] = None,
    expected_lineups: Optional[int] = None,
    stack_plan: Optional[pd.DataFrame] = None,
    projection_source_count: Optional[int] = None,
    ownership_source_count: Optional[int] = None,
    salary_cap: int = 35000,
) -> RunAuditReport:
    """Build high-signal checks for whether a run is using the intended inputs."""

    opt = optimizer_df.copy() if isinstance(optimizer_df, pd.DataFrame) else pd.DataFrame()
    lineups = lineup_df.copy() if isinstance(lineup_df, pd.DataFrame) else pd.DataFrame()
    plan = stack_plan.copy() if isinstance(stack_plan, pd.DataFrame) else pd.DataFrame()
    checks = []
    summary: Dict[str, object] = {}

    player_count = int(len(opt))
    summary["player_count"] = player_count
    summary["projection_sources"] = float(projection_source_count or 0)
    summary["ownership_sources"] = float(ownership_source_count or 0)

    if opt.empty:
        _add_check(
            checks,
            "Fail",
            "Player pool",
            "No optimizer-ready players were found.",
            "0 players",
            "Reprocess the slate after uploading the FanDuel and BallparkPal files.",
        )
        return RunAuditReport(summary, pd.DataFrame(checks), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    mean_proj = _numeric(opt, "proj_fd_mean")
    projected_count = int((mean_proj > 0).sum())
    summary["projected_players"] = projected_count
    if projected_count == 0:
        _add_check(
            checks,
            "Fail",
            "Projections",
            "No players have positive fantasy projections.",
            "0 projected",
            "Check the FanDuel/BallparkPal slate match before optimizing.",
        )
    elif projected_count < max(25, player_count * 0.5):
        _add_check(
            checks,
            "Warn",
            "Projections",
            "Less than half of the player pool has positive fantasy projections.",
            f"{projected_count}/{player_count}",
            "Check for a slate mismatch or missing projection source.",
        )
    else:
        _add_check(
            checks,
            "Good",
            "Projections",
            "Player projections are populated.",
            f"{projected_count}/{player_count}",
            "",
        )

    _add_boom_bust_checks(checks, summary, opt, projection_source_count)
    _add_ownership_checks(checks, summary, opt, ownership_source_count)

    lineup_checks = _lineup_checks(lineups, salary_cap)
    stack_mix = _stack_mix(lineups)
    stack_alignment = _stack_alignment(lineups, plan)
    _add_lineup_checks(checks, summary, lineup_checks, expected_lineups, salary_cap)
    _add_stack_alignment_checks(checks, summary, stack_alignment)

    return RunAuditReport(
        summary=summary,
        checks=pd.DataFrame(checks),
        lineup_checks=lineup_checks,
        stack_mix=stack_mix,
        stack_alignment=stack_alignment,
    )


def _add_boom_bust_checks(
    checks: list[dict],
    summary: Dict[str, object],
    opt: pd.DataFrame,
    projection_source_count: Optional[int],
) -> None:
    bust = _numeric(opt, "proj_fd_bust_rate")
    median = _numeric(opt, "proj_fd_median")
    upside = _numeric(opt, "proj_fd_upside")
    mean_proj = _numeric(opt, "proj_fd_mean")
    players = max(1, len(opt))

    bust_nonzero = int((bust > 0).sum())
    bust_coverage = bust_nonzero / players
    summary["bust_coverage_pct"] = float(bust_coverage)
    summary["avg_bust_rate"] = float(bust[bust > 0].mean()) if bust_nonzero else 0.0

    if bust_nonzero == 0:
        if projection_source_count:
            _add_check(
                checks,
                "Fail",
                "Bust input",
                "Projection/sim files were blended, but no non-zero Bust values reached the optimizer.",
                "0% coverage",
                "Confirm the source workbook has a Bust column and matched player names or IDs.",
            )
        else:
            _add_check(
                checks,
                "Warn",
                "Bust input",
                "No external projection/sim source is blended, so simulations are using generic volatility.",
                "0% coverage",
                "Upload or auto-detect the Ballpark DFS Optimizer workbook with Bust/Median/Upside.",
            )
    elif bust_coverage < 0.5:
        _add_check(
            checks,
            "Warn",
            "Bust input",
            "Bust values reached the optimizer, but coverage is light.",
            f"{bust_coverage:.0%} coverage",
            "Check player-name matching between FanDuel and the sim workbook.",
        )
    else:
        _add_check(
            checks,
            "Good",
            "Bust input",
            "Bust values are populated and available to the simulator.",
            f"{bust_coverage:.0%} coverage",
            "",
        )

    median_source_like = int(((median > 0) & ((median - mean_proj).abs() > 1e-6)).sum())
    median_coverage = float(median_source_like / players)
    summary["median_coverage_pct"] = median_coverage
    if median_source_like <= 0:
        _add_check(
            checks,
            "Warn",
            "Median input",
            "Median values are missing or equal to the mean, so distributions will fall back to mean-based shape.",
            "derived/baseline",
            "Blend the sim workbook or inspect Median column matching.",
        )
    else:
        _add_check(
            checks,
            "Good",
            "Median input",
            "Median values are available for distribution fitting.",
            f"{median_coverage:.0%} coverage",
            "",
        )

    upside_coverage = float(((upside > 0) & (upside >= mean_proj)).sum() / players)
    summary["upside_coverage_pct"] = upside_coverage
    if upside_coverage <= 0:
        _add_check(
            checks,
            "Warn",
            "Upside input",
            "No usable Upside values are populated; ceilings will rely on the baseline model.",
            "0% coverage",
            "Blend the sim workbook or inspect Upside column matching.",
        )
    elif not projection_source_count:
        _add_check(
            checks,
            "Warn",
            "Upside input",
            "Upside values are populated from the baseline ceiling model, not an external sim source.",
            f"{upside_coverage:.0%} baseline",
            "Blend the sim workbook if you want purchased Upside values driving tail outcomes.",
        )
    else:
        _add_check(
            checks,
            "Good",
            "Upside input",
            "Upside values are available for ceiling and tail modeling.",
            f"{upside_coverage:.0%} coverage",
            "",
        )


def _add_ownership_checks(
    checks: list[dict],
    summary: Dict[str, object],
    opt: pd.DataFrame,
    ownership_source_count: Optional[int],
) -> None:
    ownership = _as_fraction(_numeric(opt, "proj_fd_ownership"))
    summary["avg_player_ownership"] = float(ownership.mean()) if not ownership.empty else 0.0
    summary["max_player_ownership"] = float(ownership.max()) if not ownership.empty else 0.0
    summary["ownership_mode"] = "External" if ownership_source_count else "Fallback"
    positive = int((ownership > 0).sum())

    if ownership_source_count:
        source_covered = _ownership_source_covered(opt)
        coverage_pct = float(source_covered.mean()) if len(source_covered) else 0.0
        summary["ownership_source_coverage_pct"] = coverage_pct
        if positive == 0:
            _add_check(
                checks,
                "Fail",
                "Ownership input",
                "Ownership files were provided, but no positive ownership values reached the optimizer.",
                "0 players",
                "Check ownership source columns and player matching.",
            )
        elif coverage_pct < 0.75:
            _add_check(
                checks,
                "Warn",
                "Ownership input",
                "External ownership is present, but too much of the pool is fallback-filled.",
                f"{coverage_pct:.0%} matched",
                "Review ownership source matching before relying on leverage outputs.",
            )
        else:
            _add_check(
                checks,
                "Good",
                "Ownership input",
                "External ownership values are populated, so leverage is backed by real ownership input.",
                f"{coverage_pct:.0%} matched",
                "",
            )
            _add_check(
                checks,
                "Good",
                "Leverage confidence",
                "Leverage decisions are using an external ownership source.",
                "ownership-backed",
                "",
            )
        missing_chalk = _missing_high_projection_ownership(opt, source_covered)
        if not missing_chalk.empty:
            names = ", ".join(missing_chalk["full_name"].astype(str).head(5).tolist())
            _add_check(
                checks,
                "Warn",
                "Missing chalk ownership",
                "High-projection players are missing external ownership and are using fallback estimates.",
                names,
                "Fix name matching or add aliases before trusting chalk fades.",
            )
        else:
            _add_check(
                checks,
                "Good",
                "Missing chalk ownership",
                "High-projection players are covered by the external ownership source.",
                "covered",
                "",
            )
    else:
        _add_check(
            checks,
            "Warn",
            "Ownership input",
            "No external ownership source is blended; leverage is based on the fallback ownership model.",
            f"avg {ownership.mean():.1%}" if not ownership.empty else "no ownership",
            "Use external ownership when available for the strongest leverage decisions.",
        )
        _add_check(
            checks,
            "Warn",
            "Leverage confidence",
            "Leverage decisions are using estimated ownership only.",
            "model-estimated",
            "Treat leverage conclusions as directional until an ownership source is uploaded.",
        )

    if not ownership.empty and ownership.max() > 1.0:
        _add_check(
            checks,
            "Warn",
            "Ownership scale",
            "Ownership values look above 100% after normalization.",
            f"max {ownership.max():.1%}",
            "Check whether ownership was uploaded as both percent and decimal in different sources.",
        )
    elif not ownership.empty and ownership_source_count and ownership.max() < 0.03:
        _add_check(
            checks,
            "Warn",
            "Ownership scale",
            "External ownership looks too small; it may have been uploaded as a fraction of a percent.",
            f"max {ownership.max():.1%}",
            "Check whether the source uses 12 for 12% or 0.12 for 12%.",
        )


def _add_lineup_checks(
    checks: list[dict],
    summary: Dict[str, object],
    lineup_checks: pd.DataFrame,
    expected_lineups: Optional[int],
    salary_cap: int,
) -> None:
    if lineup_checks.empty:
        return

    lineup_count = int(lineup_checks["lineup_id"].nunique())
    unique_count = int(lineup_checks["duplicate_key"].nunique())
    duplicates = int((lineup_checks["duplicate_count"] > 1).sum())
    dup_players = int((lineup_checks["player_count"] < lineup_checks["row_count"]).sum())
    over_cap = int((lineup_checks["salary"] > salary_cap).sum())

    summary["lineup_count"] = lineup_count
    summary["unique_lineups"] = unique_count
    summary["duplicate_lineups"] = duplicates
    summary["lineups_with_duplicate_players"] = dup_players
    summary["avg_lineup_projection"] = float(lineup_checks["projection"].mean())
    summary["avg_lineup_ownership"] = float(lineup_checks["ownership"].mean())
    summary["avg_lineup_leverage"] = float(lineup_checks["avg_leverage"].mean())

    if expected_lineups is not None and lineup_count != int(expected_lineups):
        _add_check(
            checks,
            "Warn",
            "Lineup count",
            "The run produced a different lineup count than expected.",
            f"{lineup_count}/{expected_lineups}",
            "Confirm the final portfolio size before exporting.",
        )
    else:
        _add_check(
            checks,
            "Good",
            "Lineup count",
            "Lineup count matches the current run target.",
            str(lineup_count),
            "",
        )

    if duplicates:
        _add_check(
            checks,
            "Fail",
            "Duplicate lineups",
            "Some final lineups have identical player sets.",
            str(duplicates),
            "Generate a larger pool or tighten uniqueness controls before upload.",
        )
    else:
        _add_check(
            checks,
            "Good",
            "Duplicate lineups",
            "All audited lineups are unique.",
            str(unique_count),
            "",
        )

    if dup_players:
        _add_check(
            checks,
            "Fail",
            "Duplicate players",
            "Some lineups contain the same player more than once.",
            str(dup_players),
            "Do not upload until those lineups are regenerated.",
        )
    else:
        _add_check(
            checks,
            "Good",
            "Duplicate players",
            "No lineup contains a repeated player.",
            str(lineup_count),
            "",
        )

    if over_cap:
        _add_check(
            checks,
            "Fail",
            "Salary cap",
            "Some lineups exceed the FanDuel salary cap.",
            str(over_cap),
            "Regenerate with the correct salary cap before upload.",
        )
    else:
        _add_check(
            checks,
            "Good",
            "Salary cap",
            "All audited lineups are at or below the salary cap.",
            f"${salary_cap:,.0f}",
            "",
        )


def _lineup_checks(lineup_df: pd.DataFrame, salary_cap: int) -> pd.DataFrame:
    if lineup_df.empty or "lineup_id" not in lineup_df.columns:
        return pd.DataFrame()

    df = lineup_df.copy()
    for col in ("fd_player_id", "full_name", "team_code", "player_type"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)
    for col in ("salary", "proj_fd_mean", "proj_fd_ownership", "player_leverage_score"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["proj_fd_ownership"] = _as_fraction(df["proj_fd_ownership"])

    rows = []
    for lineup_id, group in df.groupby("lineup_id"):
        ids = sorted(group["fd_player_id"].astype(str).tolist())
        rows.append(
            {
                "lineup_id": lineup_id,
                "row_count": int(len(group)),
                "player_count": int(group["fd_player_id"].nunique()),
                "salary": float(group["salary"].sum()),
                "salary_unused": float(salary_cap - group["salary"].sum()),
                "projection": float(group["proj_fd_mean"].sum()),
                "ownership": float(group["proj_fd_ownership"].sum()),
                "avg_leverage": float(group["player_leverage_score"].mean()),
                "duplicate_key": "|".join(ids),
            }
        )
    checks = pd.DataFrame(rows).sort_values("lineup_id")
    if not checks.empty:
        dup_counts = checks["duplicate_key"].value_counts()
        checks["duplicate_count"] = checks["duplicate_key"].map(dup_counts).fillna(1).astype(int)
    return checks


def _stack_mix(lineup_df: pd.DataFrame) -> pd.DataFrame:
    if lineup_df.empty or "lineup_id" not in lineup_df.columns:
        return pd.DataFrame()
    df = lineup_df.copy()
    for col in ("team_code", "player_type"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)

    hitters = df[df["player_type"].str.lower() != "pitcher"]
    if hitters.empty:
        return pd.DataFrame()
    total = max(1, int(hitters["lineup_id"].nunique()))
    team_counts = hitters.groupby(["lineup_id", "team_code"]).size().reset_index(name="hitters")
    signatures = []
    for lineup_id, group in team_counts.groupby("lineup_id"):
        counts = sorted([int(value) for value in group["hitters"].tolist() if int(value) >= 2], reverse=True)
        signature = "-".join(str(value) for value in counts) if counts else "No stacks"
        signatures.append({"lineup_id": lineup_id, "stack_template": signature})
    mix = (
        pd.DataFrame(signatures)
        .groupby("stack_template")["lineup_id"]
        .nunique()
        .reset_index(name="lineups")
        .sort_values("lineups", ascending=False)
    )
    mix["lineup_pct"] = mix["lineups"] / total
    return mix


def _stack_alignment(lineup_df: pd.DataFrame, stack_plan: pd.DataFrame) -> pd.DataFrame:
    if (
        lineup_df.empty
        or stack_plan.empty
        or "lineup_id" not in lineup_df.columns
        or "team_code" not in stack_plan.columns
    ):
        return pd.DataFrame()

    df = lineup_df.copy()
    for col in ("team_code", "player_type"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).str.upper().str.strip()
    hitters = df[df["player_type"].str.lower() != "pitcher"].copy()
    if hitters.empty:
        return pd.DataFrame()

    lineup_count = max(1, int(df["lineup_id"].nunique()))
    stacks = (
        hitters.groupby(["lineup_id", "team_code"])
        .size()
        .reset_index(name="stack_hitters")
    )
    stacks = stacks[stacks["stack_hitters"] >= 3]
    if stacks.empty:
        return pd.DataFrame()

    exposure = (
        stacks.groupby("team_code")
        .agg(
            stack_lineups=("lineup_id", "nunique"),
            avg_stack_size=("stack_hitters", "mean"),
            max_stack_size=("stack_hitters", "max"),
        )
        .reset_index()
    )
    exposure["exposure_pct"] = exposure["stack_lineups"] / lineup_count

    plan = stack_plan.copy()
    plan["team_code"] = plan["team_code"].astype(str).str.upper().str.strip()
    keep_cols = [
        col
        for col in [
            "team_code",
            "tier",
            "team_runs",
            "p_runs_ge_6",
            "top5_projection",
            "top5_upside",
            "stack_ownership",
            "sim_upside_score",
            "stack_leverage_score",
            "stack_score",
            "target_exposure",
            "recommended_max_exposure",
        ]
        if col in plan.columns
    ]
    alignment = exposure.merge(plan[keep_cols], on="team_code", how="left")
    for col in [
        "team_runs",
        "p_runs_ge_6",
        "top5_projection",
        "top5_upside",
        "stack_ownership",
        "sim_upside_score",
        "stack_leverage_score",
        "stack_score",
        "target_exposure",
        "recommended_max_exposure",
    ]:
        if col not in alignment.columns:
            alignment[col] = 0.0
        alignment[col] = pd.to_numeric(alignment[col], errors="coerce").fillna(0.0)
    alignment["exposure_vs_target"] = alignment["exposure_pct"] - alignment["target_exposure"]
    alignment["over_recommended_max"] = (
        (alignment["recommended_max_exposure"] > 0)
        & (alignment["exposure_pct"] > alignment["recommended_max_exposure"])
    )

    leverage_series = pd.to_numeric(
        plan.get("stack_leverage_score", pd.Series(0.0, index=plan.index)),
        errors="coerce",
    ).fillna(0.0)
    ownership_series = pd.to_numeric(
        plan.get("stack_ownership", pd.Series(0.0, index=plan.index)),
        errors="coerce",
    ).fillna(0.0)
    upside_series = pd.to_numeric(
        plan.get("sim_upside_score", pd.Series(0.0, index=plan.index)),
        errors="coerce",
    ).fillna(0.0)
    leverage_cut = float(leverage_series.median())
    chalk_cut = float(ownership_series.quantile(0.75))
    upside_cut = float(upside_series.median())
    alignment["positive_stack_leverage"] = alignment["stack_leverage_score"] >= leverage_cut
    alignment["high_owned_stack"] = alignment["stack_ownership"] >= chalk_cut
    alignment["comparable_upside_stack"] = alignment["sim_upside_score"] >= upside_cut
    alignment["chalk_without_leverage"] = alignment["high_owned_stack"] & ~alignment["positive_stack_leverage"]
    return alignment.sort_values("exposure_pct", ascending=False).reset_index(drop=True)


def _add_stack_alignment_checks(
    checks: list[dict],
    summary: Dict[str, object],
    stack_alignment: pd.DataFrame,
) -> None:
    if stack_alignment.empty:
        return

    exposure = pd.to_numeric(stack_alignment["exposure_pct"], errors="coerce").fillna(0.0)
    positive_leverage = float(exposure[stack_alignment["positive_stack_leverage"].fillna(False)].sum())
    comparable_upside = float(exposure[stack_alignment["comparable_upside_stack"].fillna(False)].sum())
    chalk_without_leverage = float(exposure[stack_alignment["chalk_without_leverage"].fillna(False)].sum())
    over_max_count = int(stack_alignment["over_recommended_max"].fillna(False).sum())

    summary["positive_leverage_stack_exposure"] = positive_leverage
    summary["comparable_upside_stack_exposure"] = comparable_upside
    summary["chalk_without_leverage_stack_exposure"] = chalk_without_leverage

    if positive_leverage >= 0.60:
        _add_check(
            checks,
            "Good",
            "Stack leverage alignment",
            "Most stack exposure is on teams with above-slate stack leverage.",
            f"{positive_leverage:.0%}",
            "",
        )
    elif positive_leverage >= 0.40:
        _add_check(
            checks,
            "Warn",
            "Stack leverage alignment",
            "Stack exposure is only moderately tilted toward leverage-positive teams.",
            f"{positive_leverage:.0%}",
            "Review whether chalk teams are taking too much of the final portfolio.",
        )
    else:
        _add_check(
            checks,
            "Warn",
            "Stack leverage alignment",
            "Less than half of stack exposure is on leverage-positive teams.",
            f"{positive_leverage:.0%}",
            "Increase stack plan weight or lower caps on high-owned stacks.",
        )

    if comparable_upside >= 0.70:
        _add_check(
            checks,
            "Good",
            "Comparable upside stacks",
            "Final stacks are mostly on teams with comparable or better team-sim upside.",
            f"{comparable_upside:.0%}",
            "",
        )
    else:
        _add_check(
            checks,
            "Warn",
            "Comparable upside stacks",
            "A meaningful chunk of stack exposure is on lower-upside teams.",
            f"{comparable_upside:.0%}",
            "Compare final stack exposure to p_runs_ge_6 and top5_upside before upload.",
        )

    if chalk_without_leverage > 0.30:
        _add_check(
            checks,
            "Warn",
            "Chalk stack fade",
            "High-owned, low-leverage stacks are taking a large share of exposure.",
            f"{chalk_without_leverage:.0%}",
            "Lower team caps or use lesser-owned players from those stacks.",
        )
    else:
        _add_check(
            checks,
            "Good",
            "Chalk stack fade",
            "Exposure to high-owned stacks without leverage is controlled.",
            f"{chalk_without_leverage:.0%}",
            "",
        )

    if over_max_count:
        _add_check(
            checks,
            "Warn",
            "Stack max caps",
            "Some teams exceed the stack plan recommended max exposure.",
            str(over_max_count),
            "Review those teams before upload or loosen the plan deliberately.",
        )


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0).astype(float)


def _ownership_source_covered(df: pd.DataFrame) -> pd.Series:
    if "ownership_source_covered" not in df.columns:
        return pd.Series(True, index=df.index, dtype=bool)
    return df["ownership_source_covered"].fillna(False).astype(bool)


def _missing_high_projection_ownership(df: pd.DataFrame, source_covered: pd.Series) -> pd.DataFrame:
    if df.empty or source_covered.empty:
        return pd.DataFrame()
    temp = df.copy()
    temp["proj_fd_mean"] = _numeric(temp, "proj_fd_mean")
    if "player_type" in temp.columns:
        player_type = temp["player_type"].astype(str).str.lower()
    else:
        player_type = pd.Series("", index=temp.index)
    hitter_cut = temp.loc[player_type != "pitcher", "proj_fd_mean"].quantile(0.85)
    pitcher_cut = temp.loc[player_type == "pitcher", "proj_fd_mean"].quantile(0.75)
    is_high = ((player_type != "pitcher") & (temp["proj_fd_mean"] >= hitter_cut)) | (
        (player_type == "pitcher") & (temp["proj_fd_mean"] >= pitcher_cut)
    )
    missing = temp[is_high & ~source_covered.reindex(temp.index).fillna(False)]
    keep_cols = [col for col in ["full_name", "team_code", "player_type", "proj_fd_mean"] if col in missing.columns]
    return missing.sort_values("proj_fd_mean", ascending=False)[keep_cols].head(10)


def _as_fraction(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    if not values.empty and values.max() > 1.5:
        values = values / 100.0
    return values


def _add_check(
    rows: list[dict],
    status: str,
    check: str,
    detail: str,
    value: str,
    action: str,
) -> None:
    rows.append(
        {
            "status": status,
            "check": check,
            "detail": detail,
            "value": value,
            "action": action,
        }
    )


__all__ = ["RunAuditReport", "build_run_audit_report"]
