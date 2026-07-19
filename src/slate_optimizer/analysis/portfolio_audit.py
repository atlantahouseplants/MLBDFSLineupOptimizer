"""Final portfolio audit helpers for lineup review and upload validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pandas as pd


@dataclass
class PortfolioAuditReport:
    summary: Dict[str, float]
    issues: pd.DataFrame
    player_exposure: pd.DataFrame
    stack_exposure: pd.DataFrame
    lineup_checks: pd.DataFrame


def build_portfolio_audit(
    lineup_df: pd.DataFrame,
    optimizer_df: Optional[pd.DataFrame] = None,
    stack_plan: Optional[pd.DataFrame] = None,
    contest_df: Optional[pd.DataFrame] = None,
    selected_ids: Optional[Iterable[int]] = None,
    batter_chalk_threshold: float = 0.25,
    pitcher_chalk_threshold: float = 0.35,
    salary_cap: int = 35000,
) -> PortfolioAuditReport:
    """Build a compact final-portfolio audit from selected lineups."""

    df = lineup_df.copy() if lineup_df is not None else pd.DataFrame()
    if df.empty or "lineup_id" not in df.columns:
        return PortfolioAuditReport({}, _empty_issues(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    _normalize_core_columns(df)
    lineup_checks = _lineup_checks(df, salary_cap, batter_chalk_threshold, pitcher_chalk_threshold)
    player_exposure = _player_exposure(df, batter_chalk_threshold, pitcher_chalk_threshold)
    stack_exposure = _stack_exposure(df, stack_plan)
    issues = _audit_issues(
        df,
        lineup_checks,
        player_exposure,
        stack_exposure,
        salary_cap,
    )
    summary = _audit_summary(df, lineup_checks, player_exposure, stack_exposure, contest_df, selected_ids)
    return PortfolioAuditReport(
        summary=summary,
        issues=issues,
        player_exposure=player_exposure,
        stack_exposure=stack_exposure,
        lineup_checks=lineup_checks,
    )


def audit_report_to_csv(report: PortfolioAuditReport) -> str:
    """Serialize all audit sections into one readable CSV-like text file."""

    sections = []
    summary_df = pd.DataFrame([report.summary]) if report.summary else pd.DataFrame()
    for name, frame in (
        ("summary", summary_df),
        ("issues", report.issues),
        ("stack_exposure", report.stack_exposure),
        ("player_exposure", report.player_exposure),
        ("lineup_checks", report.lineup_checks),
    ):
        sections.append(f"[{name}]")
        sections.append(frame.to_csv(index=False) if frame is not None and not frame.empty else "")
    return "\n".join(sections)


def _normalize_core_columns(df: pd.DataFrame) -> None:
    for col in ("fd_player_id", "full_name", "team_code", "opponent_code", "position", "player_type"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)
    for col in ("salary", "proj_fd_mean", "proj_fd_ownership", "player_leverage_score"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if df["proj_fd_ownership"].max() > 1.5:
        df["proj_fd_ownership"] = df["proj_fd_ownership"] / 100.0
    if "is_confirmed_lineup" not in df.columns:
        df["is_confirmed_lineup"] = False
    df["is_confirmed_lineup"] = df["is_confirmed_lineup"].fillna(False).astype(bool)


def _lineup_checks(
    df: pd.DataFrame,
    salary_cap: int,
    batter_chalk_threshold: float,
    pitcher_chalk_threshold: float,
) -> pd.DataFrame:
    rows = []
    for lineup_id, group in df.groupby("lineup_id"):
        hitters = group[group["player_type"].str.lower() != "pitcher"]
        pitchers = group[group["player_type"].str.lower() == "pitcher"]
        pitcher = pitchers.iloc[0] if not pitchers.empty else None
        pitcher_opp = str(pitcher.get("opponent_code", "")) if pitcher is not None else ""
        hitters_vs_pitcher = int((hitters["team_code"].astype(str) == pitcher_opp).sum()) if pitcher_opp else 0
        team_counts = hitters["team_code"].value_counts()
        ids = sorted(group["fd_player_id"].astype(str).tolist())
        batter_chalk = hitters["proj_fd_ownership"] >= batter_chalk_threshold
        pitcher_chalk = (
            not pitchers.empty
            and float(pitchers["proj_fd_ownership"].max()) >= pitcher_chalk_threshold
        )
        rows.append(
            {
                "lineup_id": lineup_id,
                "salary": int(group["salary"].sum()),
                "salary_unused": int(salary_cap - group["salary"].sum()),
                "projection": float(group["proj_fd_mean"].sum()),
                "ownership": float(group["proj_fd_ownership"].sum()),
                "avg_leverage": float(group["player_leverage_score"].mean()),
                "pitcher": str(pitcher.get("full_name", "")) if pitcher is not None else "",
                "pitcher_team": str(pitcher.get("team_code", "")) if pitcher is not None else "",
                "pitcher_opponent": pitcher_opp,
                "hitters_vs_pitcher": hitters_vs_pitcher,
                "max_hitters_from_team": int(team_counts.max()) if not team_counts.empty else 0,
                "unconfirmed_hitters": int((~hitters["is_confirmed_lineup"]).sum()),
                "chalk_batters": int(batter_chalk.sum()),
                "chalk_pitcher": bool(pitcher_chalk),
                "player_count": int(group["fd_player_id"].nunique()),
                "duplicate_key": "|".join(ids),
            }
        )
    checks = pd.DataFrame(rows).sort_values("lineup_id")
    if checks.empty:
        return checks
    dup_counts = checks["duplicate_key"].value_counts()
    checks["duplicate_count"] = checks["duplicate_key"].map(dup_counts).fillna(1).astype(int)
    return checks


def _player_exposure(
    df: pd.DataFrame,
    batter_chalk_threshold: float,
    pitcher_chalk_threshold: float,
) -> pd.DataFrame:
    lineup_count = max(1, int(df["lineup_id"].nunique()))
    grouped = (
        df.groupby(["fd_player_id", "full_name", "team_code", "position", "player_type"])
        .agg(
            lineups=("lineup_id", "nunique"),
            projection=("proj_fd_mean", "mean"),
            ownership=("proj_fd_ownership", "mean"),
            leverage=("player_leverage_score", "mean"),
            confirmed=("is_confirmed_lineup", "max"),
        )
        .reset_index()
    )
    grouped["exposure_pct"] = grouped["lineups"] / lineup_count
    is_pitcher = grouped["player_type"].astype(str).str.lower() == "pitcher"
    grouped["chalk_threshold"] = batter_chalk_threshold
    grouped.loc[is_pitcher, "chalk_threshold"] = pitcher_chalk_threshold
    grouped["is_chalk"] = grouped["ownership"] >= grouped["chalk_threshold"]
    return grouped.sort_values(["lineups", "ownership"], ascending=[False, False])


def _stack_exposure(df: pd.DataFrame, stack_plan: Optional[pd.DataFrame]) -> pd.DataFrame:
    hitters = df[df["player_type"].str.lower() != "pitcher"]
    if hitters.empty:
        return pd.DataFrame()
    lineup_count = max(1, int(df["lineup_id"].nunique()))
    team_lineups = (
        hitters.groupby(["lineup_id", "team_code"])
        .size()
        .reset_index(name="hitters")
    )
    stacks = team_lineups[team_lineups["hitters"] >= 3]
    if stacks.empty:
        return pd.DataFrame()
    exposure = (
        stacks.groupby("team_code")
        .agg(
            stack_lineups=("lineup_id", "nunique"),
            avg_stack_size=("hitters", "mean"),
            max_stack_size=("hitters", "max"),
        )
        .reset_index()
    )
    exposure["exposure_pct"] = exposure["stack_lineups"] / lineup_count
    if stack_plan is not None and not stack_plan.empty and "team_code" in stack_plan.columns:
        keep_cols = [
            col
            for col in [
                "team_code",
                "tier",
                "target_exposure",
                "recommended_min_exposure",
                "recommended_max_exposure",
                "stack_score",
                "stack_leverage_score",
                "p_runs_ge_6",
                "stack_ownership",
            ]
            if col in stack_plan.columns
        ]
        exposure = exposure.merge(stack_plan[keep_cols], on="team_code", how="left")
        exposure["target_delta"] = exposure["exposure_pct"] - exposure.get("target_exposure", 0.0)
    return exposure.sort_values("exposure_pct", ascending=False)


def _audit_summary(
    df: pd.DataFrame,
    lineup_checks: pd.DataFrame,
    player_exposure: pd.DataFrame,
    stack_exposure: pd.DataFrame,
    contest_df: Optional[pd.DataFrame],
    selected_ids: Optional[Iterable[int]],
) -> Dict[str, float]:
    lineup_count = int(df["lineup_id"].nunique())
    summary: Dict[str, float] = {
        "lineups": lineup_count,
        "unique_players": int(df["fd_player_id"].nunique()),
        "avg_salary": float(lineup_checks["salary"].mean()) if not lineup_checks.empty else 0.0,
        "min_salary": float(lineup_checks["salary"].min()) if not lineup_checks.empty else 0.0,
        "avg_projection": float(lineup_checks["projection"].mean()) if not lineup_checks.empty else 0.0,
        "avg_lineup_ownership": float(lineup_checks["ownership"].mean()) if not lineup_checks.empty else 0.0,
        "avg_leverage": float(lineup_checks["avg_leverage"].mean()) if not lineup_checks.empty else 0.0,
        "duplicate_lineups": int((lineup_checks["duplicate_count"] > 1).sum()) if not lineup_checks.empty else 0,
        "lineups_with_hitters_vs_pitcher": int((lineup_checks["hitters_vs_pitcher"] > 0).sum()) if not lineup_checks.empty else 0,
        "lineups_with_unconfirmed_hitters": int((lineup_checks["unconfirmed_hitters"] > 0).sum()) if not lineup_checks.empty else 0,
        "teams_stacked": int(stack_exposure["team_code"].nunique()) if not stack_exposure.empty else 0,
    }
    if not player_exposure.empty:
        chalk = player_exposure[player_exposure["is_chalk"]]
        summary["chalk_players_used"] = int(len(chalk))
        summary["max_player_exposure"] = float(player_exposure["exposure_pct"].max())
    if contest_df is not None and not contest_df.empty and selected_ids:
        selected_zero_based = {int(idx) - 1 for idx in selected_ids}
        selected_rows = contest_df[contest_df["lineup_id"].astype(int).isin(selected_zero_based)]
        if not selected_rows.empty and "field_duplication_rate" in selected_rows.columns:
            summary["avg_duplication_rate"] = float(
                pd.to_numeric(selected_rows["field_duplication_rate"], errors="coerce").fillna(0.0).mean()
            )
    return summary


def _audit_issues(
    df: pd.DataFrame,
    lineup_checks: pd.DataFrame,
    player_exposure: pd.DataFrame,
    stack_exposure: pd.DataFrame,
    salary_cap: int,
) -> pd.DataFrame:
    rows = []
    if lineup_checks.empty:
        return _empty_issues()

    duplicate_lineups = int((lineup_checks["duplicate_count"] > 1).sum())
    if duplicate_lineups:
        rows.append(_issue("High", "Duplicates", f"{duplicate_lineups} lineup rows duplicate another lineup.", duplicate_lineups, "Generate a larger pool or loosen exposure constraints."))

    conflict_count = int((lineup_checks["hitters_vs_pitcher"] > 0).sum())
    if conflict_count:
        rows.append(_issue("High", "Pitcher conflicts", f"{conflict_count} lineups include hitter(s) against their pitcher.", conflict_count, "Exclude pitcher-opponent hitter combinations before upload."))

    team_limit_count = int((lineup_checks["max_hitters_from_team"] > 4).sum())
    if team_limit_count:
        rows.append(_issue("High", "Roster rules", f"{team_limit_count} lineups exceed FanDuel's four-hitter team limit.", team_limit_count, "Regenerate affected lineups."))

    short_salary = int((lineup_checks["salary"] < salary_cap - 1500).sum())
    if short_salary:
        rows.append(_issue("Medium", "Salary usage", f"{short_salary} lineups leave more than $1,500 unused.", short_salary, "Review whether the salary savings are intentional leverage."))

    unconfirmed = int((lineup_checks["unconfirmed_hitters"] > 0).sum())
    if unconfirmed:
        rows.append(_issue("Medium", "Confirmed starters", f"{unconfirmed} lineups include at least one unconfirmed hitter.", unconfirmed, "Refresh lineups or confirm those players before lock."))

    if not stack_exposure.empty and "recommended_max_exposure" in stack_exposure.columns:
        over = stack_exposure[
            pd.to_numeric(stack_exposure["recommended_max_exposure"], errors="coerce").fillna(1.0)
            < stack_exposure["exposure_pct"]
        ]
        for _, row in over.iterrows():
            rows.append(
                _issue(
                    "Medium",
                    "Stack exposure",
                    f"{row['team_code']} stack exposure is {row['exposure_pct']:.1%}, above the recommended max {row['recommended_max_exposure']:.1%}.",
                    int(row["stack_lineups"]),
                    "Trim this team or raise its cap only if you want a concentrated stand.",
                )
            )

    if not player_exposure.empty:
        over_chalk = player_exposure[
            (player_exposure["is_chalk"])
            & (player_exposure["exposure_pct"] >= 0.45)
            & (player_exposure["player_type"].astype(str).str.lower() != "pitcher")
        ]
        for _, row in over_chalk.head(10).iterrows():
            rows.append(
                _issue(
                    "Low",
                    "Chalk exposure",
                    f"{row['full_name']} is chalky and appears in {row['exposure_pct']:.1%} of lineups.",
                    int(row["lineups"]),
                    "Keep only if this is intentional chalk or a correlated stack piece.",
                )
            )

    return pd.DataFrame(rows) if rows else _empty_issues()


def _issue(severity: str, category: str, message: str, lineup_count: int, recommendation: str) -> Dict:
    return {
        "severity": severity,
        "category": category,
        "message": message,
        "lineup_count": lineup_count,
        "recommendation": recommendation,
    }


def _empty_issues() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["severity", "category", "message", "lineup_count", "recommendation"]
    )


__all__ = [
    "PortfolioAuditReport",
    "audit_report_to_csv",
    "build_portfolio_audit",
]
