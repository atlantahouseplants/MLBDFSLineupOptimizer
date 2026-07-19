"""Late-news and scratch risk checks for active MLB DFS portfolios."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from slate_optimizer.ingestion.text_utils import canonicalize_series


@dataclass
class LateNewsReport:
    issues: pd.DataFrame
    affected_lineup_ids: list[int]
    scratched_players: list[str]


def build_late_news_report(
    lineup_df: pd.DataFrame,
    optimizer_df: Optional[pd.DataFrame] = None,
    fresh_orders: Optional[pd.DataFrame] = None,
    scratched_players: Optional[Iterable[str]] = None,
    now: Optional[pd.Timestamp] = None,
    lock_warning_minutes: int = 60,
) -> LateNewsReport:
    """Flag scratches, non-starters, order changes, and lock-state risks."""

    if lineup_df is None or lineup_df.empty:
        return LateNewsReport(pd.DataFrame(), [], [])

    df = lineup_df.copy()
    _normalize_lineup_columns(df)
    now = now or pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")

    fresh_lookup = _fresh_order_lookup(fresh_orders)
    teams_with_posted_lineups = set(fresh_lookup["team_code"].unique()) if not fresh_lookup.empty else set()
    scratch_names = {str(name).strip().lower() for name in scratched_players or [] if str(name).strip()}

    issues = []
    affected: set[int] = set()
    inferred_scratches: set[str] = set()

    for _, row in df.iterrows():
        lineup_id = int(row.get("lineup_id", 0) or 0)
        player_name = str(row.get("full_name", ""))
        team = str(row.get("team_code", "")).upper()
        player_type = str(row.get("player_type", "")).lower()
        game_time = pd.to_datetime(row.get("game_start_time"), errors="coerce", utc=True)
        locked = bool(pd.notna(game_time) and game_time <= now)
        locking_soon = bool(
            pd.notna(game_time)
            and game_time > now
            and game_time <= now + pd.Timedelta(minutes=max(1, int(lock_warning_minutes)))
        )

        if player_name.lower() in scratch_names:
            affected.add(lineup_id)
            inferred_scratches.add(player_name)
            issues.append(
                _issue(
                    "High",
                    "Manual scratch",
                    lineup_id,
                    player_name,
                    team,
                    "Player is on your scratch list.",
                    "Locked lineup cannot be changed." if locked else "Re-optimize affected lineups before lock.",
                    locked,
                )
            )
            continue

        if player_type == "pitcher":
            continue

        canonical = str(row.get("_canonical_name", ""))
        fresh_team = fresh_lookup[fresh_lookup["team_code"] == team] if not fresh_lookup.empty else pd.DataFrame()
        if team in teams_with_posted_lineups and canonical:
            match = fresh_team[fresh_team["canonical_name"] == canonical]
            if match.empty:
                affected.add(lineup_id)
                inferred_scratches.add(player_name)
                issues.append(
                    _issue(
                        "High",
                        "Not in posted lineup",
                        lineup_id,
                        player_name,
                        team,
                        "Team lineup is posted and this hitter is not in it.",
                        "Locked lineup cannot be changed." if locked else "Remove or re-optimize affected lineups.",
                        locked,
                    )
                )
            else:
                new_order = int(match.iloc[0]["batting_order_position"])
                old_order = row.get("batting_order_position")
                try:
                    old_order_int = int(float(old_order))
                except (TypeError, ValueError):
                    old_order_int = 0
                if old_order_int and new_order and old_order_int != new_order:
                    affected.add(lineup_id)
                    issues.append(
                        _issue(
                            "Medium",
                            "Order changed",
                            lineup_id,
                            player_name,
                            team,
                            f"Batting order changed from {old_order_int} to {new_order}.",
                            "Review projection/order assumptions before upload.",
                            locked,
                        )
                    )
        elif not bool(row.get("is_confirmed_lineup", False)) and (locking_soon or locked):
            affected.add(lineup_id)
            issues.append(
                _issue(
                    "Medium" if not locked else "High",
                    "Unconfirmed starter",
                    lineup_id,
                    player_name,
                    team,
                    "Hitter is not confirmed and the game is locked or approaching lock.",
                    "Confirm status or swap before lock." if not locked else "Locked lineup cannot be changed.",
                    locked,
                )
            )

    issue_df = pd.DataFrame(issues)
    if not issue_df.empty:
        issue_df = issue_df.sort_values(["severity_rank", "lineup_id", "player"]).drop(columns=["severity_rank"])
    return LateNewsReport(
        issues=issue_df,
        affected_lineup_ids=sorted(idx for idx in affected if idx),
        scratched_players=sorted(inferred_scratches),
    )


def _normalize_lineup_columns(df: pd.DataFrame) -> None:
    for col in ("full_name", "team_code", "player_type"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)
    if "lineup_id" not in df.columns:
        df["lineup_id"] = 0
    if "is_confirmed_lineup" not in df.columns:
        df["is_confirmed_lineup"] = False
    if "game_start_time" not in df.columns:
        df["game_start_time"] = pd.NaT
    if "batting_order_position" not in df.columns:
        df["batting_order_position"] = 0
    df["_canonical_name"] = canonicalize_series(df["full_name"])


def _fresh_order_lookup(fresh_orders: Optional[pd.DataFrame]) -> pd.DataFrame:
    if fresh_orders is None or fresh_orders.empty:
        return pd.DataFrame(columns=["team_code", "canonical_name", "batting_order_position"])
    orders = fresh_orders.copy()
    team_col = "team_code" if "team_code" in orders.columns else "team"
    name_col = "player_name" if "player_name" in orders.columns else "full_name"
    order_col = "order_position" if "order_position" in orders.columns else "batting_order_position"
    if team_col not in orders.columns or name_col not in orders.columns or order_col not in orders.columns:
        return pd.DataFrame(columns=["team_code", "canonical_name", "batting_order_position"])
    orders["team_code"] = orders[team_col].astype(str).str.upper().str.strip()
    orders["canonical_name"] = canonicalize_series(orders[name_col])
    orders["batting_order_position"] = pd.to_numeric(orders[order_col], errors="coerce").fillna(0).astype(int)
    orders = orders[orders["batting_order_position"].between(1, 9)]
    return orders[["team_code", "canonical_name", "batting_order_position"]].drop_duplicates()


def _issue(
    severity: str,
    category: str,
    lineup_id: int,
    player: str,
    team: str,
    message: str,
    recommendation: str,
    locked: bool,
) -> dict:
    rank = {"High": 0, "Medium": 1, "Low": 2}.get(severity, 3)
    return {
        "severity_rank": rank,
        "severity": severity,
        "category": category,
        "lineup_id": lineup_id,
        "player": player,
        "team": team,
        "locked": bool(locked),
        "message": message,
        "recommendation": recommendation,
    }


__all__ = ["LateNewsReport", "build_late_news_report"]
