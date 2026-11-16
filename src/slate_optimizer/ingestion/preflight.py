"""Pre-flight checks: stop a build that is about to use the wrong inputs.

Every check here corresponds to something that actually went wrong in 2026:

* the NFL entries template was fed to the MLB pipeline and surfaced as an
  unreadable pandas error an hour later;
* about ten cron runs used a FanDuel players list from a different day
  (salaries and contest-scoped player ids were stale);
* the odds feed kept quoting games after first pitch, so in-play lines
  (a 1.5 "total") flowed into the projection multiplier.

A blocking problem raises before any lineups are built, with a message a
non-developer can act on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Set

import pandas as pd

MLB_POSITIONS = {"P", "C", "1B", "2B", "3B", "SS", "OF", "UTIL"}
OTHER_SPORT_POSITIONS = {
    "NFL": {"QB", "RB", "WR", "TE", "K", "D", "DEF", "DST", "FLEX", "MVP"},
    "NBA": {"PG", "SG", "SF", "PF"},
    "NHL": {"W", "G", "LW", "RW"},
}
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
SALARY_CAP = 35_000
VEGAS_TOTAL_RANGE = (5.5, 15.0)

_FILENAME_SPORT = re.compile(r"FanDuel[-_ ]([A-Za-z]+)[-_ ]", re.IGNORECASE)
# FanDuel writes "2026 EDT-08 EDT-07" (players list) or "2026-08-07" (template).
_FILENAME_DATE = re.compile(r"(20\d{2})\D{1,6}?(\d{2})\D{1,6}?(\d{2})(?!\d)")


def normalize_team_code(value: object) -> str:
    code = str(value or "").strip().upper()
    return TEAM_ALIASES.get(code, code)


@dataclass
class PreflightIssue:
    severity: str  # "error" blocks the run, "warning" does not
    code: str
    message: str


@dataclass
class PreflightReport:
    issues: List[PreflightIssue] = field(default_factory=list)
    fanduel_date: Optional[str] = None
    slate_date: Optional[str] = None
    fanduel_games: int = 0
    matched_games: int = 0

    @property
    def errors(self) -> List[PreflightIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> List[PreflightIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, severity: str, code: str, message: str) -> None:
        self.issues.append(PreflightIssue(severity, code, message))

    def messages(self) -> List[str]:
        prefix = {"error": "STOP", "warning": "CHECK"}
        return [f"{prefix.get(issue.severity, issue.severity.upper())}: {issue.message}" for issue in self.issues]

    def format(self) -> str:
        if not self.issues:
            return (
                f"Pre-flight OK: {self.fanduel_games} FanDuel games, all matched to BallparkPal"
                + (f" ({self.slate_date})" if self.slate_date else "")
                + "."
            )
        return "Pre-flight:\n" + "\n".join(f"  {line}" for line in self.messages())

    def raise_if_blocking(self) -> None:
        if self.errors:
            raise PreflightError(self)


class PreflightError(ValueError):
    def __init__(self, report: PreflightReport):
        self.report = report
        lines = [issue.message for issue in report.errors]
        super().__init__(
            "The inputs do not look right, so no lineups were built.\n- " + "\n- ".join(lines)
        )


def fanduel_filename_sport(path: Path | str | None) -> Optional[str]:
    if not path:
        return None
    match = _FILENAME_SPORT.search(Path(path).name)
    return match.group(1).upper() if match else None


def fanduel_filename_date(path: Path | str | None) -> Optional[str]:
    """Slate date encoded in a FanDuel export filename, as YYYY-MM-DD."""
    if not path:
        return None
    match = _FILENAME_DATE.search(Path(path).name)
    if not match:
        return None
    year, month, day = match.groups()
    if not (1 <= int(month) <= 12 and 1 <= int(day) <= 31):
        return None
    return f"{year}-{month}-{day}"


def _position_tokens(values: Iterable[object]) -> Set[str]:
    tokens: Set[str] = set()
    for value in values:
        for token in str(value or "").upper().replace("-", "/").split("/"):
            token = token.strip()
            if token:
                tokens.add(token)
    return tokens


def _game_keys(left: Iterable[object], right: Iterable[object]) -> Set[frozenset]:
    keys: Set[frozenset] = set()
    for a, b in zip(left, right):
        a_code, b_code = normalize_team_code(a), normalize_team_code(b)
        if a_code and b_code and a_code != b_code:
            keys.add(frozenset((a_code, b_code)))
    return keys


def _label(game: frozenset) -> str:
    return " vs ".join(sorted(game))


def run_preflight(
    fd_players: pd.DataFrame,
    fd_path: Path | str | None = None,
    bpp_games: Optional[pd.DataFrame] = None,
    vegas_games: Optional[pd.DataFrame] = None,
    expected_date: Optional[str] = None,
    salary_cap: int = SALARY_CAP,
) -> PreflightReport:
    """Check the FanDuel file, its match to BallparkPal, and the Vegas lines."""
    report = PreflightReport()
    players = fd_players if fd_players is not None else pd.DataFrame()

    # 1. Sport --------------------------------------------------------------
    filename_sport = fanduel_filename_sport(fd_path)
    if filename_sport and filename_sport != "MLB":
        report.add(
            "error",
            "wrong_sport",
            f"This is a FanDuel {filename_sport} file ({Path(fd_path).name}). Download the MLB file for today's slate.",
        )
    position_col = "roster_position" if "roster_position" in players.columns else "position"
    tokens = _position_tokens(players[position_col]) if position_col in players.columns else set()
    if position_col in players.columns and "position" in players.columns:
        tokens |= _position_tokens(players["position"])
    for sport, markers in OTHER_SPORT_POSITIONS.items():
        hits = tokens & markers
        if len(hits) >= 2 and not ({"SS", "OF", "2B", "3B"} & tokens):
            report.add(
                "error",
                "wrong_sport",
                f"The player list has {sport} positions ({', '.join(sorted(hits))}) and no baseball positions. "
                "This is not an MLB slate file.",
            )
            break
    if players.empty:
        report.add("error", "empty_file", "The FanDuel file has no players in it.")
        return report

    # 2. Structure ----------------------------------------------------------
    if "fd_player_id" in players.columns:
        ids = players["fd_player_id"].astype(str)
        duplicates = int(ids.duplicated().sum())
        if duplicates:
            report.add(
                "error",
                "duplicate_players",
                f"{duplicates} players appear more than once in the FanDuel file. That usually means two slates "
                "were combined or the file is not a single-slate export.",
            )
    needed = {"P": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3}
    if not report.errors and position_col in players.columns:
        position_text = players[position_col].astype(str).str.upper()
        for slot, count in needed.items():
            have = int(position_text.map(lambda text, s=slot: s in _position_tokens([text])).sum())
            if have < count:
                report.add("error", "missing_position", f"Only {have} players can fill {slot}; a lineup needs {count}.")
        first_base = int(position_text.map(lambda text: bool({"C", "1B"} & _position_tokens([text]))).sum())
        if first_base < 1:
            report.add("error", "missing_position", "No player can fill the C/1B slot.")
    if "salary" in players.columns and position_col in players.columns and not report.errors:
        salary = pd.to_numeric(players["salary"], errors="coerce").fillna(0)
        is_pitcher = players[position_col].astype(str).str.upper().map(lambda text: "P" in _position_tokens([text]))
        hitters = salary[~is_pitcher & (salary > 0)].nsmallest(8)
        pitchers = salary[is_pitcher & (salary > 0)]
        if len(hitters) == 8 and len(pitchers):
            cheapest = float(hitters.sum() + pitchers.min())
            if cheapest > salary_cap:
                report.add(
                    "error",
                    "salary_infeasible",
                    f"Even the cheapest possible lineup costs ${cheapest:,.0f}, over the ${salary_cap:,} cap. "
                    "The salaries in this file are not FanDuel MLB salaries.",
                )

    # 3. Is this file for the slate BallparkPal has? -------------------------
    report.fanduel_date = fanduel_filename_date(fd_path)
    if bpp_games is not None and not bpp_games.empty and "game_date" in bpp_games.columns:
        dates = pd.to_datetime(bpp_games["game_date"], errors="coerce").dropna()
        if not dates.empty:
            report.slate_date = dates.dt.strftime("%Y-%m-%d").mode().iloc[0]
    reference_date = expected_date or report.slate_date
    if report.fanduel_date and reference_date and report.fanduel_date != reference_date:
        report.add(
            "error",
            "date_mismatch",
            f"The FanDuel file is for {report.fanduel_date} but the BallparkPal data is for {reference_date}. "
            "Salaries and FanDuel player ids change every slate - download today's FanDuel file.",
        )

    fd_games: Set[frozenset] = set()
    if {"team", "opponent"}.issubset(players.columns):
        fd_games = _game_keys(players["team"], players["opponent"])
    report.fanduel_games = len(fd_games)
    if fd_games and bpp_games is not None and {"away_team", "home_team"}.issubset(bpp_games.columns):
        bpp_keys = _game_keys(bpp_games["away_team"], bpp_games["home_team"])
        matched = fd_games & bpp_keys
        report.matched_games = len(matched)
        missing = sorted(_label(game) for game in fd_games - bpp_keys)
        share = len(matched) / len(fd_games)
        if share < 0.6:
            report.add(
                "error",
                "slate_mismatch",
                f"Only {len(matched)} of the {len(fd_games)} games in the FanDuel file are in the BallparkPal data "
                f"(missing: {', '.join(missing[:6])}{'...' if len(missing) > 6 else ''}). "
                "The two files are for different slates.",
            )
        elif missing:
            report.add(
                "warning",
                "games_without_projections",
                f"{len(missing)} FanDuel game(s) have no BallparkPal projections and their players will be "
                f"unprojected: {', '.join(missing)}.",
            )

    # 4. Vegas lines --------------------------------------------------------
    if vegas_games is not None and not vegas_games.empty and "total" in vegas_games.columns:
        totals = pd.to_numeric(vegas_games["total"], errors="coerce")
        low, high = VEGAS_TOTAL_RANGE
        bad = vegas_games[totals.notna() & ~totals.between(low, high)]
        if not bad.empty:
            names = ", ".join(str(value) for value in bad.get("game", pd.Series(dtype=str)).astype(str).head(6))
            report.add(
                "warning",
                "in_play_lines",
                f"{len(bad)} Vegas line(s) have a total outside {low}-{high} runs ({names}). Those are in-play "
                "or broken lines from games already under way; they are ignored for projections.",
            )
        if "game" in vegas_games.columns:
            repeated = int(vegas_games["game"].astype(str).duplicated().sum())
            if repeated:
                report.add(
                    "warning",
                    "repeated_lines",
                    f"{repeated} matchup(s) appear twice in the Vegas file (tomorrow's game of the same series). "
                    "Only the first line for each matchup is used.",
                )
    return report


__all__ = [
    "PreflightError",
    "PreflightIssue",
    "PreflightReport",
    "fanduel_filename_date",
    "fanduel_filename_sport",
    "normalize_team_code",
    "run_preflight",
]
