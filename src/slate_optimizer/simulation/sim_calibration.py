"""Score simulation calibration against FanDuel My Entries results.

Joins a past run's ``*_sim_results.csv`` and the matching
``*_simulated_upload.csv`` to Geoff's FanDuel My Entries export, then
computes Brier scores (cash / top-1%), calibration-by-decile, and Spearman
rank correlation. Each scored slate is appended to
``data/output/sim_calibration_history.csv``.

FanDuel export headers vary; column mapping is flexible and fails loudly
(printing the actual headers) when rank/points cannot be resolved.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

HISTORY_COLUMNS = ["tag", "n_lineups", "brier_cash", "brier_top1", "spearman", "notes"]
DEFAULT_CASH_LINE = 0.20
DEFAULT_SELECTION_METRIC = "top_1pct_rate"
ROSTER_SLOTS = ("P", "C/1B", "2B", "3B", "SS", "OF", "UTIL")

_FD_ID_RE = re.compile(r"(\d{4,}-\d+)")
_ROSTER_COL_RE = re.compile(r"^(p|c|c_1b|1b|2b|3b|ss|of|util)(_?\d+)?$")
_RANK_ALIASES = (
    "rank",
    "place",
    "finish",
    "finishing_position",
    "entry_rank",
    "contest_rank",
    "standing",
)
_POINTS_ALIASES = (
    "points",
    "score",
    "fpts",
    "fantasy_points",
    "total_points",
    "fpts_total",
    "entry_points",
)
_LINEUP_ID_ALIASES = ("lineup_id", "lineupid", "entry_id", "entryid")
_WINNINGS_ALIASES = ("winnings", "prize", "payout", "won", "winning")
_FIELD_SIZE_ALIASES = (
    "field_size",
    "contest_size",
    "contest_entries",
    "entries",
    "entry_count",
    "n_entries",
)
# "position" is intentionally omitted from rank aliases — it collides with
# roster labels. "player" columns are roster, not identifiers.


class UnmatchedHeaderError(ValueError):
    """Raised when a required FanDuel entries column cannot be resolved."""


@dataclass
class CalibrationResult:
    tag: str
    n_lineups: int
    n_entered: int
    n_matched: int
    field_size: int
    brier_cash: float
    brier_top1: float
    spearman: float
    selection_metric: str
    cash_line: float
    notes: str = ""
    deciles: pd.DataFrame = field(default_factory=pd.DataFrame)
    joined: pd.DataFrame = field(default_factory=pd.DataFrame)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)

    def history_row(self) -> dict[str, object]:
        return {
            "tag": self.tag,
            "n_lineups": int(self.n_lineups),
            "brier_cash": _fmt_metric(self.brier_cash),
            "brier_top1": _fmt_metric(self.brier_top1),
            "spearman": _fmt_metric(self.spearman),
            "notes": self.notes or "",
        }


def lineup_signature(players: Iterable[object]) -> frozenset[str]:
    """Exact player-set signature (order-independent)."""
    tokens = [_canonical_token(player) for player in players]
    return frozenset(token for token in tokens if token)


def parse_player_ids_cell(value: object) -> list[str]:
    """Parse ``player_ids`` from sim-results (list literal, CSV, or scalar)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, np.ndarray):
        return [str(item).strip() for item in value.tolist() if str(item).strip()]
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None
    if isinstance(parsed, (list, tuple, set)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return _split_player_cell(text)


def extract_player_tokens(row: pd.Series, columns: Iterable[object] | None = None) -> list[str]:
    """Pull player IDs/names from roster cells on a row."""
    cols = list(columns) if columns is not None else list(row.index)
    tokens: list[str] = []
    for col in cols:
        if col not in row.index:
            continue
        tokens.extend(_split_player_cell(row[col]))
    return tokens


def resolve_entries_columns(
    entries: pd.DataFrame,
    *,
    rank_col: str | None = None,
    points_col: str | None = None,
    lineup_id_col: str | None = None,
    winnings_col: str | None = None,
    field_size_col: str | None = None,
) -> dict[str, str | None]:
    """Map FanDuel My Entries headers onto rank/points/(optional) extras.

    Explicit CLI names win. Otherwise aliases are matched case-insensitively.
    Rank and points are required; failure prints the actual headers.
    """
    headers = [str(col) for col in entries.columns]
    resolved = {
        "rank": _resolve_column(entries, rank_col, _RANK_ALIASES, "rank", required=True),
        "points": _resolve_column(entries, points_col, _POINTS_ALIASES, "points", required=True),
        "lineup_id": _resolve_column(
            entries, lineup_id_col, _LINEUP_ID_ALIASES, "lineup_id", required=False
        ),
        "winnings": _resolve_column(
            entries, winnings_col, _WINNINGS_ALIASES, "winnings", required=False
        ),
        "field_size": _resolve_column(
            entries, field_size_col, _FIELD_SIZE_ALIASES, "field_size", required=False
        ),
    }
    roster_cols = _roster_columns(entries.columns)
    resolved["roster"] = ",".join(str(col) for col in roster_cols) if roster_cols else None
    _ = headers  # retained for call-site error messages
    return resolved


def brier_score(predicted: Iterable[float], actual: Iterable[float]) -> float:
    """Mean squared error between predicted probabilities and 0/1 outcomes."""
    pred = pd.to_numeric(pd.Series(list(predicted)), errors="coerce")
    act = pd.to_numeric(pd.Series(list(actual)), errors="coerce")
    mask = pred.notna() & act.notna()
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean((pred[mask].to_numpy(dtype=float) - act[mask].to_numpy(dtype=float)) ** 2))


def calibration_by_decile(
    predicted: Iterable[float],
    actual: Iterable[float],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bucket lineups by predicted probability; mean predicted vs actual rate."""
    frame = pd.DataFrame(
        {
            "predicted": pd.to_numeric(pd.Series(list(predicted)), errors="coerce"),
            "actual": pd.to_numeric(pd.Series(list(actual)), errors="coerce"),
        }
    ).dropna()
    empty = pd.DataFrame(columns=["decile", "n", "predicted_mean", "actual_rate"])
    if frame.empty:
        return empty
    n_bins = max(1, min(int(n_bins), int(len(frame))))
    try:
        frame["decile"] = pd.qcut(frame["predicted"], q=n_bins, labels=False, duplicates="drop") + 1
    except ValueError:
        frame["decile"] = 1
    grouped = (
        frame.groupby("decile", sort=True)
        .agg(
            n=("predicted", "size"),
            predicted_mean=("predicted", "mean"),
            actual_rate=("actual", "mean"),
        )
        .reset_index()
    )
    grouped["decile"] = grouped["decile"].astype(int)
    grouped["n"] = grouped["n"].astype(int)
    return grouped


def spearman_correlation(predicted: Iterable[float], actual: Iterable[float]) -> float:
    """Spearman rank correlation; NaN if fewer than two paired observations."""
    pred = pd.to_numeric(pd.Series(list(predicted)), errors="coerce")
    act = pd.to_numeric(pd.Series(list(actual)), errors="coerce")
    mask = pred.notna() & act.notna()
    if int(mask.sum()) < 2:
        return float("nan")
    if pred[mask].nunique() < 2 or act[mask].nunique() < 2:
        return float("nan")
    return float(pred[mask].corr(act[mask], method="spearman"))


def load_sim_results(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Sim-results file is empty: {path}")
    return frame


def load_upload_lineups(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Upload file is empty: {path}")
    return frame


def load_entries(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Entries file is empty: {path}")
    return frame


def join_entered_lineups(
    sim_results: pd.DataFrame,
    upload: pd.DataFrame,
    entries: pd.DataFrame,
    *,
    rank_col: str | None = None,
    points_col: str | None = None,
    lineup_id_col: str | None = None,
    winnings_col: str | None = None,
    field_size_col: str | None = None,
    field_size: int | None = None,
    cash_line: float = DEFAULT_CASH_LINE,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
) -> pd.DataFrame:
    """Match entered upload lineups to sim rows and FanDuel results."""
    sim = _prepare_sim_results(sim_results, selection_metric=selection_metric)
    entered = _prepare_upload(upload)
    if entered.empty:
        raise ValueError("Upload CSV has no roster player cells to build lineup signatures.")

    # Prefer sim-results lineup_id; overlapping upload ids get a suffix that
    # would break the later entries join.
    upload_for_sig = entered.drop(columns=["lineup_id"]) if "lineup_id" in entered.columns else entered
    entered_sim = upload_for_sig.merge(sim, on="signature", how="inner", suffixes=("_upload", "_sim"))
    if entered_sim.empty and "lineup_id" in entered.columns and "lineup_id" in sim.columns:
        entered_sim = entered.merge(sim, on="lineup_id", how="inner", suffixes=("_upload", "_sim"))
    if entered_sim.empty:
        sample_upload = ", ".join(sorted(entered["signature"].astype(str).head(3)))
        sample_sim = ", ".join(sorted(sim["signature"].astype(str).head(3)))
        raise ValueError(
            "No overlap between upload lineups and sim-results. "
            f"Upload rows={len(entered)}, sim rows={len(sim)}. "
            f"Sample upload signatures: [{sample_upload}] "
            f"Sample sim signatures: [{sample_sim}]"
        )

    cols = resolve_entries_columns(
        entries,
        rank_col=rank_col,
        points_col=points_col,
        lineup_id_col=lineup_id_col,
        winnings_col=winnings_col,
        field_size_col=field_size_col,
    )
    scored = _prepare_entries(entries, cols)
    resolved_field = _resolve_field_size(scored, cols, field_size)
    scored["field_size"] = int(resolved_field)
    scored["finish_percentile"] = scored["rank"] / scored["field_size"]
    if cols["winnings"] is not None:
        scored["cashed"] = (scored["winnings"] > 0).astype(float)
    else:
        scored["cashed"] = (scored["finish_percentile"] <= float(cash_line)).astype(float)
    scored["top1"] = (scored["finish_percentile"] <= 0.01).astype(float)

    joined = _join_results(entered_sim, scored)
    if joined.empty:
        raise ValueError(
            "Upload/sim lineups did not match any FanDuel entries rows. "
            f"Entered={len(entered_sim)}, entries={len(scored)}. "
            "Check that the My Entries export is the same contest and that "
            "player cells contain FanDuel IDs or the same 9-player set."
        )
    joined["field_size"] = int(resolved_field)
    joined["finish_percentile"] = joined["rank"] / joined["field_size"]
    if "cashed" not in joined.columns:
        joined["cashed"] = (joined["finish_percentile"] <= float(cash_line)).astype(float)
    if "top1" not in joined.columns:
        joined["top1"] = (joined["finish_percentile"] <= 0.01).astype(float)
    return joined.reset_index(drop=True)


def score_calibration(
    sim_results: pd.DataFrame | Path | str,
    upload: pd.DataFrame | Path | str,
    entries: pd.DataFrame | Path | str,
    *,
    tag: str = "",
    points_col: str | None = None,
    rank_col: str | None = None,
    lineup_id_col: str | None = None,
    winnings_col: str | None = None,
    field_size: int | None = None,
    field_size_col: str | None = None,
    cash_line: float = DEFAULT_CASH_LINE,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
    history_path: Path | str | None = None,
    notes: str = "",
    append: bool = True,
) -> CalibrationResult:
    """Join, score, optionally append history, and return a result object."""
    sim_df = sim_results if isinstance(sim_results, pd.DataFrame) else load_sim_results(sim_results)
    upload_df = upload if isinstance(upload, pd.DataFrame) else load_upload_lineups(upload)
    entries_df = entries if isinstance(entries, pd.DataFrame) else load_entries(entries)
    if not tag:
        tag = _infer_tag(sim_results if not isinstance(sim_results, pd.DataFrame) else None)

    joined = join_entered_lineups(
        sim_df,
        upload_df,
        entries_df,
        rank_col=rank_col,
        points_col=points_col,
        lineup_id_col=lineup_id_col,
        winnings_col=winnings_col,
        field_size_col=field_size_col,
        field_size=field_size,
        cash_line=cash_line,
        selection_metric=selection_metric,
    )
    if selection_metric not in joined.columns:
        raise ValueError(
            f"Selection metric {selection_metric!r} missing after join. "
            f"Available: {list(joined.columns)}"
        )

    brier_cash = brier_score(joined["cash_rate"], joined["cashed"])
    brier_top1 = brier_score(joined["top_1pct_rate"], joined["top1"])
    spearman = spearman_correlation(joined[selection_metric], joined["finish_percentile"])
    deciles = calibration_by_decile(joined["cash_rate"], joined["cashed"])
    resolved_field = int(joined["field_size"].iloc[0]) if "field_size" in joined.columns else 0

    result = CalibrationResult(
        tag=str(tag),
        n_lineups=int(len(joined)),
        n_entered=int(len(upload_df)),
        n_matched=int(len(joined)),
        field_size=resolved_field,
        brier_cash=brier_cash,
        brier_top1=brier_top1,
        spearman=spearman,
        selection_metric=selection_metric,
        cash_line=float(cash_line),
        notes=notes or "",
        deciles=deciles,
        joined=joined,
    )
    if append and history_path is not None:
        result.history = append_history(history_path, result)
    elif history_path is not None:
        result.history = _read_history(Path(history_path))
    return result


def append_history(path: Path | str, result: CalibrationResult | dict[str, object]) -> pd.DataFrame:
    """Append one scored slate to the history CSV; returns the full table."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    row = result.history_row() if isinstance(result, CalibrationResult) else dict(result)
    incoming = pd.DataFrame([{col: row.get(col, "") for col in HISTORY_COLUMNS}])
    existing = _read_history(target)
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined.to_csv(target, index=False)
    return combined


def format_report(result: CalibrationResult) -> str:
    """Console-only calibration report (no dashboard)."""
    lines = [
        f"=== Sim Calibration: {result.tag or '(untagged)'} ===",
        f"Matched lineups: {result.n_matched}  (upload rows: {result.n_entered})",
        f"Field size: {result.field_size}",
        f"Cash line: top {result.cash_line:.0%}",
        f"Selection metric: {result.selection_metric}",
        f"Brier (cash): {_fmt_metric(result.brier_cash)}",
        f"Brier (top-1%): {_fmt_metric(result.brier_top1)}",
        (
            "Spearman (predicted metric vs finish percentile; "
            f"negative = better ranking): {_fmt_metric(result.spearman)}"
        ),
        "",
        "Calibration by cash_rate decile:",
    ]
    if result.deciles is None or result.deciles.empty:
        lines.append("  (no decile rows)")
    else:
        lines.append(result.deciles.to_string(index=False))
    if result.notes:
        lines.extend(["", f"Notes: {result.notes}"])
    history = result.history
    if history is not None and not history.empty:
        lines.extend(["", f"History ({len(history)} slate(s)):"] )
        numeric = history.copy()
        for col in ("brier_cash", "brier_top1", "spearman"):
            if col in numeric.columns:
                numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
        means = numeric[["brier_cash", "brier_top1", "spearman"]].mean(numeric_only=True)
        lines.append(
            "  running mean  brier_cash={bc}  brier_top1={bt}  spearman={sp}".format(
                bc=_fmt_metric(means.get("brier_cash", float("nan"))),
                bt=_fmt_metric(means.get("brier_top1", float("nan"))),
                sp=_fmt_metric(means.get("spearman", float("nan"))),
            )
        )
        lines.append(history[HISTORY_COLUMNS].to_string(index=False))
    lines.append("")
    return "\n".join(lines)


def _prepare_sim_results(sim_results: pd.DataFrame, selection_metric: str) -> pd.DataFrame:
    required = ["cash_rate", "top_1pct_rate", selection_metric]
    missing = [col for col in required if col not in sim_results.columns]
    if missing:
        raise UnmatchedHeaderError(
            _header_error(
                sim_results,
                role=", ".join(missing),
                explicit=None,
                aliases=required,
                required=True,
            )
        )
    frame = sim_results.copy()
    if "lineup_id" in frame.columns:
        frame["lineup_id"] = frame["lineup_id"]
    else:
        frame["lineup_id"] = pd.RangeIndex(start=1, stop=len(frame) + 1)
    if "player_ids" in frame.columns:
        signatures = frame["player_ids"].map(lambda cell: _signature_key(parse_player_ids_cell(cell)))
    else:
        roster = _roster_columns(frame.columns)
        if not roster:
            raise UnmatchedHeaderError(
                _header_error(
                    frame,
                    role="player_ids or roster slots",
                    explicit=None,
                    aliases=("player_ids",) + ROSTER_SLOTS,
                    required=True,
                )
            )
        signatures = frame.apply(
            lambda row: _signature_key(extract_player_tokens(row, roster)), axis=1
        )
    frame["signature"] = signatures
    keep = ["lineup_id", "signature", "cash_rate", "top_1pct_rate"]
    if selection_metric not in keep:
        keep.append(selection_metric)
    if "win_rate" in frame.columns:
        keep.append("win_rate")
    if "expected_roi" in frame.columns:
        keep.append("expected_roi")
    out = frame[keep].copy()
    for col in ("cash_rate", "top_1pct_rate", selection_metric):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[out["signature"].astype(str).str.len() > 0]
    return out.drop_duplicates(subset=["signature"], keep="first")


def _prepare_upload(upload: pd.DataFrame) -> pd.DataFrame:
    frame = upload.copy()
    roster = _roster_columns(frame.columns)
    if not roster and "player_ids" in frame.columns:
        frame["signature"] = frame["player_ids"].map(
            lambda cell: _signature_key(parse_player_ids_cell(cell))
        )
    elif roster:
        frame["signature"] = frame.apply(
            lambda row: _signature_key(extract_player_tokens(row, roster)), axis=1
        )
    else:
        raise UnmatchedHeaderError(
            _header_error(
                frame,
                role="upload roster slots",
                explicit=None,
                aliases=ROSTER_SLOTS,
                required=True,
            )
        )
    if "lineup_id" in frame.columns:
        frame["lineup_id"] = frame["lineup_id"]
    frame = frame[frame["signature"].astype(str).str.len() > 0].copy()
    keep = ["signature"]
    if "lineup_id" in frame.columns:
        keep.append("lineup_id")
    return frame[keep].reset_index(drop=True)


def _prepare_entries(entries: pd.DataFrame, cols: dict[str, str | None]) -> pd.DataFrame:
    rank_name = cols["rank"]
    points_name = cols["points"]
    assert rank_name is not None and points_name is not None
    frame = entries.copy()
    frame["rank"] = pd.to_numeric(frame[rank_name], errors="coerce")
    frame["points"] = pd.to_numeric(frame[points_name], errors="coerce")
    if cols["winnings"] is not None:
        frame["winnings"] = frame[cols["winnings"]].map(_parse_money)
    if cols["lineup_id"] is not None:
        frame["lineup_id"] = frame[cols["lineup_id"]]
    if cols["field_size"] is not None:
        frame["_field_size_col"] = pd.to_numeric(frame[cols["field_size"]], errors="coerce")
    roster = _roster_columns(frame.columns)
    if roster:
        frame["signature"] = frame.apply(
            lambda row: _signature_key(extract_player_tokens(row, roster)), axis=1
        )
    elif "player_ids" in frame.columns:
        frame["signature"] = frame["player_ids"].map(
            lambda cell: _signature_key(parse_player_ids_cell(cell))
        )
    elif "lineup_id" not in frame.columns:
        raise UnmatchedHeaderError(
            _header_error(
                frame,
                role="roster slots or lineup_id",
                explicit=None,
                aliases=ROSTER_SLOTS + ("lineup_id",),
                required=True,
            )
        )
    else:
        frame["signature"] = ""
    frame = frame[frame["rank"].notna()].copy()
    return frame.reset_index(drop=True)


def _join_results(entered_sim: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    score_keep = ["rank", "points", "finish_percentile", "cashed", "top1", "field_size"]
    if "winnings" in scored.columns:
        score_keep.append("winnings")
    if "signature" in scored.columns:
        score_keep.append("signature")
    if "lineup_id" in scored.columns:
        score_keep.append("lineup_id")

    by_id = pd.DataFrame()
    if "lineup_id" in entered_sim.columns and "lineup_id" in scored.columns:
        left_ids = entered_sim["lineup_id"].dropna()
        right_ids = scored["lineup_id"].dropna()
        if not left_ids.empty and not right_ids.empty:
            by_id = entered_sim.merge(
                scored[score_keep],
                on="lineup_id",
                how="inner",
                suffixes=("", "_entry"),
            )
    if not by_id.empty:
        return by_id

    if "signature" not in scored.columns:
        return pd.DataFrame()
    scored_sig = scored[scored["signature"].astype(str).str.len() > 0]
    return entered_sim.merge(
        scored_sig[score_keep],
        on="signature",
        how="inner",
        suffixes=("", "_entry"),
    )


def _resolve_field_size(
    scored: pd.DataFrame,
    cols: dict[str, str | None],
    field_size: int | None,
) -> int:
    if field_size is not None:
        return max(1, int(field_size))
    if "_field_size_col" in scored.columns:
        values = pd.to_numeric(scored["_field_size_col"], errors="coerce").dropna()
        if not values.empty:
            return max(1, int(values.max()))
    n_entries = int(len(scored))
    if n_entries <= 0:
        raise ValueError("Cannot infer field size: entries table is empty after rank parsing.")
    return n_entries


def _resolve_column(
    frame: pd.DataFrame,
    explicit: str | None,
    aliases: Iterable[str],
    role: str,
    required: bool,
) -> str | None:
    alias_set = {_normalize_col(name) for name in aliases}
    if explicit:
        if explicit in frame.columns:
            return explicit
        needle = _normalize_col(explicit)
        for col in frame.columns:
            if _normalize_col(col) == needle:
                return str(col)
        raise UnmatchedHeaderError(
            _header_error(frame, role=role, explicit=explicit, aliases=aliases, required=True)
        )
    for col in frame.columns:
        norm = _normalize_col(col)
        if norm in alias_set:
            return str(col)
    for col in frame.columns:
        norm = _normalize_col(col)
        if any(alias and alias in norm for alias in alias_set):
            # Avoid matching roster "P" inside unrelated headers via tiny aliases.
            if len(norm) <= 3 and norm not in alias_set:
                continue
            return str(col)
    if required:
        raise UnmatchedHeaderError(
            _header_error(frame, role=role, explicit=explicit, aliases=aliases, required=True)
        )
    return None


def _header_error(
    frame: pd.DataFrame,
    *,
    role: str,
    explicit: str | None,
    aliases: Iterable[str],
    required: bool,
) -> str:
    headers = [str(col) for col in frame.columns]
    wanted = f"explicit {explicit!r}" if explicit else f"aliases {list(aliases)}"
    return (
        f"Unable to match FanDuel entries column for {role} ({wanted}).\n"
        f"Provided headers ({len(headers)}):\n  " + ", ".join(headers) + "\n"
        "Pass --rank-col / --points-col (and optional --lineup-id-col) with an "
        "exact header from the list above."
    )


def _roster_columns(columns: Iterable[object]) -> list[str]:
    result: list[str] = []
    for col in columns:
        norm = _normalize_col(col)
        if norm in {"rank", "place", "points", "score", "fpts"}:
            continue
        if any(skip in norm for skip in ("rank", "score", "points", "owner", "entry", "prize", "winning")):
            if not _ROSTER_COL_RE.match(norm):
                continue
        if _ROSTER_COL_RE.match(norm) or norm.startswith("player") or norm.endswith("_player"):
            result.append(str(col))
    return result


def _split_player_cell(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    pieces = re.split(r"[|;/]", text)
    if len(pieces) == 1 and "," in text and len(text.split(",")) >= 6:
        pieces = text.split(",")
    tokens = [_canonical_token(piece) for piece in pieces]
    return [token for token in tokens if token]


def _canonical_token(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    match = _FD_ID_RE.search(text)
    if match:
        return match.group(1)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[^a-zA-Z0-9 .'/-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    if len(text) < 3 or text in {"p", "of", "util", "c/1b"}:
        return ""
    return text


def _signature_key(players: Iterable[str]) -> str:
    sig = lineup_signature(players)
    if len(sig) < 6:
        return ""
    return "|".join(sorted(sig))


def _parse_money(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    text = str(value).replace("$", "").replace(",", "").strip()
    if not text or text.lower() in {"nan", "none", "-"}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _normalize_col(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _infer_tag(path: Path | str | None) -> str:
    if path is None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    name = Path(path).name
    return name.replace("_sim_results.csv", "").replace(".csv", "")


def _fmt_metric(value: float) -> float | str:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return ""
    return round(float(value), 6)


def _read_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    try:
        frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    for col in HISTORY_COLUMNS:
        if col not in frame.columns:
            frame[col] = ""
    return frame[HISTORY_COLUMNS]


__all__ = [
    "HISTORY_COLUMNS",
    "UnmatchedHeaderError",
    "CalibrationResult",
    "append_history",
    "brier_score",
    "calibration_by_decile",
    "extract_player_tokens",
    "format_report",
    "join_entered_lineups",
    "lineup_signature",
    "parse_player_ids_cell",
    "resolve_entries_columns",
    "score_calibration",
    "spearman_correlation",
]
