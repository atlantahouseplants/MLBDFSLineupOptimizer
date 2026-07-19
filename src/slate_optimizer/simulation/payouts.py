"""Rank-based payout utilities for contest-specific EV simulation."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PayoutBand:
    min_rank: int
    max_rank: int
    payout: float


DEFAULT_PAYOUT_LADDER_TEXT = """1,100000
2,25000
3,10000
4-5,5000
6-10,2500
11-25,1000
26-50,500
51-100,250
101-250,100
251-500,50
501-1000,30
1001-2500,20
2501-5000,10"""


def parse_payout_ladder_text(text: str) -> list[PayoutBand]:
    """Parse lines like ``1,100000`` or ``4-10,2500`` into payout bands."""

    bands: list[PayoutBand] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        normalized = line.replace("\t", ",")
        if "," in normalized:
            rank_text, payout_text = [part.strip() for part in normalized.split(",", 1)]
        else:
            pieces = normalized.split()
            if len(pieces) < 2:
                continue
            rank_text, payout_text = pieces[0], " ".join(pieces[1:])
        rank_text = rank_text.lower().replace("rank", "").strip()
        payout_text = (
            payout_text.lower()
            .replace("payout", "")
            .replace("prize", "")
            .replace("$", "")
            .replace(",", "")
            .strip()
        )
        if not rank_text or not payout_text:
            continue
        if "-" in rank_text:
            start_text, end_text = [token.strip() for token in rank_text.split("-", 1)]
        else:
            start_text = end_text = rank_text
        try:
            min_rank = int(float(start_text))
            max_rank = int(float(end_text))
            payout = float(payout_text)
        except ValueError:
            continue
        if min_rank <= 0 or max_rank < min_rank or payout < 0:
            continue
        bands.append(PayoutBand(min_rank=min_rank, max_rank=max_rank, payout=payout))
    return normalize_payout_bands(bands)


def parse_payout_ladder_dataframe(df: pd.DataFrame) -> list[PayoutBand]:
    """Parse a FanDuel-style payout table into rank payout bands.

    Supported shapes include ``Place, Prize``, ``Rank, Payout``,
    ``Min Rank, Max Rank, Amount``, and similar exported column names.
    """

    if df is None or df.empty:
        return []
    working = df.copy()
    working.columns = [str(col).strip() for col in working.columns]
    normalized_cols = {_normalize_column_name(col): col for col in working.columns}
    payout_col = _find_column(normalized_cols, ["payout", "prize", "amount", "winnings", "paid"])
    if payout_col is None:
        return []

    min_col = _find_rank_bound_column(normalized_cols, ["min", "from", "start"])
    max_col = _find_rank_bound_column(normalized_cols, ["max", "to", "end"])
    rank_col = _find_column(normalized_cols, ["rank", "place", "position", "finish"])
    if rank_col is None and (min_col is None or max_col is None):
        non_payout_cols = [col for col in working.columns if col != payout_col]
        rank_col = non_payout_cols[0] if non_payout_cols else None

    bands: list[PayoutBand] = []
    for _, row in working.iterrows():
        payout = _parse_money(row.get(payout_col))
        if payout is None or payout < 0:
            continue
        if min_col is not None and max_col is not None:
            min_rank = _parse_rank_number(row.get(min_col))
            max_rank = _parse_rank_number(row.get(max_col))
        elif rank_col is not None:
            min_rank, max_rank = _parse_rank_range(row.get(rank_col))
        else:
            continue
        if min_rank is None or max_rank is None:
            continue
        if min_rank <= 0 or max_rank < min_rank:
            continue
        bands.append(PayoutBand(int(min_rank), int(max_rank), float(payout)))
    return normalize_payout_bands(bands)


def normalize_payout_bands(raw_bands: Iterable[PayoutBand | Mapping]) -> list[PayoutBand]:
    bands: list[PayoutBand] = []
    for raw in raw_bands or []:
        if isinstance(raw, PayoutBand):
            band = raw
        else:
            min_rank = raw.get("min_rank", raw.get("start_rank", raw.get("rank", 0)))
            max_rank = raw.get("max_rank", raw.get("end_rank", min_rank))
            payout = raw.get("payout", raw.get("prize", raw.get("amount", 0.0)))
            band = PayoutBand(int(min_rank), int(max_rank), float(payout))
        if band.min_rank <= 0 or band.max_rank < band.min_rank or band.payout < 0:
            continue
        bands.append(band)
    bands.sort(key=lambda item: (item.min_rank, item.max_rank))
    return bands


def ranks_from_percentiles(percentiles: np.ndarray, contest_entries: int) -> np.ndarray:
    entries = max(1, int(contest_entries))
    pct = np.asarray(percentiles, dtype=float)
    ranks = np.floor((1.0 - np.clip(pct, 0.0, 100.0) / 100.0) * entries).astype(int) + 1
    return np.clip(ranks, 1, entries)


def payout_from_ranks(ranks: np.ndarray, bands: Sequence[PayoutBand]) -> np.ndarray:
    rank_arr = np.asarray(ranks, dtype=int)
    payouts = np.zeros(rank_arr.shape, dtype=float)
    for band in bands:
        mask = (rank_arr >= band.min_rank) & (rank_arr <= band.max_rank)
        payouts[mask] = float(band.payout)
    return payouts


def payout_from_percentiles(
    percentiles: np.ndarray,
    bands: Sequence[PayoutBand],
    contest_entries: int,
) -> np.ndarray:
    ranks = ranks_from_percentiles(percentiles, contest_entries)
    return payout_from_ranks(ranks, bands)


def payout_ladder_to_dicts(bands: Sequence[PayoutBand]) -> list[dict[str, float]]:
    return [
        {
            "min_rank": int(band.min_rank),
            "max_rank": int(band.max_rank),
            "payout": float(band.payout),
        }
        for band in bands
    ]


def payout_ladder_to_text(bands: Sequence[PayoutBand]) -> str:
    lines = []
    for band in bands:
        rank = str(band.min_rank) if band.min_rank == band.max_rank else f"{band.min_rank}-{band.max_rank}"
        lines.append(f"{rank},{band.payout:g}")
    return "\n".join(lines)


def _normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _find_column(columns: Mapping[str, str], keywords: Sequence[str]) -> str | None:
    for key, original in columns.items():
        if any(keyword in key for keyword in keywords):
            return original
    return None


def _find_rank_bound_column(columns: Mapping[str, str], bound_keywords: Sequence[str]) -> str | None:
    for key, original in columns.items():
        has_bound = any(keyword in key for keyword in bound_keywords)
        has_rank = any(keyword in key for keyword in ["rank", "place", "position", "finish"])
        if has_bound and has_rank:
            return original
    return None


def _parse_money(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)) and not pd.isna(value):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    text = text.replace("$", "").replace(",", "").replace("USD", "").replace("usd", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_rank_range(value) -> tuple[int | None, int | None]:
    text = str(value).strip().lower()
    text = text.replace(",", "").replace("#", "")
    text = re.sub(r"\b(rank|place|finish|position|paid)\b", "", text).strip()
    text = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", text)
    numbers = [int(float(match)) for match in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return None, None
    if len(numbers) == 1:
        return numbers[0], numbers[0]
    return numbers[0], numbers[1]


def _parse_rank_number(value) -> int | None:
    start, _ = _parse_rank_range(value)
    return start


__all__ = [
    "DEFAULT_PAYOUT_LADDER_TEXT",
    "PayoutBand",
    "normalize_payout_bands",
    "parse_payout_ladder_dataframe",
    "parse_payout_ladder_text",
    "payout_from_percentiles",
    "payout_from_ranks",
    "payout_ladder_to_dicts",
    "payout_ladder_to_text",
    "ranks_from_percentiles",
]
