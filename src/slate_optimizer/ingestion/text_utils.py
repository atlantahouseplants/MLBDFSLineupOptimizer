"""Text helpers for ingestion normalization."""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

_CANONICAL_PATTERN = re.compile(r"[^a-z\s]")
_SUFFIX_PATTERN = re.compile(r"\b(jr|sr|ii|iii|iv)\s*$")


def _strip_accents(text: str) -> str:
    """Convert accented characters to their ASCII equivalents.

    e.g. Acuña → Acuna, Tatís → Tatis, José → Jose
    """
    # NFD decomposes accented chars: ñ → n + combining tilde
    nfkd = unicodedata.normalize("NFKD", text)
    # Strip combining marks (category 'Mn' = Mark, Nonspacing)
    return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")


def canonicalize_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .apply(_strip_accents)
        .str.lower()
        .str.replace(_CANONICAL_PATTERN, "", regex=True)
        .str.replace(_SUFFIX_PATTERN, "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def canonicalize_string(value: str) -> str:
    return canonicalize_series(pd.Series([value])).iloc[0]
