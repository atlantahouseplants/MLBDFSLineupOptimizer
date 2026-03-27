"""Text helpers for ingestion normalization."""
from __future__ import annotations

import pandas as pd
import re

_CANONICAL_PATTERN = re.compile(r"[^a-z\s]")
_SUFFIX_PATTERN = re.compile(r"\b(jr|sr|ii|iii|iv)\s*$")


def canonicalize_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(_CANONICAL_PATTERN, "", regex=True)
        .str.replace(_SUFFIX_PATTERN, "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

def canonicalize_string(value: str) -> str:
    return canonicalize_series(pd.Series([value])).iloc[0]
