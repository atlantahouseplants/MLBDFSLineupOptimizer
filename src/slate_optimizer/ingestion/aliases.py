"""Helpers for loading player name alias mappings."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .text_utils import canonicalize_string

CanonicalMap = Dict[str, str]


def load_alias_map(path: Path) -> CanonicalMap:
    alias_path = Path(path).expanduser().resolve()
    if not alias_path.exists():
        raise FileNotFoundError(f"Alias file not found: {alias_path}")
    data = json.loads(alias_path.read_text())
    canonical: CanonicalMap = {}
    for key, value in data.items():
        normalized_key = canonicalize_string(key)
        normalized_value = canonicalize_string(value)
        if not normalized_key or not normalized_value:
            continue
        canonical[normalized_key] = normalized_value
    return canonical

def _normalize(name: str) -> str:
    return " ".join(str(name).lower().split())

def apply_aliases(series, alias_map: CanonicalMap):
    return series.map(lambda x: alias_map.get(x, x))
