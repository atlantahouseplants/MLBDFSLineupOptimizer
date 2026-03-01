"""Synthesize a FanDuel-style CSV from BallparkPal data."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
import sys
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.ingestion.ballparkpal import BallparkPalLoader

HITTER_POSITIONS = ["C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]

FD_COLUMNS = [
    "Id",
    "Position",
    "First Name",
    "Nickname",
    "Last Name",
    "FPPG",
    "Played",
    "Salary",
    "Game",
    "Team",
    "Opponent",
    "Injury Indicator",
    "Injury Details",
    "Tier",
    "Probable Pitcher",
    "Batting Order",
    "Roster Position",
]


def _split_name(full_name: str) -> tuple[str, str]:
    parts = str(full_name).split()
    if len(parts) == 0:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def _sample_position(is_pitcher: bool) -> str:
    if is_pitcher:
        return "P"
    return random.choice(HITTER_POSITIONS)


def _generate_id(prefix: str, player_id: str) -> str:
    return f"{prefix}-{player_id}"


def _estimate_salary(points: float, is_pitcher: bool) -> int:
    base = 5200 if is_pitcher else 3200
    multiplier = 50 if is_pitcher else 70
    value = base + points * multiplier
    value = max(3000, min(value, 9000))
    return int(round(value / 100.0) * 100)


def synthesize_fanduel_csv(source_dir: Path, output_path: Path, seed: int | None = None) -> None:
    if seed is not None:
        random.seed(seed)
    loader = BallparkPalLoader(source_dir)
    bundle = loader.load_bundle()

    rows: List[dict] = []

    for table, is_pitcher in ((bundle.batters, False), (bundle.pitchers, True)):
        for _, row in table.iterrows():
            first, last = _split_name(row.get("full_name", ""))
            team = row.get("team", "")
            opponent = row.get("opponent", "")
            game = f"{team}@{opponent}" if team and opponent else ""
            points = float(row.get("points_fd", row.get("pointsfd", 0)) or 0)
            fppg = points if points > 0 else float(row.get("fppg", 0) or 0)
            salary = _estimate_salary(fppg, is_pitcher)
            position = _sample_position(is_pitcher)
            fd_row = {
                "Id": _generate_id("SYN", row.get("player_id", f"{team}_{last}")),
                "Position": position,
                "First Name": first,
                "Nickname": row.get("full_name", ""),
                "Last Name": last,
                "FPPG": round(fppg, 2),
                "Played": 0,
                "Salary": salary,
                "Game": game,
                "Team": team,
                "Opponent": opponent,
                "Injury Indicator": "",
                "Injury Details": "",
                "Tier": 0,
                "Probable Pitcher": "Yes" if is_pitcher else "",
                "Batting Order": row.get("batting_position", 0) if not is_pitcher else 0,
                "Roster Position": position,
            }
            rows.append(fd_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=FD_COLUMNS).to_csv(output_path, index=False)
    print(f"Wrote synthetic FanDuel CSV with {len(rows)} rows to {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bpp-source", required=True, help="Directory with BallparkPal Excel files.")
    parser.add_argument("--output", required=True, help="Destination CSV path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    synthesize_fanduel_csv(Path(args.bpp_source), Path(args.output), seed=args.seed)

if __name__ == "__main__":
    main()
