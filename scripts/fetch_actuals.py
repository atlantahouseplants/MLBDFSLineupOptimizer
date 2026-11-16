"""Download actual FanDuel scores from MLB box scores and cache them.

Examples:
    python scripts/fetch_actuals.py --all-live           # every date in data/live
    python scripts/fetch_actuals.py --date 2026-08-01
    python scripts/fetch_actuals.py --start 2026-07-19 --end 2026-08-11

Free MLB Stats API, no key needed. Output: data/actuals/actuals_YYYY-MM-DD.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date as date_cls, timedelta
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.analysis.outcomes import DEFAULT_ACTUALS_DIR, load_or_fetch_actuals

_DATE_RE = re.compile(r"bpp_batters_(\d{4}-\d{2}-\d{2})\.csv$")


def live_dates(live_dir: Path) -> List[str]:
    """Dates that have a saved BallparkPal batter projection file."""
    found = []
    for path in sorted(Path(live_dir).glob("bpp_batters_*.csv")):
        match = _DATE_RE.search(path.name)
        if match:
            found.append(match.group(1))
    return found


def _date_range(start: str, end: str) -> List[str]:
    first = date_cls.fromisoformat(start)
    last = date_cls.fromisoformat(end)
    days = (last - first).days
    return [(first + timedelta(days=offset)).isoformat() for offset in range(max(0, days) + 1)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", help="Single date YYYY-MM-DD.")
    parser.add_argument("--start", help="Range start YYYY-MM-DD.")
    parser.add_argument("--end", help="Range end YYYY-MM-DD.")
    parser.add_argument("--all-live", action="store_true", help="Every date with BPP data in --live-dir.")
    parser.add_argument("--live-dir", default=str(REPO_ROOT / "data" / "live"))
    parser.add_argument("--actuals-dir", default=str(DEFAULT_ACTUALS_DIR))
    parser.add_argument("--refresh", action="store_true", help="Re-download even if cached.")
    args = parser.parse_args(argv)

    dates: List[str] = []
    if args.date:
        dates.append(args.date)
    if args.start and args.end:
        dates.extend(_date_range(args.start, args.end))
    if args.all_live:
        dates.extend(live_dates(Path(args.live_dir)))
    dates = sorted(set(dates))
    if not dates:
        parser.error("Give --date, --start/--end, or --all-live.")

    today = date_cls.today().isoformat()
    total = 0
    for day in dates:
        if day >= today:
            print(f"{day}: skipped (games not final yet)")
            continue
        frame = load_or_fetch_actuals(day, actuals_dir=args.actuals_dir, refresh=args.refresh)
        games = frame["game_pk"].nunique() if not frame.empty else 0
        total += len(frame)
        print(f"{day}: {games} final games, {len(frame)} player rows")
    print(f"Done. {total} player rows cached in {args.actuals_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
