#!/usr/bin/env python3
"""Score a past simulation run against FanDuel My Entries results.

Joins ``<tag>_sim_results.csv`` + ``<tag>_simulated_upload.csv`` to a My
Entries export and prints Brier (cash / top-1%), calibration-by-decile, and
Spearman rank correlation. Appends one row to
``data/output/sim_calibration_history.csv``.

FanDuel headers vary. Pass ``--rank-col`` / ``--points-col`` when auto-detect
fails; unmatched headers print the actual CSV header list and exit non-zero.

Usage:
    python scripts/score_sim_calibration.py \\
        --sim-results data/output/2026-08-11_sim_results.csv \\
        --upload-csv data/output/2026-08-11_simulated_upload.csv \\
        --entries-csv path/to/fanduel_my_entries.csv \\
        --tag 2026-08-11
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.simulation.sim_calibration import (  # noqa: E402
    DEFAULT_CASH_LINE,
    DEFAULT_SELECTION_METRIC,
    UnmatchedHeaderError,
    format_report,
    score_calibration,
)

DEFAULT_HISTORY = REPO_ROOT / "data" / "output" / "sim_calibration_history.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sim-results",
        required=True,
        help="Path to <tag>_sim_results.csv from a simulated pipeline run.",
    )
    parser.add_argument(
        "--upload-csv",
        required=True,
        help="Matching <tag>_simulated_upload.csv (entered lineups).",
    )
    parser.add_argument(
        "--entries-csv",
        required=True,
        help="FanDuel My Entries results export for the same contest.",
    )
    parser.add_argument("--tag", default="", help="Slate tag stored in history (default: inferred from sim-results filename).")
    parser.add_argument("--rank-col", default=None, help="Exact entries-CSV header for finishing rank/place.")
    parser.add_argument("--points-col", default=None, help="Exact entries-CSV header for fantasy points.")
    parser.add_argument("--lineup-id-col", default=None, help="Optional entries-CSV header for lineup_id (join key when present).")
    parser.add_argument("--winnings-col", default=None, help="Optional entries-CSV header for prize/winnings (cashed = payout > 0).")
    parser.add_argument("--field-size-col", default=None, help="Optional entries-CSV header for contest field size.")
    parser.add_argument(
        "--field-size",
        type=int,
        default=None,
        help="Contest field size. Default: field-size column, else number of entries rows.",
    )
    parser.add_argument(
        "--cash-line",
        type=float,
        default=DEFAULT_CASH_LINE,
        help="Finish-percentile cutoff used as cash when no winnings column is present (default: 0.20).",
    )
    parser.add_argument(
        "--selection-metric",
        default=DEFAULT_SELECTION_METRIC,
        choices=["top_1pct_rate", "win_rate", "cash_rate", "expected_roi"],
        help="Predicted column correlated with actual finish percentile.",
    )
    parser.add_argument(
        "--history",
        default=str(DEFAULT_HISTORY),
        help="History CSV path (default: data/output/sim_calibration_history.csv).",
    )
    parser.add_argument("--notes", default="", help="Optional note stored on the history row.")
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Compute and print the report without appending history.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    history_path = None if args.no_history else Path(args.history)
    try:
        result = score_calibration(
            args.sim_results,
            args.upload_csv,
            args.entries_csv,
            tag=args.tag,
            points_col=args.points_col,
            rank_col=args.rank_col,
            lineup_id_col=args.lineup_id_col,
            winnings_col=args.winnings_col,
            field_size=args.field_size,
            field_size_col=args.field_size_col,
            cash_line=args.cash_line,
            selection_metric=args.selection_metric,
            history_path=history_path,
            notes=args.notes,
            append=history_path is not None,
        )
    except UnmatchedHeaderError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
