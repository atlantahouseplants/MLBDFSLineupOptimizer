"""Backtest the projection multipliers against real box scores.

Typical use (after ``python scripts/fetch_actuals.py --all-live``):

    python scripts/backtest_projections.py

Writes to data/backtest/:
    projection_backtest_summary.csv   accuracy of every variant vs raw BallparkPal
    team_total_accuracy.csv           BPP vs Vegas implied team runs vs actual
    player_games.csv.gz               projection inputs + every variant + outcome

The summary prints to the terminal.  A variant only "wins" when its
slate-clustered bootstrap interval excludes zero.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.analysis.outcomes import DEFAULT_ACTUALS_DIR
from slate_optimizer.analysis.projection_backtest import run_backtest

DISPLAY_COLUMNS = [
    "variant",
    "n",
    "bias",
    "mae",
    "rmse",
    "pearson",
    "slope",
    "top_decile_actual",
    "d_rmse_lo",
    "d_rmse_hi",
    "d_corr_lo",
    "d_corr_hi",
]


def _print_table(summary: pd.DataFrame, role: str, subset: str) -> None:
    view = summary[(summary["role"] == role) & (summary["subset"] == subset)]
    if view.empty:
        return
    columns = [col for col in DISPLAY_COLUMNS if col in view.columns]
    print(f"\n== {role}s / {subset} ==")
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(view[columns].to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live-dir", default=str(REPO_ROOT / "data" / "live"))
    parser.add_argument("--actuals-dir", default=str(DEFAULT_ACTUALS_DIR))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "backtest"))
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--no-player-games", action="store_true", help="Skip writing the joined player-game file.")
    args = parser.parse_args(argv)

    result = run_backtest(args.live_dir, args.actuals_dir, bootstrap_reps=args.bootstrap_reps)
    if result.player_games.empty:
        print("No slates with both projections and outcomes. Run scripts/fetch_actuals.py --all-live first.")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(out_dir / "projection_backtest_summary.csv", index=False)
    result.team_totals.to_csv(out_dir / "team_total_accuracy.csv", index=False)
    if not args.no_player_games:
        result.player_games.to_csv(out_dir / "player_games.csv.gz", index=False, compression="gzip")

    started = result.player_games[result.player_games["actual_started"]]
    print(
        f"Slates: {len(result.dates)} ({result.dates[0]} to {result.dates[-1]}), "
        f"{int((started['player_type'] == 'batter').sum())} hitter starts, "
        f"{int((started['player_type'] == 'pitcher').sum())} pitcher starts"
    )
    for role in ("batter", "pitcher"):
        for subset in ("all", "pregame_vegas"):
            _print_table(result.summary, role, subset)
    if not result.team_totals.empty:
        print("\n== implied team runs vs actual (pre-game lines only) ==")
        with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
            print(result.team_totals.to_string(index=False))
    print(f"\nWrote results to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
