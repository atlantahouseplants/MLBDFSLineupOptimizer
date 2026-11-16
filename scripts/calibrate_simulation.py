"""Fit the Monte Carlo simulator to real outcomes.

Run after scripts/fetch_actuals.py and scripts/backtest_projections.py:

    python scripts/calibrate_simulation.py            # report only
    python scripts/calibrate_simulation.py --write    # also save config/simulation_calibration.json

It measures how often hitters score zero, how wide hitter and pitcher scores
really are, and how correlated teammates / opponents / pitchers are, then
solves for simulator settings that reproduce those numbers.  The validation
table compares real team stack totals with the old and the fitted simulator.
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

from slate_optimizer.analysis.sim_structure import calibrate
from slate_optimizer.simulation.calibration import DEFAULT_SIM_CALIBRATION_PATH, save_simulation_calibration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--player-games", default=str(REPO_ROOT / "data" / "backtest" / "player_games.csv.gz"))
    parser.add_argument("--projection-column", default="proj_calibrated_config")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "backtest"))
    parser.add_argument("--write", action="store_true", help=f"Save the fit to {DEFAULT_SIM_CALIBRATION_PATH.name}.")
    parser.add_argument("--no-validate", action="store_true", help="Skip the (slower) stack validation table.")
    args = parser.parse_args(argv)

    path = Path(args.player_games)
    if not path.exists():
        print(f"{path} not found. Run scripts/backtest_projections.py first.")
        return 1
    player_games = pd.read_csv(path, low_memory=False)
    report = calibrate(player_games, proj_col=args.projection_column, validate=not args.no_validate)
    fit = report.calibration

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(f"Slates {fit.slates_used}, hitter starts {fit.hitter_starts}, pitcher starts {fit.pitcher_starts}")
    print("\n== measured ==")
    for key, value in report.notes.items():
        print(f"  {key}: {value:.3f}")
    print("\n== measured correlations of fantasy points (95% slate-clustered interval) ==")
    print(report.correlations.to_string(index=False))
    print("\n== fitted simulator settings ==")
    for key, value in fit.to_dict().items():
        print(f"  {key}: {value}")
    print("\n== correlations the fitted simulator produces ==")
    for key, value in report.simulated.items():
        print(f"  {key}: {value:.3f}")
    if not report.validation.empty:
        print("\n== validation: real vs simulated score distributions ==")
        print(report.validation.to_string(index=False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.correlations.to_csv(out_dir / "sim_measured_correlations.csv", index=False)
    if not report.validation.empty:
        report.validation.to_csv(out_dir / "sim_validation.csv", index=False)
    if args.write:
        save_simulation_calibration(fit)
        print(f"\nSaved {DEFAULT_SIM_CALIBRATION_PATH}")
    else:
        print("\nReport only. Re-run with --write to save the calibration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
