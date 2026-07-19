#!/usr/bin/env python3
"""Fetch live MLB data: Ballpark Pal API + batting orders (MLB Stats API) + Vegas lines (Odds API).

Usage:
    python scripts/fetch_live_data.py
    python scripts/fetch_live_data.py --date 2026-07-19
    python scripts/fetch_live_data.py --output data/live

Outputs (in --output dir):
    bpp_batters/pitchers/games/teams_YYYY-MM-DD.csv + BallparkPal_*.xlsx
    bpp_probabilities_YYYY-MM-DD.csv, bpp_hitter_park_factors_YYYY-MM-DD.csv
    batting_orders_YYYY-MM-DD.csv    — confirmed lineups overwrite BPP-derived ones
    probable_pitchers_YYYY-MM-DD.csv — probable pitchers (player_name, team, bats, throws)
    handedness_YYYY-MM-DD.csv        — batter/pitcher hands (via BPP fetch, MLB bulk API)
    vegas_lines_YYYY-MM-DD.csv       — game totals + moneylines (game,total,home_ml,away_ml)

Requires BPP_API_KEY and ODDS_API_KEY in .env (auto-loaded). Without BPP_API_KEY,
fall back to manual BallparkPal Excel exports placed in data/live/.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import os

# Auto-load .env from repo root if present
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch live MLB slate data")
    parser.add_argument(
        "--date",
        default=date.today().strftime("%Y-%m-%d"),
        help="Date to fetch (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--output",
        default="data/live",
        help="Output directory (default: data/live)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    d = args.date

    print(f"Fetching live data for {d}...")
    print()

    # ── Ballpark Pal simulation data (official API) ────────────────────────
    bundle = None
    if os.getenv("BPP_API_KEY"):
        print("[0/3] Ballpark Pal official API (simulation data)...")
        from slate_optimizer.ingestion.bpp_official_api import fetch_bpp_api_data

        bundle = fetch_bpp_api_data(date_str=d)
        if bundle is None:
            print("  Official API fetch failed — check BPP_API_KEY / API access page.")
    else:
        print("[0/3] Ballpark Pal: BPP_API_KEY not set — skipping API fetch.")
    if bundle:
        s = bundle.summary()
        print(f"  Batters: {s['batters']}  Pitchers: {s['pitchers']}  Games: {s['games']}  Teams: {s['teams']}")
        n_bo = len(bundle.batting_orders())
        n_hand = len(bundle.handedness())
        if s.get("hitter_park_factors"):
            print(f"  Hitter park factors: {s['hitter_park_factors']} players")
        if s.get("probabilities"):
            print(f"  Market probabilities: {s['probabilities']} items")
        bundle.to_csvs(str(output_dir))
        bundle.to_excels(str(output_dir))
        print(f"  Batting orders: {n_bo} lineup slots → no manual CSV needed")
        print(f"  Handedness: {n_hand} players → no manual CSV needed")
        print(f"  Saved → {output_dir}/bpp_*_{d}.csv + BallparkPal_*.xlsx")
    else:
        print("  No Ballpark Pal data fetched.")
        print("  Set BPP_API_KEY in .env, or drop manual BallparkPal_*.xlsx exports")
        print(f"  into {output_dir}/ before running the pipeline.")
    print()

    # ── MLB confirmed lineups + probable pitchers ──────────────────────────
    print("[1/3] MLB Stats API (confirmed batting orders + probable pitchers)...")
    batting_df = pd.DataFrame()
    pitchers_df = pd.DataFrame()
    try:
        from slate_optimizer.ingestion.mlb_lineups_api import fetch_batting_orders

        orders_df = fetch_batting_orders(game_date=date.fromisoformat(d))
        pitchers_df = orders_df[orders_df["order_position"] == 0].copy()
        batting_df = orders_df[orders_df["order_position"] >= 1].copy()
    except ValueError as exc:
        print(f"  {exc}")
        print("  Confirmed orders usually post 2-3 hours before first pitch.")
        print("  BPP-derived batting orders (above) already cover the slate.")

    if not batting_df.empty:
        # Confirmed orders overwrite the BPP-derived file: team, order_position, player_name
        orders_out = batting_df[["team", "order_position", "player_name"]]
        orders_path = output_dir / f"batting_orders_{d}.csv"
        orders_out.to_csv(orders_path, index=False)
        print(f"  Saved {len(batting_df)} confirmed lineup slots across {batting_df['team'].nunique()} teams → {orders_path}")

    if not pitchers_df.empty:
        # Legacy handedness-compatible shape; pitcher hands live in handedness_<d>.csv
        pitchers_out = pitchers_df.rename(columns={"player_name": "player_name"})[["player_name", "team"]]
        pitchers_out["bats"] = ""
        pitchers_out["throws"] = ""
        pitchers_path = output_dir / f"probable_pitchers_{d}.csv"
        pitchers_out[["player_name", "team", "bats", "throws"]].to_csv(pitchers_path, index=False)
        print(f"  Saved {len(pitchers_df)} probable pitchers → {pitchers_path}")
    print()

    # ── Vegas lines ────────────────────────────────────────────────────────
    print("[2/3] Odds API (Vegas lines)...")
    try:
        from slate_optimizer.ingestion.vegas_api import fetch_vegas_lines

        vegas = fetch_vegas_lines()
    except ValueError as exc:
        print(f"  {exc}")
        print("  Manual alternative: create a CSV with columns: game,total,home_ml,away_ml")
        vegas = None
    except Exception as exc:  # network/HTTP failure — non-fatal
        print(f"  Odds API request failed: {exc}")
        vegas = None

    if vegas is not None:
        if vegas.games.empty:
            print("  No games found from Odds API.")
        else:
            save_cols = ["game", "total", "home_ml", "away_ml"]
            available = [c for c in save_cols if c in vegas.games.columns]
            vegas_path = output_dir / f"vegas_lines_{d}.csv"
            vegas.games[available].to_csv(vegas_path, index=False)
            print(f"  Saved {len(vegas.games)} games → {vegas_path}")
            summary = vegas.summary()
            print(f"  Coverage: {summary['games']} games, {summary['teams']} teams")

    print()
    print("Done. Live data is in", output_dir)


if __name__ == "__main__":
    main()
