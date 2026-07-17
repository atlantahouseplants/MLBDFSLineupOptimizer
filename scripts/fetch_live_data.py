#!/usr/bin/env python3
"""Fetch live MLB data: batting orders (MLB Stats API) + Vegas lines (Odds API).

Usage:
    python scripts/fetch_live_data.py
    python scripts/fetch_live_data.py --date 2026-04-15
    python scripts/fetch_live_data.py --output data/live

Outputs (in --output dir):
    batting_orders_YYYY-MM-DD.csv   — confirmed lineups (team,order_position,player_name)
    probable_pitchers_YYYY-MM-DD.csv — probable pitchers (team,player_name,pitcher_hand)
    vegas_lines_YYYY-MM-DD.csv      — game totals + moneylines (game,total,home_ml,away_ml)

If no ODDS_API_KEY is set, Vegas lines are skipped (use manual CSV instead).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import os
from pathlib import Path

# Auto-load .env from repo root if present
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

import pandas as pd

from slate_optimizer.ingestion.mlb_api import fetch_mlb_lineups
from slate_optimizer.ingestion.bpp_api import fetch_bpp_data
from slate_optimizer.ingestion.odds_api import fetch_vegas_lines


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

    # ── BallparkPal simulation data ────────────────────────────────────────
    # Preferred source: official BallparkPal API (BPP_API_KEY). Falls back to
    # the legacy session-cookie scrape (BPP_SESSION), then to manual Excels.
    bundle = None
    if os.getenv("BPP_API_KEY"):
        print("[0/3] BallparkPal official API (simulation data)...")
        from slate_optimizer.ingestion.bpp_official_api import fetch_bpp_api_data

        bundle = fetch_bpp_api_data(date_str=d)
        if bundle is None:
            print("  Official API fetch failed — check BPP_API_KEY / API access page.")
    if bundle is None and os.getenv("BPP_SESSION"):
        if not os.getenv("BPP_API_KEY"):
            print("[0/3] BallparkPal Export Center via session cookie (legacy)...")
        else:
            print("  Falling back to legacy session-cookie scrape...")
        bundle = fetch_bpp_data(date_str=d)
    if bundle:
        s = bundle.summary()
        print(f"  Batters: {s['batters']}  Pitchers: {s['pitchers']}  Games: {s['games']}  Teams: {s['teams']}")
        n_proj = s["dfs_projections"]
        n_pf = s["park_factors"]
        n_bo = len(bundle.batting_orders())
        n_hand = len(bundle.handedness())
        if n_proj:
            print(f"  DFS projections: {n_proj} players (Bust/Median/Upside)")
        print(f"  Park factors: {n_pf} games")
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
        print("  No BallparkPal source available.")
        print("  Set BPP_API_KEY (preferred) or BPP_SESSION (legacy) in .env.")
        print("  Without this, pipeline falls back to manual Excel exports")
    print()

    # ── MLB lineups + probable pitchers ───────────────────────────────────
    print("[1/3] MLB Stats API (batting orders + probable pitchers)...")
    batting_df, pitchers_df = fetch_mlb_lineups(date_str=d)

    if not batting_df.empty:
        # Save in the format BattingOrderLoader expects: team, order_position, player_name
        orders_out = batting_df[["team_code", "order_position", "player_name"]].rename(
            columns={"team_code": "team"}
        )
        orders_path = output_dir / f"batting_orders_{d}.csv"
        orders_out.to_csv(orders_path, index=False)
        teams = batting_df["team_code"].nunique()
        players = len(batting_df)
        print(f"  Saved {players} confirmed lineup slots across {teams} teams → {orders_path}")
    else:
        print("  No confirmed lineups found yet (check closer to game time).")

    if not pitchers_df.empty:
        # Save as handedness-compatible CSV: player_name, team, bats, throws
        pitchers_out = pitchers_df.rename(columns={"team_code": "team", "pitcher_hand": "throws"})
        pitchers_out["bats"] = ""  # not available from probable pitcher API
        pitchers_path = output_dir / f"probable_pitchers_{d}.csv"
        pitchers_out[["player_name", "team", "bats", "throws"]].to_csv(pitchers_path, index=False)
        print(f"  Saved {len(pitchers_df)} probable pitchers → {pitchers_path}")
    else:
        print("  No probable pitchers found.")

    print()

    # ── Vegas lines ────────────────────────────────────────────────────────
    print("[2/3] Odds API (Vegas lines)...")
    vegas = fetch_vegas_lines()

    if vegas is None:
        print("  ODDS_API_KEY not set — skipping Vegas auto-fetch.")
        print("  To enable: get a free key at https://the-odds-api.com and set ODDS_API_KEY env var.")
        print("  Manual alternative: create a CSV with columns: game,total,home_ml,away_ml")
    elif vegas.games.empty:
        print("  No games found from Odds API.")
    else:
        # Save in VegasLoader-compatible format: game,total,home_ml,away_ml
        save_cols = ["game", "total", "home_ml", "away_ml"]
        available = [c for c in save_cols if c in vegas.games.columns]
        vegas_path = output_dir / f"vegas_lines_{d}.csv"
        vegas.games[available].to_csv(vegas_path, index=False)
        print(f"  Saved {len(vegas.games)} games → {vegas_path}")
        summary = vegas.summary()
        print(f"  Coverage: {summary['games']} games, {summary['teams']} teams")

    print()
    print("Done. Pass these files to run_daily_pipeline.py with:")
    if not batting_df.empty:
        print(f"  --batting-orders-csv {output_dir}/batting_orders_{d}.csv")
    print(f"  --handedness-csv {output_dir}/probable_pitchers_{d}.csv  (pitcher hands)")
    print(f"Or use --auto-fetch to wire them in automatically.")


if __name__ == "__main__":
    main()
