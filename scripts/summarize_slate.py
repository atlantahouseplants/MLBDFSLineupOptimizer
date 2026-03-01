"""Summarize a stored slate (team runs, pitchers, stack targets)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
import sys
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.data.storage import SlateDatabase

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default="data/slates.db", help="SQLite file storing slates.")
    parser.add_argument("--slate-id", type=int, default=None, help="Slate ID to summarize (defaults to latest).")
    parser.add_argument("--top", type=int, default=5, help="How many teams/pitchers to display.")
    parser.add_argument("--output", default=None, help="Optional Markdown file to write the summary.")
    return parser.parse_args()

def load_slate(db_path: Path, slate_id: int | None) -> tuple[pd.DataFrame, int, str]:
    db = SlateDatabase(db_path)
    slate = db.get_slate(slate_id) if slate_id is not None else db.get_latest_slate()
    if slate is None:
        raise SystemExit("No slates found. Run the pipeline first.")
    df = db.fetch_players(slate.slate_id)
    db.close()
    return df, slate.slate_id, slate.tag

def section_top_stacks(df: pd.DataFrame, top: int) -> List[str]:
    hitters = df[df["player_type"].str.lower() == "batter"].copy()
    hitters["bpp_runs"] = pd.to_numeric(hitters.get("bpp_runs"), errors="coerce")
    stack_means = (
        hitters.groupby("team_code")["bpp_runs"].mean().dropna().sort_values(ascending=False)
    )
    lines = ["### Top stacks (BallparkPal runs)"]
    if stack_means.empty:
        lines.append("No run data available.")
    else:
        for team, runs in stack_means.head(top).items():
            lines.append(f"- {team}: {runs:.2f} runs")
    print("\n".join(lines))
    print()
    return lines

def section_top_pitchers(df: pd.DataFrame, top: int) -> List[str]:
    pitchers = df[df["player_type"].str.lower() == "pitcher"].copy()
    pitchers["bpp_win_percent"] = pd.to_numeric(pitchers.get("bpp_win_percent"), errors="coerce")
    pitchers["proj_fd_mean"] = pd.to_numeric(pitchers.get("proj_fd_mean"), errors="coerce")
    board = pitchers.sort_values(by=["bpp_win_percent", "proj_fd_mean"], ascending=[False, False])
    lines = ["### Top pitchers (win% / projection)"]
    if board.empty:
        lines.append("No pitcher data available.")
    else:
        for _, row in board.head(top).iterrows():
            lines.append(
                f"- {row['full_name']} ({row['team_code']}): win% {row['bpp_win_percent']:.2f}, proj {row['proj_fd_mean']:.1f}"
            )
    print("\n".join(lines))
    print()
    return lines

def section_top_chalk(df: pd.DataFrame, top: int) -> List[str]:
    if "proj_fd_ownership" in df.columns:
        ownership = pd.to_numeric(df["proj_fd_ownership"], errors="coerce").fillna(0.0)
    else:
        ownership = pd.Series(0.0, index=df.index)
    df = df.assign(proj_fd_ownership=ownership)
    board = df.sort_values(by="proj_fd_ownership", ascending=False)
    lines = ["### Highest projected ownership"]
    if board.empty:
        lines.append("No ownership values available.")
    else:
        for _, row in board.head(top).iterrows():
            pct = row['proj_fd_ownership'] * 100
            lines.append(f"- {row['full_name']} ({row['team_code']}): {pct:.1f}% ownership proxy")
    print("\n".join(lines))
    print()
    return lines

def main() -> None:
    args = parse_args()
    df, slate_id, tag = load_slate(Path(args.db_path), args.slate_id)
    print(f"Loaded slate #{slate_id} ({tag}) with {len(df)} players.\n")

    sections = []
    sections.extend(section_top_stacks(df, args.top))
    sections.append("")
    sections.extend(section_top_pitchers(df, args.top))
    sections.append("")
    sections.extend(section_top_chalk(df, args.top))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text("\n".join(sections))
        print(f"Summary saved to {args.output}")

if __name__ == "__main__":
    main()


