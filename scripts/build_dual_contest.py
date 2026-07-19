"""Build a shared-exposure 300-lineup portfolio split into two 150-lineup contests.

Mirrors the dashboard "Process Slate" pipeline without Streamlit:
raw BPP sims + DFS Optimizer (Upside/Bust/Median) + FanDuel salaries +
pasted batting orders -> projections -> GPP solver -> two upload CSVs.

Usage:
    python scripts/build_dual_contest.py \
        --fanduel <players-list.csv> \
        --bpp-dir <dir with the 4 BallparkPal_*.xlsx> \
        --dfs-file <Ballpark DFS export> [--dfs-file <second export>] \
        --paste <lineup paste txt> \
        --total 300 --out-dir outputs
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slate_optimizer.ingestion.ballparkpal import BallparkPalLoader
from slate_optimizer.ingestion.batting_orders import BattingOrderLoader, parse_lineup_paste
from slate_optimizer.ingestion.fanduel import FanduelCSVLoader
from slate_optimizer.ingestion.slate_builder import build_player_dataset
from slate_optimizer.ingestion.text_utils import canonicalize_series
from slate_optimizer.optimizer.config import OptimizerConfig
from slate_optimizer.optimizer.dataset import build_optimizer_dataset
from slate_optimizer.optimizer.export import validate_fanduel_lineups, write_fanduel_upload
from slate_optimizer.optimizer.solver import MAX_HITTERS_PER_TEAM, SALARY_CAP, generate_lineups
from slate_optimizer.projection.baseline import compute_baseline_projections
from slate_optimizer.projection.blend import blend_projection_sources
from slate_optimizer.projection.ownership import OwnershipModelConfig, compute_ownership_series


def build_slate(fanduel_csv: Path, bpp_dir: Path, dfs_files: list[Path], paste_path: Path | None):
    bundle = BallparkPalLoader(bpp_dir).load_bundle()
    fd_players = FanduelCSVLoader(fanduel_csv).load()
    combined, diagnostics = build_player_dataset(bundle, fd_players.players)
    print(f"Matching: {diagnostics.as_dict()}")

    defaults = pd.DataFrame(
        {
            "is_confirmed_lineup": False,
            "batting_order_position": pd.array([pd.NA] * len(combined), dtype="Int64"),
            "batter_hand": "",
            "pitcher_hand": "",
            "recent_last7_fppg": 0.0,
            "recent_last14_fppg": 0.0,
            "recent_season_fppg": 0.0,
        },
        index=combined.index,
    )
    combined = pd.concat([combined, defaults], axis=1)

    confirmed_names: set[str] = set()
    confirmed_last_team: set[tuple[str, str]] = set()
    if paste_path is not None:
        paste_df = parse_lineup_paste(paste_path.read_text(encoding="utf-8"))
        if paste_df.empty:
            raise ValueError("Lineup paste parsed to zero rows — check the paste format.")
        with tempfile.TemporaryDirectory() as tmp:
            bo_csv = Path(tmp) / "batting_orders.csv"
            paste_df.to_csv(bo_csv, index=False)
            orders = BattingOrderLoader(bo_csv).load()
        combined = combined.merge(
            orders.entries.rename(columns={"batting_order_position": "_bo_pos"}),
            on=["team_code", "canonical_name"],
            how="left",
        )
        combined["batting_order_position"] = combined["_bo_pos"].combine_first(
            combined["batting_order_position"]
        )
        combined.drop(columns=["_bo_pos"], inplace=True)
        combined["is_confirmed_lineup"] = combined["batting_order_position"].notna()
        combined["batting_order_position"] = pd.to_numeric(
            combined["batting_order_position"], errors="coerce"
        ).astype("Int64")
        confirmed_names = set(canonicalize_series(paste_df["player_name"]).tolist())
        last_names = canonicalize_series(paste_df["player_name"].str.split().str[-1])
        teams = paste_df["team"].astype(str).str.upper().str.strip()
        confirmed_last_team = set(zip(last_names, teams))
        print(f"Batting orders: {orders.summary()}")

    # Backfill handedness from BPP sim columns so platoon adjustments work.
    if "bpp_batter_stand" in combined.columns:
        stand = (
            combined["bpp_batter_stand"].fillna("").astype(str).str.upper().str.strip()
        ).replace({"B": "S"})
        empty = combined["batter_hand"].fillna("").astype(str).str.strip() == ""
        fill = empty & stand.isin(["L", "R", "S"])
        combined.loc[fill, "batter_hand"] = stand.loc[fill]
    if "bpp_pitcher_hand" in combined.columns:
        hand = combined["bpp_pitcher_hand"].fillna("").astype(str).str.upper().str.strip()
        empty = combined["pitcher_hand"].fillna("").astype(str).str.strip() == ""
        fill = empty & hand.isin(["L", "R"])
        combined.loc[fill, "pitcher_hand"] = hand.loc[fill]

    # Filter to confirmed starters (paste names + last-name/team fallback).
    if confirmed_names:
        canon = canonicalize_series(combined["full_name"])
        match = canon.isin(confirmed_names)
        if confirmed_last_team:
            last = canonicalize_series(combined["full_name"].str.split().str[-1])
            team = combined["team_code"].fillna("").astype(str).str.upper().str.strip()
            pairs = pd.Series(list(zip(last, team)), index=combined.index)
            match = match | pairs.isin(confirmed_last_team)
        match = match | combined["is_confirmed_lineup"]
        # Never drop a FanDuel-probable pitcher over a paste mismatch (opener vs.
        # bulk-pitcher listings, e.g. BPP shows Englert while FanDuel lists Jax).
        if "probable_pitcher" in combined.columns:
            prob = combined["probable_pitcher"].fillna("").astype(str).str.strip().str.upper()
            fd_probable = (combined["player_type"].str.lower() == "pitcher") & prob.isin(
                {"YES", "Y", "TRUE", "1"}
            )
            rescued = fd_probable & ~match
            if rescued.any():
                print(
                    "NOTE: kept FanDuel probable pitcher(s) missing from paste: "
                    + ", ".join(combined.loc[rescued, "full_name"].astype(str))
                )
            match = match | fd_probable
        before = len(combined)
        pool = combined[match].reset_index(drop=True)
        n_pitchers = int((pool["player_type"].str.lower() == "pitcher").sum())
        print(
            f"Confirmed-starter filter: {before} -> {len(pool)} players "
            f"({n_pitchers} pitchers, {len(pool) - n_pitchers} hitters)"
        )
        if len(pool) >= 40 and n_pitchers >= 4:
            combined = pool
        else:
            print("WARNING: filtered pool not viable, using full pool")

    projections = compute_baseline_projections(combined)
    projections, blend_result = blend_projection_sources(
        combined,
        projections,
        source_paths=dfs_files,
        weights=[1.0] * len(dfs_files),
        baseline_weight=1.0,
    )
    print(f"Projection blend: baseline share {blend_result.baseline_share:.2f}")
    for src in blend_result.sources:
        print(
            f"  {src.name}: weight {src.weight:.2f}, matched {src.matched_players}/{src.source_players}, "
            f"upside={src.has_upside} bust={src.has_bust} median={src.has_median}"
        )
        if src.unmatched_players:
            print(f"    unmatched: {', '.join(src.unmatched_players[:12])}")

    ownership_result = compute_ownership_series(
        combined, projections, source_paths=[], weights=None, model_config=OwnershipModelConfig()
    )
    ownership_map = ownership_result.ownership.to_dict()
    projections["proj_fd_ownership"] = (
        projections["fd_player_id"].astype(str).map(ownership_map).fillna(0.0)
    )

    dataset = build_optimizer_dataset(combined, projections)
    return dataset


def apply_portfolio_exposure(dataset: pd.DataFrame) -> pd.DataFrame:
    """Shared-portfolio exposure: base caps + percentile-based chalk caps."""
    df = dataset.copy()
    is_pitcher = df["player_type"].str.lower() == "pitcher"
    df.loc[is_pitcher, "default_max_exposure"] = 0.35
    df.loc[~is_pitcher, "default_max_exposure"] = 0.45

    own = pd.to_numeric(df["proj_fd_ownership"], errors="coerce").fillna(0.0)
    batter_own = own[~is_pitcher]
    pitcher_own = own[is_pitcher]
    if not batter_own.empty:
        chalk_cut = float(batter_own.quantile(0.85))
        chalk = ~is_pitcher & (own >= chalk_cut)
        df.loc[chalk, "default_max_exposure"] = df.loc[chalk, "default_max_exposure"].clip(upper=0.28)
        print(f"Batter chalk cut (85th pct ownership): {chalk_cut:.3f} -> {int(chalk.sum())} batters capped at 28%")
    if not pitcher_own.empty:
        p_cut = float(pitcher_own.quantile(0.70))
        p_chalk = is_pitcher & (own >= p_cut)
        df.loc[p_chalk, "default_max_exposure"] = df.loc[p_chalk, "default_max_exposure"].clip(upper=0.30)
        print(f"Pitcher chalk cut (70th pct ownership): {p_cut:.3f} -> {int(p_chalk.sum())} pitchers capped at 30%")
    return df


def summarize(lineups, dataset: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label}: {len(lineups)} lineups ===")
    salaries = [lu.total_salary for lu in lineups]
    print(f"Salary: min {min(salaries)}, mean {sum(salaries) / len(salaries):.0f}, max {max(salaries)}")

    stack_counter: Counter[str] = Counter()
    pitcher_counter: Counter[str] = Counter()
    player_counter: Counter[str] = Counter()
    for lu in lineups:
        ldf = lu.dataframe
        hitters = ldf[ldf["player_type"].str.lower() != "pitcher"]
        team_counts = hitters["team_code"].value_counts()
        for team, cnt in team_counts.items():
            if cnt >= 3:
                stack_counter[f"{team} x{cnt}"] += 1
        pitchers = ldf[ldf["player_type"].str.lower() == "pitcher"]
        for name in pitchers["full_name"]:
            pitcher_counter[name] += 1
        for name in ldf["full_name"]:
            player_counter[name] += 1

    print("Top stacks (3+ hitters):")
    for stack, cnt in stack_counter.most_common(12):
        print(f"  {stack}: {cnt} lineups ({cnt / len(lineups):.0%})")
    print("Pitcher exposure:")
    for name, cnt in pitcher_counter.most_common(10):
        print(f"  {name}: {cnt} ({cnt / len(lineups):.0%})")
    print("Top player exposure:")
    for name, cnt in player_counter.most_common(15):
        print(f"  {name}: {cnt} ({cnt / len(lineups):.0%})")


def hard_validate(lineups, salary_cap: int, label: str) -> None:
    errors = []
    seen: set[frozenset] = set()
    for i, lu in enumerate(lineups, start=1):
        ldf = lu.dataframe
        ids = frozenset(ldf["fd_player_id"].astype(str))
        if len(ldf) != 9:
            errors.append(f"lineup {i}: {len(ldf)} players")
        if lu.total_salary > salary_cap:
            errors.append(f"lineup {i}: salary {lu.total_salary} > cap")
        n_pitchers = int((ldf["player_type"].str.lower() == "pitcher").sum())
        if n_pitchers != 1:
            errors.append(f"lineup {i}: {n_pitchers} pitchers")
        hitters = ldf[ldf["player_type"].str.lower() != "pitcher"]
        max_team = hitters["team_code"].value_counts().max()
        if max_team > MAX_HITTERS_PER_TEAM:
            errors.append(f"lineup {i}: {max_team} hitters from one team")
        if ids in seen:
            errors.append(f"lineup {i}: duplicate lineup within contest")
        seen.add(ids)
    slot_errors = validate_fanduel_lineups(lineups)
    for num, msg in slot_errors:
        errors.append(f"lineup {num}: slot assignment failed: {msg}")
    if errors:
        raise ValueError(f"{label} failed validation: " + "; ".join(errors[:10]))
    print(f"{label}: all {len(lineups)} lineups valid (cap, positions, 1P/8H, team limits, uniqueness)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fanduel", required=True, type=Path)
    ap.add_argument("--bpp-dir", required=True, type=Path)
    ap.add_argument("--dfs-file", action="append", type=Path, default=[])
    ap.add_argument("--paste", type=Path, default=None)
    ap.add_argument("--total", type=int, default=300)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    ap.add_argument("--tag", default="dual")
    ap.add_argument("--ceiling-weight", type=float, default=0.35)
    ap.add_argument("--ownership-penalty", type=float, default=0.12)
    ap.add_argument("--leverage-weight", type=float, default=0.20)
    ap.add_argument("--randomness", type=float, default=0.06)
    ap.add_argument("--min-salary", type=int, default=34200)
    ap.add_argument("--min-uniques", type=int, default=2)
    ap.add_argument("--dataset-pickle", type=Path, default=None,
                    help="Cache path: load the optimizer dataset if it exists, else build and save it")
    args = ap.parse_args()

    if args.dataset_pickle and args.dataset_pickle.exists():
        dataset = pd.read_pickle(args.dataset_pickle)
        print(f"Loaded cached dataset: {len(dataset)} players")
    else:
        dataset = build_slate(args.fanduel, args.bpp_dir, args.dfs_file, args.paste)
        dataset = apply_portfolio_exposure(dataset)
        if args.dataset_pickle:
            dataset.to_pickle(args.dataset_pickle)

    config = OptimizerConfig()  # chalk handled above; keep defaults otherwise
    dataset = config.apply_exposure_overrides(dataset)

    # Leverage-oriented stack rotation (sums to 8 hitters each):
    # heavy on 4-3-1 (primary+secondary stack), with 3-3-2 and 4-2-2 for shape
    # diversity and 4-4 for maximum correlation on a small portion.
    rotation = [
        (4, 3, 1), (4, 3, 1),
        (3, 3, 2),
        (4, 2, 2),
        (4, 4),
    ]

    print(f"\nGenerating {args.total} lineups (shared exposure portfolio)...")
    lineups = generate_lineups(
        dataset,
        num_lineups=args.total,
        salary_cap=SALARY_CAP,
        stack_rotation=list(rotation),
        bring_back_enabled=True,
        bring_back_count=1,
        leverage_weight=args.leverage_weight,
        randomness=args.randomness,
        ceiling_weight=args.ceiling_weight,
        ownership_penalty_weight=args.ownership_penalty,
        min_salary=args.min_salary,
        min_uniques=args.min_uniques,
    )
    print(f"Generated {len(lineups)} lineups")
    if len(lineups) < args.total:
        print(f"WARNING: short {args.total - len(lineups)} lineups — constraints too tight")

    dropped = sum(1 for lu in lineups if lu.stacks_dropped)
    if dropped:
        print(f"NOTE: {dropped} lineups fell back to no-stack solve")

    # Interleave split so both contests share the portfolio's exposure profile.
    contest_a = lineups[0::2][: args.total // 2]
    contest_b = lineups[1::2][: args.total // 2]

    hard_validate(contest_a, SALARY_CAP, "Contest A")
    hard_validate(contest_b, SALARY_CAP, "Contest B")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    path_a = args.out_dir / f"fanduel_upload_{stamp}_{args.tag}_contestA.csv"
    path_b = args.out_dir / f"fanduel_upload_{stamp}_{args.tag}_contestB.csv"
    write_fanduel_upload(contest_a, path_a)
    write_fanduel_upload(contest_b, path_b)

    summarize(lineups, dataset, "Combined 300-lineup portfolio")
    summarize(contest_a, dataset, "Contest A (150)")
    summarize(contest_b, dataset, "Contest B (150)")

    # Full lineup dump for review
    rows = []
    for contest, lus in (("A", contest_a), ("B", contest_b)):
        for i, lu in enumerate(lus, start=1):
            for _, r in lu.dataframe.iterrows():
                rows.append(
                    {
                        "contest": contest,
                        "lineup": i,
                        "player": r["full_name"],
                        "team": r["team_code"],
                        "pos": r.get("roster_position") or r.get("position"),
                        "salary": r["salary"],
                        "proj_mean": round(float(r["proj_fd_mean"]), 2),
                        "proj_upside": round(float(r.get("proj_fd_upside", 0.0)), 2),
                        "ownership": round(float(r.get("proj_fd_ownership", 0.0)), 3),
                        "batting_order": r.get("batting_order_position"),
                    }
                )
    detail_path = args.out_dir / f"lineups_detail_{stamp}_{args.tag}.csv"
    pd.DataFrame(rows).to_csv(detail_path, index=False)

    print(f"\nUpload CSVs:\n  {path_a}\n  {path_b}\nDetail: {detail_path}")


if __name__ == "__main__":
    main()
