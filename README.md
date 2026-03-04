# MLB DFS Slate Optimizer (Next Gen)

This repository is being rebuilt to support a fully automated workflow for the 2026 FanDuel MLB main slate grind. The legacy Flask UI is still here, but the new development happens under `src/slate_optimizer` with a data-first pipeline:

1. **Ingest** BallparkPal exports (and future data sources) into standardized tables.
2. **Analyze** each slate for weather, park factors, leverage, and stack quality.
3. **Project** custom mean/ceiling/volatility/ownership numbers for every player.
4. **Optimize** 150-lineup sets with stack, exposure, and diversification rules tuned for GPP play.

## Current status

- `scripts/ingest_ballparkpal.py` uses `slate_optimizer.ingestion.ballparkpal` to read the latest BallparkPal Excel files from a directory (e.g., `OneDrive/Desktop/DFSMLB`) and writes normalized CSVs into `data/raw/`.
- The ingestion module automatically normalizes column names to snake_case so the downstream pipeline has clean schemas to work with.

## Usage

```bash
python scripts/ingest_ballparkpal.py --source "C:\\Users\\wallg\\OneDrive\\Desktop\\DFSMLB" --output data/raw
```

Output files are timestamped by default (e.g., `20250417_ballparkpal_batters.csv`). Use `--tag some_label` to override the timestamp.

Next steps:
- Map BallparkPal player IDs to FanDuel IDs/salaries for the current slate.
- Store each slate ingestion in a lightweight database (SQLite/duckdb) for reproducibility.
- Layer in projection/analysis/optimization modules.
### Slate dataset builder

After ingesting the latest BallparkPal files, pair them with the FanDuel players list to generate a slate-ready dataset:

```bash
python scripts/build_slate_dataset.py \
    --bpp-source "C:\\Users\\wallg\\OneDrive\\Desktop\\DFSMLB" \
    --fanduel-csv cleaned_players_list.csv \
    --output data/processed \
    --tag sample
```

This writes `data/processed/<tag>_slate_players.csv` plus an `_unmatched_players.csv` report if any salaries could not be paired with BallparkPal sims. The resulting CSV is the foundation for projection adjustments, slate analytics, and the downstream optimizer.
Alias support: pass `--alias-file config/player_aliases.json` (a JSON map of `canonical_fd_name: canonical_target_name`) to smooth out naming mismatches (e.g., suffixes, diacritics). Both BallparkPal and FanDuel names are canonicalized before matching, and aliases are applied to both sides so you only have to list each quirky name once.
### Persistence preview

`SlateDatabase` (SQLite, located under `src/slate_optimizer/data/storage.py`) seeds tables for `slates` and `slate_players` so each processed slate can be versioned. Once we finalize the projection layer, we can call `SlateDatabase('data/slates.db').insert_slate(...)` and `write_players(...)` to save the combined dataset before optimization.
### Baseline projections

Once a slate is persisted (via `build_slate_dataset.py`), generate first-pass projections straight from the stored player payload:

```bash
python scripts/generate_projections.py \
    --db-path data/slates.db \
    --slate-id 1 \
    --output data/processed
```

Omit `--slate-id` to target the most recent slate in the DB. The script currently derives simple mean/floor/ceiling estimates from BallparkPal FanDuel points (falling back to FanDuel `FPPG`), and writes `<tag>_baseline_projections.csv` for downstream modeling.



Want to overweight hot streaks (or dampen them)? Use `--recency-blend season,recent` to control how much of the projection comes from 7/14 day samples versus the season average. For example, a 50/50 mix looks like:

```bash
python scripts/run_daily_pipeline.py \
    --bpp-source data/raw \
    --fanduel-csv data/processed/synthetic_fanduel.csv \
    --recent-stats-csv data/processed/sample_recent_stats.csv \
    --recency-blend 0.5,0.5
```
Need to tweak the platoon weights? Pass the optional multipliers to any projection-building script (or the Streamlit Step 1 form) to customize the boost/penalty applied to hitters based on handedness.

```bash
python scripts/generate_projections.py \
    --db-path data/slates.db \
    --slate-id 1 \
    --platoon-opposite-boost 1.08 \
    --platoon-same-penalty 0.93 \
    --platoon-switch-boost 1.02 \
    --output data/processed
```

The same `--platoon-*` trio is supported in `prepare_optimizer_dataset.py`, `run_daily_pipeline.py`, and `run_full_pipeline.py`, and the Streamlit workflow exposes matching number inputs on the Configure step so you can experiment interactively.
### Slate metadata enrichment

Each stored slate row now carries the BallparkPal team/game context (team runs, win%, side, game-level win odds, etc.), so downstream modules can reason about park/weather leverage without reparsing the Excel files. The merge flow first tries full-name matches, then falls back to a unique team + last-name combo when available, which already trimmed unmatched counts even on the cross-season sample set.

### Optimizer prep (coming up)

The persisted player payload already includes the fields the optimizer will need (`fd_player_id`, `position`, `salary`, `team_code`, `opponent_code`, projections). Next we’ll extend it with stack labels and exposure controls (e.g., `stack_key`, `max_exposure`, `ceiling`, `ownership`) so we can feed a single denormalized table into the lineup builder. With the DB in place, each slate can be re-hydrated, enriched, and sent through optimization experiments without re-running ingestion.
### Optimizer dataset

After persisting a slate you can build a single denormalized table that the upcoming lineup optimizer will consume:

```bash
python scripts/prepare_optimizer_dataset.py \
    --db-path data/slates.db \
    --slate-id 3 \
    --output data/processed
```

The dataset joins the stored player payload with the baseline projections, adds `stack_key`/`game_key`, assigns a `stack_priority` tier from BallparkPal team runs, and sets a default exposure cap (0.65 for bats, 0.40 for pitchers). The resulting CSV (e.g., `sample_db3_optimizer_dataset.csv`) is ready for stack/exposure experiments without re-reading the raw spreadsheets.
### Running the ILP solver

```bash
python scripts/run_optimizer.py \
    --dataset data/processed/sample_db3_optimizer_dataset.csv \
    --num-lineups 5 \
    --output data/processed/sample_db3_lineups.csv
```

The solver enforces FanDuel roster rules (P, C/1B, 2B, 3B, SS, 3 OF, 1 UTIL via the total player count) under a $35k cap, applies the per-player exposure caps from the dataset (default 65% for bats, 40% for pitchers), and iteratively adds uniqueness constraints so each lineup differs by at least one player. The CSV output tags every row with `lineup_id`, ready for exports or further analysis.
You can also tweak the solver via `--min-stack-size` (default 4 hitters from one team; set to 0 to disable) and every run prints the top player/stack exposures so you can sanity-check diversification before exporting.
### FanDuel upload helper

### Correlation-aware lineup controls

The PuLP solver now models key MLB correlations directly:

- **Pitcher vs. opposing hitters**: every lineup must choose a single pitcher, and that pitcher can never be stacked with hitters from the team he faces. This is enforced automatically inside generate_lineups() with no flags required.
- **Bring-back stacks** (--bring-back, --bring-back-count): when enabled, any qualifying team stack must include at least ring_back_count hitters from the opposing team in that same game. This is configurable via both the CLI and the Streamlit Configure tab.
- **Minimum game total for stacks** (--min-game-total): limit primary stacks to games with a Vegas total at or above the threshold. Players from lower-total games can still appear as one-offs, but they will not satisfy the stack constraint.

These options are available in both scripts/run_optimizer.py and the full scripts/run_daily_pipeline.py workflow (look for --bring-back, --bring-back-count, and --min-game-total). Streamlit exposes the same toggles under *Configure & Optimize* so you can experiment interactively.

### Verifying correlation constraints

Use any optimizer dataset (e.g., the ones in data/processed/) to validate the new behavior:

`ash
# Require a four-man stack plus a one-player bring-back from the opponent
python scripts/run_optimizer.py \
    --dataset data/processed/sample_pipeline_optimizer_dataset.csv \
    --num-lineups 5 \
    --min-stack-size 4 \
    --bring-back \
    --bring-back-count 1 \
    --output data/processed/sample_pipeline_lineups_bringback.csv

# Restrict stacks to the highest-total games (10+ implied runs)
python scripts/run_optimizer.py \
    --dataset data/processed/sample_pipeline_optimizer_dataset.csv \
    --num-lineups 5 \
    --min-stack-size 3 \
    --min-game-total 10 \
    --output data/processed/sample_pipeline_lineups_high_total.csv
`

Inspect the resulting CSVs (or the console exposure summaries) to confirm each lineup includes the mandated bring-back hitter(s), that all stacks originate from high-total games, and that no pitcher ever appears with hitters from the team he faces. The Streamlit dashboard reflects these same constraints when you export lineups from the Configure step.

After generating lineups, convert them into FanDuel's upload format (P, C/1B, 2B, 3B, SS, OF×3, UTIL):

```bash
python scripts/export_fanduel_upload.py \
    --lineups data/processed/sample_pipeline_lineups.csv \
    --output data/processed/sample_pipeline_fanduel_upload.csv
```

The exporter greedily assigns each lineup's players to the required roster slots (preferring true Cs for the C/1B slot, pure SS players for SS, etc.) and emits a CSV with columns `lineup_id,P,C1,SS,3B,2B,OF1,OF2,OF3,UTIL` ready to paste/upload into FanDuel.
### Full run orchestration

`python scripts/run_full_pipeline.py --bpp-source ... --fanduel-csv ... [--stack-templates 4,3] [--adjust]` wraps the whole workflow: slate ingest/persist, (optional) projection adjustment, leverage computation, optimizer run, and FanDuel upload export. All artifacts land under `data/processed/<tag>_...` so a single command reproduces the entire slate day.





