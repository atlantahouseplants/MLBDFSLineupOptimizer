# Dashboard Redesign Plan

## Goal

Replace the current developer-style dashboard with a simple three-tab daily workflow for one FanDuel MLB DFS player:

1. `Today's Slate`
2. `Run Optimizer`
3. `Review Lineups`

The dashboard should assume the user wants to play today’s FanDuel slate with as little setup as possible. The only daily required action is uploading the FanDuel salary CSV. Everything else should auto-load from `data/live/` and `data/output/`.

## Core Design Rules

- The user is a DFS player, not a developer.
- No sidebar full of filters.
- No raw file-picking unless there is no valid file to auto-detect.
- No optimizer jargon unless it maps directly to DFS decisions.
- Defaults should be strong enough that most days the user clicks upload, then run, then review.

## Three-Tab Structure

### 1. `Today's Slate`

Purpose: answer “What does today’s slate look like?” before the user runs anything.

What it shows:

- A clear slate status strip:
  - FanDuel salary file status
  - BallparkPal data status
  - Vegas lines status
  - Batting orders status
- Top team stack targets
  - Based on team run environment and stack quality
- Best pitchers
  - Based on projection, win odds, and ownership context
- Best leverage bats
  - Strong plays with lower ownership than their projection suggests
- Chalk warnings
  - Highest-owned hitters and pitchers likely to be popular
- Simple player pool table
  - Searchable
  - Focused columns only: player, team, position, salary, projection, ownership, leverage

What it should not show:

- Team multiselects
- Position multiselects
- Projection sliders
- Ownership sliders
- Dataset dropdowns
- Debug output

Daily user outcome:

- The user can scan the slate in 30 seconds and understand where the best stacks, pitchers, leverage plays, and chalk are.

### 2. `Run Optimizer`

Purpose: one place to upload today’s FanDuel CSV and generate lineups with minimal decisions.

What it shows:

- One prominent upload control for the FanDuel salary CSV
- A short “ready to run” checklist that auto-validates:
  - FanDuel CSV uploaded
  - BallparkPal files found in `data/live/`
  - Vegas lines found in `data/live/`
  - Batting orders found in `data/live/`
- A compact settings block with only the few user-facing controls worth exposing
- One primary `Run Optimizer` button
- A progress/log area with plain-English stages:
  - loaded salary file
  - loaded live data
  - built projections
  - estimated ownership
  - generated lineups
  - saved FanDuel upload file
- A success panel after completion with:
  - number of lineups generated
  - path/name of FanDuel upload CSV
  - path/name of lineup review CSV

### 3. `Review Lineups`

Purpose: answer “What lineups did I get, and do I want to play them?”

What it shows:

- Auto-load the newest generated lineup set from `data/output/`
- Summary cards:
  - number of lineups
  - average projection
  - average ownership
  - top stack teams
  - most-used pitcher
- Exposure board
  - player exposure percentages across all generated lineups
- Stack exposure board
  - team stack frequency
- Lineup-by-lineup table
  - lineup id
  - total salary
  - total projection
  - total ownership
  - key stack/team summary
- Expandable lineup details
  - each player in that lineup with projection and ownership
- Download actions:
  - FanDuel upload CSV
  - full lineup breakdown CSV

What it should not show:

- Historical output file picker unless auto-detection fails
- Raw processed dataset selection
- Internal config details

## Settings: Exposed vs Hidden

### Exposed settings

These are the only settings worth showing because they reflect real DFS choices:

- `Number of lineups`
  - Default: `20`
  - Reason: already matches pipeline default and common MME flow
- `Stack style`
  - Default: `4-3`
  - Maps to stack templates without making the user think in optimizer internals
- `Bring-back`
  - Default: `On`
  - Simple toggle
- `Maximum lineup ownership`
  - Default: prefilled recommended value, but optional
  - This is the only advanced lever regular DFS players may actually care about

### Hidden settings

These should not be visible in the dashboard:

- `salary_cap`
  - Hidden because FanDuel cap is fixed; keep default `35000`
- `min_stack_size`
  - Hidden behind stack style
- raw `stack_templates`
  - Hidden behind stack style labels
- `chalk_threshold`
  - Hidden
- `chalk_exposure_cap`
  - Hidden
- `config`
  - Hidden
- `db_path`
  - Hidden
- `output_dir`
  - Hidden
- `alias_file`
  - Hidden
- `ownership_sources`
  - Hidden
- `ownership_weights`
  - Hidden
- `recency_blend`
  - Hidden
- platoon tuning args
  - Hidden
- `min_game_total`
  - Hidden unless later promoted as an advanced optional toggle
- `bring_back_count`
  - Hidden behind the bring-back toggle, default to `1`
- `write_intermediate`
  - Hidden and enabled by app behavior if needed for review screens
- `auto_fetch`
  - Hidden for now because the product assumption is that live files already exist in `data/live/`

## Data File Map

### `Today's Slate` reads from

Primary live inputs:

- `data/live/bpp_batters_<date>.csv`
- `data/live/bpp_pitchers_<date>.csv`
- `data/live/bpp_dfs_projections_<date>.csv`
- `data/live/batting_orders_<date>.csv`
- `data/live/vegas_lines_<date>.csv`
- `data/live/handedness_<date>.csv`
- optional supporting files if present:
  - `data/live/bpp_games_<date>.csv`
  - `data/live/bpp_teams_<date>.csv`
  - `data/live/bpp_park_factors_<date>.csv`
  - `data/live/probable_pitchers_<date>.csv`

Preferred derived file if already available:

- newest `data/output/*_optimizer_dataset.csv`

Why:

- `*_optimizer_dataset.csv` already has the best unified slate view for projections, ownership, leverage, Vegas context, stack priority, and BPP run data.
- If a current optimizer dataset exists for today, `Today's Slate` should prefer it.
- If not, the tab should still surface slate context from `data/live/` as a pre-run preview.

### `Run Optimizer` reads from

- uploaded FanDuel salary CSV from the user
- `data/live/` auto-detected files:
  - `bpp_*`
  - `batting_orders_*`
  - `vegas_lines_*`
  - `handedness_*`
- optional existing outputs only for showing last-run context:
  - newest `data/output/*_lineups.csv`
  - newest `data/output/*_fanduel_upload.csv`

### `Review Lineups` reads from

- newest `data/output/*_lineups.csv`
- newest `data/output/*_fanduel_upload.csv`
- newest matching `data/output/*_optimizer_dataset.csv`
- newest matching `data/output/*_ownership_predicted.csv`

Why:

- `*_lineups.csv` provides lineup-level and exposure-level review
- `*_fanduel_upload.csv` is the file the user actually needs to upload to FanDuel
- `*_optimizer_dataset.csv` adds player projection and leverage context during lineup review
- `*_ownership_predicted.csv` is useful if ownership needs to be shown separately or validated against the optimizer dataset

## How `Run Optimizer` Calls the Pipeline

The dashboard should call `scripts/run_daily_pipeline.py` through a subprocess using the uploaded FanDuel CSV plus auto-detected live files.

Expected command shape:

```bash
python scripts/run_daily_pipeline.py \
  --bpp-source data/live \
  --fanduel-csv <uploaded_temp_or_saved_csv_path> \
  --vegas-csv data/live/vegas_lines_<date>.csv \
  --batting-orders-csv data/live/batting_orders_<date>.csv \
  --handedness-csv data/live/handedness_<date>.csv \
  --output-dir data/output \
  --write-intermediate \
  --num-lineups <ui_value> \
  --salary-cap 35000 \
  --stack-templates <mapped_from_stack_style> \
  [--bring-back] \
  [--bring-back-count 1] \
  [--max-lineup-ownership <ui_value_if_set>] \
  --tag <generated_tag>
```

Notes:

- `--bpp-source` should always point to `data/live`
- The uploaded FanDuel CSV should be saved to a stable temporary or live input path before execution
- `--write-intermediate` should be enabled so the review tab has consistent outputs to read
- `stack style` UI labels should map cleanly to pipeline values:
  - `4-3` -> `--stack-templates 4,3`
  - `4-4` -> `--stack-templates 4,4`
  - `5-3` -> `--stack-templates 5,3` if supported by roster constraints
- `bring-back` toggle should add `--bring-back`
- The app should generate a clear slate tag for the run, ideally based on date and mode
- The dashboard should capture stdout and show plain-English progress, not raw traceback unless the run fails

Failure behavior:

- If required live files are missing, block the run and show exactly which file type is missing
- If the optimizer returns no feasible lineups, show that clearly and suggest the user retry with default ownership cap behavior
- If the run succeeds, auto-refresh the `Review Lineups` tab state

## Color and Style Philosophy

Theme direction:

- Dark theme throughout
- Clean sportsbook-style look, not notebook-style analytics
- Big, high-contrast cards and simple tables
- One primary accent color for actions and positive signals

Color rules:

- Green = good
  - strong stack spots
  - positive leverage
  - successful run state
  - completed data checks
- Red = bad
  - bad leverage
  - missing required files
  - optimizer failure
  - chalk warnings when a player is extremely popular
- Yellow/amber = caution
  - incomplete batting orders
  - optional files missing
  - ownership cap too restrictive
- Neutral grays = labels, secondary data, dividers

Usage philosophy:

- Do not color everything
- Reserve strong color for decisions and alerts
- Tables should use color to answer “is this good, bad, or risky?” at a glance
- Positive leverage and favorable stacks should visually stand out first

## Recommended UX Flow

1. User opens the app.
2. `Today's Slate` already shows the day’s context from auto-detected files.
3. User goes to `Run Optimizer`.
4. User uploads the FanDuel salary CSV.
5. User keeps the defaults.
6. User clicks `Run Optimizer`.
7. User lands on `Review Lineups`.
8. User checks exposure, stacks, and lineup quality.
9. User downloads the FanDuel upload CSV and enters contests.

## Implementation Guardrails

- Keep the app to three tabs only.
- No left-sidebar settings panel.
- No processed-data dropdowns.
- Auto-detect the newest valid files by date and type.
- Prefer one recommended path over optionality.
- Use DFS language:
  - “Top Stacks”
  - “Best Pitchers”
  - “Leverage Bats”
  - “Chalk”
  - “Lineup Exposures”
- Avoid engineering language:
  - “dataset”
  - “intermediate output”
  - “ownership sources”
  - “config overrides”

## Final Product Definition

If the redesign is successful, the user should be able to do the full daily workflow with this mental model:

- Check the slate
- Upload FanDuel salaries
- Run lineups
- Review and download

No other daily decisions should be required unless the user explicitly wants them.
