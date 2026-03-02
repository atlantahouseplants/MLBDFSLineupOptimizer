# MLB DFS Slate Optimizer V2 — Product Requirements Document

**Contrarian Lineup Construction System | FanDuel MLB**
**Version 2.0 | March 2026 | Status: Ready for Development**

---

## 1. Executive Summary

### 1.1 What Exists Today

The current system is a functional MLB DFS lineup optimizer built in Python with a multi-stage pipeline: BallparkPal + FanDuel data ingestion, baseline projection engine, leverage scoring (player + team level), a PuLP-based integer linear programming solver with stacking and exposure controls, and two front-end interfaces (Flask web app and Streamlit dashboard). The contrarian philosophy is encoded via leverage scores, chalk thresholds, ownership caps, and max-lineup-ownership constraints.

### 1.2 Core Problem

The system has strong architecture but weak data inputs. The projection model is naive (simple BallparkPal multipliers), there is no dedicated ownership projection model, no backtesting framework, no live-slate workflow, and the two frontends are redundant. For a contrarian strategy, accurate ownership projections are existential — if ownership numbers are wrong, every leverage score is meaningless.

### 1.3 V2 Vision

Transform this from a prototype into a daily-use competitive tool by upgrading the projection and ownership models, adding a backtesting engine, building a streamlined single-UI daily workflow, and integrating late-swap support. The guiding principle: get the inputs right (projections + ownership), and the existing optimizer + contrarian framework will do the rest.

---

## 2. DFS Philosophy & Strategy Context

> **This section exists so the coding agent understands WHY the system is designed this way. Every technical decision should serve this strategy.**

**Core Thesis:** Baseball is a high-variance fail sport. The best hitters fail 70% of the time. On any given slate, the chalk (highest-owned players) will frequently underperform their ownership. When chalk fails, every lineup that used those players loses. Lineups built with under-owned alternatives (leveraged off chalk) gain a massive edge by avoiding the field's most common failure points.

**Ownership Pivots:** The key move is identifying where the field is overweight on a player relative to their true probability of outperforming, and finding alternatives with similar or better upside at lower ownership. This is a leverage play — you're not trying to be different for its own sake, you're trying to capture the same upside at a lower cost in field duplication.

**Stacking in MLB:** Batting orders are correlated. When a team scores, multiple hitters in the lineup tend to produce. Stacking 4–5 hitters from the same team captures this correlation. The contrarian angle: finding the RIGHT stacks (high run environment, weak opposing pitcher) that the field is underweight on.

**What This Means for the System:** Ownership projection accuracy is the #1 priority. The optimizer is already good at building lineups — it just needs better inputs. Every feature should be evaluated through the lens of: does this help me find better ownership pivots?

---

## 3. Existing System Architecture (Preserve & Extend)

The following components are solid and should be preserved. All V2 work extends or enhances these — **do not rewrite from scratch.**

| Component | Current Implementation | Status |
|---|---|---|
| **Data Ingestion** | BallparkPal Excel loader + FanDuel CSV parser + alias system | ✅ Solid — keep as-is, add new data sources alongside |
| **Projection Engine** | Baseline model using BPP multipliers (team runs, pitcher win%, park/weather adjustments) | ⚠️ Functional but needs major upgrade (see Section 4) |
| **Leverage Scoring** | Player-level and team-level leverage (projection rank minus ownership rank) | 🔵 Good concept, accuracy depends on input quality |
| **Optimizer / Solver** | PuLP ILP solver with FanDuel constraints, stacking, exposure limits, uniqueness | ✅ Strong — keep as-is |
| **Config System** | JSON-based config for chalk thresholds, exposure caps, ownership caps, stack templates | ✅ Good — extend with new parameters |
| **Frontends** | Flask web app (upload + optimize) and Streamlit dashboard (analysis + visualization) | ⚠️ Consolidate into single Streamlit app (see Section 7) |
| **Pipeline Scripts** | run_daily_pipeline.py, run_optimizer.py, compute_leverage.py, run_full_pipeline.py | ✅ Good orchestration — extend |
| **Storage** | SQLite slate database + CSV outputs | ✅ Adequate for V2 — extend schema for results tracking |

---

## 4. V2 Feature Requirements

Features are organized into prioritized workstreams. P0 features are blocking for season use. P1 features provide significant competitive edge. P2 features are quality-of-life improvements.

---

### 4.1 Enhanced Projection Model (P0 — Critical)

The current projection model uses simple BallparkPal multipliers. V2 needs a multi-factor model that blends multiple data sources to produce more accurate mean, floor, and ceiling projections.

#### 4.1.1 Vegas Integration

- Ingest game-level Vegas data: total (over/under), moneyline, and run line for each game on the slate.
- Derive implied team totals from the over/under and moneyline (standard conversion formula).
- Use implied team total as the primary scaling factor for hitter projections — this replaces the naive BPP team run multiplier.
- Map Vegas data to each player via team/game matching in the optimizer dataset.

**Data Source Options:** Accept manual CSV upload of Vegas lines (simplest for V2), or scrape from a free API like The Odds API. Do not hardcode a single source — use an adapter pattern so sources can be swapped.

**CSV Format Expected:**
```
game,total,home_ml,away_ml
NYY@BOS,9.5,-130,+110
LAD@SF,8.0,-150,+130
```

#### 4.1.2 Batting Order Integration

- Ingest confirmed (or projected) batting orders for each team.
- Apply a batting order position multiplier to hitter projections. Top-of-order hitters (1–4) get more plate appearances and higher run/RBI opportunity.
- Standard multiplier curve: 1st=1.12, 2nd=1.08, 3rd=1.10, 4th=1.06, 5th=1.02, 6th=1.00, 7th=0.96, 8th=0.94, 9th=0.92
- Flag players with no confirmed batting order position (bench risk).

**Data Source:** Accept CSV upload of batting orders, or integrate with a site like RotoGrinders, BaseballPress, or MLB.com lineups page.

**CSV Format Expected:**
```
team,order_position,player_name
BOS,1,Jarren Duran
BOS,2,Rafael Devers
BOS,3,Masataka Yoshida
```

#### 4.1.3 Platoon Splits

- Track whether each batter hits left or right, and whether the opposing starting pitcher throws left or right.
- Apply a platoon adjustment factor: batters facing opposite-hand pitchers get a boost (~5-8%), same-hand get a penalty (~3-5%). These percentages are configurable.
- Store handedness data alongside the player record in the optimizer dataset.

#### 4.1.4 Recent Performance Weighting

- Incorporate a rolling 7-day and 14-day performance metric (FanDuel points per game) as a recency signal.
- Blend recency signal with season-long projection at a configurable ratio (default: 70% season / 30% recent).
- This captures hot/cold streaks that BallparkPal's season-level model misses.

**Data Source:** CSV upload of recent player stats, or pull from a free stats API.

#### 4.1.5 Projection Output

Final projection for each player should include:
- `proj_fd_mean` — blended projection
- `proj_fd_floor` — 10th percentile estimate
- `proj_fd_ceiling` — 90th percentile estimate
- `proj_fd_value` — projection / salary * 1000
- `vegas_factor` — implied team total adjustment
- `order_factor` — batting order multiplier
- `platoon_factor` — handedness adjustment
- `recency_factor` — hot/cold streak adjustment

All factors should be logged and visible in the dashboard so the user can audit why a player is projected where they are.

---

### 4.2 Ownership Projection Model (P0 — Critical)

**This is the single most important feature for a contrarian strategy.** If ownership projections are inaccurate, every leverage score, chalk threshold, and ownership cap in the system produces garbage output. The current system appears to pass through BallparkPal's ownership numbers without a dedicated model.

#### 4.2.1 Ownership Model Requirements

- Build a model that estimates projected ownership percentage for every player on the slate.
- Inputs to the model should include:
  - Salary (higher salary = higher ownership, strong correlation)
  - Projection rank (top projected players attract more ownership)
  - Value rank (pts/$ — the field chases value)
  - Team implied total (high-implied-total teams attract stacks)
  - Name recognition factor (stars get overowned relative to projection)
  - Positional scarcity (thin positions force ownership concentration)
- Output: projected ownership % for each player.

#### 4.2.2 Ownership Sources & Blending

- Accept ownership projections from multiple sources as CSV inputs: BallparkPal, RotoGrinders, FantasyLabs, SaberSim, or any other source the user subscribes to.
- Implement a configurable weighted blend of multiple ownership sources (e.g., 40% SaberSim, 30% FantasyLabs, 30% BallparkPal).
- If only one source is available, use it directly. The system should never break because a source is missing.

**CSV Format Expected (per source):**
```
player_name,team,position,projected_ownership
Shohei Ohtani,LAD,OF,28.5
Mookie Betts,LAD,OF,22.1
```

#### 4.2.3 Ownership-Adjusted Leverage Scores

- Once ownership projections are upgraded, recalculate the existing `leverage_score` and `team_leverage_score` columns using the blended ownership numbers.
- Add a new `ownership_edge` metric: (player's projected ceiling / ownership%) to quantify upside per unit of ownership cost. This is the core contrarian metric.

---

### 4.3 Backtesting & Results Tracking (P1 — High)

Without backtesting, you can't know if the contrarian approach is actually working. This is essential for tuning the system's parameters (chalk thresholds, exposure caps, ownership caps) based on real results rather than guessing.

#### 4.3.1 Historical Results Storage

Extend the SQLite database (slates.db) with tables for:

**slate_results:**
```sql
CREATE TABLE slate_results (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    contest_type TEXT,
    entry_fee REAL,
    num_entries INTEGER,
    winning_score REAL,
    cash_line REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**actual_scores:**
```sql
CREATE TABLE actual_scores (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    fd_player_id TEXT,
    player_name TEXT,
    actual_fd_points REAL,
    actual_ownership_pct REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**lineup_results:**
```sql
CREATE TABLE lineup_results (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    lineup_id INTEGER,
    total_actual_points REAL,
    rank INTEGER,
    payout REAL,
    roi REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

After each slate, the user should be able to upload or input actual results and actual ownership percentages.

#### 4.3.2 Key Metrics to Track

- **Ownership projection accuracy:** mean absolute error of projected vs. actual ownership for each source and the blend.
- **Projection accuracy:** mean absolute error of projected vs. actual FanDuel points.
- **Leverage ROI:** track whether high-leverage (low ownership, high projection) players outperform on a per-dollar basis over time.
- **Lineup performance:** cash rate (% of lineups that cash), average ROI, max score, min score per slate.
- **Chalk vs. contrarian split:** when chalk performs well, how do contrarian lineups do? When chalk busts, how much edge is gained?

#### 4.3.3 Backtest Dashboard

- New Streamlit dashboard tab showing cumulative performance over time.
- Charts: ownership projection accuracy trend, projection accuracy trend, ROI over time, leverage score correlation with actual outperformance.
- Filterable by date range, contest type, and strategy settings used.

---

### 4.4 Streamlined Daily Workflow (P1 — High)

The current system requires running multiple CLI scripts with various arguments. V2 should provide a single-screen daily workflow that a non-technical user can operate.

#### 4.4.1 Unified Streamlit App

- Consolidate the Flask web app and the Streamlit dashboard into a single Streamlit application. **Retire the Flask app.**
- The app should present a step-by-step daily workflow as tabs or a sidebar wizard.

#### 4.4.2 Daily Workflow Steps

**Step 1 — Slate Setup:**
- Upload FanDuel player CSV
- Upload BallparkPal Excel files
- Optionally upload Vegas lines CSV
- Optionally upload batting orders CSV
- Optionally upload ownership projections from external sources
- System validates all inputs and reports any missing/mismatched players

**Step 2 — Review Projections:**
- Display the full optimizer dataset with all projection factors visible
- Sortable/filterable by team, position, salary, projection, ownership, leverage
- Allow manual overrides: the user should be able to adjust any player's projection or ownership directly in the UI

**Step 3 — Configure & Optimize:**
- Set number of lineups, stack templates, chalk threshold, chalk exposure cap, max lineup ownership, and any player-specific exposure overrides
- Load/save configurations as JSON presets
- Run optimizer and display results

**Step 4 — Review Lineups:**
- Display generated lineups with per-lineup stats (total projection, total salary, total ownership, leverage score)
- Player exposure summary
- Stack exposure summary
- Ability to lock/exclude players and re-run
- Export to FanDuel-compatible CSV for direct upload

**Step 5 — Post-Slate Review (after games):**
- Upload actual results and actual ownership
- View performance metrics
- Data is stored for backtesting

---

### 4.5 Late Swap & Lineup Confirmation (P1 — High)

MLB slates have staggered lock times. Players get scratched from lineups. This feature prevents you from getting stuck with a scratched player in a locked lineup.

#### 4.5.1 Requirements

- Track lock times per game on the slate (derivable from the FanDuel CSV or game start times).
- Before lock: flag any players in generated lineups who are NOT in confirmed batting orders (bench risk).
- Provide a re-optimization workflow: remove scratched/unconfirmed players, re-run the optimizer for affected lineups only, and regenerate with the same strategy settings.
- Display a countdown or status indicator for each game's lock time.

---

### 4.6 FanDuel Export Compatibility (P0 — Critical)

Generated lineups must be directly uploadable to FanDuel's bulk lineup upload feature.

- Output a CSV in FanDuel's exact required format: columns must match their template (player IDs, positions, etc.).
- The current system outputs lineup CSVs but they may not match FanDuel's upload format exactly. Verify and fix this.
- Include a one-click download button in the Streamlit UI.

**FanDuel's expected format uses player IDs (the numeric ID from their CSV) in positional columns:**
```
P,C/1B,2B,3B,SS,OF,OF,OF,UTIL
12345,23456,34567,45678,56789,67890,78901,89012,90123
```

---

### 4.7 Correlation-Aware Optimizer Enhancements (P2 — Medium)

The current optimizer enforces team stacking but doesn't model other correlations. These enhancements improve lineup construction quality.

#### 4.7.1 Bring-Back / Opposing Stack Correlation

- In high-total games, both teams tend to score. Add an optional constraint: for the primary stack team, also include 1–2 hitters from the opposing team ("bring-back").
- This is configurable (on/off, number of bring-back players) per optimization run.

#### 4.7.2 Pitcher-Stack Anti-Correlation

- The system should never put a starting pitcher and hitters from the opposing team in the same lineup (the pitcher's success means the opposing hitters fail, and vice versa).
- Verify this constraint exists in `solver.py`; add if missing.

#### 4.7.3 Game Environment Targeting

- Add a constraint or preference to target hitters from games with the highest Vegas totals.
- Allow the user to set a minimum game total threshold (e.g., only stack teams in games with 8.5+ total).

---

## 5. Priority & Sequencing Matrix

**Build in this order. Each phase should be shippable and usable independently.**

| Phase | Feature | Priority | Effort Est. | Depends On |
|---|---|---|---|---|
| 1 | FanDuel Export Fix | P0 — Critical | Small (1–2 days) | None |
| 1 | Vegas Line Integration | P0 — Critical | Medium (3–5 days) | None |
| 1 | Ownership Projection Model | P0 — Critical | Medium (3–5 days) | None |
| 2 | Batting Order Integration | P1 — High | Small (2–3 days) | None |
| 2 | Platoon Splits | P1 — High | Small (1–2 days) | Batting Orders |
| 2 | Recent Performance Weighting | P1 — High | Small (1–2 days) | None |
| 3 | Unified Streamlit Workflow | P1 — High | Large (5–8 days) | Phases 1–2 |
| 3 | Late Swap Support | P1 — High | Medium (3–4 days) | Unified UI |
| 4 | Backtesting & Results Tracking | P1 — High | Large (5–7 days) | Unified UI |
| 5 | Correlation Enhancements | P2 — Medium | Medium (3–4 days) | None |

---

## 6. Technical Constraints & Guidelines

### 6.1 Preserve What Works

- **DO NOT rewrite** the PuLP optimizer/solver (`src/slate_optimizer/optimizer/solver.py`). It works. Extend the dataset it receives.
- **DO NOT rewrite** the BallparkPal or FanDuel ingestion modules. Add new data source adapters alongside them.
- **Keep** the `OptimizerConfig` JSON system. Add new config keys as needed.
- **Keep** the SQLite storage layer. Extend the schema for new tables.

### 6.2 Code Architecture

- All new data source integrations should follow the same adapter pattern as `BallparkPalLoader` and `FanduelCSVLoader`: a loader class with a `load()` method that returns a standardized DataFrame.
- New projection factors should be added as composable functions in `src/slate_optimizer/projection/`, following the pattern of `_team_run_multiplier` and `_pitcher_win_multiplier`.
- The `build_optimizer_dataset` function in `src/slate_optimizer/optimizer/dataset.py` should be extended (not replaced) to include new columns.
- All new features should have corresponding entries in the `OPTIMIZER_COLUMNS` list.

### 6.3 Technology Stack

- Python 3.10+. All existing dependencies in `requirements.txt` should be preserved.
- Add new dependencies as needed but prefer standard libraries and well-maintained packages.
- **Streamlit** for all UI work (retire Flask). Use Streamlit's session state for workflow state management.
- **SQLite** for all persistent storage (no need for Postgres in V2).
- **PuLP** remains the solver. Do not switch to OR-Tools or Gurobi unless PuLP performance becomes a bottleneck (unlikely at 20–150 lineups).

### 6.4 File & Data Format Standards

- All intermediate data should be saved as CSV for debuggability.
- The optimizer dataset CSV is the single source of truth that the solver consumes. All enrichments (Vegas, batting order, ownership, etc.) must be merged into this file before optimization.
- FanDuel export CSVs must match FanDuel's exact upload format (test this with an actual FanDuel contest entry).

---

## 7. Acceptance Criteria

Each feature is considered complete when the following criteria are met:

**Vegas Integration:**
- User can upload a Vegas lines CSV and see implied team totals reflected in player projections.
- Hitter projections demonstrably shift up/down based on game total.

**Ownership Model:**
- System produces an ownership projection for every player on the slate.
- User can upload multiple ownership sources and configure blend weights.
- Leverage scores use the blended ownership values.

**Backtesting:**
- User can input actual results after a slate and see performance metrics.
- Ownership projection accuracy is tracked and graphed over time.

**Daily Workflow:**
- A user can go from raw data files to exported FanDuel lineups in a single Streamlit session without touching the CLI.
- The entire workflow (upload → review → configure → optimize → export) takes under 10 minutes.

**Late Swap:**
- Scratched players are flagged before lock. Affected lineups can be re-optimized with one click.

**FanDuel Export:**
- Exported CSV uploads to FanDuel without errors on the first attempt.

---

## 8. Out of Scope for V2

- Machine learning projection models (save for V3 when backtesting data is accumulated).
- Multi-site support (DraftKings, etc.) — FanDuel only for V2.
- Automated contest entry or API integration with FanDuel (manual upload is fine).
- Real-time odds streaming (manual CSV upload of Vegas lines is sufficient for V2).
- Multi-sport support — MLB only.
- Cloud deployment — local machine is fine for V2.

---

## 9. Key File Reference for Development

**Read these files first to understand the current system before making changes.**

| File Path | Purpose |
|---|---|
| `src/slate_optimizer/optimizer/solver.py` | The PuLP lineup solver. **DO NOT REWRITE.** Extend constraints only. |
| `src/slate_optimizer/optimizer/dataset.py` | Builds the optimizer-ready dataset. **ADD** new columns here for new data. |
| `src/slate_optimizer/optimizer/config.py` | OptimizerConfig dataclass. **ADD** new config keys here. |
| `src/slate_optimizer/projection/baseline.py` | Current projection engine. **ENHANCE** with new factor functions. |
| `src/slate_optimizer/ingestion/ballparkpal.py` | BallparkPal loader. **KEEP.** Use as a pattern for new data adapters. |
| `src/slate_optimizer/ingestion/fanduel.py` | FanDuel CSV loader. **KEEP.** Use as a pattern for new data adapters. |
| `src/slate_optimizer/ingestion/slate_builder.py` | Merges FanDuel + BPP data into player dataset. **EXTEND** for new sources. |
| `scripts/run_daily_pipeline.py` | Full pipeline orchestration. **UPDATE** to include new data steps. |
| `scripts/compute_leverage.py` | Standalone leverage analysis. **UPDATE** formulas when ownership model changes. |
| `dashboard/app.py` | Streamlit dashboard. **EVOLVE** into the unified daily workflow UI. |
