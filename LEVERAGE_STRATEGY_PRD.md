# MLB DFS Leverage Strategy Overhaul — Implementation PRD

**Project:** MLBDFSLineupOptimizer  
**Repo:** `github.com/atlantahouseplants/MLBDFSLineupOptimizer`  
**Local path:** `C:\Users\wallg\MLBDFSLineupOptimizer`  
**Date:** March 31, 2026  
**Status:** Ready for implementation

---

## 0. Read This First — Agent Instructions

**You are implementing a leverage-based GPP strategy for an MLB DFS lineup optimizer.** Before writing any code, read this entire document. Then read the existing codebase files listed in Section 1.2. Do NOT rewrite modules from scratch — extend and modify existing code.

**Implementation order matters.** Follow the numbered sections in order (Section 3 → 4 → 5 → 6 → 7 → 8). Each section builds on the previous. After each section, run existing tests and any new tests to confirm nothing broke.

**Core philosophy in one sentence:** The optimizer should build lineups that maximize *leverage-weighted upside* — not raw projected points — by preferring under-owned players with high ceilings and stacking teams the field is ignoring.

---

## 1. Context & Architecture

### 1.1 What Exists Today

The system is a multi-stage MLB DFS pipeline:

1. **Ingestion** — BallparkPal Excel files + FanDuel CSV → merged player dataset
2. **Projections** — `baseline.py` computes `proj_fd_mean`, `proj_fd_floor`, `proj_fd_ceiling` using BPP data with batting order, platoon, recency, Vegas, park, and weather adjustments
3. **Ownership** — `ownership.py` either blends external CSV ownership sources or falls back to a synthetic model that estimates ownership from salary/projection/value/Vegas/name/scarcity ranks
4. **Dataset builder** — `dataset.py` merges everything into an optimizer-ready DataFrame with leverage scores
5. **ILP Solver** — `solver.py` uses PuLP to generate lineups maximizing `proj_fd_mean` subject to salary, position, stacking, exposure, and uniqueness constraints
6. **Simulation** — Monte Carlo engine with correlated player distributions (t-copula), field simulation, and contest simulation
7. **Portfolio selection** — Greedy selector picks final lineups from simulation candidates based on top-1% rate, win rate, or cash rate with diversity/exposure constraints
8. **Dashboard** — Streamlit UI (`dashboard/daily_workflow.py`) orchestrates the full pipeline

### 1.2 Files You Must Read Before Coding

Read these files completely to understand the current architecture:

```
src/slate_optimizer/optimizer/solver.py        — ILP solver (THE MAIN TARGET)
src/slate_optimizer/optimizer/dataset.py       — Dataset builder with leverage columns
src/slate_optimizer/optimizer/config.py        — Optimizer config dataclass
src/slate_optimizer/projection/ownership.py    — Ownership model (MAJOR TARGET)
src/slate_optimizer/projection/baseline.py     — Projection engine
src/slate_optimizer/simulation/correlation.py  — Correlation matrix builder
src/slate_optimizer/simulation/distributions.py — Player distribution fitting
src/slate_optimizer/simulation/lineup_selector.py — Portfolio selection
src/slate_optimizer/simulation/config.py       — Simulation config
dashboard/daily_workflow.py                    — Streamlit dashboard (UI changes)
```

### 1.3 What's Wrong With The Current System

These are the specific problems this PRD fixes:

| Problem | Where | Impact |
|---------|-------|--------|
| **Solver maximizes only `proj_fd_mean`** | `solver.py` line 116 | Builds chalk lineups. No awareness of ceiling or ownership. A pure mean optimizer is a cash-game tool, not a GPP tool. |
| **"Auto" stack mode has zero constraints** | `solver.py` line 548 | When user picks Auto, solver picks 9 best players regardless of team. Produces uncorrelated hodgepodge lineups with random 1-1-1-1-1-1-1-1 team distributions. |
| **Ownership model is too crude for leverage** | `ownership.py` lines 93-167 | Fallback model uses rank-based heuristics with equal-ish weights. Doesn't model field behavior accurately enough — salary influence is underweighted, chalk dynamics are missing. |
| **Ceiling doesn't drive lineup construction** | `solver.py` | `proj_fd_ceiling` is computed but only used in simulation distributions. The optimizer never sees it. |
| **No game-environment scoring** | nowhere | No system identifies which games are high-leverage (good Vegas total + low ownership). Stack selection is projection-driven, not leverage-driven. |
| **Correlation slider doesn't affect lineup building** | `correlation.py` + `solver.py` | Users think raising correlation to 0.5 will produce better stacks. It only affects simulation scoring, not lineup construction. |

---

## 2. DFS Strategy — Why These Changes Matter

### 2.1 The Leverage Theory

In a large-field GPP (1000+ entries), you don't win by having the highest projected lineup. You win by having a high-scoring lineup that *nobody else has*. The math:

- If you and 50 other people stack the Yankees and the Yankees score 10 runs, you split the top payout slots 50 ways.
- If you're the only person who stacked the Guardians and the Guardians score 10 runs, you get those top slots alone.

**Tournament value ≈ (probability of ceiling outcome) / (ownership%)**

This means the optimizer should actively seek players and stacks where the upside-to-ownership ratio is favorable.

### 2.2 Strategy Rules

These rules define how the optimizer should behave. They are non-negotiable for GPP mode.

1. **Never build lineups by mean alone.** The objective function must incorporate ceiling and ownership leverage.
2. **Eat the chalk strategically.** When a player is so dominant that fading them is reckless (e.g., top pitcher, 2+ point projection gap), roster them — but differentiate elsewhere.
3. **Stack from underowned games.** Prefer teams with high Vegas implied totals and low projected ownership. The field chases the obvious stacks (NYY, LAD, etc.). Find the games where the field is sleeping.
4. **On chalky teams, play the underowned batters.** If you must stack the Dodgers, avoid the highest-owned Dodgers. Play the 6th-7th hitters in the lineup who have the same team correlation upside but fraction of the ownership.
5. **Boom-bust distributions matter.** For GPP, we want players who have high variance. A safe 10-point floor with a 14-point ceiling is a cash play. A 2-point floor with a 25-point ceiling is a GPP play.
6. **Smart leverage, not stupid leverage.** Never play a player just because they're low-owned. The player must have a legitimate path to a ceiling game — confirmed in the lineup, favorable matchup, good park/weather, batting in the top 6.

### 2.3 What "Good" Looks Like

A well-constructed GPP portfolio of 20 lineups should have:

- **2-4 distinct primary stacks** (the team you have 3-4 batters from) spread across different games
- **60-70% of primary stacks from games with team implied totals >= 4.5 AND team ownership < 15%** (leverage stacks)
- **At most 20-30% of lineups stacking the highest-owned team** on the slate
- **Average lineup ownership between 40-80%** (not too chalky, not too contrarian)
- **High ceiling lineups** — the median lineup ceiling projection should be 15%+ above the median lineup mean projection
- **Player exposure diversity** — no single batter in more than 35-40% of lineups, no single pitcher in more than 55-60%

---

## 3. Enhanced Ownership Model

**Files to modify:** `src/slate_optimizer/projection/ownership.py`  
**Files to update:** `src/slate_optimizer/optimizer/dataset.py`

### 3.1 Problem

The current `_estimate_fallback()` function produces ownership estimates that are too flat and don't model real field behavior well. In actual FanDuel contests, ownership is heavily driven by salary (the field chases expensive players), name recognition (stars get overowned), and recent narratives (hot streaks, injuries to starters ahead of a player, etc.).

### 3.2 New Ownership Model Design

Replace the fallback model internals with a more realistic estimation that accounts for how DFS players actually behave. Keep the existing external-source blending pipeline (`compute_ownership_series`) and the `OwnershipModelConfig` dataclass — just improve the fallback estimation.

#### 3.2.1 Salary-Driven Ownership Curve

Real MLB DFS ownership follows a power-law relationship with salary. The most expensive players get disproportionately high ownership. Replace the linear rank-based approach with an exponential salary curve:

```
salary_signal = (salary / max_salary_on_slate) ^ salary_exponent
```

Where `salary_exponent` defaults to 1.8 (steeper than linear, reflecting real field behavior). This creates a curve where $10K players get much more ownership than $7K players, which matches reality.

#### 3.2.2 Value Trap Detection

The field chases "value" (points per dollar). When a player is cheap but has a high projection, their ownership spikes. The model should detect and inflate ownership for high-value players:

```
value_signal = (proj_fd_mean / salary * 1000).rank(pct=True)
value_bonus = max(0, (value_signal - 0.7)) * value_multiplier
```

Where `value_multiplier` defaults to 0.5. Players in the top 30% of value get an ownership bump.

#### 3.2.3 Stack Magnetism

When a team has a high implied total, the field piles onto that team's hitters. The model should inflate ownership for all batters on high-implied-total teams:

```
team_ownership_base = normalized_rank(vegas_team_total)
per_player_team_bonus = team_ownership_base * team_magnetism_factor
```

Where `team_magnetism_factor` defaults to 0.3. The top implied-total team on the slate gets the largest bonus.

#### 3.2.4 Positional Concentration

At thin positions (C, SS), the field concentrates on the top 2-3 options. The model should inflate ownership for top-projected players at thin positions:

```
if position is scarce (C, SS, 2B):
    if player is top-2 at position by projection:
        ownership += scarcity_bonus (default 0.04 = 4%)
```

#### 3.2.5 Pitcher Ownership

Pitchers have unique ownership dynamics. The top 2-3 pitchers on a slate get massive ownership (25-40%), while mid-tier pitchers get 5-15%, and bad pitchers get 1-3%. The model should use a sharper distribution for pitchers:

```
pitcher_salary_rank = salary.rank(pct=True) among pitchers only
pitcher_ownership = sigmoid(pitcher_salary_rank, steepness=4) * max_pitcher_ownership
```

Where `max_pitcher_ownership` defaults to 0.35.

#### 3.2.6 Config

Add these new parameters to `OwnershipModelConfig`:

```python
salary_exponent: float = 1.8
value_multiplier: float = 0.5
team_magnetism_factor: float = 0.3
scarcity_bonus: float = 0.04
max_pitcher_ownership: float = 0.35
pitcher_steepness: float = 4.0
```

Keep backward compatibility: the existing weight fields (`salary_weight`, `projection_weight`, etc.) should still work, but the new model should be the default when no external sources are provided.

### 3.3 Normalization

After computing raw ownership signals, normalize so that total slate ownership sums to approximately `num_roster_spots / num_players_on_slate * 100`. For FanDuel MLB (9 roster spots, ~100-130 players per slate), average ownership per player should be around 7-9%. Ensure no player exceeds `max_pct` and no player falls below `min_pct`.

### 3.4 Dashboard Integration

In `daily_workflow.py`, add a new "Ownership Model" expander in the projection step that shows:
- The computed ownership for each player (sortable table)
- A flag column showing "chalk" (>20% ownership), "mid" (8-20%), "leverage" (3-8%), "deep leverage" (<3%)
- Team-level aggregate ownership (sum of all batters on each team)

---

## 4. Game Environment Scoring

**New file:** `src/slate_optimizer/projection/game_environment.py`  
**Files to update:** `src/slate_optimizer/optimizer/dataset.py`

### 4.1 Purpose

Before lineup construction, score every game on the slate for GPP attractiveness. This answers the question: "Which games should I be targeting for stacks?"

### 4.2 Game Leverage Score

For each game on the slate, compute:

```python
@dataclass
class GameEnvironment:
    game_key: str
    home_team: str
    away_team: str
    vegas_game_total: float       # Over/under
    home_implied_total: float
    away_implied_total: float
    home_team_ownership: float    # Sum of projected ownership for all home batters
    away_team_ownership: float
    game_leverage_score: float    # The key metric
    environment_tier: str         # "prime", "good", "neutral", "avoid"
```

Compute `game_leverage_score` as:

```
game_leverage_score = vegas_game_total_rank * (1 - team_ownership_rank)
```

Where both ranks are percentile ranks across the slate (0-1). A game with a high total and low ownership gets a high leverage score. A game with a high total and high ownership (the obvious stack) gets a lower score.

### 4.3 Environment Tiers

Assign tiers based on the game leverage score:

- **"prime"** — Top 25% of game leverage scores AND vegas_game_total >= 8.5. These are the games to target for primary stacks.
- **"good"** — Top 50% of game leverage scores AND vegas_game_total >= 7.5. Secondary stack targets.
- **"neutral"** — Everything else with vegas_game_total >= 6.5.
- **"avoid"** — vegas_game_total < 6.5 OR pitcher matchup suggests low scoring. Don't stack these unless there's a specific player narrative.

### 4.4 Per-Team Leverage

For each team, compute:

```python
team_leverage = team_implied_total / max(team_aggregate_ownership, 0.01)
```

This is the core "upside per unit of field duplication" metric. Store as `team_gpp_leverage` in the optimizer dataset.

### 4.5 Integration

Add `game_leverage_score`, `environment_tier`, and `team_gpp_leverage` columns to the optimizer dataset in `dataset.py`. These will be used by the solver (Section 5) and the dashboard (Section 8).

---

## 5. Leverage-Weighted ILP Solver

**Files to modify:** `src/slate_optimizer/optimizer/solver.py`, `src/slate_optimizer/optimizer/config.py`

**This is the most important change in the entire PRD.** The solver is the brain that decides which players go into each lineup.

### 5.1 New Objective Function

Replace the current objective (line 116 of `solver.py`):

```python
# CURRENT — pure mean maximization (this is the problem)
prob += lpSum(pool.loc[idx, "proj_fd_mean"] * var for idx, var in decision_vars.items())
```

With a leverage-weighted objective:

```python
# NEW — leverage-weighted GPP objective
for idx, var in decision_vars.items():
    mean = pool.loc[idx, "proj_fd_mean"]
    ceiling = pool.loc[idx, "proj_fd_ceiling"]
    ownership = pool.loc[idx, "proj_fd_ownership"]

    # Ceiling component: reward high-ceiling players
    ceiling_bonus = ceiling * ceiling_weight

    # Ownership leverage: penalize heavily-owned players, reward low-owned
    # Use log to prevent extreme penalties at very low ownership
    own_pct = max(ownership, 0.01)  # floor to prevent division issues
    leverage_bonus = -ownership_penalty * own_pct

    # Combined GPP score
    gpp_score = mean + ceiling_bonus + leverage_bonus

    # Store for the objective
    prob += gpp_score * var  # This line replaces the old lpSum
```

Wait — that changes the lpSum pattern. Here's the correct PuLP syntax:

```python
prob += lpSum(
    _compute_gpp_score(pool, idx, config) * var
    for idx, var in decision_vars.items()
)
```

Where `_compute_gpp_score` is a new helper:

```python
def _compute_gpp_score(
    pool: pd.DataFrame,
    idx: int,
    config: "LeverageConfig",
) -> float:
    mean = float(pool.loc[idx, "proj_fd_mean"])
    ceiling = float(pool.loc[idx, "proj_fd_ceiling"])
    ownership = max(float(pool.loc[idx, "proj_fd_ownership"]), 0.005)

    score = mean
    score += config.ceiling_weight * ceiling
    score -= config.ownership_penalty * ownership

    # Boom potential: (ceiling - mean) / mean gives upside percentage
    if mean > 0:
        boom_pct = (ceiling - mean) / mean
        score += config.boom_weight * boom_pct * mean

    return score
```

### 5.2 LeverageConfig

Add a new config dataclass in `config.py`:

```python
@dataclass
class LeverageConfig:
    """Controls for the leverage-weighted GPP objective function."""

    # How much to reward high-ceiling players (0 = ignore ceiling, 0.3 = moderate, 0.6 = aggressive)
    ceiling_weight: float = 0.25

    # How much to penalize high-ownership players (0 = ignore ownership, 0.5 = moderate, 1.0 = aggressive)
    ownership_penalty: float = 0.40

    # How much to reward boom potential (ceiling-mean spread) (0 = ignore, 0.15 = moderate)
    boom_weight: float = 0.10

    # Stack leverage: bonus for stacking teams from "prime" or "good" environment games
    stack_leverage_bonus: float = 0.5

    # Minimum projection threshold — never play someone below this no matter how low-owned
    # (prevents "stupid leverage" — rostering bad players just because they're contrarian)
    min_viable_projection_pct: float = 0.40  # Bottom 40% of projections are excluded

    # Chalk ceiling — if a player's ownership exceeds this, apply extra penalty
    chalk_threshold: float = 0.25  # 25% ownership
    chalk_extra_penalty: float = 0.15

    # Mode toggle
    mode: str = "gpp"  # "gpp" = leverage objective, "cash" = pure mean (original behavior)
```

### 5.3 Smart Auto-Stack Mode

Replace the current "Auto" mode (which has zero constraints) with a leverage-aware auto-stacker. When the user selects "Auto (optimizer's choice)", the system should:

1. **Score all teams** by `team_gpp_leverage` (from Section 4).
2. **Select a stack template** based on the number of attractive games:
   - If 2+ "prime" games exist → use 4-4 (two big leverage stacks)
   - If 1 "prime" + 1 "good" → use 4-3-1
   - If 1 "prime" only → use 4-2-2 or 4-2-1-1
   - Fallback → use 3-3-2
3. **Bias the stack assignment** toward high-leverage teams. Add a `_stack_preference_bonus` to the LP that gives a small objective bonus when a stack is assigned to a high-leverage team:

```python
# For each team assignment variable in the stack constraints:
if team_gpp_leverage is available:
    prob += stack_leverage_bonus * team_gpp_leverage * assign_var
```

This doesn't force the optimizer to pick a specific team (it's still a soft preference), but it tilts the optimizer toward leverage stacks when projections are close.

### 5.4 Player Floor Filter

Before running the solver, filter out players who have no realistic path to a ceiling game. This prevents "stupid leverage":

```python
if config.mode == "gpp":
    projection_cutoff = pool["proj_fd_mean"].quantile(config.min_viable_projection_pct)
    # Keep all pitchers (don't filter those) and batters above the cutoff
    viable_mask = (pool["player_type"].str.lower() == "pitcher") | (pool["proj_fd_mean"] >= projection_cutoff)
    # Also keep batters who have high ceiling even if mean is low (boom candidates)
    boom_mask = pool["proj_fd_ceiling"] >= pool["proj_fd_ceiling"].quantile(0.6)
    pool = pool[viable_mask | boom_mask].reset_index(drop=True)
```

### 5.5 Chalk Avoidance on Stacked Teams

When the optimizer stacks a team, it should prefer the *underowned* players on that team. Add a within-stack ownership penalty:

For batters on a team that is being stacked (i.e., has a stack assignment variable = 1), add a small per-player penalty proportional to their individual ownership:

```python
# Inside _add_stack_constraints or after, for each team being stacked:
for idx in team_batter_indices[tc]:
    player_own = pool.loc[idx, "proj_fd_ownership"]
    if player_own > chalk_threshold:
        # Small penalty for choosing chalky players within a stack
        prob += -within_stack_chalk_penalty * player_own * decision_vars[idx] * assign_var
```

This makes the optimizer prefer the 6th-hitter at 4% ownership over the 3rd-hitter at 22% ownership on the same team, when their projections are close.

### 5.6 Backward Compatibility

The `generate_lineups()` function signature must remain backward compatible. Add `leverage_config: Optional[LeverageConfig] = None` as a parameter. When `None` or when `leverage_config.mode == "cash"`, use the original `proj_fd_mean` objective. When `mode == "gpp"`, use the new leverage objective.

The Streamlit dashboard should pass the leverage config based on a new "Strategy Mode" selector (GPP Leverage / Cash / Custom).

---

## 6. Bring-Back Improvements

**Files to modify:** `src/slate_optimizer/optimizer/solver.py`

### 6.1 Current Problem

Bring-back currently requires at least N batters from the opposing team of your primary stack. This is good but doesn't consider ownership. A bring-back from the opposing team's highest-owned batter doesn't add differentiation.

### 6.2 Change

When bring-back is enabled and leverage mode is active, add a soft preference for low-owned bring-back players. Don't hard-constrain it (the ILP needs flexibility), but add a small objective bonus for bring-back players who are under the median ownership:

```python
# After selecting bring-back candidates from the opposing team:
for idx in opposing_team_indices:
    own = pool.loc[idx, "proj_fd_ownership"]
    if own < median_ownership:
        prob += bring_back_leverage_bonus * (median_ownership - own) * decision_vars[idx]
```

Default `bring_back_leverage_bonus = 0.3`.

---

## 7. Simulation & Portfolio Selection Enhancements

**Files to modify:**  
- `src/slate_optimizer/simulation/lineup_selector.py`  
- `src/slate_optimizer/simulation/distributions.py`

### 7.1 Leverage-Aware Portfolio Selection

The current `select_portfolio()` function uses a greedy approach that picks lineups by a single metric (top_1pct_rate or win_rate) with diversity penalties. Enhance this to incorporate leverage:

Add a new `selection_metric` option: `"leverage_adjusted_top1"` that scores candidates as:

```python
score = top_1pct_rate * (1 + ownership_leverage_bonus)
```

Where `ownership_leverage_bonus` is computed from the lineup's average ownership:

```python
avg_ownership = mean(player ownership for all 9 players in lineup)
ownership_leverage_bonus = max(0, (target_ownership - avg_ownership) / target_ownership)
```

With `target_ownership = 0.10` (10% average), a lineup with 7% average ownership gets a 30% score bonus. A lineup with 15% average gets no bonus. This ensures the portfolio tilts toward unique lineups even when simulation scores are similar.

### 7.2 Stack Diversity in Portfolio

Add a constraint to `select_portfolio` that limits how many lineups can share the same primary stack team. The current `max_stack_exposure` handles this, but the default (1.0 = unlimited) should change to 0.40 for GPP mode. No single team stack should appear in more than 40% of your lineups.

### 7.3 Boom-Bust Distribution Tuning

In `distributions.py`, the batter lognormal fitting uses a `salary_factor` that scales variance based on salary. For GPP mode, increase the base variance for cheap players (who tend to be more volatile):

```python
if gpp_mode:
    # Cheap players are more volatile (higher boom/bust)
    salary_factor = _clamp(3500.0 / salary, 0.8, 1.5)  # was 4000/salary clamped 0.7-1.3
```

This widens the distribution for cheap players, making the simulation properly reward their boom potential.

---

## 8. Dashboard UI Changes

**Files to modify:** `dashboard/daily_workflow.py`

### 8.1 Strategy Mode Selector

At the top of the dashboard (above Step 1), add a prominent mode selector:

```
Strategy Mode: [GPP Leverage] [Cash/Safe] [Custom]
```

- **GPP Leverage** — Activates the leverage-weighted objective, smart auto-stacking, ownership penalties, and boom-weighted distributions. Uses all the defaults from `LeverageConfig`.
- **Cash/Safe** — Uses the original `proj_fd_mean` objective with no ownership penalty. Good for 50/50 and double-up contests.
- **Custom** — Exposes all `LeverageConfig` parameters as sliders so the user can tune manually.

### 8.2 Leverage Dashboard Panel

In Step 3 (Optimize), add a new section after lineup generation that shows:

**Game Environment Table:**
| Game | Total | Home Implied | Away Implied | Home Own% | Away Own% | Leverage Score | Tier |
|------|-------|-------------|-------------|-----------|-----------|---------------|------|

**Stack Leverage Ranking:**
| Team | Implied Total | Team Own% | GPP Leverage | Tier | Your Exposure |
|------|--------------|-----------|-------------|------|--------------|

**Lineup Ownership Summary:**
- Average lineup ownership across all generated lineups
- Distribution chart (histogram) of lineup ownership levels
- Count of lineups per primary stack team
- Chalk/leverage/deep-leverage player exposure table

### 8.3 Player Leverage Table Enhancement

In the existing player projection table, add these visible columns:
- `ownership_tier` — "chalk" / "mid" / "leverage" / "deep"
- `gpp_score` — the leverage-weighted score used by the solver
- `team_gpp_leverage` — the team-level leverage metric

Sort by `gpp_score` descending by default when in GPP Leverage mode.

---

## 9. New & Modified Columns Reference

### 9.1 New Columns in Optimizer Dataset

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| `gpp_score` | float | Section 5 solver | Leverage-weighted objective score per player |
| `game_leverage_score` | float | Section 4 | Game-level leverage metric |
| `environment_tier` | str | Section 4 | "prime" / "good" / "neutral" / "avoid" |
| `team_gpp_leverage` | float | Section 4 | Team-level upside-per-ownership metric |
| `ownership_tier` | str | Section 3 | "chalk" / "mid" / "leverage" / "deep" |
| `boom_pct` | float | Section 5 | (ceiling - mean) / mean — upside percentage |

### 9.2 Modified Columns

| Column | What Changes |
|--------|-------------|
| `proj_fd_ownership` | More accurate estimation when no external sources (Section 3) |
| `player_leverage_score` | Recalculated using improved ownership (existing in dataset.py) |
| `team_leverage_score` | Recalculated using improved ownership (existing in dataset.py) |

---

## 10. Config File Updates

### 10.1 New Config: `config/leverage_config.json`

Create a default config file:

```json
{
  "mode": "gpp",
  "ceiling_weight": 0.25,
  "ownership_penalty": 0.40,
  "boom_weight": 0.10,
  "stack_leverage_bonus": 0.50,
  "min_viable_projection_pct": 0.40,
  "chalk_threshold": 0.25,
  "chalk_extra_penalty": 0.15,
  "within_stack_chalk_penalty": 0.10,
  "bring_back_leverage_bonus": 0.30,
  "target_avg_lineup_ownership": 0.10,
  "max_stack_exposure_gpp": 0.40,
  "max_batter_exposure_gpp": 0.35,
  "max_pitcher_exposure_gpp": 0.55
}
```

### 10.2 GPP Presets

Add presets for different GPP approaches:

**"Contrarian Heavy"** — Maximum leverage, for large-field GPPs (5000+ entries):
```json
{
  "ceiling_weight": 0.35,
  "ownership_penalty": 0.60,
  "boom_weight": 0.15,
  "chalk_threshold": 0.20,
  "target_avg_lineup_ownership": 0.08
}
```

**"Balanced GPP"** — Moderate leverage, for medium-field GPPs (500-5000 entries):
```json
{
  "ceiling_weight": 0.25,
  "ownership_penalty": 0.35,
  "boom_weight": 0.10,
  "chalk_threshold": 0.25,
  "target_avg_lineup_ownership": 0.11
}
```

**"Light Leverage"** — Slight contrarian tilt, for small-field GPPs or single-entry:
```json
{
  "ceiling_weight": 0.15,
  "ownership_penalty": 0.20,
  "boom_weight": 0.05,
  "chalk_threshold": 0.30,
  "target_avg_lineup_ownership": 0.13
}
```

---

## 11. Testing Requirements

### 11.1 Unit Tests

Add tests in `tests/`:

**`tests/test_leverage_solver.py`:**
- Test that GPP mode produces different lineups than cash mode on the same dataset
- Test that higher `ownership_penalty` produces lower average lineup ownership
- Test that `min_viable_projection_pct` filter excludes bottom players
- Test that smart auto-stack selects a reasonable template (not random)
- Test that the leverage objective prefers low-owned players when projections are close
- Test backward compatibility: `leverage_config=None` produces identical output to current behavior

**`tests/test_ownership_model.py`:**
- Test that the enhanced fallback model produces ownership that sums to reasonable total
- Test that high-salary players get higher ownership than low-salary players
- Test that pitcher ownership is sharper (top pitcher gets 25%+, bottom gets <5%)
- Test that `chalk_threshold` classification works correctly
- Test backward compatibility with existing external source blending

**`tests/test_game_environment.py`:**
- Test that high-total, low-ownership games get "prime" tier
- Test that low-total games get "avoid" regardless of ownership
- Test that team_gpp_leverage is higher for underowned high-implied teams

### 11.2 Integration Test

Create a test that runs the full pipeline on the sample data files in the repo (`BallparkPal_Batters (97).xlsx`, `FanDuel-MLB-2026...csv`):

1. Ingest data
2. Compute projections
3. Compute ownership (fallback model)
4. Compute game environments
5. Generate 20 lineups in GPP mode
6. Generate 20 lineups in cash mode
7. Assert: GPP lineups have lower average ownership than cash lineups
8. Assert: GPP lineups have more distinct primary stack teams
9. Assert: All GPP lineups have valid stack structures (not hodgepodge 1-1-1-1...)
10. Assert: No player in GPP lineups is below the viable projection cutoff

---

## 12. Implementation Order

Execute these in sequence. After each step, run tests.

| Step | Section | What | Est. Effort |
|------|---------|------|-------------|
| 1 | Section 3 | Enhanced ownership model | Medium |
| 2 | Section 4 | Game environment scoring (new file) | Small |
| 3 | Section 10 | LeverageConfig dataclass + JSON presets | Small |
| 4 | Section 5.1-5.2 | New objective function in solver | Medium-Large |
| 5 | Section 5.3 | Smart auto-stack | Medium |
| 6 | Section 5.4-5.5 | Player floor filter + within-stack chalk penalty | Small |
| 7 | Section 6 | Bring-back leverage preference | Small |
| 8 | Section 7 | Portfolio selection enhancements | Medium |
| 9 | Section 8 | Dashboard UI updates | Medium |
| 10 | Section 11 | Full test suite | Medium |

**Total estimated effort:** ~4-6 hours of focused Claude Code work.

---

## 13. What NOT To Change

Preserve these as-is:

- **BallparkPal ingestion** (`ingestion/ballparkpal.py`) — data source is fine
- **FanDuel CSV parsing** (`ingestion/fanduel.py`) — working correctly
- **Position constraints** in solver — FanDuel rules are fixed
- **Salary cap logic** — $35K cap is correct
- **Export/upload format** (`optimizer/export.py`) — FanDuel template format is fixed
- **Simulation correlation matrix builder** (`simulation/correlation.py`) — correlation params are sound, just not wired to lineup construction. The simulation layer is fine as-is; it's the solver that needs the leverage overhaul.
- **SQLite storage** (`data/storage.py`) — no changes needed

---

## 14. Success Criteria

The implementation is successful when:

1. Running in GPP Leverage mode produces lineups that are measurably different from Cash mode (lower avg ownership, more diverse stacks)
2. The "Auto" stack mode produces clean 4-4, 4-3-1, or similar structures — never hodgepodge 1-1-1-1-1-1-1-1
3. Ownership tiers are visible in the dashboard and make intuitive sense (expensive stars = chalk, cheap role players = leverage)
4. Game environment tiers correctly identify high-leverage games
5. All existing tests continue to pass (backward compatibility)
6. The dashboard Strategy Mode selector switches between GPP and Cash cleanly
7. A user can go from data upload to exported FanDuel CSV in under 10 minutes using the GPP workflow
