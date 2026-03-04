# MLB DFS Simulation Engine — Product Requirements Document

**Monte Carlo Contest Simulation for GPP Edge | FanDuel MLB**
**Version 1.0 | March 2026 | Status: Ready for Development**

---

## 1. Executive Summary

### 1.1 What Exists Today

The current system is a functional MLB DFS lineup optimizer with a multi-stage pipeline:

- **Data ingestion**: BallparkPal Excel + FanDuel CSV + Vegas lines + batting orders + handedness + recent stats
- **Projection engine** (`src/slate_optimizer/projection/baseline.py`): Multi-factor model producing `proj_fd_mean`, `proj_fd_floor`, `proj_fd_ceiling` using park/weather/platoon/order/recency/vegas adjustments
- **Ownership model** (`src/slate_optimizer/projection/ownership.py`): Blends multiple external ownership sources with a fallback heuristic
- **ILP solver** (`src/slate_optimizer/optimizer/solver.py`): PuLP-based integer linear programming optimizer with stacking, exposure caps, bring-back, pitcher anti-correlation, and uniqueness constraints
- **Leverage scoring** (`scripts/compute_leverage.py`): `projection_rank - ownership_rank` to identify contrarian plays
- **Streamlit dashboard** (`dashboard/daily_workflow.py`): Full 4-step upload-to-export workflow
- **SQLite persistence** (`src/slate_optimizer/data/storage.py`): Slate versioning and backtest result storage

### 1.2 Core Problem

The optimizer maximizes projected points subject to constraints. This works for cash games (where median performance wins), but **GPPs (tournaments) are won by tail outcomes, not averages**. The current system:

1. Uses **point estimates** (mean/floor/ceiling) instead of full probability distributions
2. Treats player outcomes as **independent** — stacking is enforced by hard constraints, not by modeling the actual correlation in fantasy point outcomes
3. Has **no model of the contest field** — it can't answer "how often does this lineup set beat 150,000 other entries?"
4. Cannot distinguish between a lineup that's **optimal on average** vs. one that **wins GPPs** — these are fundamentally different objectives
5. Calculates floor/ceiling as naive `mean * 0.8` and `mean * 1.2` — not from actual distributional parameters

### 1.3 What We're Building

A **Monte Carlo contest simulation engine** layered on top of the existing optimizer. The simulation engine:

1. Models each player's fantasy point outcome as a **parameterized probability distribution** (not a point estimate)
2. Simulates **correlated outcomes** across all players using a **Student-t copula** (capturing teammate correlation, game environment correlation, and tail dependence)
3. Runs **10,000–50,000 simulated slates** where every player gets a drawn fantasy score
4. Simulates the **contest field** (what 150,000 other entries look like) using ownership projections
5. Scores candidate lineups against the simulated field to estimate **GPP win rate, ROI, and cash rate**
6. Selects the final lineup set that maximizes **simulated tournament equity** instead of raw projected points

This is not a replacement for the existing optimizer — it's a **selection and evaluation layer on top of it**. The PuLP solver generates candidate lineups. The simulation engine evaluates which candidates perform best across thousands of possible outcomes.

### 1.4 Guiding Principles

- **DO NOT rewrite** the PuLP solver, ingestion modules, or existing pipeline. Extend them.
- **All new code** goes under `src/slate_optimizer/simulation/` as a new subpackage.
- **The optimizer dataset CSV remains the single source of truth** — simulation reads from it.
- **Every new module must be callable from CLI scripts AND the Streamlit UI.**
- **Performance target**: 10,000 simulations of a 100-player slate must complete in under 60 seconds on a modern laptop (use NumPy vectorization, not Python loops).

---

## 2. Why This Matters: The Math of GPP Edge

### 2.1 The Contrarian Paradox

The existing leverage score (`projection_rank - ownership_rank`) identifies under-owned players with high projections. But it can't answer the critical question: **how much is that leverage worth in a specific contest?**

If a player is 5% owned and projected for 15 FD points, the leverage score says "good." But:
- If his **ceiling is 25 points** and his outcome distribution is right-skewed, he's a GPP goldmine
- If his **ceiling is 18 points** with a tight distribution, the low ownership doesn't help — he can't separate from the field

The simulation engine answers this by modeling the full distribution and measuring actual separation.

### 2.2 Correlation Is Everything in Stacking

The current optimizer enforces "put 4 guys from Team X together." But it doesn't model **why** that works:

- When the Dodgers score 8 runs, Ohtani (25 pts), Betts (22 pts), and Freeman (20 pts) all produce simultaneously
- A Gaussian correlation model underestimates these blowup games — **tail dependence** (the Student-t copula) captures the reality that when a team goes off, they REALLY go off
- The bring-back constraint is a binary approximation of game-level correlation: if the Dodgers score 8, the opposing team often scores 5+ too

Simulation with a copula replaces these approximations with actual probabilistic modeling.

### 2.3 Simulating the Field

Without a model of what other people's lineups look like, you can't measure GPP edge. A lineup that's "unique" by the current leverage metric might still overlap heavily with the field on 6 of 9 players. The field simulation uses ownership projections to generate opponent lineups and measure true differentiation.

---

## 3. Architecture Overview

```
EXISTING PIPELINE (unchanged)
─────────────────────────────
FanDuel CSV + BallparkPal + Vegas + etc.
    │
    ▼
build_player_dataset() → combined_df
    │
    ▼
compute_baseline_projections() → projections_df
    │
    ▼
compute_ownership_series() → ownership
    │
    ▼
build_optimizer_dataset() → optimizer_df
    │
    ▼
generate_lineups() → List[LineupResult]    ◄── candidates (e.g., 500 lineups)
    │
    ▼
NEW SIMULATION LAYER
─────────────────────────────
    │
    ▼
┌─ PlayerDistributionModel ──────────────┐
│  Fits Beta/LogNormal per player from   │
│  proj_fd_mean, floor, ceiling, salary  │
│  Output: distribution params per player│
└────────────────┬───────────────────────┘
                 │
                 ▼
┌─ CorrelationModel (Student-t Copula) ──┐
│  Builds correlation matrix from:       │
│  - stack_key (teammate pairs)          │
│  - game_key (same-game pairs)          │
│  - pitcher vs opposing hitters         │
│  - opponent_code (bring-back)          │
│  Output: correlation matrix + ν (DoF)  │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌─ SlateSimulator (Monte Carlo Engine) ──┐
│  Draws N correlated outcome vectors    │
│  using t-copula + marginal CDFs        │
│  Each draw: one fantasy score per player│
│  Output: (N × P) matrix of FD points  │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌─ FieldSimulator ───────────────────────┐
│  Generates M simulated opponent lineups│
│  using ownership % as selection probs  │
│  Output: (M × 9) matrix of player IDs │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌─ ContestSimulator ─────────────────────┐
│  Scores candidate lineups vs field     │
│  across all N simulated outcomes       │
│  Output: per-lineup GPP metrics        │
│  (win_rate, top1pct_rate, cash_rate,   │
│   expected_roi, ceiling_score)         │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌─ LineupSelector ───────────────────────┐
│  Picks final K lineups from candidates │
│  maximizing portfolio-level sim equity │
│  with exposure + diversity constraints │
│  Output: final List[LineupResult]      │
└────────────────┬───────────────────────┘
                 │
                 ▼
write_fanduel_upload() → FanDuel CSV (existing export, unchanged)
```

---

## 4. File Structure

All new code lives under `src/slate_optimizer/simulation/`. Do NOT scatter files into existing directories.

```
src/slate_optimizer/simulation/
├── __init__.py
├── distributions.py      # Phase 1: Player outcome distributions
├── correlation.py         # Phase 2: Copula-based correlation model
├── slate_simulator.py     # Phase 3: Monte Carlo slate outcome engine
├── field_simulator.py     # Phase 4: Contest field generation
├── contest_simulator.py   # Phase 5: Full contest evaluation
├── lineup_selector.py     # Phase 6: Portfolio-optimal lineup selection
├── variance_reduction.py  # Phase 7: Antithetic variates, stratification
└── config.py              # Simulation-specific configuration

scripts/
├── run_simulation.py          # CLI entry point for simulation
└── run_simulated_pipeline.py  # Full pipeline with simulation layer

dashboard/
└── simulation_tab.py          # Streamlit tab for simulation controls + results
```

---

## 5. Phase 1: Player Outcome Distributions

**File:** `src/slate_optimizer/simulation/distributions.py`

### 5.1 Purpose

Replace the naive `floor = mean * 0.8` / `ceiling = mean * 1.2` with parameterized probability distributions that accurately model fantasy point outcomes for each player type.

### 5.2 Distribution Selection

**Hitters — Use a shifted LogNormal distribution.**

Rationale: Hitter fantasy points are zero-bounded, right-skewed (most outcomes are 0–10 FD pts, but blowup games of 30+ exist), and have a point mass near zero (0-for-4 with a strikeout = ~0 pts). LogNormal captures all of this.

Parameters derived from existing projection fields:
- `proj_fd_mean` → sets the distribution mean
- `proj_fd_ceiling` → sets the 90th percentile (right tail)
- `salary` → higher salary = tighter distribution (more consistent, less volatile)
- `player_type == "batter"` → apply batter-specific shape

**Pitchers — Use a Normal distribution (truncated at 0).**

Rationale: Pitcher fantasy points are more symmetric around the mean. A quality start (6 IP, 2 ER, 6 K) ≈ 35 pts. A blowup (3 IP, 6 ER) ≈ 5 pts. Less skew than hitters.

Parameters:
- `proj_fd_mean` → sets the distribution mean
- `proj_fd_ceiling` → derives the standard deviation
- `proj_fd_floor` → validates the lower tail

### 5.3 Interface

```python
# src/slate_optimizer/simulation/distributions.py

from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd


@dataclass
class PlayerDistribution:
    """Parameterized fantasy point distribution for one player."""
    fd_player_id: str
    full_name: str
    player_type: Literal["batter", "pitcher"]
    dist_type: Literal["lognormal", "truncated_normal"]
    # Distribution parameters
    mu: float       # location parameter
    sigma: float    # scale parameter
    shift: float    # shift (floor offset, usually 0)
    # Source values used to fit
    proj_mean: float
    proj_floor: float
    proj_ceiling: float
    salary: int

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n samples from this player's distribution.
        Returns array of shape (n,) with fantasy point outcomes."""
        ...

    def ppf(self, quantiles: np.ndarray) -> np.ndarray:
        """Inverse CDF — convert uniform(0,1) samples to fantasy points.
        Used by the copula to transform correlated uniforms into outcomes."""
        ...

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """CDF — convert fantasy points to uniform(0,1) space.
        Used for copula probability integral transform."""
        ...

    def mean(self) -> float:
        """Analytical mean of the distribution."""
        ...

    def percentile(self, p: float) -> float:
        """p-th percentile (0-100 scale)."""
        ...


def fit_player_distributions(
    optimizer_df: pd.DataFrame,
    volatility_scale: float = 1.0,
) -> dict[str, PlayerDistribution]:
    """Fit a distribution for every player in the optimizer dataset.

    Args:
        optimizer_df: The standard optimizer dataset DataFrame with columns:
            fd_player_id, full_name, player_type, salary,
            proj_fd_mean, proj_fd_floor, proj_fd_ceiling
        volatility_scale: Global multiplier on all distribution widths.
            >1.0 = more volatile outcomes (GPP-friendly).
            <1.0 = tighter outcomes (cash-friendly).

    Returns:
        Dict mapping fd_player_id -> PlayerDistribution
    """
    ...


def _fit_batter_lognormal(
    mean: float, floor: float, ceiling: float, salary: int
) -> tuple[float, float, float]:
    """Fit lognormal (mu, sigma, shift) for a batter.

    The key insight: ceiling/mean ratio determines the skew (sigma).
    Higher ceiling relative to mean = more right-skew = more GPP upside.

    Salary acts as a variance damper: expensive players are more consistent.
    A $5500 hitter has tighter sigma than a $2800 hitter at the same proj.

    Method:
        1. Set shift = max(0, floor * 0.5) as the minimum possible score
        2. Compute sigma from ceiling-to-mean ratio:
           sigma = sqrt(ln((ceiling - shift) / (mean - shift)))
           (derived from lognormal 90th percentile relationship)
        3. Compute mu from mean:
           mu = ln(mean - shift) - sigma^2 / 2
        4. Apply salary-based variance adjustment:
           sigma *= salary_factor where salary_factor = clamp(4000/salary, 0.7, 1.3)

    Returns: (mu, sigma, shift)
    """
    ...


def _fit_pitcher_normal(
    mean: float, floor: float, ceiling: float, salary: int
) -> tuple[float, float, float]:
    """Fit truncated normal (mu, sigma, shift=0) for a pitcher.

    Method:
        1. sigma = (ceiling - floor) / (2 * 1.645)
           (assumes floor ≈ 5th percentile, ceiling ≈ 95th percentile)
        2. mu = mean (symmetric distribution)
        3. Truncate at 0 (pitcher can't score negative FD points)

    Returns: (mu, sigma, 0.0)
    """
    ...
```

### 5.4 Fitting Logic Details

**Batter LogNormal fitting:**

The existing projections give us three anchor points: mean, floor (≈10th percentile), and ceiling (≈90th percentile). Currently these are `mean * 0.8` and `mean * 1.2`, which is flat and uninformative. The V2 plan (Section 4.1.5) defines floor as 10th percentile and ceiling as 90th percentile — this module should honor that interpretation even if the current values are naive.

For a lognormal with parameters (μ, σ, shift):
- `mean = shift + exp(μ + σ²/2)`
- `90th percentile = shift + exp(μ + 1.2816 * σ)`
- `10th percentile = shift + exp(μ - 1.2816 * σ)`

Given (mean, floor, ceiling), solve for (μ, σ, shift). If the current floor/ceiling are just `0.8*mean` / `1.2*mean`, use reasonable defaults: `σ ≈ 0.6` for hitters (high variance) scaled by salary.

**Salary-variance relationship:**
```
salary_factor = clamp(4000 / salary, 0.7, 1.3)
# $2800 hitter: factor = 1.3 (more volatile, boom-bust)
# $4000 hitter: factor = 1.0 (baseline)
# $6000 hitter: factor = 0.7 (more consistent, lower variance)
```

This encodes a real DFS dynamic: cheap hitters are cheap because they're volatile. That volatility is what makes them GPP-viable.

### 5.5 Validation

After fitting, validate each distribution:
1. `abs(dist.mean() - proj_fd_mean) < 0.5` — analytical mean matches projection
2. `dist.percentile(50) > 0` — median is positive
3. `dist.percentile(90) > dist.percentile(50)` — right tail exists
4. `dist.sample(1000).min() >= 0` — no negative FD point draws

Print a summary table: `player_name | proj_mean | dist_mean | dist_p10 | dist_p50 | dist_p90 | sigma`

---

## 6. Phase 2: Correlation Model

**File:** `src/slate_optimizer/simulation/correlation.py`

### 6.1 Purpose

Build a correlation matrix that encodes the relationships between all players on the slate, then use a Student-t copula to generate correlated uniform random variables that preserve tail dependence.

### 6.2 Correlation Structure

MLB DFS has five types of correlation that matter:

| Relationship | Correlation | Rationale |
|---|---|---|
| **Teammates** (same `stack_key`) | +0.15 to +0.40 | When a team scores, multiple hitters produce. Batting order adjacency increases correlation. |
| **Same game, opposing teams** (same `game_key`, different `stack_key`) | +0.05 to +0.15 | High-scoring games benefit both sides (game environment effect). |
| **Pitcher vs own hitters** (pitcher's `team_code` == hitter's `team_code`) | +0.05 | Mild positive: if pitcher is winning, team is likely scoring too. |
| **Pitcher vs opposing hitters** (pitcher's `team_code` == hitter's `opponent_code`) | -0.10 to -0.20 | Negative: pitcher dominance = opposing hitters fail. |
| **Unrelated players** (different `game_key`) | 0.00 | Players in different games are independent. |

### 6.3 Correlation Matrix Construction

```python
# src/slate_optimizer/simulation/correlation.py

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CorrelationConfig:
    """Configurable correlation strengths."""
    teammate_base: float = 0.25          # base correlation between teammates
    teammate_adjacent_bonus: float = 0.10 # extra correlation for adjacent batting order
    same_game_opponent: float = 0.10     # same game, different team
    pitcher_own_hitters: float = 0.05    # pitcher and hitters on his team
    pitcher_vs_opposing: float = -0.15   # pitcher vs opposing hitters
    cross_game: float = 0.00            # different games (independent)
    copula_nu: int = 5                   # Student-t degrees of freedom (lower = fatter tails)
    # nu=5 gives meaningful tail dependence without extreme behavior
    # nu=4 is more aggressive (article default), nu=8 is closer to Gaussian


@dataclass
class CorrelationModel:
    """Correlation matrix + copula parameters for a slate."""
    player_ids: list[str]           # ordered list of fd_player_ids
    matrix: np.ndarray              # (P x P) correlation matrix
    nu: int                         # Student-t degrees of freedom
    cholesky: np.ndarray            # Cholesky decomposition of matrix (precomputed)

    def validate(self) -> bool:
        """Check that the matrix is positive semi-definite and valid."""
        ...


def build_correlation_matrix(
    optimizer_df: pd.DataFrame,
    config: Optional[CorrelationConfig] = None,
) -> CorrelationModel:
    """Build the player-player correlation matrix from the optimizer dataset.

    Uses columns: fd_player_id, player_type, team_code, opponent_code,
                  stack_key, game_key, batting_order_position

    Args:
        optimizer_df: The standard optimizer dataset.
        config: Correlation strengths. Uses defaults if None.

    Returns:
        CorrelationModel with the full P×P matrix and Cholesky factor.

    Algorithm:
        1. Initialize P×P identity matrix
        2. For each pair (i, j):
           - If same stack_key and both batters: teammate_base
             + teammate_adjacent_bonus if |order_i - order_j| <= 1
           - If same game_key, different stack_key, both batters: same_game_opponent
           - If one is pitcher, other is batter on same team: pitcher_own_hitters
           - If one is pitcher, other is batter on opposing team: pitcher_vs_opposing
           - If different game_key: cross_game (0)
        3. Ensure positive semi-definiteness:
           - Compute eigenvalues
           - If any eigenvalue < 0, apply nearest PSD correction
             (set negative eigenvalues to small positive ε, reconstruct)
        4. Precompute Cholesky decomposition for fast sampling
    """
    ...


def _nearest_positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix to the nearest PSD matrix.
    Uses the Higham (2002) alternating projections algorithm."""
    ...
```

### 6.4 Correlation Modifiers from Data

The base correlations above should be **adjusted** using data already in the optimizer dataset:

- **Vegas game total** (`vegas_game_total`): Higher game totals = more run scoring = higher teammate correlation. Scale teammate correlation by `min(1.3, game_total / 8.5)`. An 11-run game total gets 1.29x correlation boost.
- **Stack priority** (`stack_priority`): "high" priority stacks get a correlation boost of +0.05 on teammate pairs.
- **Batting order adjacency**: Players batting 1-2, 2-3, 3-4 are more correlated than 1-8. The adjacent bonus captures this.

### 6.5 Why Student-t, Not Gaussian

The article is explicit: Gaussian copulas have **zero tail dependence** (λ_U = λ_L = 0). This means the probability of extreme co-movement (a team's entire lineup going off simultaneously) is modeled as essentially zero. That's wrong for baseball — when the Cubs score 14 runs, the top 5 hitters ALL produce massive games.

The Student-t copula with ν=5 has tail dependence coefficient ≈ 0.10–0.20 depending on correlation, meaning there's a meaningful probability of extreme co-movement. This is critical for GPP simulation where you need to model the "team goes nuclear" scenarios accurately.

---

## 7. Phase 3: Slate Simulator (Monte Carlo Engine)

**File:** `src/slate_optimizer/simulation/slate_simulator.py`

### 7.1 Purpose

The core simulation engine. Draws N complete slate outcomes (one fantasy score per player per simulation) using the copula for correlation.

### 7.2 Interface

```python
# src/slate_optimizer/simulation/slate_simulator.py

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from .distributions import PlayerDistribution
from .correlation import CorrelationModel


@dataclass
class SlateSimulation:
    """Result of simulating a full slate N times."""
    scores: np.ndarray
    # Shape: (N, P) where N = num_simulations, P = num_players
    # scores[i, j] = fantasy points for player j in simulation i

    player_ids: list[str]
    # Ordered list of fd_player_ids, matching columns of scores

    player_id_to_index: dict[str, int]
    # Fast lookup: fd_player_id -> column index in scores

    num_simulations: int
    num_players: int

    def player_scores(self, fd_player_id: str) -> np.ndarray:
        """Get all N simulated scores for one player. Shape: (N,)"""
        return self.scores[:, self.player_id_to_index[fd_player_id]]

    def lineup_scores(self, player_ids: list[str]) -> np.ndarray:
        """Score a lineup (list of fd_player_ids) across all simulations.
        Returns array of shape (N,) — total fantasy points per sim."""
        indices = [self.player_id_to_index[pid] for pid in player_ids]
        return self.scores[:, indices].sum(axis=1)

    def percentile_score(self, player_ids: list[str], pct: float) -> float:
        """The p-th percentile total score for a lineup across simulations."""
        return float(np.percentile(self.lineup_scores(player_ids), pct))

    def summary_stats(self, player_ids: list[str]) -> dict:
        """Summary statistics for a lineup across simulations."""
        scores = self.lineup_scores(player_ids)
        return {
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "std": float(scores.std()),
            "p10": float(np.percentile(scores, 10)),
            "p25": float(np.percentile(scores, 25)),
            "p75": float(np.percentile(scores, 75)),
            "p90": float(np.percentile(scores, 90)),
            "p99": float(np.percentile(scores, 99)),
            "max": float(scores.max()),
            "min": float(scores.min()),
        }


def simulate_slate(
    distributions: dict[str, PlayerDistribution],
    correlation_model: CorrelationModel,
    num_simulations: int = 10_000,
    seed: Optional[int] = None,
    use_antithetic: bool = True,
) -> SlateSimulation:
    """Run the full Monte Carlo slate simulation.

    Algorithm:
        1. Generate (N × P) matrix of correlated draws from Student-t copula:
           a. Draw Z ~ N(0, I) of shape (N, P)
           b. Correlate: X = Z @ L^T  where L = Cholesky(correlation_matrix)
           c. Draw S ~ chi-squared(nu) / nu  of shape (N,)
           d. Compute T = X / sqrt(S[:, None])  to get t-distributed marginals
           e. Apply Student-t CDF to get uniform marginals: U = t_cdf(T, nu)

        2. Transform uniforms to fantasy points using each player's inverse CDF:
           For each player j: scores[:, j] = distributions[player_ids[j]].ppf(U[:, j])

        3. If use_antithetic: also compute scores for (1 - U) and average.
           This gives ~30-50% variance reduction for free.

    Args:
        distributions: Dict of fd_player_id -> PlayerDistribution (from Phase 1)
        correlation_model: CorrelationModel with matrix + nu (from Phase 2)
        num_simulations: Number of Monte Carlo draws. 10k = fast, 50k = precise.
        seed: Random seed for reproducibility.
        use_antithetic: Enable antithetic variate variance reduction.

    Returns:
        SlateSimulation with the full (N × P) score matrix.

    Performance:
        Must complete in <60 seconds for 100 players × 10,000 sims.
        Use NumPy vectorized operations throughout — NO Python for-loops over sims.
    """
    ...
```

### 7.3 Antithetic Variates (Built-in Variance Reduction)

From the article (Part V): when you draw uniform samples U, also evaluate at (1 - U). For monotone payoff functions (more points = better), the antithetic sample is negatively correlated with the original, reducing variance.

Implementation: if `num_simulations = 10,000` and `use_antithetic = True`:
1. Draw 5,000 correlated uniform matrices U (shape 5000 × P)
2. Compute antithetic: U_anti = 1 - U (shape 5000 × P)
3. Transform both through inverse CDFs
4. Stack: scores = vstack([scores_original, scores_antithetic])
5. Result: 10,000 effective simulations from 5,000 random draws, with lower variance

### 7.4 Performance Requirements

The simulation engine is the computational bottleneck. Requirements:

| Operation | Target Time (100 players, 10k sims) |
|---|---|
| Correlated draw generation (copula) | < 5 seconds |
| Inverse CDF transform (all players) | < 10 seconds |
| Lineup scoring (500 candidate lineups) | < 5 seconds |
| Total end-to-end | < 30 seconds |

Use `numpy` and `scipy.stats` throughout. Pre-compute the Cholesky factor once. Vectorize all CDF/PPF operations across the full (N, P) matrix.

---

## 8. Phase 4: Field Simulator

**File:** `src/slate_optimizer/simulation/field_simulator.py`

### 8.1 Purpose

Generate simulated contest fields — what the other 150,000 entries in a GPP look like. This is critical: you can't measure GPP edge without knowing what you're competing against.

### 8.2 Interface

```python
# src/slate_optimizer/simulation/field_simulator.py

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulatedField:
    """A simulated contest field."""
    lineups: np.ndarray
    # Shape: (M, 9) — M opponent lineups, each with 9 player indices
    # Values are indices into player_ids list

    player_ids: list[str]
    # Ordered list of fd_player_ids matching the index space

    num_lineups: int
    # M — how many opponent lineups were generated

    ownership_used: np.ndarray
    # Shape: (P,) — the ownership percentages used for generation


def simulate_field(
    optimizer_df: pd.DataFrame,
    num_opponent_lineups: int = 1000,
    salary_cap: int = 35_000,
    seed: Optional[int] = None,
    position_constraints: bool = True,
) -> SimulatedField:
    """Generate a simulated contest field using ownership as selection probability.

    This models what the average GPP field looks like. Higher-owned players
    appear in more simulated lineups. The result is a set of M "opponent" lineups
    that your candidate lineups will be scored against.

    Algorithm:
        1. Normalize ownership percentages to selection probabilities per position.
        2. For each simulated lineup:
           a. Select 1 pitcher: weighted random by pitcher ownership
           b. For each position slot (C/1B, 2B, 3B, SS, 3×OF, UTIL):
              - Select from eligible players weighted by ownership
              - Remove selected players from pool (no duplicates)
           c. Validate salary <= salary_cap. If over, resample UTIL slot with
              cheaper options until valid (max 50 attempts, then discard lineup).
        3. Return the valid lineups as player index arrays.

    Simplification for speed: this does NOT need to perfectly replicate
    FanDuel position rules. A reasonable approximation (1P + 8 hitters
    weighted by ownership with salary cap) is sufficient — the field
    simulation is a statistical model, not an exact replica.

    Args:
        optimizer_df: Standard optimizer dataset with columns:
            fd_player_id, position, player_type, salary, proj_fd_ownership
        num_opponent_lineups: How many opponent lineups to generate.
            1000 is fast, 5000 is more accurate, 10000+ for precision.
        salary_cap: FanDuel salary cap (default 35000).
        seed: Random seed.
        position_constraints: If True, enforce position eligibility.
            If False, just pick 1 pitcher + 8 hitters by ownership weight.

    Returns:
        SimulatedField with the generated opponent lineups.

    Performance: 1000 lineups should generate in < 5 seconds.
    """
    ...
```

### 8.3 Ownership-to-Selection Mapping

The ownership percentage from `proj_fd_ownership` maps to selection probability, but not linearly:

```python
# Within each position pool, normalize ownership to sum to a valid probability
# Players with 0% ownership get a small floor (0.5%) so they can still appear
# Players with >40% ownership get capped at 40% to prevent field simulation
# from being dominated by a single player

def _ownership_to_selection_probs(ownership: np.ndarray) -> np.ndarray:
    """Convert ownership % to selection probabilities.
    Floor at 0.5%, cap at 40%, then normalize to sum to 1."""
    probs = np.clip(ownership, 0.005, 0.40)
    return probs / probs.sum()
```

### 8.4 Field Quality Tiers

Not all 150,000 entries are built equally. The field contains:
- **Sharks (~10%)**: Optimizer-built lineups with good stacking and leverage
- **Recreational (~60%)**: Star-heavy lineups that chase names and value
- **Random (~30%)**: Auto-fill, last-minute entries, novelty picks

The field simulator should support an optional `field_quality` parameter:

```python
@dataclass
class FieldQualityMix:
    """Composition of the simulated field."""
    shark_pct: float = 0.10    # use optimizer-quality construction
    rec_pct: float = 0.60      # weight heavily toward high-salary + high-ownership
    random_pct: float = 0.30   # mostly random with loose constraints

# Sharks: use ownership but also boost high-projection players
# Recreational: double the weight on salary rank and name recognition
# Random: near-uniform selection with basic position constraints
```

---

## 9. Phase 5: Contest Simulator

**File:** `src/slate_optimizer/simulation/contest_simulator.py`

### 9.1 Purpose

The integration layer. Takes candidate lineups, slate simulations, and field simulations, and produces GPP performance metrics for each candidate.

### 9.2 Interface

```python
# src/slate_optimizer/simulation/contest_simulator.py

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Sequence

from ..optimizer.solver import LineupResult
from .slate_simulator import SlateSimulation
from .field_simulator import SimulatedField


@dataclass
class LineupSimResult:
    """Simulation results for one candidate lineup."""
    lineup_id: int
    player_ids: list[str]

    # Core GPP metrics (across all N simulations)
    mean_score: float          # average total FD points
    median_score: float
    std_score: float
    p90_score: float           # 90th percentile score
    p99_score: float           # 99th percentile score (GPP ceiling)
    max_score: float           # absolute ceiling

    # Contest placement metrics (vs simulated field)
    win_rate: float            # % of sims where this lineup scores #1
    top_1pct_rate: float       # % of sims finishing top 1%
    top_10pct_rate: float      # % of sims finishing top 10%
    cash_rate: float           # % of sims where this lineup cashes (top ~20%)
    expected_roi: float        # E[payout / entry_fee] across sims

    # Ownership/leverage
    total_ownership: float     # sum of ownership % for all 9 players
    leverage_score: float      # lineup-level leverage (from existing compute_leverage)

    # Uniqueness
    field_duplication_rate: float  # % of field lineups that share 6+ of 9 players


@dataclass
class ContestSimResult:
    """Full contest simulation output."""
    lineup_results: list[LineupSimResult]
    num_simulations: int
    num_field_lineups: int
    num_candidates: int

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all lineup results to a DataFrame for analysis/display."""
        ...

    def rank_by(self, metric: str = "top_1pct_rate") -> list[LineupSimResult]:
        """Rank candidate lineups by a simulation metric."""
        ...


def simulate_contest(
    candidates: Sequence[LineupResult],
    slate_sim: SlateSimulation,
    field_sim: SimulatedField,
    entry_fee: float = 20.0,
    payout_structure: Optional[dict] = None,
) -> ContestSimResult:
    """Score all candidate lineups against the simulated field
    across all simulated slate outcomes.

    Algorithm:
        For each simulation i (out of N):
            1. Score all candidate lineups: sum player scores for sim i
            2. Score all field lineups: sum player scores for sim i
            3. Rank all lineups (candidates + field) by total score
            4. Record each candidate's percentile rank
            5. Map rank to payout using payout_structure

    Payout structure (simplified GPP model):
        If no payout_structure provided, use default FanDuel GPP:
        - Top 0.1%: 100x entry fee
        - Top 0.5%: 20x
        - Top 1%:   10x
        - Top 5%:   3x
        - Top 20%:  1.5x (cash line)
        - Bottom 80%: 0x
        These are rough approximations. User can supply exact structure.

    Args:
        candidates: List of LineupResult from generate_lineups()
            (access player IDs from candidates[i].dataframe["fd_player_id"])
        slate_sim: SlateSimulation from simulate_slate()
        field_sim: SimulatedField from simulate_field()
        entry_fee: Contest entry fee for ROI calculation.
        payout_structure: Dict mapping percentile thresholds to payout multipliers.

    Returns:
        ContestSimResult with per-lineup GPP metrics.

    Performance:
        500 candidates × 10,000 sims × 1,000 field lineups in < 30 seconds.
        Key optimization: vectorize lineup scoring using matrix indexing, not loops.
    """
    ...
```

### 9.3 Scoring Optimization

The performance-critical operation is scoring lineups across simulations. Use NumPy advanced indexing:

```python
# Pseudocode for vectorized lineup scoring:

# slate_sim.scores shape: (N, P)
# candidate has 9 player indices: [i1, i2, ..., i9]
# For one candidate:
candidate_scores = slate_sim.scores[:, candidate_indices].sum(axis=1)  # shape: (N,)

# For all M field lineups at once:
# field_sim.lineups shape: (M, 9) — indices into player axis
# We need: for each sim, for each field lineup, sum 9 player scores
# field_scores shape: (N, M)

# IMPORTANT: do NOT loop over simulations. Use:
# scores_3d shape: (N, M, 9) via fancy indexing, then sum over axis=2
field_indices = field_sim.lineups  # (M, 9)
field_scores = np.zeros((num_sims, num_field))
for sim_idx in range(num_sims):  # this loop is acceptable: N iterations, vectorized inside
    sim_player_scores = slate_sim.scores[sim_idx]  # shape (P,)
    field_scores[sim_idx] = sim_player_scores[field_indices].sum(axis=1)  # (M,)

# OR fully vectorized (more memory but faster):
# player_scores_expanded: (N, 1, P) broadcast with field_indices (1, M, 9)
# This uses more memory but avoids the loop entirely.
```

### 9.4 Default Payout Structure

```python
DEFAULT_GPP_PAYOUTS = {
    # (min_percentile, max_percentile): payout_multiplier
    (99.9, 100.0): 100.0,   # 1st place
    (99.5, 99.9):  20.0,    # top 0.5%
    (99.0, 99.5):  10.0,    # top 1%
    (95.0, 99.0):  3.0,     # top 5%
    (80.0, 95.0):  1.5,     # cash (top 20%)
    (0.0,  80.0):  0.0,     # out of the money
}
```

---

## 10. Phase 6: Lineup Selector

**File:** `src/slate_optimizer/simulation/lineup_selector.py`

### 10.1 Purpose

Given N candidate lineups scored by the contest simulator, select the final K lineups to enter in the contest. This is NOT just "pick the top K by win rate" — it's a **portfolio optimization problem**. You want K lineups that collectively maximize equity, which means they should be **diverse** (not all correlated with each other).

### 10.2 Interface

```python
# src/slate_optimizer/simulation/lineup_selector.py

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Sequence

from .contest_simulator import ContestSimResult, LineupSimResult


@dataclass
class PortfolioSelection:
    """The final selected lineup set."""
    selected: list[LineupSimResult]
    num_selected: int

    # Portfolio-level metrics (aggregated across selected lineups)
    portfolio_win_rate: float       # P(at least one lineup wins) per sim
    portfolio_top1pct_rate: float   # P(at least one lineup in top 1%)
    portfolio_cash_rate: float      # P(at least one lineup cashes)
    portfolio_expected_roi: float   # expected ROI across all entries
    portfolio_total_cost: float     # num_selected × entry_fee

    # Diversity metrics
    avg_pairwise_overlap: float     # avg number of shared players between pairs
    unique_players_used: int        # total distinct players across all lineups
    max_player_exposure: float      # highest single-player exposure in set

    def to_dataframe(self) -> pd.DataFrame:
        """Selected lineups as a DataFrame."""
        ...


def select_portfolio(
    contest_result: ContestSimResult,
    num_lineups: int = 20,
    selection_metric: str = "top_1pct_rate",
    max_overlap: int = 5,
    max_player_exposure: float = 0.60,
    diversity_weight: float = 0.3,
) -> PortfolioSelection:
    """Select the final lineup portfolio from candidates.

    Algorithm (greedy with diversity penalty):
        1. Rank all candidates by selection_metric (descending)
        2. Select the top candidate as lineup #1
        3. For each subsequent selection:
           a. For each remaining candidate, compute:
              - sim_score = normalized selection_metric value
              - overlap_penalty = avg overlap with already-selected lineups / 9
              - exposure_penalty = max(0, max_exposure_of_any_player - max_player_exposure)
              - combined_score = sim_score * (1 - diversity_weight * overlap_penalty)
                                - exposure_penalty * 10
           b. Select the candidate with highest combined_score
           c. Skip candidates that would push any player above max_player_exposure
        4. Repeat until num_lineups are selected or candidates exhausted

    Args:
        contest_result: Full contest simulation output
        num_lineups: How many lineups to select (K)
        selection_metric: Which sim metric to optimize. Options:
            "top_1pct_rate" (GPP tournaments — recommended)
            "win_rate" (single-entry GPPs)
            "expected_roi" (balanced)
            "cash_rate" (cash games — use this if entering a 50/50)
            "p99_score" (ceiling chasing)
        max_overlap: Maximum shared players between any two selected lineups
        max_player_exposure: No player in more than this % of selected lineups
        diversity_weight: 0.0 = pure metric ranking, 1.0 = heavy diversity preference

    Returns:
        PortfolioSelection with the final lineup set and portfolio metrics.
    """
    ...
```

### 10.3 The Portfolio Insight

This is the key concept that connects the article's prediction market portfolio theory to DFS:

- In prediction markets, you don't want all your bets correlated — if one fails, they all fail
- In DFS GPPs, you don't want all your lineups correlated — if one lineup's stack busts, you want other lineups with different stacks that can still win

The `diversity_weight` parameter controls this tradeoff. At 0.0, you just pick the 20 lineups with the highest individual win rate (which might all contain the same core of players). At 0.3–0.5, you sacrifice some individual lineup quality for portfolio diversification.

The optimal `diversity_weight` depends on contest size and entries:
- **Single entry**: 0.0 (just pick the best lineup)
- **20 entries in a 150-entry contest**: 0.1 (mild diversity)
- **150 entries in a 150,000-entry contest**: 0.4–0.5 (heavy diversity)

---

## 11. Phase 7: Variance Reduction

**File:** `src/slate_optimizer/simulation/variance_reduction.py`

### 11.1 Purpose

Implement the three variance reduction techniques from the article (Part V) that stack multiplicatively. These make the simulation more precise at the same computational cost.

### 11.2 Techniques

```python
# src/slate_optimizer/simulation/variance_reduction.py

import numpy as np
from scipy.stats import norm


def antithetic_uniforms(U: np.ndarray) -> np.ndarray:
    """Antithetic variates: given (N/2, P) uniform draws,
    return (N, P) by stacking U and (1 - U).

    Variance reduction: ~30-50% for monotone payoffs (fantasy points).
    Cost: zero — same number of random draws, double the effective samples.
    """
    return np.vstack([U, 1.0 - U])


def stratified_uniforms(
    num_strata: int,
    num_per_stratum: int,
    num_players: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stratified sampling: divide [0,1] into J strata for the FIRST
    player dimension (typically the highest-correlated group), then
    sample uniformly within each stratum.

    This ensures the simulation covers the full range of outcomes
    evenly, rather than clustering around the center.

    Returns: (J * num_per_stratum, num_players) uniform draws where
    the first column is stratified.

    Variance reduction: ~20-40% depending on payoff structure.
    """
    ...


def control_variate_adjustment(
    raw_estimates: np.ndarray,
    control_values: np.ndarray,
    control_expected: float,
) -> np.ndarray:
    """Control variate: if we know E[C] analytically, use the
    correlation between our estimator and C to reduce variance.

    For DFS: use the sum of proj_fd_mean values as the control variate.
    We know the expected lineup score (sum of means), so any deviation
    of the simulation average from this known value can be corrected.

    raw_estimates: (N,) simulated lineup scores
    control_values: (N,) sum of player means in each sim (known E[X])
    control_expected: the true expected score (sum of proj_fd_mean)

    Returns: adjusted estimates with lower variance.

    Formula: Y_adj = Y - beta * (C - E[C])
    where beta = Cov(Y, C) / Var(C)
    """
    cov_yc = np.cov(raw_estimates, control_values)[0, 1]
    var_c = control_values.var()
    if var_c < 1e-10:
        return raw_estimates
    beta = cov_yc / var_c
    return raw_estimates - beta * (control_values - control_expected)
```

---

## 12. Simulation Configuration

**File:** `src/slate_optimizer/simulation/config.py`

### 12.1 Interface

```python
# src/slate_optimizer/simulation/config.py

from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path
import json

from .correlation import CorrelationConfig


@dataclass
class SimulationConfig:
    """All simulation parameters in one place."""

    # Monte Carlo settings
    num_simulations: int = 10_000
    seed: Optional[int] = None
    use_antithetic: bool = True
    use_stratified: bool = False
    num_strata: int = 10

    # Distribution settings
    volatility_scale: float = 1.0
    # >1.0 widens all distributions (more boom-bust, GPP-friendly)
    # <1.0 tightens distributions (more consistent, cash-friendly)

    # Correlation settings
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)

    # Field simulation
    num_field_lineups: int = 1000
    field_quality_shark_pct: float = 0.10
    field_quality_rec_pct: float = 0.60
    field_quality_random_pct: float = 0.30

    # Contest settings
    entry_fee: float = 20.0
    payout_structure: Optional[dict] = None  # None = use default GPP

    # Selection settings
    num_candidates: int = 500
    # Generate this many candidates from the ILP solver before simulation
    selection_metric: str = "top_1pct_rate"
    max_overlap: int = 5
    max_player_exposure: float = 0.60
    diversity_weight: float = 0.3

    @classmethod
    def load(cls, path: Path) -> "SimulationConfig":
        """Load from JSON file."""
        ...

    def save(self, path: Path) -> None:
        """Save to JSON file."""
        ...

    @classmethod
    def gpp_preset(cls) -> "SimulationConfig":
        """Preset for large GPP tournaments."""
        return cls(
            num_simulations=20_000,
            volatility_scale=1.1,
            num_field_lineups=2000,
            selection_metric="top_1pct_rate",
            diversity_weight=0.4,
            max_player_exposure=0.50,
        )

    @classmethod
    def cash_preset(cls) -> "SimulationConfig":
        """Preset for cash games (50/50, double-ups)."""
        return cls(
            num_simulations=10_000,
            volatility_scale=0.8,
            num_field_lineups=500,
            selection_metric="cash_rate",
            diversity_weight=0.1,
            max_player_exposure=0.80,
        )

    @classmethod
    def single_entry_preset(cls) -> "SimulationConfig":
        """Preset for single-entry GPPs."""
        return cls(
            num_simulations=20_000,
            volatility_scale=1.0,
            num_field_lineups=2000,
            selection_metric="win_rate",
            diversity_weight=0.0,  # no diversity needed for 1 lineup
            max_player_exposure=1.0,
        )
```

---

## 13. CLI Scripts

### 13.1 Standalone Simulation Runner

**File:** `scripts/run_simulation.py`

```
Usage:
    python scripts/run_simulation.py \
        --dataset data/processed/sample_optimizer_dataset.csv \
        --candidates data/processed/sample_lineups.csv \
        --num-sims 20000 \
        --num-field 2000 \
        --num-select 20 \
        --metric top_1pct_rate \
        --output data/processed/sample_sim_results.csv \
        --config config/sim_gpp.json
```

This script:
1. Loads the optimizer dataset CSV
2. Loads candidate lineups CSV (from a prior `run_optimizer.py` run)
3. Fits player distributions (Phase 1)
4. Builds correlation model (Phase 2)
5. Runs slate simulation (Phase 3)
6. Simulates the field (Phase 4)
7. Runs contest simulation (Phase 5)
8. Selects final portfolio (Phase 6)
9. Writes results CSV + prints summary

If `--candidates` is omitted, run the ILP solver first to generate `--num-candidates` candidate lineups with relaxed constraints (lower min-stack, higher exposure caps) to maximize the candidate pool for simulation to evaluate.

### 13.2 Full Pipeline with Simulation

**File:** `scripts/run_simulated_pipeline.py`

This extends `run_daily_pipeline.py` by adding the simulation layer after lineup generation. It should import and reuse the existing pipeline functions, then append simulation steps.

```
Usage:
    python scripts/run_simulated_pipeline.py \
        --bpp-source "C:\\Users\\wallg\\OneDrive\\Desktop\\DFSMLB" \
        --fanduel-csv cleaned_players_list.csv \
        --num-lineups 20 \
        --num-candidates 500 \
        --sim-config config/sim_gpp.json \
        --output-dir data/processed
```

Pipeline flow:
1. All existing daily pipeline steps (ingest → project → own → dataset → config)
2. Generate 500 candidate lineups (relaxed constraints) via existing `generate_lineups()`
3. Fit distributions
4. Build correlation model
5. Simulate slate
6. Simulate field
7. Run contest simulation
8. Select final 20 lineups via portfolio optimizer
9. Export FanDuel upload CSV
10. Write simulation report CSV

---

## 14. Streamlit Integration

**File:** `dashboard/simulation_tab.py`

Add a new tab/step to the existing `daily_workflow.py` Streamlit app. This sits between "Configure & Optimize" (Step 3) and the final export.

### 14.1 UI Layout

**Simulation Settings Panel (sidebar or expander):**
- Number of simulations: slider (1000–50000, default 10000)
- Volatility scale: slider (0.5–2.0, default 1.0)
- Copula degrees of freedom (ν): slider (3–20, default 5)
- Correlation strength: sliders for teammate, same-game, pitcher-vs-opposing
- Field size: slider (500–5000, default 1000)
- Selection metric: dropdown (top_1pct_rate, win_rate, cash_rate, expected_roi)
- Diversity weight: slider (0.0–1.0, default 0.3)
- Preset buttons: "GPP", "Cash", "Single Entry"
- "Run Simulation" button

**Results Display:**

1. **Candidate Lineup Rankings Table** — sortable by any sim metric:
   | lineup_id | mean | p90 | p99 | win_rate | top1% | cash% | E[ROI] | ownership | selected |
   |---|---|---|---|---|---|---|---|---|---|
   | 14 | 112.3 | 145.6 | 178.2 | 0.3% | 4.2% | 62% | 1.85 | 48.2% | YES |
   | 7 | 108.1 | 148.9 | 182.4 | 0.4% | 3.8% | 55% | 1.92 | 39.1% | YES |

2. **Player Distribution Viewer** — select a player, see their simulated outcome distribution as a histogram. Overlay the current point projection as a vertical line.

3. **Portfolio Summary Panel:**
   - Portfolio win rate (P any lineup wins)
   - Portfolio cash rate
   - Portfolio expected ROI
   - Unique players used / total player exposure chart
   - Avg pairwise overlap between selected lineups

4. **Correlation Heatmap** — show the correlation matrix for the top 20 players, color-coded.

5. **Simulation Convergence Chart** — running average of a key metric (e.g., mean win rate) as N increases, to show the user that 10k sims was sufficient.

### 14.2 Integration with Existing Workflow

The simulation tab should be optional. The existing workflow (generate lineups via ILP solver, export directly) must still work. Simulation is an enhancement step that the user opts into.

In `daily_workflow.py`, after Step 3 (Configure & Optimize), add:
- "Run Simulation Analysis" button
- If clicked, run the simulation pipeline and show results
- "Use Simulation Selections" button to replace the ILP-only lineups with sim-selected lineups
- The export step (Step 4) should work with either the ILP lineups or the sim-selected lineups

---

## 15. Backtest Extensions

### 15.1 Brier Score for Projections

Extend `src/slate_optimizer/analysis/backtest.py` with:

```python
def calculate_brier_score(
    date: str,
    db_path: Path,
    threshold: float = 15.0,
) -> dict[str, float]:
    """Brier score: how well did our projected distributions predict
    which players would exceed the threshold?

    For each player:
        prediction = P(player > threshold) from their fitted distribution
        outcome = 1 if actual_fd_points > threshold, else 0

    Brier score = mean((prediction - outcome)^2)

    This extends the existing calculate_projection_accuracy() with a
    distributional calibration metric.

    Returns:
        brier_score, num_players, pct_above_threshold
    """
    ...
```

### 15.2 Simulation Accuracy Tracking

After each slate, when actual results are uploaded, compute:

1. **Distribution calibration**: For each percentile bucket (10th, 25th, 50th, 75th, 90th), what % of players actually scored below that percentile? Perfect calibration = the percentages match (10% below p10, 25% below p25, etc.).

2. **Correlation accuracy**: Did teammates actually co-produce at the rate the copula predicted? Measure the actual correlation between teammate fantasy points and compare to the model's assumed correlation.

3. **Field accuracy**: How close was the simulated field to actual contest results? Compare simulated winning score distribution to actual winning scores.

Store these in a new SQLite table:

```sql
CREATE TABLE simulation_accuracy (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    metric_name TEXT NOT NULL,       -- 'brier_score', 'dist_calibration_p50', 'teammate_corr', etc.
    metric_value REAL NOT NULL,
    num_players INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 16. Dependencies

Add to `requirements.txt`:

```
scipy>=1.11.0       # Already likely installed (used by copula CDF/PPF, chi-squared, t-dist)
```

**No other new dependencies required.** The entire simulation engine uses NumPy + SciPy, which are standard scientific Python and likely already in the environment. Do NOT introduce heavy packages like PyMC, Stan, or pyvinecopulib — keep it lightweight.

Verify that `scipy.stats` (for `t`, `norm`, `lognorm`, `truncnorm`, `chi2`) and `scipy.linalg` (for `cholesky`) are available. They are part of base SciPy.

---

## 17. Priority & Sequencing

**Build in this order. Each phase is testable independently.**

| Phase | Module | What It Does | Depends On | Test |
|---|---|---|---|---|
| 1 | `distributions.py` | Fit player outcome distributions | None | Validate mean ≈ proj_mean, sample stats match |
| 2 | `correlation.py` | Build correlation matrix from optimizer dataset | None | Matrix is PSD, dimensions match player count |
| 3 | `slate_simulator.py` | Monte Carlo correlated outcome draws | Phases 1+2 | Scores are non-negative, correlations match model |
| 4 | `field_simulator.py` | Generate simulated opponent lineups | None | Lineups are valid (9 players, salary cap, positions) |
| 5 | `contest_simulator.py` | Score candidates vs field across sims | Phases 3+4 | Win rates sum correctly, ROI is reasonable |
| 6 | `lineup_selector.py` | Portfolio-optimal lineup selection | Phase 5 | Exposure constraints respected, diversity > 0 |
| 7 | `variance_reduction.py` | Antithetic/stratified/control variate | Phase 3 | Variance of estimates lower than crude MC |
| 8 | `config.py` | Configuration dataclass + presets | None | Load/save round-trips, presets instantiate |
| 9 | `scripts/run_simulation.py` | CLI entry point | All above | End-to-end run produces valid output CSV |
| 10 | `scripts/run_simulated_pipeline.py` | Full pipeline + simulation | Phase 9 | Pipeline runs from raw data to sim-selected export |
| 11 | `dashboard/simulation_tab.py` | Streamlit UI | Phase 9 | Renders without errors, buttons trigger sim |
| 12 | Backtest extensions | Brier score + calibration tracking | Phase 1 | Metrics compute with synthetic data |

---

## 18. Acceptance Criteria

### Phase 1 — Distributions
- [ ] `fit_player_distributions()` returns a distribution for every player in the optimizer dataset
- [ ] Batter distributions are lognormal, pitcher distributions are truncated normal
- [ ] `dist.mean()` is within 0.5 FD points of `proj_fd_mean` for all players
- [ ] `dist.sample(10000)` produces no negative values
- [ ] Salary-variance relationship is visible: cheap hitters have higher sigma than expensive ones

### Phase 2 — Correlation
- [ ] Correlation matrix has correct dimensions (P × P)
- [ ] Teammate pairs have positive correlation (0.15–0.40)
- [ ] Pitcher vs opposing hitters have negative correlation (-0.10 to -0.20)
- [ ] Different-game pairs have zero correlation
- [ ] Matrix is positive semi-definite (Cholesky succeeds without correction, or correction is applied transparently)
- [ ] Vegas game total modulates teammate correlation strength

### Phase 3 — Slate Simulator
- [ ] `simulate_slate()` returns (N × P) matrix of non-negative scores
- [ ] Teammate scores are correlated (empirical correlation ≈ model correlation ± 0.05)
- [ ] `lineup_scores()` correctly sums 9 player scores per simulation
- [ ] 10,000 simulations × 100 players completes in < 60 seconds
- [ ] Antithetic variates reduce variance by 20%+ vs crude MC (measurable)

### Phase 4 — Field Simulator
- [ ] All generated lineups have exactly 9 players
- [ ] All lineups are under the salary cap
- [ ] Player appearance rate in generated lineups correlates with ownership % (r > 0.7)
- [ ] No duplicate players within a single lineup
- [ ] 1000 lineups generate in < 5 seconds

### Phase 5 — Contest Simulator
- [ ] Every candidate lineup gets a `win_rate`, `top_1pct_rate`, `cash_rate`, `expected_roi`
- [ ] Win rates across all candidates + field sum to ~100% per simulation
- [ ] High-ownership lineups tend to have higher `cash_rate` but lower `win_rate` (validates contrarian thesis)
- [ ] Low-ownership, high-ceiling lineups tend to have higher `top_1pct_rate`
- [ ] Metric calculations are consistent: if lineup A beats lineup B in more simulations, A's win_rate > B's

### Phase 6 — Lineup Selector
- [ ] Selected portfolio respects `max_player_exposure` constraint
- [ ] Selected portfolio respects `max_overlap` constraint
- [ ] `diversity_weight = 0` produces the K lineups with highest individual metric
- [ ] `diversity_weight = 0.5` produces a more diverse set (measurably lower avg overlap)
- [ ] Portfolio-level metrics (portfolio_win_rate, etc.) are computed correctly

### Phase 7 — Variance Reduction
- [ ] Antithetic variates produce estimates with measurably lower standard error
- [ ] Control variate adjustment doesn't bias the estimates (mean is preserved)
- [ ] Stratified sampling covers the full outcome range (no gaps)

### CLI Scripts
- [ ] `run_simulation.py` runs end-to-end and produces a CSV with lineup sim metrics
- [ ] `run_simulated_pipeline.py` runs from raw data to sim-selected FanDuel export
- [ ] Both scripts accept `--sim-config` JSON path

### Streamlit Integration
- [ ] Simulation tab appears in the daily workflow
- [ ] "Run Simulation" button triggers the pipeline and displays results
- [ ] Player distribution histograms render correctly
- [ ] Portfolio summary panel shows correct aggregate metrics
- [ ] "Use Simulation Selections" replaces ILP lineups with sim-selected ones
- [ ] Export still produces valid FanDuel upload CSV

---

## 19. What NOT to Do

1. **DO NOT rewrite the PuLP solver.** The ILP optimizer generates candidates. Simulation evaluates and selects from them. They are complementary, not competing.

2. **DO NOT modify** existing files in `src/slate_optimizer/projection/`, `src/slate_optimizer/optimizer/`, or `src/slate_optimizer/ingestion/` unless strictly necessary (e.g., adding an import). All new code goes in `src/slate_optimizer/simulation/`.

3. **DO NOT use PyMC, Stan, TensorFlow, or any heavyweight Bayesian inference library.** The simulation engine is pure NumPy + SciPy. Keep it fast and dependency-light.

4. **DO NOT implement vine copulas.** The Student-t copula is sufficient for V1. Vine copulas add complexity for marginal gain on a 100-player slate.

5. **DO NOT build a real-time streaming system.** This is a batch process: run simulation once per slate, review results, export. Real-time updating is out of scope.

6. **DO NOT add agent-based ownership modeling.** The field simulator (Phase 4) uses ownership percentages directly. Agent-based modeling is a future enhancement.

7. **DO NOT over-optimize performance prematurely.** Get correct results first, then optimize if needed. NumPy vectorization should be sufficient — avoid Cython, Numba, or multiprocessing unless profiling shows a specific bottleneck.

8. **DO NOT introduce new database tables** beyond the single `simulation_accuracy` table in Section 15.2. Simulation results go to CSV files, not the database.

---

## 20. Testing Strategy

### Unit Tests

Place tests in `tests/simulation/`. Test each module independently with synthetic data.

```
tests/simulation/
├── test_distributions.py
├── test_correlation.py
├── test_slate_simulator.py
├── test_field_simulator.py
├── test_contest_simulator.py
├── test_lineup_selector.py
└── test_variance_reduction.py
```

**Key synthetic test data:**
- Create a small optimizer dataset (10 players: 2 pitchers, 8 hitters across 2 teams in 1 game)
- Known projections, known correlations → verify simulation statistics match analytical expectations
- Example: if two players have correlation 0.3 and we simulate 100k times, empirical correlation should be 0.30 ± 0.02

### Integration Test

`tests/test_simulation_pipeline.py`:
1. Load a real optimizer dataset from `data/processed/`
2. Run the full simulation pipeline (distributions → correlation → slate sim → field → contest → selection)
3. Verify the output has the expected columns and reasonable values
4. Verify it completes in < 120 seconds

---

## 21. Key File Reference

**Read these BEFORE writing any code:**

| File | Why |
|---|---|
| `src/slate_optimizer/optimizer/solver.py` | Understand `generate_lineups()` signature and `LineupResult` dataclass — your simulation consumes these |
| `src/slate_optimizer/optimizer/dataset.py` | Understand the optimizer dataset schema — `OPTIMIZER_COLUMNS` is your input contract |
| `src/slate_optimizer/projection/baseline.py` | Understand how `proj_fd_mean`, `proj_fd_floor`, `proj_fd_ceiling` are computed — your distributions extend these |
| `src/slate_optimizer/optimizer/config.py` | Understand `OptimizerConfig` pattern — model your `SimulationConfig` the same way |
| `scripts/run_daily_pipeline.py` | Understand the full pipeline flow — your `run_simulated_pipeline.py` extends this |
| `dashboard/daily_workflow.py` | Understand the Streamlit workflow — your simulation tab integrates here |
| `src/slate_optimizer/data/storage.py` | Understand the DB schema — your backtest extensions add one table here |

---

## 22. Glossary

| Term | Definition |
|---|---|
| **GPP** | Guaranteed Prize Pool — a tournament-style DFS contest where top-heavy payouts reward high scores. The contrarian strategy targets GPPs. |
| **Cash game** | 50/50 or double-up contest where ~50% of entries win. Consistency beats upside. |
| **Copula** | A function that models the dependency structure between random variables, separate from their marginal distributions. |
| **Tail dependence** | The probability of extreme co-movement. Gaussian copulas have zero tail dependence; Student-t copulas have positive tail dependence. |
| **Antithetic variates** | Variance reduction technique: for each random draw U, also evaluate at (1-U). Exploits negative correlation to reduce estimator variance. |
| **Ownership** | The % of contest entries that roster a particular player. High ownership = "chalk." Low ownership = contrarian value. |
| **Leverage** | The gap between a player's projection rank and ownership rank. Positive leverage = projected better than owned = contrarian edge. |
| **Slate** | The set of all games and players available for a single DFS contest day. |
| **Stack** | Rostering 3-5 hitters from the same team to capture correlated scoring. |
| **Bring-back** | Including 1-2 hitters from the opposing team in a game stack to capture game-environment correlation. |
| **Exposure** | The % of your lineup set that includes a particular player. 40% exposure = player is in 8 of 20 lineups. |
