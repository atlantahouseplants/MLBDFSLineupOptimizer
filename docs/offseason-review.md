# MLB DFS Optimizer: Offseason Review (2026 → 2027)

Branch `offseason-review`, cut from `windows-ui-jul2026`. Tag `season-2026-final` is untouched.
Last updated 2026-11-16. Section 8 lists what is done and what is not.

## 1. Verdict

The architecture is sound. The numbers feeding it were not, and nobody could have known,
because the system never recorded a single actual result: `actual_scores`,
`ownership_history`, `slate_results` and `lineup_results` in `data/slates.db` all have zero
rows after a full season.

So I pulled real box scores for the 24 slates that have saved BallparkPal data (Jul 19 to
Aug 11: 5,733 hitter starts, 648 pitcher starts) and tested the system against them.

- **Projections:** every hand-set multiplier made projections worse than raw BallparkPal.
- **Simulator:** scores varied half as much as real baseball (hitters) or a third as much
  (pitchers), no hitter could ever score zero, and teammates were assumed about twice as
  correlated as they are.
- **Bugs:** five real ones, three of them in the numbers the leverage strategy runs on.
- **The main button was broken.** The dashboard's "Run Analysis" button could not complete.

All of that is now fixed on this branch and backed by tests. What is still unvalidated is the
part that matters most for a contrarian player: the ownership model and the simulated field.
No actual FanDuel ownership was ever saved, so there was nothing to calibrate them against.

What is good: PuLP lineup generation with real FanDuel constraints, a correlated Monte Carlo
sim, a sampled field, payout-ladder support, and a UI a non-developer can run. That is the
shape a serious GPP tool has.

What was cargo-cult: the multiplier layer (it re-applied batting order, platoon, park and run
environment that BallparkPal's simulation already contains), the "weather score" (it is the
first-inning run percentage, not weather), correlation bonuses for adjacent hitters and
"high priority" stacks (the data shows neither), and about a dozen hand-tuned selection
weights never checked against an outcome.

What a pro would pay for: calibrated ownership, a field model built from real contest data,
and a portfolio objective tied to the payout ladder. None of the three exists yet in a
validated form.

## 2. How it actually works (one lineup, end to end)

1. `scripts/fetch_live_data.py` calls the BallparkPal API and writes
   `data/live/bpp_{batters,pitchers,games,teams}_DATE.csv`, plus MLB Stats API batting orders
   and Odds API lines.
2. **New:** `ingestion/preflight.py` checks the FanDuel file is MLB, is for the same day and
   the same games as the BallparkPal data, and stops with a plain-English reason if not.
3. `ingestion/slate_builder.build_player_dataset` joins BallparkPal to the FanDuel CSV by name.
4. `projection/baseline.compute_baseline_projections` turns `bpp_points_fd` into
   `proj_fd_mean`, now under `config/projection_model.json`.
5. `projection/ownership_model` scores each player and spreads 100% pitcher / 800% hitter
   ownership with a softmax. Weights are hand-set priors; nothing has ever been fitted.
6. `optimizer/dataset.build_optimizer_dataset` adds leverage scores;
   `optimizer/solver.generate_lineups` solves one ILP per lineup with stack templates.
7. `simulation/`: per-player score distributions, a copula for correlation, a field sampled
   from the ownership model, then `simulate_contest` places each candidate in that field.
   Distributions and correlations now come from `config/simulation_calibration.json`.
8. `simulation/lineup_selector.select_portfolio` picks the final set: an ILP when the
   marginal-value weight is zero, otherwise greedy.
9. `optimizer/export` writes `*_simulated_upload.csv`.

There are **two different pipelines**. The CLI (`run_simulated_pipeline.py`, used by the cron
and the Run Analysis button) builds the pool with leverage weight 0 and no ownership penalty,
so it is pure projection maximisation. The stepwise dashboard path calls the solver directly
with leverage, ceiling and ownership-penalty weights. Same tool, different strategy.

## 3. Math review

### 3a. Projections (measured, fixed)

Hitters, games whose Vegas line was fetched before first pitch (3,754 starts):

| Variant | RMSE | Slope | Top-decile actual |
|---|---|---|---|
| Raw BallparkPal | 9.236 | 0.83 | 11.59 |
| 2026 dashboard path (all multipliers) | 9.482 | 0.37 | 11.31 |
| 2026 cron path | 9.369 | 0.46 | 11.20 |
| Batting-order multiplier only | 9.289 | 0.61 | 11.64 |
| Platoon multiplier only | 9.266 | 0.74 | 11.43 |
| Market ratio, alpha 0.5 (now shipped) | 9.209 | 0.93 | 11.99 |

Slope is how much actual points move per projected point; 1.0 is correct, 0.37 means the
projections were spread about 2.7x too wide. Slate-clustered bootstrap, 95% interval on RMSE
change vs raw: 2026 dashboard +0.18 to +0.31 (reliably worse); market ratio -0.048 to -0.006
(reliably, but only slightly, better). The correlation gain for the market ratio is not
significant (-0.001 to +0.017).

Pitchers: raw BallparkPal RMSE 14.24; the 2026 win multiplier and additive adjustment made it
14.84 (interval +0.26 to +0.97). Raw is now used.

Team runs (434 team-games): BallparkPal ran 6% hot (4.54 projected vs 4.28 actual) with slope
0.70. Vegas was unbiased (4.29) with slope 0.89. This matches the season archive finding that
the market beats BallparkPal where they disagree.

Tables: `data/backtest/projection_backtest_summary.csv`. The `_2026_as_shipped` file was
produced before the bug fixes and preserves the original behaviour.

### 3b. Simulator (measured, fixed)

Real versus simulated score distributions, pooled over all 24 slates
(`data/backtest/sim_validation.csv`):

| Quantity | | Mean | SD | 10th pct | 90th pct | 99th pct |
|---|---|---|---|---|---|---|
| Hitter score | Actual | 8.9 | 9.4 | 0.0 | 22.2 | 39.0 |
| | 2026 simulator | 9.1 | 4.9 | 5.0 | 14.5 | 27.7 |
| | Calibrated | 9.1 | 9.5 | 0.0 | 21.2 | 42.8 |
| Pitcher score | Actual | 25.9 | 15.6 | 6.0 | 46.0 | 58.0 |
| | 2026 simulator | 25.3 | 8.7 | 15.3 | 36.7 | 48.4 |
| | Calibrated | 25.3 | 15.7 | 5.4 | 45.9 | 63.2 |
| Team's 9 hitters | Actual | 79.4 | 40.3 | 33.7 | 134.6 | 192.8 |
| | 2026 simulator | 81.9 | 26.5 | 56.9 | 112.0 | 178.0 |
| | Calibrated | 81.8 | 40.4 | 36.3 | 134.6 | 205.5 |
| Top-4 stack | Actual | 40.2 | 24.3 | 12.0 | 74.8 | 109.8 |
| | 2026 simulator | 40.4 | 14.8 | 26.7 | 57.2 | 94.6 |
| | Calibrated | 40.3 | 23.6 | 14.4 | 71.2 | 114.3 |

The pitcher SD row understates the old problem: around a given pitcher's projection the 2026
simulator's SD was 4.3 against a real 14.3. In the old simulator the pitcher was close to a
fixed number, a hitter never scored under about 4, and a stack never busted.

Measured correlations of fantasy points (95% slate-clustered interval) against the 2026
copula settings:

| Pair | Measured | 2026 setting |
|---|---|---|
| Teammates | 0.12 (0.10 to 0.14), flat across batting-order distance | 0.25 base, +0.10 adjacent, +0.05 "high" stacks, scaled up with Vegas total |
| Opposing hitters, same game | -0.01 (-0.03 to 0.01) | +0.10 |
| Pitcher vs each opposing hitter | -0.31 (-0.33 to -0.28) | -0.15 |
| Pitcher vs own hitters | +0.06 (0.02 to 0.09) | +0.05 |
| Hitters in different games | about 0.01 | 0.04 (side effect of copula nu = 5) |

The new model: a hitter scores zero with probability `exp(-projection / 6.1)`, otherwise 3
points plus a gamma; a pitcher is a normal with SD `8.7 + 0.22 x projection`, bounded below at
-20 and above at 2.3 SDs. Copula parameters are solved by simulation so that simulated
fantasy-point correlations equal the measured ones. The adjacency, "high priority" and
Vegas-scaling bonuses are off.

Caveats, plainly: the validation is in-sample (fitted and checked on the same 24 slates,
though with only 8 parameters against 6,381 starts); the hitter 99th percentile is still
about 10% too fat; and the model is driven by the projection alone, so two hitters with the
same projection get the same shape even if one is a power bat. BallparkPal's website
Bust/Upside numbers, when uploaded, no longer change the simulated shape.

### 3c. Contest simulation and the field

"10,000 sims" is fine. Two things were not:

- **Fixed:** `simulate_contest` ranked candidates and field together. With 500 candidates and
  1,000 field lineups a third of the "contest" was your own pool, so lineups sharing a popular
  stack pushed each other out of the top 1%. Candidates are now placed against the field only.
- **Not fixed:** the field is far too weak. In a test run on the Mar 26 slate the simulator
  said the average candidate lineup finishes top-1% 7.8% of the time and returns +259% per
  entry. Nobody has that edge. The field is 60% "recreational" and 30% random lineups, sampled
  from our own ownership estimates and scored with our own projections. Treat every ROI,
  win-rate and cash-rate figure the tool prints as a ranking device, not as a forecast.

### 3d. Selection objective

Expected payout is additive across lineups, so a pure-EV player needs no diversification term.
Diversification matters for bankroll risk, for duplicates, and because picking the top N by a
noisy estimate selects for the noise. The current objective is a weighted sum of
rank-normalised per-lineup scores, which discards magnitudes. The reported "portfolio win
rate" (`1 - prod(1 - p)`) assumes lineups are independent, which stacked lineups are not, and
"expected ROI 22x" is a sum over entries, not a return.

A proper joint objective at 300 lineups: keep the sims x candidates payout matrix and greedily
add the lineup with the largest marginal gain in expected utility of total payout, with a
concave utility (log of bankroll plus winnings). Linear utility gives pure EV; strong concavity
gives "cover as many winning worlds as possible". It replaces about six hand-set weights with
one risk setting. It is worth building, and the simulator is now accurate enough to make it
meaningful, but it should come after the field is fixed: optimising harder against a field
that hands every lineup +259% just amplifies that error.

## 4. Bugs found and fixed

1. **Pitchers lost 20% on any run without a Vegas file.** Missing odds arrive as 0.0, which
   was read as a 0% win chance. No cron run in `slates.db` has Vegas columns, so every cron
   run had this. Measured bias: -5.2 points per pitcher, now -0.1.
2. **Vegas implied team totals were `total x win probability`.** An 8.5 total with a 72%
   favourite gave 6.1 / 2.4; the correct split is about 5.3 / 3.2. This fed a hitter
   multiplier clipped at +/-30%, inflating exactly the favourite stack you intend to fade.
3. **`bpp_runs` is a batter's own projected runs (about 0.5) on batter rows.** Team leverage
   score, stack priority, the stack exposure plan and the ownership model's team-total feature
   all read it as the team total, so teams were ranked by whichever hitter sorted first.
4. **Candidates were ranked against each other** (section 3c).
5. **The Run Analysis button could not finish.** It passed `--stack-rotation`, which no script
   defined (argument error); it called `st.rerun()` on the first line of output, abandoning the
   run; and it looked for results in `data/output` while the script wrote to `data/processed`.
6. `ownership_model.py` contained the same three-line block pasted 46 times (harmless).

## 5. Ranked improvements (EV per build-hour)

| # | Item | Status |
|---|---|---|
| 1 | Outcome data + projection backtest + calibrated projection config | **Built** (`374455f`) |
| 2 | Fix pitcher haircut, Vegas implied totals, team-runs column | **Built** (`374455f`) |
| 3 | Fit simulator distributions and correlations to real outcomes | **Built** (`60a76dd`) |
| 4 | Pre-flight check for wrong sport / day / slate; in-play Vegas lines | **Built** (`60a76dd`) |
| 5 | Rank candidates against the field only | **Built** (`60a76dd`) |
| 6 | Repair the one-click Run Analysis path | **Built** (`60a76dd`), button itself not clicked |
| 7 | Save actual ownership and contest standings every slate, then fit the ownership model (`scripts/fit_ownership_model.py` already exists and has never had data) | Next season: needs data that does not exist yet |
| 8 | Rebuild the field from real contest lineups (duplication, stack shapes, salary left on the table) | Next season, after 7 |
| 9 | Joint portfolio objective on the payout matrix | Next season, after 8 |
| 10 | One pipeline instead of two (CLI and dashboard disagree on the objective) | Next season |
| 11 | Player-specific score shapes (power vs contact) in the simulator | Next season, testable now with the saved data |
| 12 | Split the 350KB dashboard file | Low EV; leave it unless it blocks 10 |

## 6. What was built

Three commits on `offseason-review`; 155 tests pass (112 before, 43 new). The original 10
smoke tests are unchanged and green.

**`374455f` Projection backtest and fixes**
- `scripts/fetch_actuals.py`: FanDuel points for every final game from the free MLB Stats API,
  joined to BallparkPal on exact game and player ids.
- `scripts/backtest_projections.py`: scores every projection variant with a slate-clustered
  bootstrap.
- `config/projection_model.json`: the settings the backtest chose (multipliers off, market
  ratio alpha 0.5). Delete the file to restore 2026 behaviour. The dashboard's platoon sliders
  do nothing while this file is in place, and the slate summary says so.
- `data/backtest/player_games.csv.gz`: every projected player-game with inputs, every
  variant's projection and the real outcome.

**`60a76dd` Simulator calibration, ranking fix, first-run safety**
- `scripts/calibrate_simulation.py` and `config/simulation_calibration.json`: the fitted
  simulator. Delete the file to restore the hand-set one. The dashboard's correlation inputs
  start from the fitted values and their ranges were widened to allow them.
- `ingestion/preflight.py`: wired into the CLI and the dashboard's slate setup.
- `--stack-rotation` and `--skip-preflight` on the CLI.

**Next-season routine (three commands, monthly):**

```
.venv\Scripts\python.exe scripts\fetch_actuals.py --all-live
.venv\Scripts\python.exe scripts\backtest_projections.py
.venv\Scripts\python.exe scripts\calibrate_simulation.py --write
```

## 7. Blind spots

1. **You were tuning without a scoreboard.** A season of settings changes and no recorded
   result to judge any of them. Capturing outcomes is worth more than any model change, and
   for ownership it is the only way forward.
2. **The tool's ROI numbers are fiction.** +259% per entry in a test run. Any decision made
   because "the sim says this portfolio is +EV" was made on a field that does not exist.
3. **Your leverage signal was partly noise, and partly backwards.** Team leverage ranked teams
   on one hitter's projected runs, and the Vegas multiplier pushed projections toward the
   heavy favourite, the opposite of the stated strategy.
4. **The cron often ran on the wrong FanDuel file.** In `slates.db`, about 10 of 33 cron runs
   used a players list whose filename date is not the slate date (the Jul 20 run used a Jul 4
   file). The pre-flight check now stops that.
5. **The only copy of the season's BallparkPal projections is an untracked folder.**
   `data/live` (34 MB) is not in git and BallparkPal's API does not serve history. The joined
   backtest file is now committed, but the raw folder should be backed up.
6. **BallparkPal has no demonstrated edge over the market.** Your archive showed this on 136k
   props and the team-run numbers here agree. Any edge in this system has to come from
   ownership and construction, which are the two parts still uncalibrated.

## 8. Not done, not verified

- The Run Analysis button was repaired and the exact command it issues was run successfully
  from the command line, but the button was not clicked in a browser. Every dashboard page was
  loaded headlessly without errors; the simulation inputs on step 4 only render with a slate
  loaded, so their new ranges were not rendered.
- No contest was entered with any of this. The fixes are validated against outcomes, not
  against results in a real FanDuel contest.
- Ownership model and field model: untouched, for lack of data (items 7 and 8).
- The skill manual (`SKILL.md`) still describes an ownership model with percentage weights and
  a 3-tab UI that do not match this tree. It was not updated.
- `data/actuals/*.csv` (the per-day box-score cache) is not committed; one command rebuilds it.
