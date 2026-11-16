# MLB DFS Optimizer: Offseason Review (2026 → 2027)

Branch `offseason-review`, cut from `windows-ui-jul2026` (tag `season-2026-final` untouched).
Written 2026-11-15. **This review is incomplete**: the session hit its usage limit after the
first build. Section 7 lists exactly what is done and what is not.

## 1. Verdict

The architecture is sound. The inputs to it were not, and nobody could have known, because the
system never recorded a single actual result: `actual_scores`, `ownership_history`,
`slate_results` and `lineup_results` in `data/slates.db` all have zero rows after a full season.

So I pulled real box scores for the 24 slates that have saved BallparkPal data (Jul 19 to Aug 11:
5,733 hitter starts, 648 pitcher starts) and tested the projection layer against them. Result:

- **Every hand-set multiplier made projections worse than raw BallparkPal.** The full 2026
  stack raised hitter RMSE from 9.24 to 9.48 and spread projections about 2.7x too wide
  (calibration slope 0.37, where 1.0 is correct).
- **Three real bugs** were distorting the numbers the leverage strategy runs on (section 3).
- The one adjustment that helped is small: nudging hitters toward the Vegas team total where
  Vegas and BallparkPal disagree.

What is good: PuLP lineup generation with real FanDuel constraints, a correlated Monte Carlo
sim, a sampled field, payout-ladder support, and a UI a non-developer can run. That is the
shape a serious GPP tool has.

What is cargo-cult: the multiplier layer (it re-applies batting order, platoon, park and run
environment that BallparkPal's simulation already contains), the "weather score" (it is the
first-inning run percentage, not weather), and roughly a dozen hand-tuned selection weights
that were never checked against an outcome.

What a pro would pay for: calibrated ownership, a field model built from real contest data,
and a portfolio objective tied to the payout ladder. None of the three exists yet in a
validated form.

## 2. How it actually works (one lineup, end to end)

1. `scripts/fetch_live_data.py` calls the BallparkPal API and writes
   `data/live/bpp_{batters,pitchers,games,teams}_DATE.csv`, plus MLB Stats API batting orders
   and Odds API lines.
2. `ingestion/slate_builder.build_player_dataset` joins BallparkPal to the FanDuel CSV by name.
3. `projection/baseline.compute_baseline_projections` turns `bpp_points_fd` into `proj_fd_mean`.
4. `projection/ownership_model` scores each player and spreads 100% pitcher / 800% hitter
   ownership with a softmax. Weights are hand-set priors; nothing has ever been fitted.
5. `optimizer/dataset.build_optimizer_dataset` adds leverage scores;
   `optimizer/solver.generate_lineups` solves one ILP per lineup with stack templates.
6. `simulation/`: lognormal hitters and truncated-normal pitchers, Student-t copula for
   correlation, a field sampled from the ownership model, then `simulate_contest` ranks
   candidates and field together.
7. `simulation/lineup_selector.select_portfolio` picks the final set: an ILP when the
   marginal-value weight is zero, otherwise greedy.
8. `optimizer/export` writes `*_simulated_upload.csv`.

There are **two different pipelines**. The CLI (`run_simulated_pipeline.py`, used by the cron)
generates the pool with leverage weight 0 and no ownership penalty, so it is pure projection
maximisation. The dashboard (`daily_workflow.py`) calls the solver directly with leverage,
ceiling and ownership-penalty weights. Same name, different strategy.

## 3. Math review

### Projections (measured)

Hitters, games whose Vegas line was fetched before first pitch (3,754 starts):

| Variant | RMSE | Slope | Top-decile actual |
|---|---|---|---|
| Raw BallparkPal | 9.236 | 0.83 | 11.59 |
| 2026 dashboard path (all multipliers) | 9.482 | 0.37 | 11.31 |
| 2026 cron path | 9.369 | 0.46 | 11.20 |
| Batting-order multiplier only | 9.289 | 0.61 | 11.64 |
| Platoon multiplier only | 9.266 | 0.74 | 11.43 |
| Market ratio, alpha 0.5 (now shipped) | 9.209 | 0.93 | 11.99 |

Slate-clustered bootstrap, 95% interval on RMSE change vs raw: 2026 dashboard +0.18 to +0.31
(reliably worse); market ratio -0.048 to -0.006 (reliably, but only slightly, better). The
correlation gain for the market ratio is not significant (-0.001 to +0.017).

Pitchers: raw BallparkPal RMSE 14.24; the 2026 win multiplier and additive adjustment made it
14.84 (interval +0.26 to +0.97). Raw is now used.

Team runs (434 team-games): BallparkPal ran 6% hot (4.54 projected vs 4.28 actual) with slope
0.70. Vegas was unbiased (4.29) with slope 0.89. This matches the season archive finding that
the market beats BallparkPal where they disagree.

Full tables: `data/backtest/projection_backtest_summary.csv`. The file with the suffix
`_2026_as_shipped` was produced before the bug fixes and preserves the original behaviour.

### Bugs found and fixed

1. **Pitchers lost 20% on any run without a Vegas file.** Missing odds arrive as 0.0, which
   was read as a 0% win chance. Every cron run in `slates.db` has no Vegas columns, so every
   cron run had this. Measured bias: -5.2 points per pitcher, now -0.1.
2. **Vegas implied team totals were `total x win probability`.** An 8.5 total with a 72%
   favourite gave 6.1 / 2.4; the correct split is about 5.3 / 3.2. This fed a hitter
   multiplier clipped at +/-30%, inflating exactly the favourite stack you intend to fade.
3. **`bpp_runs` is a batter's own projected runs (about 0.5) on batter rows.** Team leverage
   score, stack priority, the stack exposure plan and the ownership model's team-total feature
   all read it as the team total, so teams were ranked by whichever hitter sorted first.
4. `ownership_model.py` contained the same three-line block pasted 46 times (harmless, removed).

### Simulation (measured, not yet fixed)

Residual correlations from the same 24 slates versus the simulator's settings. The simulator's
numbers are copula parameters, so they are not exactly comparable, but the gaps are too large
to be a units issue:

| Pair | Measured | Simulator assumes |
|---|---|---|
| Teammates, any batting-order distance | 0.13 (flat from distance 1 to 4) | 0.25 base, +0.10 adjacent, +0.05 "high" stacks |
| Opposing hitters, same game | -0.01 | +0.10 |
| Pitcher vs each opposing hitter | -0.32 | -0.15 |
| Pitcher vs own hitters | +0.06 | +0.05 |

Consequences: stacks are over-rewarded, bring-backs are rewarded for a correlation that is not
there, and pitcher-versus-stack conflict is under-penalised.

Marginals: 23.3% of hitter starts score exactly zero and 40% score 3 or less. The simulator
uses a lognormal shifted up by half the floor, which can never produce a zero. Real tails:
15% of starts reach 2x projection, 4.9% reach 3x, 1.4% reach 4x.

"10,000 sims" is fine. The weak points are the field and the ranking: 1,000 field lineups by
default (not 10,000), and `simulate_contest` ranks candidates against each other as well as the
field, so a 500-lineup pool is a third of the "contest" and popular stacks in your own pool
cannibalise each other's top-1% rate.

### Selection objective

Expected payout is additive across lineups, so a pure-EV player needs no diversification term.
Diversification matters for bankroll risk, for duplicates, and because picking the top N by a
noisy estimate selects for the noise. The current objective is a weighted sum of rank-normalised
per-lineup scores, which discards magnitudes, and the reported portfolio win rate
(`1 - prod(1 - p)`) assumes lineups are independent, which stacked lineups are not.

A proper joint objective at 300 lineups: keep the sims x candidates payout matrix and greedily
add the lineup with the largest marginal gain in expected utility of total payout, with a
concave utility (log of bankroll plus winnings). Linear utility gives pure EV; strong concavity
gives "cover as many winning worlds as possible". It replaces about six hand-set weights with
one risk setting. Worth building, but only after the sim's correlations and distributions are
corrected: optimising harder against a mis-specified sim amplifies its errors.

## 4. Ranked improvements (EV per build-hour)

| # | Item | Status |
|---|---|---|
| 1 | Outcome data + projection backtest + calibrated projection config | **Built** (`374455f`) |
| 2 | Fix pitcher haircut, Vegas implied totals, team-runs column | **Built** (`374455f`) |
| 3 | Apply measured correlations to the simulator; zero-inflated hitter distribution | Build now, not done |
| 4 | Pre-flight check: refuse a FanDuel file whose date or sport does not match the slate; reject in-play Vegas lines at fetch time | Build now, not done |
| 5 | Rank candidates against the field only in `simulate_contest` | Build now, not done |
| 6 | Record actual ownership and contest results every slate, then fit `fit_ownership_model.py` | Next season (needs data that does not exist yet) |
| 7 | Joint portfolio objective on the payout matrix | Next season, after 3 and 5 |
| 8 | Merge the CLI and dashboard pipelines into one code path | Next season |
| 9 | Split the 350KB dashboard file | Low EV; leave it unless it blocks 8 |

## 5. What was built

Commit `374455f` on `offseason-review`. 134 tests pass (112 before, 22 new).

- `scripts/fetch_actuals.py`: FanDuel points for every final game from the free MLB Stats API,
  joined to BallparkPal on exact game and player ids.
- `scripts/backtest_projections.py`: scores every projection variant with a slate-clustered
  bootstrap. Re-run it any time; next season it should be run monthly.
- `config/projection_model.json`: the settings the backtest chose (multipliers off, market
  ratio alpha 0.5). The pipeline and dashboard load it. Delete the file to restore 2026
  behaviour. The dashboard's platoon sliders now do nothing while this file is in place, and
  the slate summary says so.
- `data/backtest/player_games.csv.gz`: every projected player-game with inputs, every
  variant's projection and the real outcome.

## 6. Blind spots

1. **You were tuning without a scoreboard.** A season of settings changes and no recorded
   result to judge any of them. Capturing outcomes is worth more than any model change.
2. **The cron often ran on the wrong FanDuel file.** In `slates.db`, about 10 of 33 cron runs
   used a players list whose filename date is not the slate date (for example the Jul 20 run
   used a Jul 4 file). Salaries and player ids were stale on those runs.
3. **Your leverage signal was partly noise.** Team leverage ranked teams on one hitter's
   projected runs, and the Vegas multiplier pushed projections toward the heavy favourite, the
   opposite of the stated strategy.
4. **The only copy of the season's BallparkPal projections is an untracked folder.**
   `data/live` (34 MB) is not in git and BallparkPal's API does not serve history. Back it up.
5. **BallparkPal has no demonstrated edge over the market.** Your own archive showed this on
   136k props, and the team-run numbers here agree. The edge in this system, if there is one,
   has to come from ownership and construction, which are the two parts that are uncalibrated.

## 7. Not done

- Items 3, 4 and 5 in the table were intended for this pass and were not built.
- The simulation numbers in section 3 are measurements from an exploratory script that was not
  committed; no simulator setting was changed.
- Not verified by running: the dashboard's one-click analysis passes `--stack-rotation` to
  `run_simulated_pipeline.py`, and no script in this tree defines that flag (found by search).
  If so, that button fails with an argument error.
- The skill manual (`SKILL.md`) describes an ownership model and a `--stack-rotation` flag
  that do not match this tree. It was not updated.
- The dashboard was not launched to check the changes visually.
