# MLB DFS Simulation Engine — User Guide

**A plain-English walkthrough of what this is, why it matters, and how to use it**

---

## Table of Contents

1. [What Is This?](#1-what-is-this)
2. [Why Does This Matter for GPPs?](#2-why-does-this-matter-for-gpps)
3. [The Big Picture: How It Works](#3-the-big-picture-how-it-works)
4. [Under the Hood: What Each Step Actually Does](#4-under-the-hood-what-each-step-actually-does)
5. [How to Use It: Step by Step](#5-how-to-use-it-step-by-step)
6. [Reading Your Results](#6-reading-your-results)
7. [The Key Settings and What They Do](#7-the-key-settings-and-what-they-do)
8. [Real Examples: What Good Output Looks Like](#8-real-examples-what-good-output-looks-like)
9. [FAQ and Troubleshooting](#9-faq-and-troubleshooting)

---

## 1. What Is This?

Your optimizer already builds lineups. It picks the best combination of players that fits the salary cap, stacks teammates together, and maximizes projected fantasy points. That part hasn't changed.

The simulation engine is a **new layer on top of the optimizer**. It answers a fundamentally different question:

> **Old question:** "What lineup scores the most points on average?"
>
> **New question:** "What lineup set gives me the best chance of finishing in the top 1% of a 150,000-person GPP?"

These are not the same question. The lineup that scores the most points on average is the lineup everyone else is also building — which means it rarely wins a tournament. The lineup that wins a tournament is the one that goes off on the right night while most of the field's lineups don't.

The simulation engine figures out which lineups those are by literally playing out the slate thousands of times and counting who wins.

---

## 2. Why Does This Matter for GPPs?

### The cash game vs. GPP difference

In a **cash game** (50/50, double-up), you need to beat roughly half the field. Consistency wins. Pick the best projected players, don't get cute, cash your ticket.

In a **GPP** (tournament), you need to beat 99%+ of the field to make real money. The payout structure is top-heavy — first place might win $100,000 while 50th place wins $50. To win, you need:

1. **A lineup that goes off** — not just a good score, but a great one
2. **A lineup that's different from the field** — because if your lineup is the same as 10,000 other entries and you all score high, you split the prize money 10,000 ways

### What the optimizer alone can't tell you

Your PuLP optimizer maximizes projected points. But it can't answer:

- "If I run this lineup 10,000 times, how often does it score 200+?"
- "How often does this lineup actually finish in the top 1% when I account for what everyone else is playing?"
- "Is this player's upside real or just noise? What does his outcome distribution actually look like?"
- "Am I too correlated with the field? Will I just tie with 5,000 other optimized lineups?"

The simulation engine answers all of these by running the slate thousands of times.

### The stacking insight

You already stack 4-5 hitters from the same team. But the optimizer treats this as a binary constraint: "put 4 Dodgers together." It doesn't model WHY stacking works.

The reason stacking works is **correlation**. When the Dodgers score 8 runs, Ohtani doesn't score 25 FD points while Betts scores 3. They both go off. Their outcomes are linked. And importantly, the linkage is strongest in blowup games — when a team explodes for 12 runs, the correlation between teammates is even higher than usual.

The simulation models this explicitly. It knows that teammate outcomes are correlated, that high-scoring games tend to benefit both teams, and that when teams go off, they REALLY go off (tail dependence). This produces more realistic simulations than treating each player as independent.

---

## 3. The Big Picture: How It Works

Here's the full flow in plain English:

```
STEP 1: You run your normal pipeline
         (upload FanDuel CSV, BallparkPal data, Vegas lines, etc.)
         Result: an optimizer dataset with projections for every player
                                    │
                                    ▼
STEP 2: The optimizer generates 500 candidate lineups
         (more than you need — these are the "options on the table")
                                    │
                                    ▼
STEP 3: The simulation engine gives every player a "distribution"
         (not just "he'll score 15 points" but "he'll score between
          3 and 35 points, with 15 being most likely, and a 10%
          chance of going over 25")
                                    │
                                    ▼
STEP 4: The simulation engine models correlations
         (teammates go off together, pitchers suppress opposing
          hitters, high-total games produce runs for both sides)
                                    │
                                    ▼
STEP 5: Simulate the slate 10,000 times
         (each simulation: every player gets a random fantasy score
          drawn from their distribution, with correlations respected.
          One sim might have the Dodgers exploding for 40 combined FD
          points. Another might have them going 0-for-30.)
                                    │
                                    ▼
STEP 6: Simulate the contest field
         (generate 1,000 fake opponent lineups using ownership %
          as the selection probability — high-owned players appear
          in more opponent lineups, low-owned players in fewer)
                                    │
                                    ▼
STEP 7: Score every candidate lineup against every simulated outcome
         AND against the simulated field
         (for each of 10,000 simulations: score all 500 candidates,
          score all 1,000 field lineups, rank them, record where
          each candidate finished)
                                    │
                                    ▼
STEP 8: Pick the best 20 lineups for your contest
         (not just the 20 with the highest average score — the 20
          that TOGETHER give you the best tournament equity, meaning
          they're diverse enough that you have multiple paths to winning)
                                    │
                                    ▼
STEP 9: Export to FanDuel and enter your contest
```

---

## 4. Under the Hood: What Each Step Actually Does

### Step 3: Player Distributions — "Not just a number, a range of possibilities"

Your current projections say something like "Ronald Acuna Jr. is projected for 16.0 FD points." That's a single number — a mean. But on any given night, Acuna might score:

- 0 points (0-for-4, no walks, nothing)
- 8 points (1-for-4 with a single)
- 16 points (2-for-4 with a double and a run scored — the "average" night)
- 35 points (3-for-4 with a homer, 2 RBIs, 2 runs, a stolen base — the blowup)

The simulation models this entire range using a **probability distribution**. For hitters, it uses a "lognormal" distribution — a bell curve that's shifted to the right, meaning there's a floor near zero and a long tail of possible blowup games. This matches reality: most hitter performances are mediocre, but occasionally someone goes nuclear.

**The salary-variance relationship:** Cheap hitters ($2,800-$3,500) have wider distributions — they're cheap because they're inconsistent. They might score 0 or they might score 30. Expensive hitters ($5,500+) have tighter distributions — they're expensive because they consistently deliver. The simulation encodes this: a $2,800 hitter's distribution is wider (more boom-bust) than a $5,500 hitter's at the same projection.

This is why cheap hitters are GPP gold. Their high variance means they occasionally score way above expectation, and when they do, nobody else has them in their lineup (because the ownership is low). The simulation captures this dynamic.

For **pitchers**, the distribution is a "truncated normal" — more symmetric, since pitcher performances don't skew as heavily as hitters.

### Step 4: Correlations — "Players don't exist in a vacuum"

This is where the math from the article you read comes in. The simulation uses something called a **Student-t copula** to model how player outcomes are linked.

**What's a copula?** It's a mathematical tool that lets you say "these two things are correlated" without changing what each thing looks like individually. Acuna's distribution stays the same, Betts' distribution stays the same, but when we simulate them together, the copula ensures that when Acuna has a huge game, Betts is more likely to also have a huge game.

**Why Student-t instead of Gaussian (normal)?** This is the key insight from the article. A Gaussian/normal copula says "things are correlated, but extreme co-movements basically never happen." A Student-t copula says "things are correlated, AND when extreme events happen, they tend to happen together."

In baseball terms:
- **Gaussian** would say: "There's a 25% correlation between Dodger hitters. Sometimes they both do well."
- **Student-t** would say: "There's a 25% correlation between Dodger hitters. AND when the Dodgers score 12 runs, there's a much higher chance that ALL of them went off, not just one."

The Student-t copula has a parameter called **degrees of freedom (nu/ν)**. Lower nu = fatter tails = more extreme co-movement. The default is ν=5, which provides meaningful tail dependence. At ν=5, when one teammate has a blowup game, there's roughly a 15-20% chance the other teammate also has an extreme game — much higher than the ~0% a Gaussian model would predict.

**The five types of correlation the model uses:**

| Relationship | Example | Correlation | Why |
|---|---|---|---|
| Teammates | Acuna ↔ Albies (both ATL batters) | +0.25 to +0.40 | When a team scores, multiple hitters produce |
| Adjacent batting order | Acuna (1st) ↔ Riley (3rd) | Extra +0.10 | Adjacent hitters create runs together |
| Same game, opposing teams | ATL hitter ↔ PHI hitter | +0.10 | High-scoring games benefit both sides |
| Pitcher + own hitters | ATL pitcher ↔ ATL hitter | +0.05 | Pitcher winning often means team scoring |
| Pitcher vs opposing hitters | ATL pitcher ↔ PHI hitter | -0.15 | Pitcher dominance means opposing hitters struggle |

**Vegas modifier:** When the game total is high (say, 11.5 runs), the teammate correlation gets boosted by up to 30%. This makes intuitive sense — a high-total game environment means more run-scoring, which means more correlated production.

### Step 5: Slate Simulation — "Playing out the night 10,000 times"

This is the Monte Carlo engine. Here's what happens for each simulation:

1. Generate 870 random numbers (one per player) that are **correlated** according to the copula
2. Transform each random number through the player's distribution to get a fantasy point score
3. Result: one complete slate outcome — every player has a fantasy score

Do this 10,000 times and you have 10,000 alternate realities of how tonight's slate could play out. In some simulations, the chalk (high-owned favorites) goes off. In others, the chalk busts and the low-owned sleepers dominate. The simulation captures the full range of possibilities.

**Antithetic variates** (variance reduction): For every set of random numbers we draw, we also evaluate the "mirror image" (if a random number was 0.8, the mirror is 0.2). This is a free trick that reduces the noise in our estimates by ~30-50%. It's like getting 15,000 simulations for the price of 10,000.

### Step 6: Field Simulation — "What is everyone else playing?"

You can't measure tournament edge without knowing what you're competing against. The field simulator generates 1,000 fake opponent lineups that represent what the other 150,000 entries in your GPP roughly look like.

Each fake lineup is built by randomly selecting players weighted by their projected ownership percentage. A player projected at 25% ownership appears in roughly 25% of simulated opponent lineups. A player at 3% ownership appears in roughly 3%.

The field has three tiers:
- **Sharks (10%)** — Well-built lineups that also weight by projection (other optimizers)
- **Recreational players (60%)** — Heavily weight expensive/popular players (star chasers)
- **Random entries (30%)** — Near-random selections (auto-fill, last-minute entries)

### Step 7: Contest Simulation — "Who wins?"

For each of the 10,000 simulated slates:
1. Score all 500 candidate lineups (sum up the 9 players' simulated fantasy points)
2. Score all 1,000 field lineups the same way
3. Rank everyone — candidate lineups + field lineups
4. Record where each candidate finished (top 1%? top 10%? out of the money?)
5. Map the finish to a payout based on the GPP payout structure

After 10,000 simulations, each candidate lineup has a full performance profile:
- "This lineup finished in the top 1% in 4.2% of simulations"
- "This lineup cashed in 62% of simulations"
- "This lineup's expected ROI is 1.85x"

### Step 8: Portfolio Selection — "Picking the final 20"

This is where it gets clever. You don't just pick the 20 lineups with the highest individual win rate. You pick the 20 that **together** maximize your total tournament equity.

Why? Because if your top 20 lineups all share the same 6 players, they'll all win on the same nights and all lose on the same nights. You're essentially entering 20 copies of the same bet. That's wasted entries.

The portfolio selector uses a **greedy algorithm with a diversity penalty**:
1. Pick the best lineup
2. For the next pick, consider each remaining candidate's individual merit MINUS a penalty for how much it overlaps with lineups already selected
3. Repeat until you have 20

This ensures your 20 lineups cover different scenarios. Some might be built around a Dodgers stack, others around a Cubs stack. If the Dodgers bust, your Cubs lineups still have a shot.

The **diversity_weight** parameter controls this tradeoff:
- 0.0 = just pick the top 20 by individual metric (maximum individual quality, minimum diversity)
- 0.3 = moderate diversity (recommended for 20-entry GPPs)
- 0.5 = heavy diversity (for 150-entry mass multi-entry)

---

## 5. How to Use It: Step by Step

### Option A: Command Line

**Step 1:** Run your normal daily pipeline to get an optimizer dataset:
```bash
python scripts/run_daily_pipeline.py \
    --bpp-source "C:\Users\wallg\OneDrive\Desktop\DFSMLB" \
    --fanduel-csv cleaned_players_list.csv \
    --output-dir data/processed \
    --write-intermediate
```

**Step 2:** Run the simulation on that dataset:
```bash
PYTHONPATH=src python scripts/run_simulation.py \
    --dataset data/processed/YOUR_TAG_optimizer_dataset.csv \
    --num-sims 10000 \
    --num-field 1000 \
    --num-candidates 200 \
    --num-select 20 \
    --metric top_1pct_rate \
    --output data/processed/sim_results.csv
```

**Step 3:** The output CSV has one row per candidate lineup with all simulation metrics. The selected lineups are the ones the portfolio optimizer chose.

**Or run everything in one shot:**
```bash
PYTHONPATH=src python scripts/run_simulated_pipeline.py \
    --bpp-source "C:\Users\wallg\OneDrive\Desktop\DFSMLB" \
    --fanduel-csv cleaned_players_list.csv \
    --num-lineups 20 \
    --num-candidates 200 \
    --output-dir data/processed
```

### Option B: Streamlit Dashboard

1. Run `streamlit run dashboard/daily_workflow.py`
2. Complete Steps 1-3 as normal (upload data, review projections, configure optimizer)
3. In the simulation tab, choose a preset (GPP, Cash, or Single Entry) or customize the sliders
4. Click "Run Simulation"
5. Review the results table and portfolio summary
6. Click "Use Simulation Selections" to replace the ILP-only lineups
7. Export to FanDuel

### Key CLI Parameters

| Parameter | What It Does | Default | Recommendation |
|---|---|---|---|
| `--num-sims` | How many times to simulate the slate | 10,000 | 10k for speed, 20k for precision |
| `--num-field` | How many opponent lineups to simulate | 1,000 | 1k is fine, 2k for large GPPs |
| `--num-candidates` | How many lineups the optimizer generates before simulation picks from them | 500 | More = better selection pool, but slower |
| `--num-select` | How many lineups to actually enter | 20 | Match your contest entry count |
| `--metric` | What to optimize for | top_1pct_rate | See "Selection Metrics" below |
| `--volatility-scale` | Widen or tighten all distributions | 1.0 | 1.0-1.2 for GPPs, 0.8 for cash |
| `--copula-nu` | Student-t degrees of freedom | 5 | 4-6 for GPPs (more tail dependence) |

---

## 6. Reading Your Results

The simulation outputs a CSV where each row is a candidate lineup. Here's what the columns mean:

### Score Distribution Columns

| Column | What It Means | What to Look For |
|---|---|---|
| `mean_score` | Average total FD points across all sims | Higher is better, but not the whole story for GPPs |
| `median_score` | The "typical" score (50th percentile) | More stable than mean |
| `std_score` | How much the score varies | Higher = more volatile = more GPP-friendly |
| `p90_score` | Score at the 90th percentile | "When this lineup has a good night, it scores this much" |
| `p99_score` | Score at the 99th percentile | **This is your GPP ceiling** — the score when everything breaks right |
| `max_score` | Absolute best simulation | The dream scenario |

### Contest Placement Columns

| Column | What It Means | What to Look For |
|---|---|---|
| `win_rate` | % of sims where this lineup is #1 overall | Even 0.1% is meaningful in a 150k-entry contest |
| `top_1pct_rate` | % of sims finishing in the top 1% | **The key GPP metric.** 3-5% is good. |
| `top_10pct_rate` | % of sims finishing in the top 10% | Shows consistent upside |
| `cash_rate` | % of sims where this lineup would cash (~top 20%) | 50%+ is solid |
| `expected_roi` | Expected return on investment | >1.0 means profitable on average |

### Ownership/Leverage Columns

| Column | What It Means | What to Look For |
|---|---|---|
| `total_ownership` | Sum of all 9 players' projected ownership % | Lower = more contrarian. Under 100% is very contrarian. |
| `leverage_score` | Sum of individual player leverage scores | Positive = lineup is projected better than it's owned |
| `field_duplication_rate` | % of field lineups that share 6+ players with this one | Lower = more unique. Under 5% is great. |

### What a great GPP lineup looks like in the results:

- `top_1pct_rate` of 4-6% (top 1% finish in 4-6 out of 100 simulations)
- `p99_score` above 200 (ceiling is high enough to win)
- `total_ownership` under 120% (differentiated from the field)
- `field_duplication_rate` under 5% (not many opponents have a similar lineup)
- `cash_rate` above 40% (still cashes reasonably often — not pure dart throw)

### What the portfolio summary tells you:

- **Portfolio win rate:** "If I enter all 20 lineups, at least one of them wins the whole contest in X% of simulations." Even 5-10% is extremely strong.
- **Portfolio top 1% rate:** "At least one of my 20 lineups finishes in the top 1% in X% of simulations." You want this above 30-40%.
- **Portfolio cash rate:** "At least one of my lineups cashes in X% of simulations." This should be 90%+.
- **Avg pairwise overlap:** Average number of shared players between any two of your 20 lineups. Under 4 is good diversity.
- **Unique players used:** Total distinct players across all 20 lineups. Higher = more diversified.

---

## 7. The Key Settings and What They Do

### Selection Metrics (what you're optimizing for)

| Metric | Use When | How It Plays |
|---|---|---|
| `top_1pct_rate` | **Large GPPs (150+ entries, 150k+ field)** | Maximizes chance of a top-1% finish. The default and usually the right choice. |
| `win_rate` | **Single-entry GPPs** | Maximizes the probability of winning outright. More aggressive, more volatile. |
| `cash_rate` | **Cash games (50/50s, double-ups)** | Maximizes the probability of cashing. Picks consistent, high-floor lineups. |
| `expected_roi` | **Balanced approach** | Optimizes dollar-weighted expected return. Balances ceiling and floor. |
| `p99_score` | **Ceiling chasing** | Picks the lineups with the absolute highest possible scores. Very aggressive. |

### Presets

| Preset | Sims | Volatility | Field | Metric | Diversity | Best For |
|---|---|---|---|---|---|---|
| **GPP** | 20,000 | 1.1 | 2,000 | top_1pct_rate | 0.4 | 20-entry tournaments |
| **Cash** | 10,000 | 0.8 | 500 | cash_rate | 0.1 | 50/50s, double-ups |
| **Single Entry** | 20,000 | 1.0 | 2,000 | win_rate | 0.0 | Single-entry GPPs |

### Correlation Settings (advanced)

| Setting | Default | Effect of Increasing | Effect of Decreasing |
|---|---|---|---|
| Teammate correlation | 0.25 | Stacks become more boom-bust (higher ceiling, lower floor) | Players behave more independently |
| Copula nu (ν) | 5 | Less tail dependence (closer to Gaussian) | More tail dependence (more blowup games) |
| Pitcher vs opposing | -0.15 | Stronger anti-correlation (pitcher dominance hurts opposing hitters more) | Weaker anti-correlation |
| Volatility scale | 1.0 | All distributions widen (more variance, more boom-bust) | Distributions tighten (more consistent) |

### When to adjust these:

- **Rainy day with uncertain weather?** Increase volatility_scale to 1.2-1.3 (more uncertainty)
- **Small slate (4-5 games)?** Increase teammate correlation to 0.30 (fewer games = more stacking importance)
- **Huge slate (15 games)?** Decrease diversity_weight to 0.2 (more lineup options available naturally)
- **Feeling aggressive?** Decrease copula_nu to 4, increase volatility_scale to 1.2
- **Playing it safer?** Increase copula_nu to 8, decrease volatility_scale to 0.9

---

## 8. Real Examples: What Good Output Looks Like

### Example: A strong GPP portfolio selection

```
Selected 20 lineups using metric=top_1pct_rate
Portfolio win rate: 8.4%, top1%: 38.2%, cash: 97.1%
Expected ROI (sum across entries): 12.3x
Unique players used: 47, Avg pairwise overlap: 3.2
```

**Reading this:** Across 10,000 simulated nights, at least one of your 20 lineups wins the whole contest in 8.4% of those scenarios. At least one finishes in the top 1% in 38.2% of scenarios. You cash at least one lineup 97.1% of the time. Your 20 lineups use 47 different players (high diversity) with an average of only 3.2 shared players between any two lineups.

### Example: Comparing a chalk lineup vs. a contrarian lineup

```
Lineup A (chalk):  mean=118, p99=165, top_1pct=1.2%, cash=71%, ownership=185%
Lineup B (contra): mean=108, p99=182, top_1pct=4.8%, cash=48%, ownership=72%
```

**Reading this:** Lineup A scores more on average (118 vs 108) and cashes more often (71% vs 48%). But Lineup B has a much higher 99th percentile ceiling (182 vs 165) and finishes in the top 1% four times as often (4.8% vs 1.2%). In a GPP, Lineup B is far more valuable despite scoring less on average. This is the contrarian thesis in action — Lineup B's low ownership means that when it goes off, few other entries share that score.

---

## 9. FAQ and Troubleshooting

### "How long does it take to run?"

With default settings (10k sims, 870 players):
- Distribution fitting: instant
- Correlation matrix: ~0.5 seconds
- Slate simulation: ~6 seconds
- Field simulation: ~2 seconds
- Contest simulation: ~5 seconds
- **Total: roughly 15-20 seconds**

At 20k sims it's about 30-40 seconds. At 50k sims, about 90 seconds.

### "How many simulations do I need?"

- **1,000**: Quick and dirty. Good for testing settings. Noisy estimates.
- **10,000**: The sweet spot. Reliable estimates, fast runtime.
- **20,000**: More precise. Worth it for your final selection on game day.
- **50,000+**: Diminishing returns for most use cases. Only if you're fine-tuning specific settings.

The estimates follow a square-root law: doubling precision requires 4x the simulations. Going from 10k to 20k improves precision by ~40%, not 100%.

### "Why did it only select 15 lineups when I asked for 20?"

The portfolio selector enforces diversity constraints (max overlap, max exposure). If the candidate pool doesn't have 20 sufficiently different lineups, it stops early rather than adding lineups that are near-duplicates of what's already selected. This is by design — 15 diverse lineups beat 20 overlapping ones.

To get more selections, increase `--num-candidates` (give it more options to pick from) or decrease `diversity_weight`.

### "The ownership and leverage columns are showing 0.0"

This happens when the optimizer dataset doesn't have `proj_fd_ownership` or `player_leverage_score` columns. These require ownership data to be loaded during the daily pipeline (via `--ownership-sources`). Without external ownership data, the system falls back to a heuristic that may not populate these columns in the lineup DataFrames. The simulation still works — it just can't report ownership-based metrics for individual lineups.

### "What's the difference between running the simulation and just using the optimizer?"

| | Optimizer Only | Optimizer + Simulation |
|---|---|---|
| **Objective** | Maximize projected points | Maximize tournament equity |
| **Player model** | Point estimate (mean) | Full distribution (mean + variance + skew) |
| **Correlation** | Hard constraint ("put 4 teammates together") | Probabilistic model (teammates have correlated outcomes) |
| **Field awareness** | None (doesn't know what others are playing) | Simulates 1,000+ opponent lineups |
| **Lineup selection** | Top N by projected score | Portfolio that maximizes simulated win rate across diverse scenarios |
| **Best for** | Cash games | GPP tournaments |

### "Can I still use the optimizer without simulation?"

Yes, absolutely. The simulation is optional. The existing pipeline (upload → project → optimize → export) works exactly as before. Simulation is an extra step you opt into.

### "I don't understand what 'tail dependence' means. Do I need to?"

Not really. The practical implication is: when you use this tool, the simulations will correctly model "team blowup games" where an entire batting order goes off simultaneously. A simpler model would underestimate how often this happens, which would undervalue stacking. The Student-t copula gets this right. You don't need to understand the math to benefit from it — just know that it makes the stacking analysis more realistic.

### "What should I do on a typical game day?"

1. Download FanDuel player list CSV
2. Get your BallparkPal data
3. Optionally: get Vegas lines, batting orders, ownership projections
4. Run the daily pipeline to build your optimizer dataset
5. Run the simulation with GPP preset settings
6. Review the portfolio summary — check that portfolio top1% rate is above 25-30%
7. Export to FanDuel
8. After games: upload actual results for backtesting (optional but valuable over time)

The whole process from data upload to FanDuel export should take under 10 minutes once you're familiar with it. The simulation itself runs in under 30 seconds.
