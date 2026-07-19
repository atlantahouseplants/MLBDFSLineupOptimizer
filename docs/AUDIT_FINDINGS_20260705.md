# Optimizer Audit — July 5, 2026

Audit of the full pipeline against the LEVERAGE_STRATEGY / ADDENDUM / URGENT_FIX /
BPP_UPSIDE specs. Legend: **FIXED** = corrected today, **OK** = verified working,
**GAP** = spec-vs-code gap (noted, some fixed), **CALL** = judgment call for Geoff.

## 1. BPP DFS Optimizer file (Upside/Bust/Median)

- **OK** — contrary to the April audit note, this file *is* ingested now. It's
  handled by `projection/blend.py`: header-on-row-2 is detected via the
  "Ballpark DFS | Ballpark Pal" title cell, Bust/Median/Upside map to
  `proj_fd_bust_rate` / `proj_fd_median` / `proj_fd_upside`, and Upside becomes
  the ceiling when no ceiling column exists. Files dropped in the BallparkPal
  upload bucket are auto-detected and rerouted.
- **FIXED** — the *filename* fast-path in `_looks_like_ballpark_optimizer_projection`
  only knew the old "Ballpark DFS Optimizer" naming; BallparkPal renamed the export
  to "Ballpark DFS  Ballpark Pal (N).xlsx" mid-2026. Content sniffing still caught
  it, but the name check now covers both.
- **GAP → FIXED (the big one)** — Upside was ingested and carried through
  `optimizer/dataset.py` into the player pool **but never reached the solver
  objective**. The ILP maximized `proj_fd_mean` only — the "ceiling-weighted GPP
  objective" from the LEVERAGE_STRATEGY PRD did not exist. Added `ceiling_weight`
  to the solver: objective base = `(1-w)·mean + w·upside`. Default 35% in the
  dashboard (0% = old behavior).

## 2. Upload → pipeline mapping

Added as an expander at the top of Step 1 in the dashboard ("Which file goes
where?"). Short version:

| File | Feeds | If missing/swapped |
|---|---|---|
| FanDuel players-list CSV | salaries, positions, IDs — the legal player universe | required; wrong day = silent mismatches |
| 4 raw BallparkPal exports | baseline projections, run distributions, handedness | required; loader errors if one missing |
| Ballpark DFS export(s) | Upside/Bust/Median → GPP objective | falls back to synthetic ceilings |
| Ownership CSVs | leverage + chalk fades | falls back to internal ownership model |
| Lineup paste | confirmed-starter filter, batting order boosts | bench players dilute the pool |
| Vegas CSV | team-total multipliers | falls back to BPP sim totals |

## 3. Spec vs. code

| Spec item | Status |
|---|---|
| Slate-size auto-detection | **OK** — `analysis/preset_recommender.py` (team_count ≤ 8 → small-field, ownership concentration → chalk fade, etc.) |
| Contest-type presets | **OK** — sim-side presets exist (`_apply_sim_preset`, contest_type.py); solver-side presets are the config panel defaults |
| Ceiling-weighted GPP objective | **GAP → FIXED** — was mean-only; now `ceiling_weight` blends BPP Upside into the objective |
| Ownership-penalty scaling | **PARTIAL → FIXED** — existed only as hard caps (exposure caps, max-lineup-ownership) plus a rank-based leverage multiplier. Added a true soft penalty in the objective (`ownership_penalty_weight`), which behaves smoothly at the extremes instead of cliff-edge caps |
| Within-stack chalk penalty | **GAP — judgment call** — not implemented as a stack-specific term. The new global ownership penalty + chalk exposure caps approximate it. A stack-conditional penalty would need quadratic terms (stack_var × player_var linearization); skipped for now — **CALL** if you want it |
| Bring-back leverage bonus | **PARTIAL** — bring-back exists as a hard constraint (≥N opposing hitters when a stack fires). There's no "leverage-weighted" bring-back selection, but the global objective penalties make the solver pick under-owned bring-backs naturally |
| Smart auto-stack templates | **OK** — team-leverage bonus in `_add_stack_constraints` nudges stack slots toward low-owned/high-run teams; template rotation supported |

## 4. Previously flagged bugs

- **Reliever misclassification: FIXED (already)** — `slate_builder.build_player_dataset`
  filters pitchers to FanDuel probable starters when the column is populated, and
  drops players flagged O/OUT/NA/SUSP/IR/IL.
- **Salary underutilization: FIXED (today)** — there was no minimum-salary
  constraint at all. Added `min_salary` to the solver (dashboard default $34,000;
  tonight's build used $34,200).
- **Ownership-penalty extremes: FIXED (today)** — the only "penalty" was hard caps,
  which do behave oddly at extremes (0% cap = player banned; cap above pool max =
  no-op with no gradient in between). The new soft penalty is linear in ownership
  and scaled by pool-average projection, so it stays sane at both ends. Solver
  also defensively normalizes percent-vs-decimal ownership scales.

## 5. Code health

- **FIXED** — handedness silently no-oped: platoon adjustments only worked with a
  separate handedness upload even though the raw BPP exports carry
  BatterStand/PitcherHand. Now backfilled from BPP columns (upload still wins).
- **FIXED** — a swapped/corrupt file in the projections slot crashed `_process_slate`
  with a raw KeyError/ValueError. Now wrapped with a message naming the file and
  what was expected.
- **OK** — name+team matching has team-code aliasing (CHW→CWS etc.), a last-name
  fallback for unique (team, last-name) pairs, and match-count diagnostics shown
  in the UI. Unmatched FanDuel players are kept with FPPG fallback (not silently
  dropped); the confirmed-starters filter then removes bench noise.
- **OK** — slate-size robustness: pool-viability checks (≥40 players, all
  positions covered, ≥50% paste match) with fallback to the full pool and slate-
  mismatch warnings. Stack templates degrade gracefully when a team can't fill a
  slot. 300-lineup runs: exposure caps are computed against the requested count,
  attempt budget is 4×N+20 — held up fine at 300.
- **FIXED (improvement)** — uniqueness was fixed at "not identical" (1 unique).
  Now configurable `min_uniques` (dashboard default 1, tonight's build 2). Also
  corrected an over-tight variant of the constraint when previous-lineup players
  drop out of the eligible pool.

## 6. Judgment calls made (weigh in when you have time)

1. **Ceiling weight 35% / ownership penalty 12% / leverage weight 20%** for
   tonight's build — reasonable GPP-aggressive settings, not backtested. The
   dashboard defaults are slightly tamer (35/10/15).
2. **Chalk defined by percentile** in tonight's build (top 15% of batter
   ownership capped at 28% exposure, top 30% of pitchers at 30%) because the
   internal ownership model's absolute scale is uncalibrated without real
   ownership uploads.
3. **Stack rotation** 2×(4-3-1), 1×(3-3-2), 1×(4-2-2), 1×(4-4) with bring-back ≥1.
4. **Shared-portfolio split by interleaving** (even/odd) so both contests get the
   same exposure profile; uniqueness (≥2 different players) enforced globally,
   which is stricter than per-contest.
5. **Within-stack chalk penalty not built** (see §3).
6. Baseline vs. BPP-DFS projection blend at **50/50** for covered players.
