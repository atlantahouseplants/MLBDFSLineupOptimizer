# MLB DFS Optimizer — Full System Audit + 300-Lineup Dual-Contest Build

## Context

This is a custom MLB DFS lineup optimizer for FanDuel GPP tournaments — a Streamlit
dashboard backed by a Python pipeline (PuLP-based ILP solver, not `pydfs`). Rough
architecture, as of the last known-good state:

```
scripts/run_daily_pipeline.py         # orchestrates the daily run
src/slate_optimizer/
    ingestion/ballparkpal.py          # loads BallparkPal Batters/Pitchers/Games/Teams
    ingestion/fanduel.py              # loads FanDuel salary CSV
    ingestion/slate_builder.py        # matches players across sources (name + team)
    data/storage.py                   # persists to SQLite
    projection/baseline.py            # turns BPP sim data into fantasy projections
    optimizer/dataset.py              # builds optimizer-ready player pool
    optimizer/solver.py               # ILP solver — salary cap, positions, stacks,
                                       # exposure limits, uniqueness across lineups
    simulation/lineup_selector.py     # portfolio selection from generated lineups
    simulation/distributions.py
config/optimizer_settings.json        # stack size, exposure caps, leverage params
scripts/export_fanduel_upload.py      # formats final lineups for FanDuel upload
```

There have been several rounds of PRD-driven changes to this codebase: a main
**LEVERAGE_STRATEGY** PRD (ceiling-weighted GPP objective, ownership penalties,
smart auto-stacking), an **ADDENDUM** (contest-size × slate-size strategy matrix,
auto slate-profile detection, dashboard strategy summary banner), a **URGENT_FIX**
pass, and a **BPP_UPSIDE** PRD (wiring in Upside/Bust/Median data). These were
written as specs at different points — **do not assume they were fully or
correctly implemented.** Verify against the actual code.

## Your job, in two parts

### Part 1 — Full audit (do this first, before generating anything)

Read through the whole pipeline end to end — ingestion, matching, projections,
solver objective, stacking, exposure/uniqueness, simulation/portfolio selection,
the Streamlit UI, and the export step. I want an honest assessment of what's
solid, what's broken, and what's just not built optimally. Specifically check:

1. **BPP DFS Optimizer file ingestion.** There's a second BallparkPal export
   (separate from the standard Batters/Pitchers/Games/Teams files) with columns
   `Tm, Pos, Players, $, Points, Bust, Median, Upside, Pts/$, Med/$, Ups/$, PA, HR,
   3B, 2B, 1B, RBI, R, BB, SB, ..., PlatformName, SlateName, PlayerType,
   StackCount, GameDescription, FullName, FirstName, LastName` (header on row 2,
   not row 1 — watch for that when reading it). As of an April audit, this file's
   Upside/Bust/Median numbers were **not** being ingested anywhere — only the raw
   sim exports were used for projections. Confirm the current state. If it's still
   unused, wire it in properly: this is the actual ceiling/floor data a leverage
   strategy should be built on, not something to leave on the table.

2. **Upload-to-pipeline mapping.** I genuinely don't have a clear picture of which
   uploaded file feeds which step of "Process Slate" in the Streamlit UI, or
   whether I'm dropping files in the right slots. Produce a short, plain-language
   map: *file name/type → what it's used for → what breaks if it's missing or
   swapped with another file*. This should also live as a comment or help-text
   in the Streamlit upload step itself, not just in a doc I'll lose track of.

3. **Do the LEVERAGE_STRATEGY / ADDENDUM / URGENT_FIX / BPP_UPSIDE specs actually
   match the code?** Check specifically: slate-size auto-detection, contest-type
   presets (Single Entry / Small Field / Large Field), the ceiling-weighted GPP
   objective vs. the old cash objective, ownership-penalty scaling, within-stack
   chalk penalties, bring-back leverage bonus, and smart auto-stack template
   selection. Report gaps between spec and implementation as clearly as bugs.

4. **Previously flagged bugs** — confirm fixed or still present: reliever
   misclassification in projections, salary underutilization in the solver,
   and ownership-penalty scaling behaving oddly at the extremes.

5. **General code health** — error handling on bad/missing/swapped uploads,
   validation of the name+team player matching across BPP and FanDuel sources
   (mismatches silently dropping players would be bad), and anything that's
   hard-coded in a way that'll break on unusual slate sizes (2-game slates,
   15-game slates) or unusually large lineup counts (see Part 2 — 300 lineups
   in one run is more than this has likely been stress-tested at).

Fix what you find. Don't just hand back a list — implement corrections, and
call out anywhere you made a judgment call so I can weigh in.

### Part 2 — Build tonight's lineups

Goal: **300 total lineups — two independent sets of 150** — for two separate
GPP tournaments on the same slate, using the sample data provided (today's
6-game FanDuel main slate). Strategy is smart contrarian/leverage: target
under-owned high-ceiling stacks in good scoring environments, fade obvious
chalk where the ceiling doesn't justify the ownership, use bring-backs that
add differentiation rather than just correlation.

**Before finalizing, confirm one thing with me directly rather than assuming:**
since both contests share the exact same real-world slate outcome, should the
150/150 pools be built with *shared* exposure limits across both contests
combined (treating it as one 300-lineup portfolio split in two), or as two
*fully independent* 150-lineup builds each optimized on its own? This changes
how the exposure/uniqueness constraints in the solver should be configured —
ask me before you commit to one.

Deliverables:
- Two separate FanDuel-upload-ready CSVs (150 lineups each), clearly labeled
  by contest.
- A short summary of the stacking/leverage decisions — which games/teams were
  prioritized and why (Vegas totals, BPP upside, ownership signals), and which
  were faded — so I can sanity-check before entering.

## Sample data provided alongside this prompt

- `BallparkPal_Pitchers_...xlsx`, `BallparkPal_Batters_...xlsx`,
  `BallparkPal_Teams_...xlsx`, `BallparkPal_Games_...xlsx` — raw sim exports
- `Ballpark_DFS__Ballpark_Pal__45_.xlsx`, `_46_.xlsx` — the DFS Optimizer
  exports with Points/Bust/Median/Upside (see item 1 above)
- `FanDuel-MLB-...players-list.csv` — today's FanDuel salary file, 6-game main slate

## Acceptance criteria

- Audit findings written up plainly: what was broken, what was fixed, what's a
  spec-vs-code gap, what's a judgment call I need to weigh in on.
- Both 150-lineup CSVs generated and validated against FanDuel's salary cap and
  position rules.
- The upload-mapping clarification is visible in the Streamlit UI itself, not
  just in a separate doc.
