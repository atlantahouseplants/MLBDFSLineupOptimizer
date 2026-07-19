# Daily Build — the short version (updated 2026-07-05)

## The 4-click daily flow

1. **Download today's files** (anywhere in Downloads, no renaming needed):
   - BallparkPal: Batters, Pitchers, Games, Teams exports + the "Ballpark DFS" export(s)
   - FanDuel: **enter your contests first, then download the entries-upload-template**
     from "Enter lineups via CSV" (fallback: the plain players-list works, but you'll
     be back to pasting into the template by hand)
2. Open the dashboard → Step 1 shows **⚡ Quick load from Downloads** with the
   detected files and their age. Paste the BallparkPal lineups block into the box.
3. Click **⚡ Process detected files**. The app now auto-detects the slate size,
   picks a contest preset, and applies the slate-size overlay — the blue strategy
   banner in Step 3 shows exactly what it chose and why. Adjust anything or just
   click **Run Optimizer**.
4. Step 5 → **Download FanDuel Upload CSV**. If you uploaded the entries template,
   this file contains your entry IDs — upload it straight back to FanDuel. Done.

## What got automated (and where to override it)

| Decision | Automated by | Override |
|---|---|---|
| Which files to use | Quick-load scans Downloads for the newest complete set, warns if stale (>18 h) | Classic uploaders below the panel |
| Slate size handling | 2–4 games = small, 5–9 = medium, 10+ = large profile (chalk caps, randomness, ceiling weight, uniques, bring-backs, stack templates) | Any field in Step 3 after the banner |
| Contest strategy | `recommend_slate_preset` picks from the 5 contest presets using ownership concentration + stack shape | Preset dropdown in Step 3 |
| GPP objective | ceiling weight / ownership penalty / min salary / min uniques set by preset + overlay | Step 3 sliders |
| Entry assignment | Entries template → direct-upload CSV with entry IDs, shuffled across buy-ins | Shuffle toggle in Step 5 |

## Slate-size profiles (the "why")

- **Small (2–4 games):** you cannot dodge chalk when there are only 4 stacks worth
  playing — chalk caps loosen (50%/60%), randomness and ceiling weight go UP for
  separation, bring-backs on, uniqueness stays at 1 (the pool is too small for more).
- **Medium (5–9 games):** the standard main-slate calibration — 30%/50% chalk caps,
  2-unique separation, bring-backs on, 4-3-1 / 4-2-2 / 3-3-2 rotation.
- **Large (10+ games):** the field spreads thin, so obvious chalk gets extra-owned —
  chalk caps tighten (25%/45%), stacks filtered to games with 8.5+ Vegas totals,
  forced bring-backs off (correlation is cheaper to find), leverage up.

Numbers live in `SLATE_SIZE_PROFILES` in `dashboard/daily_workflow.py` — edit there
to retune.

## Dual-contest nights (two GPPs, same slate)

Use the CLI portfolio builder — one shared-exposure portfolio split into two upload
files (this is the same engine the dashboard uses, plus global uniqueness):

```
python scripts/build_dual_contest.py \
  --fanduel <players-list or template csv> --bpp-dir <folder with the 4 exports> \
  --dfs-file <Ballpark DFS batters xlsx> --dfs-file <Ballpark DFS pitchers xlsx> \
  --paste <lineup paste txt> --total 300
```
