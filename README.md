# MLB DFS Lineup Optimizer

Fully automated FanDuel MLB lineup generator. Uses BallparkPal 3,000-simulation projections through a PuLP ILP solver to build GPP tournament lineups optimized for contrarian leverage.

---

## Daily Workflow (2 minutes)

### 1 — Get your FanDuel salary CSV
Go to fanduel.com → Lobby → today's MLB slate → Download player list → save it

### 2 — Launch the dashboard
```
Windows: double-click run_dashboard.bat
Mac/Linux: ./run_dashboard.sh
```
Opens at http://localhost:8501

### 3 — Refresh live data
On **Today's Slate** tab → click **🔄 Refresh Live Data**
Pulls BallparkPal sims, Vegas lines, and batting orders automatically.

### 4 — Run the optimizer
**Run Optimizer** tab → Upload FD salary CSV → Set lineup count → **🚀 Run Optimizer**

### 5 — Review and submit
**Review Lineups** tab → Check exposure + stacks → **⬇️ Download FanDuel Upload CSV** → upload to FanDuel

---

## One-Time Setup

```bash
git clone https://github.com/atlantahouseplants/MLBDFSLineupOptimizer.git
cd MLBDFSLineupOptimizer
pip install -r requirements.txt
```

Create `.env` in the repo root:
```
BPP_SESSION=<your BallparkPal PHPSESSID cookie>
ODDS_API_KEY=6d20401fd47d415664f3d50f1b0a0849
```

### BPP Session Cookie — refresh every ~7 days
1. Log in at ballparkpal.com in your browser
2. F12 → Application tab → Cookies → ballparkpal.com → copy **PHPSESSID** value
3. Update `.env`: `BPP_SESSION=<new value>`

---

## What Gets Fetched Automatically

| Source | Data | Auth |
|--------|------|------|
| BallparkPal | Simulations, DFS projections, batting orders, handedness, park factors | Session cookie (~7 day) |
| The Odds API | Vegas totals, moneylines, implied team totals | API key (free, 500/month) |
| MLB Stats API | Confirmed lineups (2-3hrs before game), probable pitchers + handedness | Free, no key needed |

---

## Architecture

```
dashboard/app.py               # Streamlit UI — 3 tabs: Slate / Run / Review
scripts/fetch_live_data.py     # Fetches all live data → data/live/
scripts/run_daily_pipeline.py  # Full optimizer (called by dashboard)

src/slate_optimizer/
  ingestion/
    bpp_api.py          # BallparkPal Export Center auto-fetch
    odds_api.py         # The Odds API Vegas lines
    mlb_api.py          # MLB Stats API batting orders + pitchers
  projection/
    baseline.py         # BPP projections → FD point estimates
    ownership_model.py  # Ownership from BPP signals (bust%, upside, etc.)
  optimizer/
    solver.py           # PuLP ILP solver with FanDuel constraints
  simulation/           # Contest simulation for GPP portfolio scoring
```

## FanDuel MLB Lineup Format
- 9 players: 1P + 1C/1B + 1 2B + 1 3B + 1 SS + 3 OF + 1 UTIL
- $35,000 salary cap | Max 4 hitters from same team
- Stack styles: 4-3 (default), 4-4, 5-3, 3-3-2

## Ownership Model Signals
Batters: value score (pts/$), Vegas implied total, bust%, upside ratio, HR/hit probability, batting order slot
Pitchers: FD projection, win%, strikeouts, quality start %

After each run, predicted ownership saves to `data/output/TAG_ownership_predicted.csv`.
Compare vs real contest ownership (downloadable from FanDuel) to calibrate the model.
