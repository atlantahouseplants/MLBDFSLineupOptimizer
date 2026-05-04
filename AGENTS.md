# MLB DFS Lineup Optimizer

Fully automated FanDuel MLB lineup generator. Uses BallparkPal 3,000-simulation projections through a PuLP ILP solver to build GPP tournament lineups optimized for contrarian leverage.

## Tech Stack
- Language: Python 3.11+
- Package manager: `uv` (NOT pip — PEP 668 on this machine)
- Key dependencies: pulp, pandas, numpy, streamlit, scipy
- Solver: PuLP ILP (integer linear programming)
- Data: BallparkPal projections, FanDuel salary CSVs, Vegas lines, MLB Stats API

## Project Structure
```
MLBDFSLineupOptimizer/
├── app.py                      # Top-level streamlit entry
├── dashboard/                  # Streamlit dashboard modules
│   ├── app.py                  # Main dashboard
│   └── daily_workflow.py       # Daily workflow orchestration
├── scripts/                    # Pipeline scripts
│   ├── run_daily_pipeline.py   # Daily automated pipeline
│   ├── run_full_pipeline.py    # Full end-to-end
│   ├── fetch_live_data.py      # Fetch BallparkPal/Vegas/lineups
│   ├── ingest_ballparkpal.py   # BallparkPal sim ingestion
│   ├── generate_projections.py # Projection generation
│   ├── build_slate_dataset.py  # Slate dataset builder
│   ├── prepare_optimizer_dataset.py # Optimizer input prep
│   ├── run_optimizer.py        # ILP solver runner
│   ├── compute_leverage.py     # Contrarian leverage scoring
│   └── export_fanduel_upload.py # FanDuel CSV export
├── data/
│   ├── live/                   # Today's fetched data
│   └── output/                 # Pipeline run outputs
├── tests/                      # Test suite
└── pyproject.toml              # Project config + deps
```

## Key Commands
- Install: `uv venv .venv && uv pip install --python .venv -e .`
- Dashboard: `streamlit run dashboard/app.py --server.port 8502`
- Full pipeline: `uv run python scripts/run_daily_pipeline.py`
- Simulated pipeline (no live data): `uv run python scripts/run_simulated_pipeline.py`
- Tests: `uv run python -m pytest tests/ -v`
- Validate lineup: `uv run python validate_lineup.py`

## Dashboard
- Port 8502 (8501 had stale connection issues)
- Main entry: `dashboard/app.py`
- Tabs: Today's Slate, Run Optimizer, Review Lineups

## Code Standards
- Type hints encouraged but not strictly enforced
- Scripts are mostly functional pipeline steps
- `.venv/bin/python` for running, NOT system `python3`
- Test files: `tests/test_*.py`, run with pytest

## Environment Notes
- `uv` for package management (PEP 668 — pip blocked globally)
- Dashboard runs on Windows host, scripts run in WSL
- `.streamlit/config.toml` for Streamlit config

## Known Gotchas
- **PEP 668** — use `uv` not `pip`. Create venv with `uv venv .venv`
- **Streamlit port** — use 8502, not 8501 (stale connection issues)
- **BallparkPal requires auth** — cookies/session needed for simulated pipeline
- **FanDuel CSV is manual step** — must download from FanDuel website, can't automate
- **`write_file` tool fails on wallg-owned dirs** — use `terminal cat >` heredoc instead
- **ILP solver can hang** on large slates — set realistic lineup count (20-50)
- **Pipeline scripts are long-running** — use background mode with notify_on_complete=true
