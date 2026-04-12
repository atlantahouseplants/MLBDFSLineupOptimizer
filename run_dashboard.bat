@echo off
REM MLB DFS Lineup Optimizer Dashboard — Launch Script
REM Double-click this file to start. Opens at http://localhost:8501

cd /d "%~dp0"

REM ── Create .env with API keys if it doesn't exist ──────────────────────────
if not exist ".env" (
    echo Creating .env with API keys...
    (
        echo BPP_SESSION=p3g7jubi4i49s30qkpcfr0hbtq
        echo ODDS_API_KEY=6d20401fd47d415664f3d50f1b0a0849
        echo BPP_EMAIL=sarah@atlantahouseplant.com
    ) > .env
    echo .env created.
)

REM ── Check Python ────────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM ── Install dependencies if needed ─────────────────────────────────────────
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies ^(first run only^)...
    pip install -r requirements.txt -q
)

REM ── Set Python path ─────────────────────────────────────────────────────────
set PYTHONPATH=%~dp0src;%PYTHONPATH%

echo.
echo ==========================================
echo   MLB DFS Optimizer Dashboard
echo ==========================================
echo   Opening at: http://localhost:8501
echo   Press Ctrl+C to stop
echo ==========================================
echo.

python -m streamlit run dashboard/app.py ^
    --server.headless false ^
    --browser.gatherUsageStats false ^
    --server.port 8501
