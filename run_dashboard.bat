@echo off
REM Launch the MLB DFS Lineup Optimizer Dashboard (Windows)
REM Usage: double-click this file or run from command prompt

cd /d "%~dp0"

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM Install dependencies if needed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Set Python path
set PYTHONPATH=%~dp0src;%PYTHONPATH%

echo.
echo ==========================================
echo   MLB DFS Lineup Optimizer Dashboard
echo ==========================================
echo.
echo Opening in your browser...
echo.

streamlit run dashboard/app.py --server.headless true --browser.gatherUsageStats false %*
