@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0src;%PYTHONPATH%
"%~dp0.venv\Scripts\python.exe" -m streamlit run dashboard/daily_workflow.py --server.headless true --server.port 8600 --server.fileWatcherType none --server.runOnSave false --browser.gatherUsageStats false >> "%~dp0dashboard_server.log" 2>&1
