@echo off
title Streamlit App Run
REM Double-click this from standalone_sql_perf_pipeline: SQL Query Explorer on http://localhost:8765
cd /d "%~dp0sql_query_explorer"
if not exist "app.py" (
    echo ERROR: app.py not found. Expected: sql_query_explorer\app.py next to this file.
    pause
    exit /b 1
)
python -m streamlit run app.py --server.port 8765
