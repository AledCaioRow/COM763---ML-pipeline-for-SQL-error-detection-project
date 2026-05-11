@echo off
REM Fixed port 8765 — same as SQL_Query_Explorer.url (double-click that to open the app in a browser once this is running).
cd /d "%~dp0"
python -m streamlit run app.py --server.port 8765
