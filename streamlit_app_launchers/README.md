# SQL Query Explorer

Streamlit app to run **live SQL** against two BIRD Mini-Dev SQLite databases (**`formula_1`** and **`financial`**), see **measured wall-clock runtime**, and compare with an **ML predicted runtime** (per-database regression when artifacts are present).

---

## What you need

| Requirement | Notes |
|-------------|--------|
| **Python 3.10+** | Add Python to PATH on Windows. |
| **Dependencies** | From this folder: `pip install -r requirements.txt` (ideally in a virtual environment). |
| **Repo layout** | The app expects the **repository root** (parent of `standalone_sql_perf_pipeline`) to contain **`Mini Dev/MINIDEV/dev_databases/<db_id>/<db_id>.sqlite`**. Without those files, query execution fails. |
| **Regression models (optional)** | For full runtime prediction, train/export artifacts to `standalone_sql_perf_pipeline/artifacts/regression_by_db/` (see main pipeline README / `run_schema_stats_model.py`). If files are missing, the UI still runs but prediction may fall back or show errors—check the in-app messages. |

---

## How to run

### Option A — From this folder (recommended)

1. Open a terminal in **`sql_query_explorer`** (this directory).
2. Create and activate a venv (first time only):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Start Streamlit on the **fixed port 8765** (matches `SQL_Query_Explorer.url`):

   ```powershell
   python -m streamlit run app.py --server.port 8765
   ```

4. Open a browser at **http://localhost:8765/**  
   Or double-click **`SQL_Query_Explorer.url`** after the server has started.

### Option B — Double-click launcher (this folder)

- **`Launch_Streamlit.bat`** — same as above (`app.py`, port **8765**).

### Option C — From `standalone_sql_perf_pipeline`

- **`Streamlit App Run.bat`** (one level up) — changes into this folder and runs the same command.

---

## Using the app

1. **Sidebar** — Choose **`formula_1`** or **`financial`**.
2. **SQL box** — Paste or type SQLite SQL, then click **Run Query & Predict**.
3. **Results** — You get:
   - **Measured runtime** (from executing your query on the selected `.sqlite` file).
   - **Predicted runtime** (from the ML pipeline when models are available).
   - **Query result grid** (if the query returns rows).
4. **Example queries** — Right column has copy-paste examples per database.
5. **Second page** — In the sidebar, open **Model Performance** (`pages/2_Model_Performance.py`) for history and metrics from past runs (stored locally in **`query_history.db`** in this folder).

---

## Sharing with someone else (e.g. a supervisor)

They need a **copy of the repo** (or zip) with **`Mini Dev`** databases and your **`requirements.txt`** install—not just Streamlit installed globally.

- They run **Option A** or **B** on **their own machine**.
- **`localhost:8765`** always means “the computer where Streamlit is running”; it does not connect to your laptop automatically.

For a **public URL** without local setup, deploy to **Streamlit Community Cloud** (or similar) and share that link.

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| `python` not found | Reinstall Python and enable **Add to PATH**, or use `py -m streamlit ...`. |
| Database file not found | Confirm paths: `…/Mini Dev/MINIDEV/dev_databases/formula_1/formula_1.sqlite` (and same for `financial`) relative to **repo root**. |
| Port already in use | Stop other Streamlit servers, or run with another port, e.g. `python -m streamlit run app.py --server.port 8766` and open that port in the browser (update a local bookmark if you change it). |
| Prediction errors | Ensure `artifacts/regression_by_db/` contains the expected per-database models; re-run training/export from the parent pipeline if needed. |

---

## Files in this folder (quick reference)

| File / folder | Role |
|---------------|------|
| `app.py` | Main Streamlit page (query runner). |
| `pages/2_Model_Performance.py` | History and performance charts. |
| `utils/db_runner.py` | SQLite execution and timing. |
| `utils/predictor.py` | Loads regression artifacts and predicts runtime. |
| `utils/history_db.py` | Local SQLite DB for submitted queries. |
| `requirements.txt` | Python dependencies. |
| `Launch_Streamlit.bat` | Windows one-click start (port 8765). |
| `SQL_Query_Explorer.url` | Opens **http://localhost:8765/** in the default browser (Windows). |
