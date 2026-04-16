# Standalone SQL query timing & ML pipeline

This codebase:

1. **Loads BIRD Mini-Dev SQL questions** and resolves them against SQLite databases
2. **Executes each query repeatedly** (with timeout) and records **wall-clock runtime**
3. **Parses SQL text** into structural features and builds **fast/slow labels** from runtime quantiles
4. **Trains and evaluates classifiers** (iteration 1: hold-out databases) and **additional experiment modes** (iterations 2–7)

---

## Directory layout (this package only)


| Path                                      | Purpose                                                                                                               |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `data/`                                   | Copied pipeline CSVs (`query_dataset_raw.csv`, `query_dataset_features.csv`) + `data/README.md`.                      |
| `main.py`                                 | Single entry: `python main.py` or `python main.py -n <1–7>` (see iterations).                                         |
| `config.py`                               | Paths, `HOLDOUT_DATABASES`, label method, feature list, seeds.                                                        |
| `setup_bird.py`                           | Sanity-check JSON + SQLite DB files before a long run.                                                                |
| `src/data/load_bird.py`                   | Reads BIRD JSON, runs queries on SQLite, builds raw timing table.                                                     |
| `src/features/extract_features.py`        | `sqlparse` structural features; quantile/median **labels**.                                                           |
| `src/models/train.py`                     | Database-aware or random split; model zoo + CV selection.                                                             |
| `src/models/predict.py`                   | Load `artifacts/best_model.joblib` and score new rows (optional).                                                     |
| `src/evaluation/evaluate.py`              | Test metrics, plots, CSV/text reports under `reports/`.                                                               |
| `rerun_commit_comparison.py`              | Iteration 2: git worktrees + seen/unseen classifier geometries; tree ablations (needs git + `sql_runtime_predictor`). |
| `sql_runtime_predictor/`                  | Minimal subtree: **plan-tree** feature extraction for iteration-2 extras inside `rerun_commit_comparison.py`.         |
| `export_classifier_geometries_summary.py` | One CSV row: unseen vs **seen** metrics on **current** tree only.                                                     |
| `run_within_db_logistic.py`               | Iteration 3: per-DB 80/20, SQL features only, LogisticRegression.                                                     |
| `run_schema_stats_model.py`               | Iteration 4: per-DB regression with SQL + **schema stats** from SQLite.                                               |
| `run_full_regression.py`                  | Cross-DB regression bridge (holdout DBs); prints metrics to stdout.                                                   |
| `run_phase6_figures.py`                   | Figures 13–14 from `within_db_schema_metrics.csv`.                                                                    |
| `run_report_graphs.py`                    | Broader narrative EDA figures under `reports/narrative_figures/`.                                                     |
| `build_report_evidence_bundle.py`         | Collates key metrics into `reports/report_evidence_bundle.md` + `report_metrics_long.csv`.                            |


---

## What produces the queries and measures runtime?

**Queries** come from the BIRD Mini-Dev **SQLite JSON** (paths resolved in `config.py`: `BIRD_SQLITE_JSON`, `BIRD_DATA_DIR`, `BIRD_DB_DIR`).

**Runtime measurement** is implemented in `src/data/load_bird.py`: for each question it connects to the right `<db_id>.sqlite`, runs the SQL **multiple times** (`TIMING_RUNS` in `config.py`), and stores median (or aggregated) seconds per query. Failed/timed-out runs are filtered; results are written to `data/query_dataset_raw.csv` when you run **iteration 1**.

**Features and labels** are added in `src/features/extract_features.py` (`add_parsed_features`, `add_labels`) using `LABEL_METHOD` in `config.py` (e.g. quantile split: fast / slow / drop middle).

---

**Direct runs** (same as subprocessed by `main.py`):

```bash
python run_within_db_logistic.py
python export_classifier_geometries_summary.py   # current tree: one wide row unseen vs seen
python build_report_evidence_bundle.py
```

---

## Data you must provide

1. **Bundled CSV snapshot** — `data/query_dataset_raw.csv` and `data/query_dataset_features.csv` are **copies** of the parent project outputs so you can run downstream scripts immediately. Replace them by running `python main.py -n 1` after refreshing BIRD timing. See `data/README.md`.
2. **BIRD Mini-Dev** under one of the layouts `config.py` searches (typical clone):
  - `Mini Dev/MINIDEV/` with `data/mini_dev_sqlite-*.json` (or `mini_dev_sqlite.json`)  
  - `Mini Dev/MINIDEV/dev_databases/<db_id>/<db_id>.sqlite`
   Or mirror the same structure under `data/bird/mini_dev_data/` (see `_BIRD_DIR_CANDIDATES` in `config.py`).
3. Create empty dirs on first run (or let the pipeline create them):
  - `data/`, `reports/`, `artifacts/`

---

## Dependencies

```bash
pip install -r requirements.txt
```

`PyYAML` is required for `sql_runtime_predictor/src/extract_features.py` when iteration 2 pulls plan-tree features.

---

## Turning this into its own repository

1. Copy **only** this `standalone_sql_perf_pipeline/` folder to a new repo root (or rename it to the repo root).
2. Add your BIRD data (or document where to download it).
3. Run `python setup_bird.py`, then `python main.py -n 1`.
4. Do **not** rely on any path outside this folder; scripts here use `Path(__file__).resolve().parent` for project root (replacing the old machine-specific absolute paths from the course repo).

**Optional:** Remove `rerun_commit_comparison.py` and `sql_runtime_predictor/` if you only need iteration 1 and analysis scripts on a single dataset snapshot (iteration 2 will not run).

---

## SQL Query Explorer — run order for deploy

The interactive Streamlit app lives in `sql_query_explorer/`.

> **Note on iteration naming:** `run_schema_stats_model.py` is called
> "Iteration 4" in `main.py` but corresponds to "Iteration 3" in the
> project report (per-database regression with schema statistics).

### One-time setup — generate the regression artifacts

```bash
# 1. (Optional) Re-time only formula_1 and financial for a fresher training set
python retime_subset.py                          # writes data/*.csv

# 2. Train per-database regression models + dump joblib artifacts
python run_schema_stats_model.py
#    → artifacts/regression_by_db/<db_id>.joblib  (one per database)
#    → artifacts/regression_by_db/manifest.json
#    → reports/within_db_schema_metrics.csv
```

### Launch the app

```bash
cd sql_query_explorer
pip install -r requirements.txt
streamlit run app.py
```

`requirements.txt` pins `scikit-learn==1.8.0` to match the version used when
the joblib artifacts were created. If you retrain with a different version,
update the pin accordingly.

### Shared infrastructure


| Module                | Purpose                                                                                                                                                                                                |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/schema_stats.py` | Single `schema_stats(db_id)` function used by both `run_schema_stats_model.py` (training) and `sql_query_explorer/utils/predictor.py` (inference). Both resolve SQLite paths via `config.BIRD_DB_DIR`. |
| `config.py`           | Now searches the repo root (`../Mini Dev/MINIDEV/`) as a candidate for `BIRD_DATA_DIR` in addition to the standalone pipeline directory.                                                               |


---

## Outputs (typical)

- `data/query_dataset_raw.csv` — SQL, `db_id`, runtimes  
- `data/query_dataset_features.csv` — + structural features + `label_binary`  
- `artifacts/best_model.joblib` — best CV classifier (iteration 1)  
- `reports/*.csv`, `reports/model_results.txt`, plots — evaluation artefacts  
- Further CSVs from iterations 3–4 and `reports/narrative_figures/*.png` from 6–7

This README plus the copied `.py` files are intended to be sufficient to understand and operate the pipeline **without** the parent project.