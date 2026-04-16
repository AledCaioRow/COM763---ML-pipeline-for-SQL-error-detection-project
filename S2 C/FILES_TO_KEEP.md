# Files to keep (evidence bundle / report reproduction)

Flat checklist of paths relative to project root  
`COM763---ML-pipeline-for-SQL-error-detection-project/`.

---

## A. Entry scripts (what you run)

- `main.py`
- `setup_bird.py`
- `rerun_commit_comparison.py`
- `run_schema_stats_model.py`
- `run_full_regression.py`
- `run_phase6_figures.py`
- `run_report_graphs.py`

---

## B. Config and dependencies under `main.py`

- `config.py`
- `requirements.txt`
- `src/data/load_bird.py`
- `src/features/extract_features.py`
- `src/models/train.py`
- `src/models/predict.py`
- `src/evaluation/evaluate.py`

---

## C. BIRD inputs (needed to rebuild from scratch)

- `Mini Dev/MINIDEV/` (or `data/bird/mini_dev_data/` if that is your active layout per `config.py`)
  - SQLite JSON (e.g. `mini_dev_sqlite.json` or sharded `data/mini_dev_sqlite-*.json`)
  - `dev_databases/<db_id>/<db_id>.sqlite` trees used at runtime

---

## D. Generated data and model (Iteration 1 + downstream scripts)

- `data/query_dataset_raw.csv`
- `data/query_dataset_features.csv`
- `artifacts/best_model.joblib`

---

## E. Reports and metrics CSVs (copy into evidence bundle as needed)

**From `python main.py` (Iteration 1):**

- `reports/split_summary.csv`
- `reports/all_models_test_comparison.csv`
- `reports/model_results.txt`
- `reports/per_database_results.csv`
- `reports/confusion_matrix.csv`
- `reports/classification_report.csv`
- `reports/cv_fold_scores.csv` (optional)
- `reports/class_distribution.csv` (if produced in your run)

**From `python rerun_commit_comparison.py` (Iteration 2 / seen metrics):**

- `reports/commit_rerun_metrics.csv`

**From `python run_schema_stats_model.py` (Iteration 4 + fig inputs):**

- `reports/within_db_schema_metrics.csv`
- `reports/schema_stats.csv` (if present for your run)

**From `python run_phase6_figures.py`:**

- `reports/narrative_figures/fig13_within_db_r2.png`
- `reports/narrative_figures/fig14_schema_vs_r2.png`

**From `python run_report_graphs.py` (EDA / narrative figures):**

- `reports/narrative_figures/` (other `fig*.png` your report cites)

**Optional / bridge:**

- `markdowns/FULL_PROJECT_REPORT.md` (only if you treat it as cached narrative to cross-check `run_full_regression.py` stdout)
- `reports/within_db_logistic_metrics.csv` (only if you add Iteration 3 grid per plan)

**Cross-schema regression:**

- No default CSV; keep a **saved terminal log** if you rely on `python run_full_regression.py` stdout.

---

## F. Standalone analysis scripts (no extra local package tree)

These are single-file drivers; keep the `.py` file itself:

- `run_schema_stats_model.py`
- `run_full_regression.py`
- `run_phase6_figures.py`
- `run_report_graphs.py`
- `rerun_commit_comparison.py`

---

## G. Not required for the evidence table alone (optional product)

- `streamlit_app/` (UI only; not needed to regenerate the CSVs above)
