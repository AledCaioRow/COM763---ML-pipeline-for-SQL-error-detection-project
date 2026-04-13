# SQPP Streamlit dashboard

Read-only UI for the **repository root legacy classifier** pipeline: explore timed BIRD queries, inspect training/evaluation reports, and run **fast vs slow** predictions using `artifacts/best_model.joblib`.

This app does **not** load or visualize `sql_runtime_predictor/` (PyTorch plan-tree runtime regression). For that workflow, see `sql_runtime_predictor/README.md`.

## What the app does

| Area | Purpose |
|------|---------|
| **Data** | Reads CSVs produced by `main.py` (`query_dataset_raw`, `query_dataset_features`). |
| **Reports** | Parses `reports/model_results.txt` and shows per-database / per-difficulty tables. |
| **Model** | Loads sklearn `best_model.joblib` for metrics charts, importances, and **Predict**. |
| **Config** | Imports root `config.py` for `FEATURE_COLS`, split mode, timing, and label settings (shown in the sidebar). |

On startup, the app resolves the **project root** (parent of `streamlit_app/`) unless `SQPP_PROJECT_ROOT` points elsewhere, then loads files **once** at import time. Missing paths are listed in the sidebar.

## Prerequisites

From the **course project root** (parent of `streamlit_app/`):

```bash
python setup_bird.py
python -u main.py
```

That produces `data/`, `reports/`, and ideally `artifacts/best_model.joblib` next to `streamlit_app/`.

## Install and run

```bash
cd streamlit_app
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the URL Streamlit prints (usually [http://localhost:8501](http://localhost:8501)).

## Point at another project directory

```powershell
$env:SQPP_PROJECT_ROOT = "c:\path\to\Advanced ML Course"
python -m streamlit run app.py
```

Resolution order: valid `SQPP_PROJECT_ROOT` directory → else parent of `streamlit_app/`.

## Files the app expects

| Key | Path (under project root) |
|-----|---------------------------|
| Raw timed queries | `data/query_dataset_raw.csv` |
| Features + labels | `data/query_dataset_features.csv` |
| Text report | `reports/model_results.txt` |
| Breakdown CSVs | `reports/per_database_results.csv`, `reports/per_difficulty_results.csv` |
| Saved model | `artifacts/best_model.joblib` |
| Pipeline settings | `config.py` |

If `config.py` is missing or `FEATURE_COLS` is empty, feature names are inferred from the features CSV (columns other than metadata: `question_id`, `db_id`, `sql`, `difficulty`, `runtime_s`, `label`, `label_binary`).

## Layout (code)

- **`app.py`** — Page router, dataset loads, `FEATURE_COLS` resolution, predict helpers.
- **`components/sidebar.py`** — Project root, pipeline settings from `config.py`, missing-file list, page radio.
- **`components/charts.py`** — Plotly figures (histograms, bars, heatmaps, etc.).
- **`components/metrics.py`** — Small layout helpers for metrics.
- **`utils/paths.py`** — `get_project_root()`, optional `SQPP_PROJECT_ROOT`.
- **`utils/data_loader.py`** — Safe CSV reads, regex parse of `model_results.txt` (CV F1 lines, test metrics, class metrics, optional top-features block).
- **`utils/model_loader.py`** — Load joblib model; feature importances when the estimator exposes them.
- **`utils/predictor.py`** — `predict_from_features`: builds one-row `DataFrame`, `predict` / optional `predict_proba` → label and **P(slow)**.

## Pages (current behavior)

### 1. Overview

- Counts: raw rows, labeled rows, distinct `db_id`, number of model features.
- Numbered **pipeline** steps (load BIRD → time on SQLite → raw CSV → features → labels → train → reports).
- **Latest model summary** from parsed `model_results.txt` (best model name, test F1, ROC-AUC, accuracy).
- Bar charts: difficulty and database counts from the raw CSV (if present).

### 2. Data Explorer

Requires both raw and feature CSVs.

- **Runtime** — Histogram of `runtime_s` (optional log scale) with p50/p75 reference; box plots by `difficulty` and `db_id`.
- **Distributions** — Pie chart of `label` from the features table.
- **Features** — Multiselect correlation heatmap (numeric feature columns); scatter of `runtime_s` vs one feature, colored by `label` / `difficulty` / `db_id`.

### 3. Model Results

- CV model F1 bar chart (from parsed report).
- Per-class precision/recall/F1 bars when available.
- Test F1 by difficulty (from `per_difficulty_results.csv` when `f1` column exists).
- Feature importance: from the loaded sklearn model if supported; else top features from the report text if parsed.
- Side-by-side dataframes: per-database and per-difficulty CSVs.
- Expander with full raw `model_results.txt` text.

### 4. Predict

Requires `best_model.joblib` and a non-empty `FEATURE_COLS`.

- **Manual features** — Form: binary `has_*` checkboxes; numeric features as sliders (range from dataset min/max when available, else a safe default). Submit runs `predict_from_features`; shows label, **P(slow)** if available, and an optional contribution-style bar chart (feature values × importances).
- **From SQL** — Text area; **Extract features & predict** temporarily adds project root to `sys.path` and calls `src.features.extract_features.extract_features(sql)`. Missing or failed import shows a warning (ensure root `src/` package matches the trained feature schema).

## Troubleshooting

- **Sidebar lists missing files** — Run the root pipeline from project root; confirm paths are under the same root Streamlit uses (`SQPP_PROJECT_ROOT` if set).
- **Predict → SQL fails** — Root must contain `src/features/extract_features.py` and the SQL must be parseable by that module; feature names must align with `FEATURE_COLS` / training.
- **Charts empty on Model Results** — Regenerate `reports/model_results.txt` with `main.py`; encoding quirks in the report are tolerated via UTF-8 replace when reading.

## Scope note

The dashboard is intentionally scoped to the **binary fast/slow classifier** and its artifacts. Continuous runtime prediction (q-error, Spearman, etc.) lives under `sql_runtime_predictor/` and is not wired into this UI.
