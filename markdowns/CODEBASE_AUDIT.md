# SQPP Codebase Audit: 2026-04-13

## File inventory

- `config.py` — Central root-pipeline configuration for paths, timing, split mode, labels, and `FEATURE_COLS` (25 features). **MODIFIED**
- `main.py` — Root pipeline orchestrator (`load_and_time_bird_queries` -> `add_parsed_features` -> `add_labels` -> train -> evaluate -> save report). **MODIFIED**
- `setup_bird.py` — BIRD Mini-Dev setup/data verifier with expected DB list and download guidance. **MODIFIED**
- `src/data/load_bird.py` — Loads BIRD JSON, optional MySQL->SQLite conversion, times queries (median of runs), logs failure categories. **MODIFIED**
- `src/features/extract_features.py` — Extracts SQL structural features and applies quantile/median labeling for binary slow/fast targets. **MODIFIED**
- `src/models/train.py` — Train/test split (`random` or `database_aware`), model registry (LR/RF/GB/XGBoost), 5-fold CV model selection and persistence. **MODIFIED**
- `src/models/predict.py` — Inference helpers for single/batch SQL prediction using saved sklearn model. **MODIFIED**
- `src/evaluation/evaluate.py` — Root evaluation/reporting: classification report, confusion matrix, F1/ROC-AUC, per-db/per-difficulty CSV outputs. **MODIFIED**
- `data/query_dataset_raw.csv` — Root timed-query dataset (`question_id, db_id, sql, difficulty, runtime_s`). **MODIFIED**
- `data/query_dataset_features.csv` — Root feature matrix with extracted SQL features and labels. **MODIFIED**
- `reports/model_results.txt` — Root model selection and test-evaluation report text. **MODIFIED**
- `reports/per_database_results.csv` — Root held-out per-database metrics. **MODIFIED**
- `reports/per_difficulty_results.csv` — Root held-out per-difficulty metrics. **MODIFIED**
- `artifacts/best_model.joblib` — Expected root persisted best model artifact. **[CHECK] NOT FOUND in current tree**
- `streamlit_app/app.py` — Streamlit dashboard entrypoint with pages: Overview, Data Explorer, Model Results, Predict. **MODIFIED**
- `streamlit_app/README.md` — Streamlit documentation; explicitly states it targets the root legacy classifier pipeline. **MODIFIED**
- `streamlit_app/requirements.txt` — Streamlit-side dependency list. **NEW**
- `streamlit_app/components/sidebar.py` — Sidebar rendering for root path, pipeline settings, missing artifact indicators, page routing. **NEW**
- `streamlit_app/components/charts.py` — Plotly chart builder utilities used across pages. **NEW**
- `streamlit_app/components/metrics.py` — Metric row helper utilities for Streamlit layouts. **NEW**
- `streamlit_app/utils/paths.py` — Project-root resolution (`SQPP_PROJECT_ROOT` env support). **NEW**
- `streamlit_app/utils/data_loader.py` — CSV/report parser + root config loader for dashboard. **NEW**
- `streamlit_app/utils/model_loader.py` — Safe model loading + feature importance extraction for sklearn objects. **NEW**
- `streamlit_app/utils/predictor.py` — In-dashboard prediction from feature vectors. **NEW**
- `sql_runtime_predictor/README.md` — New end-to-end runtime regression workflow documentation (synthetic gen -> runtime collection -> plan-tree features -> PyTorch training/eval). **NEW**
- `sql_runtime_predictor/requirements.txt` — Dependencies for the runtime-regression subsystem (PyTorch, sklearn, scipy, etc.). **NEW**
- `sql_runtime_predictor/configs/default.yaml` — Runtime-regression config (BIRD roots, synthetic volume, timing, training, eval percentiles). **NEW**
- `sql_runtime_predictor/data/bird_mini_dev/README.txt` — Data-location note for optional local BIRD Mini-Dev copy/symlink. **NEW**
- `sql_runtime_predictor/notebooks/eda.ipynb` — Notebook stub for runtime collection exploratory analysis. **NEW**
- `sql_runtime_predictor/notebooks/results.ipynb` — Notebook stub for evaluation report visualization. **NEW**
- `sql_runtime_predictor/src/__init__.py` — Package marker for runtime predictor modules. **NEW**
- `sql_runtime_predictor/src/generate_queries.py` — Phase 1 synthetic SQL generator (schema-aware joins/predicates/aggregates/subqueries/set-ops; validates executable SQL). **NEW**
- `sql_runtime_predictor/src/collect_runtimes.py` — Phase 2 runtime collection (warmup + timed runs, timeout checks, metadata + JSONL outputs). **NEW**
- `sql_runtime_predictor/src/extract_features.py` — Phase 3 feature extraction using `EXPLAIN QUERY PLAN` tree encoding + global query features. **NEW**
- `sql_runtime_predictor/src/model.py` — QPPNet-style recursive tree encoder (`RuntimePredictor`) implemented in PyTorch. **NEW**
- `sql_runtime_predictor/src/train.py` — Phase 4 training loop (log-runtime target, stratified-by-database split, AdamW + cosine LR + early stopping). **NEW**
- `sql_runtime_predictor/src/evaluate.py` — Phase 5 evaluation (q-error, p90 q-error, MAE(log), Spearman, median-split accuracy; plus Ridge baselines). **NEW**
- `sql_runtime_predictor/src/utils.py` — Shared schema inspection, config/path resolution, JSONL utilities, MySQL->SQLite conversion. **NEW**
- `sql_runtime_predictor/src/creation/__init__.py` — Creation namespace package marker. **NEW**
- `sql_runtime_predictor/src/creation/generate_queries.py` — Compatibility entry point re-exporting `src.generate_queries`. **NEW**
- `sql_runtime_predictor/src/creation/collect_runtimes.py` — Compatibility entry point re-exporting `src.collect_runtimes`. **NEW**
- `sql_runtime_predictor/src/modeling/__init__.py` — Modeling namespace package marker. **NEW**
- `sql_runtime_predictor/src/modeling/extract_features.py` — Compatibility entry point re-exporting `src.extract_features`. **NEW**
- `sql_runtime_predictor/src/modeling/model.py` — Compatibility exports re-exporting `src.model`. **NEW**
- `sql_runtime_predictor/src/modeling/train.py` — Compatibility entry point re-exporting `src.train`. **NEW**
- `sql_runtime_predictor/src/performance/__init__.py` — Performance namespace package marker. **NEW**
- `sql_runtime_predictor/src/performance/evaluate.py` — Compatibility entry point re-exporting `src.evaluate`. **NEW**
- `sql_runtime_predictor/src/database/__init__.py` — Database namespace package marker. **NEW**
- `sql_runtime_predictor/src/database/utils.py` — Compatibility exports re-exporting `src.utils`. **NEW**
- `sql_runtime_predictor/src/data_queries/__init__.py` — Data/query namespace package marker. **NEW**
- `sql_runtime_predictor/src/data_queries/README.md` — Notes for future query-template/data-query helper modules. **NEW**

## New or changed features

- Root pipeline feature set remains 25 columns (no detected additions): `n_tokens`, `query_length`, `n_joins`, `n_tables_approx`, `n_where_predicates`, `has_group_by`, `has_order_by`, `has_having`, `has_distinct`, `has_limit`, `has_union`, `n_subqueries`, `has_subquery`, `max_nesting_depth`, `n_aggregations`, `n_count`, `n_sum`, `n_avg`, `n_max`, `n_min`, `has_between`, `has_in_clause`, `has_like`, `has_exists`, `has_correlated_subquery`.
- Root `src/features/extract_features.py` still uses quantile labeling logic (`p75` slow, `p50` fast, drop middle) when `LABEL_METHOD == "quantile"`; median labeling path remains available.
- New runtime-regression subsystem does not use `FEATURE_COLS`; it defines structural vector dimensions in `sql_runtime_predictor/src/extract_features.py`:
  - `NODE_FEATURE_DIM = len(OPERATOR_TYPES) + len(ACCESS_TYPES) + 8`
  - `GLOBAL_FEATURE_DIM = 16`
- New plan-tree features include one-hot operator/access classes plus numeric node attributes (table normalization, estimated row-scale proxy, predicate/index signals, selectivity heuristics).

## Pipeline changes

- Root legacy classifier pipeline (baseline-aligned) is still present (`main.py`, `src/data`, `src/features`, `src/models`, `src/evaluation`) with key baseline settings preserved in `config.py`:
  - `SPLIT_METHOD = "database_aware"`
  - `HOLDOUT_DATABASES = ["formula_1", "financial"]`
  - `TIMING_RUNS = 3`
  - `QUERY_TIMEOUT_S = 30`
- Root loading/timing in `load_and_time_bird_queries()` retains SQLite-first JSON choice via `pick_bird_json_path()`, MySQL fallback conversion via `convert_mysql_to_sqlite()`, median-of-runs runtime, and timeout handling.
- Major NEW parallel pipeline introduced under `sql_runtime_predictor/`:
  - **Phase 1** `src/generate_queries.py`: synthetic, schema-driven query generation (joins/predicates/aggregates/subqueries/set operations).
  - **Phase 2** `src/collect_runtimes.py`: runtime collection from synthetic JSONL with timing metadata.
  - **Phase 3** `src/extract_features.py`: `EXPLAIN QUERY PLAN` tree extraction + global feature construction.
  - **Phase 4** `src/train.py`: PyTorch `RuntimePredictor` training on `log1p(runtime)`.
  - **Phase 5** `src/evaluate.py`: runtime-regression metrics + optional baselines (`flat_ridge`, `tfidf_ridge`, `explain_opcode_ridge`).
- Data format has diverged for the new subsystem: legacy root uses CSV (`data/query_dataset_*.csv`), while runtime predictor uses JSONL shards (`data/synthetic_queries`, `data/collected_runtimes`, `data/features`).
- Streamlit pipeline integration changed architecture: from single-app expectations to modular files (`components/` + `utils/`), while still reading legacy root artifacts only.

## Current metrics

- Source: `reports/model_results.txt`, `reports/per_database_results.csv`, `reports/per_difficulty_results.csv`.
- Cross-validation F1 (training set):
  - `XGBoost`: `0.5362 ± 0.0515`
  - `Random Forest`: `0.5126 ± 0.0459`
  - `Gradient Boosting`: `0.4728 ± 0.0998`
  - `Logistic Regression`: `0.4138 ± 0.0638`
- Best model: `XGBoost`
- Held-out test metrics:
  - Test F1: `0.1765`
  - ROC-AUC: `0.5442`
  - Accuracy: `0.49`
- Per-database F1:
  - `financial`: `0.16666666666666666` (22 queries)
  - `formula_1`: `0.18181818181818182` (33 queries)
- Per-difficulty F1:
  - `challenging`: `0.4444444444444444` (9 queries)
  - `moderate`: `0.16666666666666666` (23 queries)
  - `simple`: `0.0` (23 queries)
- Runtime-regression metrics (`sql_runtime_predictor/artifacts/eval_report.json`) are **[CHECK] not present** in current tree, so no exact values can be extracted for that subsystem.

## Streamlit app state

- Current app is multi-page and modular:
  - Pages (`PAGE_OPTIONS`): `Overview`, `Data Explorer`, `Model Results`, `Predict`.
  - Components/util modules split across `streamlit_app/components/*.py` and `streamlit_app/utils/*.py`.
- Data sources are legacy root artifacts, not runtime-regression artifacts:
  - Reads `data/query_dataset_raw.csv`, `data/query_dataset_features.csv`.
  - Reads `reports/model_results.txt`, `reports/per_database_results.csv`, `reports/per_difficulty_results.csv`.
  - Loads `artifacts/best_model.joblib` (if present).
  - Loads root `config.py` for `FEATURE_COLS` and split/timing settings.
- `Predict` page uses:
  - Manual feature vector prediction via loaded sklearn model.
  - SQL-to-feature extraction by importing `src.features.extract_features.extract_features` from the root project.
- `streamlit_app/README.md` explicitly states the dashboard is for the legacy root classifier and does not connect to `sql_runtime_predictor/`.
- Baseline layout expected `streamlit_app/snapshots/`; no snapshot files were found in current visible tree. **[CHECK]**

## Code quality notes

- Generated cache artifacts appear tracked in VCS (`sql_runtime_predictor/src/**/__pycache__/*.pyc` per git status), which should usually be excluded.
- Duplicate path representations in git status (same files with `/` and `\`) suggest path-normalization/noise that can complicate review and diffs.
- `reports/model_results.txt` contains mojibake for special characters (`�` where `±` appears), indicating encoding inconsistency in report writing/reading.
- Legacy and new pipelines coexist with overlapping entrypoint names (`train.py`, `evaluate.py`, etc.); compatibility wrappers help, but naming overlap increases maintenance risk.
- Root report files exist while `artifacts/best_model.joblib` is currently missing in tree. **[CHECK]** Verify artifact generation/output path before dashboard inference use.
