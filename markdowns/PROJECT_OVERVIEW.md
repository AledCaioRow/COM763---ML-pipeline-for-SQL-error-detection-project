# Advanced ML Course — Project overview

This repository is the **SQL query performance** coursework codebase: a **legacy root pipeline** that classifies queries as fast vs slow, a **separate runtime-regression subsystem** under `sql_runtime_predictor/` that predicts continuous runtimes from SQLite plan trees, and a **Streamlit dashboard** that primarily visualizes the legacy pipeline while also exposing optional runtime prediction flows.

Use **`sql_runtime_predictor/`** for plan-tree runtime prediction; use the **repository root** plus **`streamlit_app/`** for the sklearn classifier and its UI.

---

## Repository layout (high level)

| Area | Role |
|------|------|
| **Root** | `main.py`, `config.py`, `setup_bird.py`, `requirements.txt` — BIRD load/timing, structural features, sklearn training/evaluation |
| **`src/`** | Legacy pipeline: `data/load_bird.py`, `features/extract_features.py`, `models/{train,predict}.py`, `evaluation/evaluate.py` |
| **`data/`** | Root pipeline CSVs (`query_dataset_raw.csv`, `query_dataset_features.csv`) after running `main.py` |
| **`reports/`** | Root evaluation text/CSV outputs |
| **`artifacts/`** | Expected location for `best_model.joblib` after training (create by running `main.py`) |
| **`streamlit_app/`** | Multi-page Streamlit app (`app.py`, `components/`, `utils/`) for the legacy classifier plus optional runtime prediction / live comparison surfaces |
| **`sql_runtime_predictor/`** | Self-contained PyTorch runtime pipeline, own `requirements.txt`, `configs/default.yaml`, `data/`, `artifacts/` |
| **`Mini Dev/`** | Local BIRD Mini-Dev layout (`MINIDEV/dev_databases/...`, JSON task files) when present |
| **`markdowns/`** | Course / design notes (`AI_CONTEXT.md`, `SDD.md`, etc.) |

A file-level inventory lives in **`CODEBASE_AUDIT.md`** (optional deep dive).

---

## 1) Root pipeline: fast / slow classification

Trains a **binary classifier** (e.g. XGBoost, random forest, gradient boosting, logistic regression) on **hand-crafted SQL structure features** and labels derived from measured runtimes on BIRD Mini-Dev SQLite databases.

**Flow** (`main.py`):

1. Resolve BIRD JSON (`mini_dev_sqlite.json` preferred; MySQL JSON can be converted) via `src/data/load_bird.py`.
2. Time each query on `dev_databases/<db_id>/<db_id>.sqlite` (median over repeated runs; timeout from `config.py`).
3. Write `data/query_dataset_raw.csv`.
4. Parse SQL and add feature columns + labels in `src/features/extract_features.py`.
5. Split (e.g. database-aware holdout) and train in `src/models/train.py`; persist **`artifacts/best_model.joblib`**.
6. Evaluate in `src/evaluation/evaluate.py` → `reports/model_results.txt`, `reports/per_database_results.csv`, `reports/per_difficulty_results.csv`.

**Key files**

- `config.py` — paths, timing, split mode, `FEATURE_COLS` (25 structural features)
- `setup_bird.py` — verify BIRD Mini-Dev assets
- `src/data/load_bird.py`, `src/features/extract_features.py`, `src/models/train.py`, `src/models/predict.py`, `src/evaluation/evaluate.py`

**Run**

```bash
python setup_bird.py
python -u main.py
```

---

## 2) `sql_runtime_predictor/`: continuous runtime regression

Predicts **`log1p(runtime_seconds)`** using **`EXPLAIN QUERY PLAN`** tree encodings and a PyTorch **QPPNet-style** tree model (`RuntimePredictor`), with optional Ridge baselines at evaluation time. Trained primarily on **synthetic** schema-driven SQL; evaluated on held-out **BIRD Mini-Dev** queries.

**Recommended commands** (from inside `sql_runtime_predictor/`):

```bash
python -m pip install -r requirements.txt
python -m src.creation.generate_queries --per-db 500
python -m src.creation.collect_runtimes
python -m src.modeling.extract_features
python -m src.modeling.train
python -m src.performance.evaluate
```

Equivalent **`python -m src.<module>`** forms (e.g. `src.generate_queries`) remain supported for compatibility.

**BIRD resolution** (see `sql_runtime_predictor/README.md`):

1. `BIRD_ROOT` or `SRP_BIRD_ROOT`
2. Paths in `configs/default.yaml` → `bird_root_candidates`
3. Default fallback: `../Mini Dev/MINIDEV` relative to `sql_runtime_predictor/`

Optional local copy: `sql_runtime_predictor/data/bird_mini_dev/` with `BIRD_ROOT` pointing there.

**Generated layout** (under `sql_runtime_predictor/`)

- `data/synthetic_queries/`, `data/collected_runtimes/`, `data/features/` (JSONL shards, feature files)
- `artifacts/runtime_predictor.pt`, `artifacts/train_meta.json`, `artifacts/eval_report.json` (after train/eval)

**Code organization** (`sql_runtime_predictor/src/`)

- **`creation/`** — synthetic query generation and runtime collection entry points
- **`modeling/`** — plan-tree feature extraction, model definition, training
- **`performance/`** — evaluation and baselines
- **`database/`** — shared config/schema/path helpers (re-exports from flat `utils.py`)
- **`data_queries/`** — namespace reserved for query/data helpers (`README.md` inside)
- **Top-level `*.py`** — implementations re-exported by the packages above

Primary subsystem documentation: **`sql_runtime_predictor/README.md`**.

---

## 3) Streamlit dashboard (`streamlit_app/`)

Hybrid UI centered on the **root legacy classifier**: it explores timed-query CSVs, report files, and predictions from **`artifacts/best_model.joblib`**, and can also use `sql_runtime_predictor` checkpoints for runtime prediction and tier display when those runtime artifacts exist.

**Layout**

- `app.py` — entrypoint and page routing
- `components/` — `sidebar.py`, `charts.py`, `metrics.py`
- `utils/` — `paths.py` (`SQPP_PROJECT_ROOT`), `data_loader.py`, `model_loader.py`, `predictor.py`

**Pages**: Overview, Data Explorer, Model Results, Predict, Live Compare.

**Runtime integration**:

- `Predict` includes a runtime tab when `sql_runtime_predictor/artifacts/runtime_predictor.pt` is present.
- `Live Compare` measures observed runtime on a selected SQLite DB and compares it with predicted runtime / tier.
- Core data exploration and evaluation pages still rely on the root classifier outputs (`data/`, `reports/`, `artifacts/best_model.joblib`).

**Run** (from repo root):

```bash
cd streamlit_app
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Details: **`streamlit_app/README.md`**.

---

## Supporting data and environment

- BIRD Mini-Dev may live under **`Mini Dev/MINIDEV`**, **`data/bird/`**, or any path you set with **`BIRD_ROOT`** / **`SRP_BIRD_ROOT`** (runtime predictor) alongside the root pipeline’s expectations in `config.py`.
- Run each pipeline from its **intended working directory** (repo root for `main.py` and Streamlit parent paths; `sql_runtime_predictor/` for the PyTorch pipeline) so relative paths in config and YAML resolve correctly.
