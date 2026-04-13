# Codebase overview

This repository implements **SQL query performance** coursework in three parts: a **root legacy pipeline** (classify queries as fast vs slow on BIRD Mini-Dev), a **Streamlit dashboard** that visualizes that pipeline only, and a **separate runtime-regression subsystem** under `sql_runtime_predictor/` (synthetic queries, plan-tree features, PyTorch).

For a detailed file-by-file inventory and metrics snapshot, see `CODEBASE_AUDIT.md`. For high-level project intent, see `PROJECT_OVERVIEW.md`.

## Repository layout

| Path | Role |
|------|------|
| `config.py` | Root pipeline: paths, timing, train/test split, labeling, `FEATURE_COLS`. |
| `main.py` | Orchestrates load → feature extraction → labels → train → evaluate → reports. |
| `setup_bird.py` | BIRD Mini-Dev setup / verification. |
| `src/data/` | Load BIRD JSON, SQLite timing, optional MySQL→SQLite conversion. |
| `src/features/` | Structural SQL features + label logic used by root pipeline and Streamlit SQL tab. |
| `src/models/` | Training, model registry, persistence (`best_model.joblib`). |
| `src/evaluation/` | Classification metrics, `reports/*.txt` and `*.csv`. |
| `data/` | `query_dataset_raw.csv`, `query_dataset_features.csv` (root pipeline outputs). |
| `reports/` | `model_results.txt`, `per_database_results.csv`, `per_difficulty_results.csv`. |
| `artifacts/` | `best_model.joblib` (root classifier). |
| `streamlit_app/` | Streamlit UI: reads the artifacts above only. |
| `sql_runtime_predictor/` | Synthetic query generation, runtime collection, plan features, PyTorch train/eval (no Streamlit integration). |

## Entry points

**Root classifier**

```bash
python setup_bird.py
python -u main.py
```

**Streamlit (legacy dashboard)**

```bash
cd streamlit_app && python -m pip install -r requirements.txt && python -m streamlit run app.py
```

See `streamlit_app/README.md` for pages, env vars (`SQPP_PROJECT_ROOT`), and artifact contract.

**Runtime predictor (parallel pipeline)**

```bash
cd sql_runtime_predictor
pip install -r requirements.txt
# Typical flow: creation → modeling → performance modules (see sql_runtime_predictor/README.md)
```

## Streamlit vs runtime predictor

- **Streamlit** uses sklearn + CSV + text reports from the **root** tree. It does not read `sql_runtime_predictor/artifacts/` or PyTorch checkpoints.
- **sql_runtime_predictor** targets **continuous** runtime (e.g. log-scale regression, q-error metrics) from `EXPLAIN QUERY PLAN`–style features, not the binary fast/slow UI.

Keeping the two systems separate avoids mixing feature schemas and evaluation metrics.

## Documentation map

| Document | Contents |
|----------|----------|
| `streamlit_app/README.md` | Current Streamlit behavior, pages, files, troubleshooting. |
| `sql_runtime_predictor/README.md` | Runtime regression workflow and configs. |
| `CODEBASE_AUDIT.md` | Dated audit: full file list, feature dimensions, pipeline notes, Streamlit state. |
| `PROJECT_OVERVIEW.md` | Narrative project description. |
