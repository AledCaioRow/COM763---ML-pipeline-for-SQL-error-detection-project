# Software design document — SQL performance repository

## 1. Purpose

This repository implements two related systems:

1. **Legacy classification**: predict fast vs slow from static SQL features and timed BIRD Mini-Dev data (`main.py` at repo root)
2. **Current regression**: predict runtime in seconds from `EXPLAIN QUERY PLAN` trees + global features (`sql_runtime_predictor/`)

---

## 2. System A — root classifier (legacy)

### 2.1 Components

- `config.py` — paths, timing, split mode, holdout DBs, `FEATURE_COLS`
- `setup_bird.py` — verifies BIRD JSON + SQLite assets
- `src/data/load_bird.py` — query loading and runtime timing
- `src/features/extract_features.py` — structural SQL features + labels
- `src/models/train.py` — split strategies, CV, model persistence
- `src/evaluation/evaluate.py` — metrics and grouped reports
- `src/models/predict.py` — inference helpers  

### 2.2 Data flow

1. Read `mini_dev_sqlite.json` or `mini_dev_mysql.json`.  
2. Execute on matching SQLite files; record median runtime.  
3. `query_dataset_raw.csv` → features + labels → `query_dataset_features.csv`.  
4. Train → `artifacts/best_model.joblib`.  
5. Evaluate → `reports/*`.

### 2.3 Split strategy

- Random stratified split or **database-aware** holdout (entire `db_id` groups in test), per `config.py`.

### 2.4 Evaluation outputs

Classification outputs include precision/recall/F1, optional ROC-AUC, grouped CSVs, and `model_results.txt`.

---

## 3. System B — `sql_runtime_predictor` (current plan-tree regression)

### 3.1 Components

- `configs/default.yaml` — path candidates, data dirs, synthetic volume, timing, training hyperparameters, q-error settings
- `src/database/utils.py` — config loading, BIRD root resolution (`BIRD_ROOT` / `SRP_BIRD_ROOT`), schema helpers
- `src/creation/generate_queries.py` — schema-driven synthetic SQL generation and validation
- `src/creation/collect_runtimes.py` — warm-up + repeated timing, median runtime, timeout/error tracking
- `src/modeling/extract_features.py` — `EXPLAIN QUERY PLAN` parsing to plan-tree and global vectors (`--bird-only` supported)
- `src/modeling/model.py` — `RuntimePredictor` tree encoder + regression head
- `src/modeling/train.py` — stratified-by-db split, AdamW, cosine scheduler, early stopping, checkpoint save
- `src/performance/evaluate.py` — BIRD dev retiming and metrics (q-error, Spearman, MAE-log, median-split accuracy) + Ridge baselines
- `src/data_queries/` — namespace for query/data-query helper modules
- `src/*.py` — compatibility entry points for older command paths

### 3.2 Data flow

1. Synthetic JSONL in `data/synthetic_queries/`
2. Timed JSONL in `data/collected_runtimes/` (+ `collection_meta.json`)
3. Feature JSONL in `data/features/` (`train_all.jsonl`, shards, `bird_dev_features.jsonl`)
4. Checkpoint `artifacts/runtime_predictor.pt` + metadata `train_meta.json`
5. Evaluation report `artifacts/eval_report.json`

### 3.3 Design notes

- Training target is `log1p(median_runtime_seconds)` for successful synthetic runs
- Evaluation compares `expm1(prediction)` to freshly timed BIRD runtimes
- Baselines run only when optional deps are available (`scikit-learn`, `scipy`)

---

## 4. Streamlit dashboard

`streamlit_app/` is a read-only UI for System A (root classifier outputs). It does not load `sql_runtime_predictor` checkpoints.
