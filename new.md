# Project explanation (beginner friendly)

## Two projects in one repo

| What | Where | Question it answers |
|------|--------|----------------------|
| **Fast vs slow classifier (legacy)** | Repository root (`main.py`) | Will this query be slow or fast? |
| **Runtime predictor (new)** | `sql_runtime_predictor/` | About how many seconds will this query take? |

Both use BIRD Mini-Dev style assets. They do **not** share model files, feature formats, or artifacts.

---

## A) Fast/slow classifier (root, legacy)

**Idea:** Convert measured runtimes into classes (for example fast vs slow), extract SQL structure features, train sklearn models, and report classification metrics.

**Steps**

1. `python setup_bird.py` — checks that BIRD files exist.  
2. `python -u main.py` — load queries, time them, featurize, train, evaluate.

**Outputs (project root)**

- `data/query_dataset_raw.csv`, `data/query_dataset_features.csv`  
- `artifacts/best_model.joblib`  
- `reports/model_results.txt`, `reports/per_database_results.csv`, `reports/per_difficulty_results.csv`

**Key code**

- `src/data/load_bird.py` — load JSON, optional MySQL→SQLite tweaks, timing  
- `src/features/extract_features.py` — features + labels  
- `src/models/train.py` — splits (including holding out whole databases)  
- `src/evaluation/evaluate.py` — metrics and reports  

The **Streamlit app** in `streamlit_app/` is built for this pipeline (CSVs + joblib + `config.py`).

---

## B) Runtime regression (`sql_runtime_predictor/`, current)

**Idea:** Generate synthetic queries per database, time them, build features from SQLite query plans, train a PyTorch tree model, and compare predictions to real BIRD dev runtimes plus simple baselines.

**Steps** (run from inside `sql_runtime_predictor`)

```bash
cd sql_runtime_predictor
python -m pip install -r requirements.txt
python -m src.generate_queries --per-db 500
python -m src.collect_runtimes
python -m src.extract_features
python -m src.train
python -m src.evaluate
```

**Outputs**

- `sql_runtime_predictor/data/synthetic_queries/*.jsonl`
- `sql_runtime_predictor/data/collected_runtimes/*.jsonl`
- `sql_runtime_predictor/data/features/train_all.jsonl` (+ shards and bird dev features)
- `sql_runtime_predictor/artifacts/runtime_predictor.pt`
- `sql_runtime_predictor/artifacts/train_meta.json`
- `sql_runtime_predictor/artifacts/eval_report.json`

**Config**

- `sql_runtime_predictor/configs/default.yaml` controls paths, timing, and training hyperparameters

Set **`BIRD_ROOT`** or **`SRP_BIRD_ROOT`** to point at your Mini-Dev root, or rely on `bird_root_candidates` in the YAML config.

---

## Which should you use?

- Use **A** if you need binary decisions and the existing Streamlit dashboard.
- Use **B** if you need continuous runtime estimates, plan-based modeling, and the new codebase.
