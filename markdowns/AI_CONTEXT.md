# AI Context -- SQL Query Performance Predictor (BIRD Version)

## What this project does

This project predicts whether a SQL query will be **slow** or **fast** using 25 structural SQL features.  
It now uses **BIRD Mini-Dev** query data (real benchmark queries), not synthetic TPC-H templates.

## Current pipeline

1. `setup_bird.py` verifies dataset files.
2. `src/data/load_bird.py` loads queries and times them on SQLite databases.
3. `src/features/extract_features.py` extracts features and labels fast/slow.
4. `src/models/train.py` splits data and trains models with CV.
5. `src/evaluation/evaluate.py` evaluates and writes reports.

Run with:

```bash
python setup_bird.py
python -u main.py
```

## Key split behavior

- `SPLIT_METHOD = "random"`: stratified random split.
- `SPLIT_METHOD = "database_aware"`: holds out entire databases using `HOLDOUT_DATABASES`.

`database_aware` is the preferred honesty check because the model never sees holdout schemas during training.

## Core metadata rules

- Model features are only `FEATURE_COLS` from `config.py`.
- Metadata columns (`question_id`, `db_id`, `difficulty`, `sql`) are for split/evaluation only.

## Main outputs

- `data/query_dataset_raw.csv`
- `data/query_dataset_features.csv`
- `artifacts/best_model.joblib`
- `reports/model_results.txt`
- `reports/per_database_results.csv`
- `reports/per_difficulty_results.csv`
