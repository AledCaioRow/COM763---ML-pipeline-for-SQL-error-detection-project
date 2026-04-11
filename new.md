# Project Explanation (Beginner-Friendly)

## What this project is now

This project trains a machine learning model to predict whether a SQL query is **fast** or **slow**.

It now uses **BIRD Mini-Dev** queries (real benchmark queries), not the old synthetic TPC-H generator.

## Why this matters

The old synthetic workflow had repeated query patterns, so results could look better than real-world performance.

The new workflow uses real queries across multiple databases, which gives a more realistic test of generalization.

## Current pipeline (BIRD version)

The pipeline is run from `main.py`:

1. Load BIRD queries from JSON (`mini_dev_sqlite.json` preferred, fallback `mini_dev_mysql.json`)
2. Time each query on its matching SQLite database (`src/data/load_bird.py`)
3. Save `data/query_dataset_raw.csv`
4. Extract SQL structural features (`src/features/extract_features.py`)
5. Label queries as fast/slow from runtime (`src/features/extract_features.py`)
6. Split data and train models (`src/models/train.py`)
7. Evaluate and save reports (`src/evaluation/evaluate.py`)

## Key files

- `setup_bird.py`  
  Checks whether required BIRD files are present.

- `src/data/load_bird.py`  
  Loads BIRD JSON, performs optional MySQL->SQLite query conversion, times queries, and builds raw dataset.

- `src/features/extract_features.py`  
  Parses SQL into model features; keeps metadata (`db_id`, `difficulty`) for analysis.

- `src/models/train.py`  
  Supports `random` split and `database_aware` split (hold out whole databases).

- `src/evaluation/evaluate.py`  
  Produces overall metrics plus per-database and per-difficulty breakdowns.

## Outputs

- `data/query_dataset_raw.csv`
- `data/query_dataset_features.csv`
- `artifacts/best_model.joblib`
- `reports/model_results.txt`
- `reports/per_database_results.csv`
- `reports/per_difficulty_results.csv`

## How to run

1. Verify data:
```bash
python setup_bird.py
```

2. Run pipeline:
```bash
python -u main.py
```

If BIRD files are missing, `main.py` now exits with a clear setup message.
