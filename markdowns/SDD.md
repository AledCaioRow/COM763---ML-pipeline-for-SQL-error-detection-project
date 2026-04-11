# Software Design Document -- SQL Query Performance Predictor

## 1. Purpose

This system predicts SQL runtime class (**fast** vs **slow**) using static SQL structure features.
The data source is BIRD Mini-Dev with per-query timing on SQLite databases.

## 2. Architecture

- `config.py`: central configuration
- `setup_bird.py`: setup verification for BIRD assets
- `src/data/load_bird.py`: load + optional MySQL-to-SQLite conversion + runtime timing
- `src/features/extract_features.py`: 25 feature extraction + runtime-based labels
- `src/models/train.py`: split, CV model selection, save best model
- `src/evaluation/evaluate.py`: test metrics + grouped reporting
- `src/models/predict.py`: inference API for new SQL queries

## 3. Data Flow

1. Load BIRD query JSON (`mini_dev_sqlite.json`, fallback `mini_dev_mysql.json`)
2. Resolve database file by `db_id` (`dev_databases/<db_id>/<db_id>.sqlite`)
3. Execute each query with timeout and record median runtime
4. Save raw dataset (`query_dataset_raw.csv`)
5. Extract features and apply labels (`query_dataset_features.csv`)
6. Train models and persist best artifact (`best_model.joblib`)
7. Evaluate on held-out test data and write reports

## 4. Split Strategy

- `random`: stratified 80/20 split (baseline)
- `database_aware`: hold out complete databases (`HOLDOUT_DATABASES`) for testing

The `database_aware` split is the primary method because it tests schema-level generalization.

## 5. Evaluation Outputs

- Overall classification report, F1, ROC-AUC
- Per-database metrics (`reports/per_database_results.csv`)
- Per-difficulty metrics (`reports/per_difficulty_results.csv`)
- Human-readable summary (`reports/model_results.txt`)

Edge cases (single-class or too-small groups) are reported as `N/A`.
