# SQL Query Performance Predictor - Current Project Overview

## What this project is doing right now

This project trains a machine learning classifier that predicts whether a SQL query will be `fast` or `slow` before running it.

It uses the BIRD Mini-Dev benchmark data, executes queries on SQLite databases to measure runtime, extracts structural SQL features, and trains multiple models to pick the best one.

## End-to-end pipeline

The pipeline is orchestrated from `main.py`:

1. **Load query data**
   - Picks query source using `src/data/load_bird.py`:
     - Prefer `mini_dev_sqlite.json`
     - Fallback to `mini_dev_mysql.json` and convert syntax to SQLite where possible
2. **Measure runtime**
   - Runs each query against its matching database:
     - Database path pattern: `dev_databases/<db_id>/<db_id>.sqlite`
     - Uses median runtime across 3 runs (`TIMING_RUNS = 3`)
     - Uses timeout guard (`QUERY_TIMEOUT_S = 30`)
3. **Save raw timed dataset**
   - Writes `data/query_dataset_raw.csv`
4. **Extract SQL features**
   - `src/features/extract_features.py` computes 25 structural features
   - Examples: joins, predicates, group/order by flags, aggregates, subqueries, nesting depth
5. **Create labels**
   - Current method: `quantile`
   - Slow = top 25% runtime, Fast = bottom 50%, middle dropped
   - Writes `data/query_dataset_features.csv`
6. **Split data**
   - Current method: `database_aware`
   - Holds out full databases for testing: `formula_1`, `financial`
7. **Train and select model**
   - Candidates: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
   - Selection by 5-fold CV F1 score on training set
   - Best model saved to `artifacts/best_model.joblib`
8. **Evaluate and report**
   - Writes:
     - `reports/model_results.txt`
     - `reports/per_database_results.csv`
     - `reports/per_difficulty_results.csv`

## Current data state

From the generated datasets currently in the repo:

- Raw timed dataset: **425 queries** (`query_dataset_raw.csv`)
- Feature/labeled dataset: **320 queries** (`query_dataset_features.csv`)
- Label distribution (after quantile filtering):
  - `fast`: 213
  - `slow`: 107

Difficulty distribution in raw data:

- `moderate`: 209
- `simple`: 138
- `challenging`: 78

Runtime summary (`runtime_s`, raw data):

- Median: ~0.00248 s
- 75th percentile: ~0.0674 s
- Max: ~1.8010 s

## Current model results (latest run)

From `reports/model_results.txt`:

- Train set: 265
- Test set: 55
- Best model selected: **XGBoost**

Cross-validation F1 on training set:

- XGBoost: 0.5362 +/- 0.0515
- Random Forest: 0.5126 +/- 0.0459
- Gradient Boosting: 0.4728 +/- 0.0998
- Logistic Regression: 0.4138 +/- 0.0638

Held-out test performance:

- F1 (slow class): **0.1765**
- ROC-AUC: **0.5442**
- Accuracy: **0.49**

Class behavior on test set:

- Fast queries: high precision, lower recall
- Slow queries: low precision, moderate recall
- This indicates the model is currently weak at cleanly identifying slow queries on unseen databases.

Per-database test breakdown:

- `financial`: F1 = 0.1667
- `formula_1`: F1 = 0.1818

Per-difficulty test breakdown:

- `challenging`: F1 = 0.4444
- `moderate`: F1 = 0.1667
- `simple`: F1 = 0.0000

## Project structure and important files

- `config.py`: all configuration (paths, split mode, holdout DBs, feature list)
- `setup_bird.py`: checks that BIRD data files and SQLite DB files exist
- `src/data/load_bird.py`: query loading, optional SQL conversion, query timing
- `src/features/extract_features.py`: feature extraction and label generation
- `src/models/train.py`: split logic, model registry, CV selection, model persistence
- `src/models/predict.py`: inference helpers for single query or batch
- `src/evaluation/evaluate.py`: metrics and report generation

## Supporting data bundle in this repo

Under `Mini Dev/` the project includes:

- 11 SQLite benchmark databases
- SQL files for benchmark content
- Many CSV schema/description files for the benchmark databases

These files provide the benchmark workload and schemas used for runtime timing and evaluation.

## How to run the project

1. Verify dataset setup:

```bash
python setup_bird.py
```

2. Run full pipeline:

```bash
python -u main.py
```

## Bottom line

The project is currently in a working end-to-end state: data loading, timing, feature extraction, training, evaluation, and report generation all run.  
The main current gap is predictive quality on truly unseen databases, especially for the `slow` class.

