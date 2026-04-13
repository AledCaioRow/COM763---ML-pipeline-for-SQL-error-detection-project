# SQL runtime predictor

End-to-end runtime regression pipeline for BIRD Mini-Dev:

synthetic SQL generation -> runtime collection -> plan-tree feature extraction -> PyTorch training -> held-out BIRD evaluation.

The model predicts `log1p(runtime_seconds)` and reports runtime metrics in seconds.

## Quickstart

```bash
cd sql_runtime_predictor
python -m pip install -r requirements.txt
python -m src.creation.generate_queries --per-db 500
python -m src.creation.collect_runtimes
python -m src.modeling.extract_features
python -m src.modeling.train
python -m src.performance.evaluate
```

Run commands from inside `sql_runtime_predictor`.
Backward-compatible entry points (`python -m src.generate_queries`, etc.) still work.

## BIRD data location

The project expects a directory containing:

- `dev_databases/<db_id>/<db_id>.sqlite`
- `mini_dev_sqlite.json` or `mini_dev_mysql.json`

Resolution order:

1. `BIRD_ROOT` or `SRP_BIRD_ROOT` (environment variable)
2. First existing path in `configs/default.yaml` `bird_root_candidates`
3. Fallback to `../Mini Dev/MINIDEV`

You can also keep a local copy under `data/bird_mini_dev/` and point `BIRD_ROOT` there.

## Pipeline commands

1) Generate synthetic training SQL

```bash
python -m src.creation.generate_queries
```

- `--per-db N`: override per-database count
- `--db <db_id>`: generate only one database
- Uses schema inspection + template diversity (joins, predicates, aggregation, ordering, limits, optional subquery/set-op variants)

2) Collect runtimes

```bash
python -m src.creation.collect_runtimes
```

- `--input path/to/file.jsonl`: run a single synthetic file
- Stores median runtime, all run times, timeout flag, and errors
- Writes `data/collected_runtimes/*.jsonl` + `collection_meta.json`

3) Extract plan-tree features

```bash
python -m src.modeling.extract_features
```

- Normal mode writes per-file training shards and `data/features/train_all.jsonl`
- Also refreshes `data/features/bird_dev_features.jsonl` when BIRD JSON is available
- `--bird-only` writes only `bird_dev_features.jsonl`

4) Train model

```bash
python -m src.modeling.train
```

- `--features path/to/train_all.jsonl`
- Stratified-by-database train/validation split
- Saves `artifacts/runtime_predictor.pt` + `artifacts/train_meta.json`

5) Evaluate model and baselines

```bash
python -m src.performance.evaluate
```

- `--checkpoint path/to/runtime_predictor.pt`
- Retimes BIRD dev queries and evaluates:
  - tree model (`RuntimePredictor`)
  - flattened-plan Ridge baseline (if available)
  - TF-IDF SQL Ridge baseline (if available)
  - EXPLAIN opcode-count Ridge baseline (if available)
- Writes `artifacts/eval_report.json`

## Key configuration

`configs/default.yaml` controls:

- synthetic volume (`queries_per_db_min`, `queries_per_db_max`)
- timing (`timing_runs`, `cache_size_pages`, `timeout_seconds`)
- training (`hidden_dim`, `dropout`, `learning_rate`, `weight_decay`, `epochs`, `batch_size`, `early_stopping_patience`, `train_fraction`)
- evaluation percentiles (`q_error_percentiles`)

## Project layout

| Path | Purpose |
|------|---------|
| `src/creation/` | query creation and runtime collection entry points |
| `src/modeling/` | feature extraction, model, and training entry points |
| `src/performance/` | evaluation and baseline entry points |
| `src/database/` | shared utility entry points for config/schema/path helpers |
| `src/data_queries/` | namespace for query/data-query helper modules |
| `src/*.py` | backward-compatible module paths for existing scripts |
| `data/` | generated synthetic/runtimes/features JSONL |
| `artifacts/` | model checkpoint + training/evaluation reports |
| `notebooks/` | optional EDA and result exploration |