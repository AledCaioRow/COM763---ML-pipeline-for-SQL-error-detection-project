# AI context — SQL performance repo

Use this document to keep tooling and contributors aligned with the current codebase structure.

The repository has two pipelines with different artifacts; keep them separated in docs and code changes.

---

## System A (legacy): root fast/slow classifier

- **Task:** classification from SQL structure features + measured runtimes
- **Entry:** `python setup_bird.py` then `python -u main.py`
- **Config:** `config.py` (`FEATURE_COLS`, split mode, holdout DBs, timing)
- **Artifacts:** `artifacts/best_model.joblib`, `data/query_dataset_*.csv`, `reports/*`
- **Streamlit:** `streamlit_app/` expects these root outputs.

---

## System B (current): `sql_runtime_predictor/` runtime regression

- **Task:** regress query runtime using `EXPLAIN QUERY PLAN` trees + global features
- **Working directory:** `sql_runtime_predictor/`
- **Order:** `src.creation.generate_queries` -> `src.creation.collect_runtimes` -> `src.modeling.extract_features` -> `src.modeling.train` -> `src.performance.evaluate`
- **Config:** `configs/default.yaml`
- **BIRD path resolution:** `BIRD_ROOT` or `SRP_BIRD_ROOT`, else `bird_root_candidates`
- **Artifacts:** `artifacts/runtime_predictor.pt`, `train_meta.json`, `eval_report.json`
- **Data folders:** `data/synthetic_queries`, `data/collected_runtimes`, `data/features`
- **CLI flags:** `--per-db`, `--db`, `--input`, `--bird-only`, `--features`, `--checkpoint`

---

## Shared concepts

- BIRD layout: `dev_databases/<db_id>/<db_id>.sqlite` + `mini_dev_sqlite.json` or `mini_dev_mysql.json`
- `sql_runtime_predictor/src/database/utils.py`: `resolve_bird_root`, `convert_mysql_to_sqlite`, `list_sqlite_databases`
- Legacy module paths (`src.generate_queries`, `src.train`, etc.) remain available for compatibility.

When editing docs or code, always state which system you are changing (root legacy vs `sql_runtime_predictor` current).
