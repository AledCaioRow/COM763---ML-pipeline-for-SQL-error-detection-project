# Data (bundled copy)

This folder holds a **snapshot copy** of the parent project’s pipeline outputs:

- `query_dataset_raw.csv` — BIRD questions + measured runtimes (from `main.py` iteration 1, stage 1).
- `query_dataset_features.csv` — same rows + parsed SQL features + `label_binary`.

Regenerate from scratch (after placing BIRD Mini-Dev where `config.py` can find it):

```bash
cd ..   # standalone_sql_perf_pipeline root
python setup_bird.py
python main.py -n 1
```

The copies here let you run iterations **3–7** or analysis **without** re-timing all queries, until you refresh them.

**Not copied:** BIRD JSON/SQLite under `Mini Dev/` — still required on disk for iteration 1 and for any script that opens `.sqlite` (e.g. `run_schema_stats_model.py`, `run_report_graphs.py`).
