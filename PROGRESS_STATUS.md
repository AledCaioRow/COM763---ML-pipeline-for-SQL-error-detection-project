# SQPP Progress Status: 2026-04-14

## Changes since last checkpoint
- Dataset size increased from 425 raw / 320 labelled to 498 raw / 374 labelled rows (`data/query_dataset_raw.csv`, `data/query_dataset_features.csv`).
- Evaluation/reporting was expanded and now includes all-model comparison, classification report CSV, per-database and per-difficulty CSVs, confusion matrices, ROC/PR plots, calibration curve, CV fold scores, feature importance outputs, class/runtime distributions, split summary, and error analysis (`reports/` outputs present).
- Model training persists all four fitted models plus best-model copy (`artifacts/*_model.joblib`, `artifacts/best_model.joblib`).
- Quick iterative experiments were added and executed via `rerun_quick_improvements.py`.
- New experiment outputs were generated: `reports/quick_experiments_summary.csv` and `reports/quick_experiments_summary.md`.

## Current results
- Baseline best model (`XGBoost`) CV F1: `0.5359 ± 0.0406` (`reports/model_results.txt`).
- Baseline held-out test metrics: F1 `0.1860`, ROC-AUC `0.4610`, accuracy `0.5205`.
- Baseline per-database F1: `financial=0.1818` (support `32`), `formula_1=0.1905` (support `41`).
- Baseline per-difficulty F1: `challenging=0.2000` (support `13`), `moderate=0.2609` (support `33`), `simple=0.0000` (support `27`).
- Quick experiment table (`reports/quick_experiments_summary.csv`):
  - Baseline (`best_model.joblib`, threshold `0.50`): F1_slow `0.1860`, Precision_slow `0.1111`, Recall_slow `0.5714`, ROC-AUC `0.4610`.
  - Exp1 Weighted Logistic Regression (`class_weight=balanced`, threshold `0.50`): F1_slow `0.2632`, Precision_slow `0.1613`, Recall_slow `0.7143`, ROC-AUC `0.6602`.
  - Exp2 Weighted Logistic + tuned threshold (`0.38`): F1_slow `0.1587`, Precision_slow `0.0893`, Recall_slow `0.7143`, ROC-AUC `0.6602`.

## Decisions and reasoning
- Kept database-aware holdout (`financial`, `formula_1`) to preserve evaluation integrity under schema shift rather than inflate scores with random-split leakage.
- Prioritized fast, defensible iterative experiments (class weighting + threshold tuning) instead of large architectural changes due to submission-time constraints.
- Accepted that weighted logistic regression (Exp1) improved target-class performance versus baseline, while threshold tuning (Exp2) did not improve F1 in this run; both are retained as evidence of tested alternatives.
- Framed System B (`sql_runtime_predictor/`) as designed/partially implemented direction because code/modules exist but final runtime-regression artifacts and reported end-to-end metrics are not currently present in the repository.

## Outstanding work
- Finalize the 2000-word Task 1 report using consistent, file-backed metrics only (avoid mixing older snapshots).
- Add the Baseline vs Exp1 vs Exp2 comparison table and discuss why Exp1 was retained.
- Provide Streamlit Community Cloud deployment URL and reproducible run/deploy commands in the submission.
- Resolve remaining documentation consistency issues (for example, any README statements that conflict with current code behavior).
- [UNCONFIRMED] If claiming full System B results, run System B end-to-end and produce/report its artifacts and metrics; otherwise keep it as future direction.

## Current blockers
- Assignment brief/rubric files are present as `.docx`/`.xlsx` in `markdowns/ADVANCED MML BREIFS`, but no extracted markdown text exists in-repo yet for direct quote-level cross-checking.
- Streamlit deployment URL is not yet recorded in repository docs.
