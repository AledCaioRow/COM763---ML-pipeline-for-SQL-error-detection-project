# Report evidence bundle (Part B)

Generated: `2026-04-16T11:46:00Z` (UTC). Paths relative to project root.

This file consolidates **where each number lives** and how it maps to report iterations. 
Machine-readable key/value rows: `reports/report_metrics_long.csv`.

## Producer commands (refresh)

| Iteration | Command | Primary outputs |
| --- | --- | --- |
| 1 | `python main.py` or `python main.py -n 1` | `reports/model_results.txt`, `all_models_test_comparison.csv`, `per_database_results.csv`, `split_summary.csv`, … |
| 2 | `python main.py -n 2` or `rerun_commit_comparison.py` | `commit_rerun_metrics.csv` |
| 2 (one row, current tree) | `python export_classifier_geometries_summary.py` | `classifier_geometries_summary.csv` |
| 3 | `python main.py -n 3` or `run_within_db_logistic.py` | `within_db_logistic_metrics.csv` |
| 4 | `python main.py -n 4` or `run_schema_stats_model.py` | `within_db_schema_metrics.csv` |
| Bridge | `python main.py -n 5` or `run_full_regression.py` | stdout → `cross_schema_regression_log.txt` (captured by bundle builder) |
| Figures | `python main.py -n 6` / `-n 7` | `reports/narrative_figures/` |

## Iteration 1 — Global classifier (held-out DBs)

| Metric | Source file | Fields |
| --- | --- | --- |
| Train/test counts, DB lists | `reports/split_summary.csv` | `n_queries`, `fast`, `slow`, `databases` |
| Model comparison | `reports/all_models_test_comparison.csv` | `Model`, `Test F1`, `Test ROC-AUC`, … |
| Narrative | `reports/model_results.txt` | summary + per-DB breakdown |
| Per-DB test | `reports/per_database_results.csv` | `db_id`, `f1`, `support`, … |

### Snapshot (from bundle CSV)

| bundle_key | value | source_file |
| --- | --- | --- |
| iter1_split_train_n | 310 | reports/split_summary.csv |
| iter1_split_train_fast | 194 | reports/split_summary.csv |
| iter1_split_train_slow | 116 | reports/split_summary.csv |
| iter1_split_test_n | 64 | reports/split_summary.csv |
| iter1_split_test_fast | 55 | reports/split_summary.csv |
| iter1_split_test_slow | 9 | reports/split_summary.csv |
| iter1_holdout_dbs | financial, formula_1 | reports/split_summary.csv |
| iter1_best_model | Logistic Regression | reports/all_models_test_comparison.csv |
| iter1_best_test_f1 | 0.5217391304347826 | reports/all_models_test_comparison.csv |
| iter1_best_test_roc_auc | 0.6444444444444444 | reports/all_models_test_comparison.csv |
| iter1_best_test_acc | 0.828125 | reports/all_models_test_comparison.csv |
| iter1_perdb_financial_f1 | 0.125 | reports/per_database_results.csv |
| iter1_perdb_financial_support | 30 | reports/per_database_results.csv |
| iter1_perdb_formula_1_f1 | 0.3809523809523809 | reports/per_database_results.csv |
| iter1_perdb_formula_1_support | 34 | reports/per_database_results.csv |
| geometry_raw_rows | 498 | reports/classifier_geometries_summary.csv |
| geometry_labelled_rows | 374 | reports/classifier_geometries_summary.csv |
| geometry_fast_rows | 249 | reports/classifier_geometries_summary.csv |
| geometry_slow_rows | 125 | reports/classifier_geometries_summary.csv |
| geometry_unseen_train_rows | 310 | reports/classifier_geometries_summary.csv |
| geometry_unseen_test_rows | 64 | reports/classifier_geometries_summary.csv |
| geometry_seen_train_rows | 294 | reports/classifier_geometries_summary.csv |
| geometry_seen_test_rows | 80 | reports/classifier_geometries_summary.csv |
| geometry_unseen_best_model | Random Forest | reports/classifier_geometries_summary.csv |
| geometry_unseen_f1 | 0.2702702702702703 | reports/classifier_geometries_summary.csv |
| geometry_unseen_roc_auc | 0.5686868686868687 | reports/classifier_geometries_summary.csv |
| geometry_unseen_accuracy | 0.578125 | reports/classifier_geometries_summary.csv |
| geometry_seen_best_model | Gradient Boosting | reports/classifier_geometries_summary.csv |
| geometry_seen_f1 | 0.2553191489361702 | reports/classifier_geometries_summary.csv |
| geometry_seen_roc_auc | 0.4567272727272727 | reports/classifier_geometries_summary.csv |
| geometry_seen_accuracy | 0.5625 | reports/classifier_geometries_summary.csv |


## Iteration 2 — Seen split (pooled) + multi-commit table

| Metric | Source | Fields |
| --- | --- | --- |
| Seen / unseen metrics | `reports/commit_rerun_metrics.csv` | `seen_*`, `unseen_*`, row counts |
| **Single source of truth (current tree)** | `reports/classifier_geometries_summary.csv` | same columns, one row |

### Degenerate-label check (features CSV)

| db_id | n | slow_rate | degenerate |
| --- | ---: | ---: | --- |
| california_schools | 3 | 0.0000 | True |
| card_games | 46 | 0.8478 | False |
| codebase_community | 37 | 0.9189 | False |
| debit_card_specializing | 20 | 0.3500 | False |
| european_football_2 | 41 | 0.8049 | False |
| financial | 30 | 0.1333 | False |
| formula_1 | 34 | 0.1471 | False |
| student_club | 47 | 0.0000 | True |
| superhero | 52 | 0.0385 | False |
| thrombosis_prediction | 38 | 0.0000 | True |
| toxicology | 26 | 0.0385 | False |


## Iteration 3 — Per-DB logistic (SQL features)

Artefact: `reports/within_db_logistic_metrics.csv`. Use rows with `status=ok`; 
degenerate / skipped databases are explicit for the “ill-posed class” argument.

| bundle_key | value | source_file |
| --- | --- | --- |
| iter3_db_california_schools_n | 3 | reports/within_db_logistic_metrics.csv |
| iter3_db_california_schools_n_train |  | reports/within_db_logistic_metrics.csv |
| iter3_db_california_schools_n_test |  | reports/within_db_logistic_metrics.csv |
| iter3_db_california_schools_slow_pct | 0 | reports/within_db_logistic_metrics.csv |
| iter3_db_california_schools_f1_slow |  | reports/within_db_logistic_metrics.csv |
| iter3_db_california_schools_roc_auc |  | reports/within_db_logistic_metrics.csv |
| iter3_db_california_schools_accuracy |  | reports/within_db_logistic_metrics.csv |
| iter3_db_california_schools_status | skipped_small | reports/within_db_logistic_metrics.csv |
| iter3_db_card_games_n | 46 | reports/within_db_logistic_metrics.csv |
| iter3_db_card_games_n_train | 36.0 | reports/within_db_logistic_metrics.csv |
| iter3_db_card_games_n_test | 10.0 | reports/within_db_logistic_metrics.csv |
| iter3_db_card_games_slow_pct | 85 | reports/within_db_logistic_metrics.csv |
| iter3_db_card_games_f1_slow | 0.888889 | reports/within_db_logistic_metrics.csv |
| iter3_db_card_games_roc_auc | 0.3125 | reports/within_db_logistic_metrics.csv |
| iter3_db_card_games_accuracy | 0.8 | reports/within_db_logistic_metrics.csv |
| iter3_db_card_games_status | ok | reports/within_db_logistic_metrics.csv |
| iter3_db_codebase_community_n | 37 | reports/within_db_logistic_metrics.csv |
| iter3_db_codebase_community_n_train | 29.0 | reports/within_db_logistic_metrics.csv |
| iter3_db_codebase_community_n_test | 8.0 | reports/within_db_logistic_metrics.csv |
| iter3_db_codebase_community_slow_pct | 92 | reports/within_db_logistic_metrics.csv |
| iter3_db_codebase_community_f1_slow | 0.933333 | reports/within_db_logistic_metrics.csv |
| iter3_db_codebase_community_roc_auc | 1.0 | reports/within_db_logistic_metrics.csv |
| iter3_db_codebase_community_accuracy | 0.875 | reports/within_db_logistic_metrics.csv |
| iter3_db_codebase_community_status | ok | reports/within_db_logistic_metrics.csv |
| iter3_db_debit_card_specializing_n | 20 | reports/within_db_logistic_metrics.csv |
| iter3_db_debit_card_specializing_n_train | 16.0 | reports/within_db_logistic_metrics.csv |
| iter3_db_debit_card_specializing_n_test | 4.0 | reports/within_db_logistic_metrics.csv |
| iter3_db_debit_card_specializing_slow_pct | 35 | reports/within_db_logistic_metrics.csv |
| iter3_db_debit_card_specializing_f1_slow | 0.0 | reports/within_db_logistic_metrics.csv |
| iter3_db_debit_card_specializing_roc_auc | 0.666667 | reports/within_db_logistic_metrics.csv |


## Iteration 4 — Per-DB regression + schema stats

| Metric | Source | Fields |
| --- | --- | --- |
| Grid | `reports/within_db_schema_metrics.csv` | `db_id`, `model`, `r2_log`, `mae_s`, … |
| Best per DB | same | row with max `r2_log` per `db_id` |
| Figures | `reports/narrative_figures/fig13_*.png`, `fig14_*.png` | after `run_phase6_figures.py` |

| bundle_key | value | source_file |
| --- | --- | --- |
| iter4_db_card_games_best_model | RF | reports/within_db_schema_metrics.csv |
| iter4_db_card_games_r2_log | 0.1889 | reports/within_db_schema_metrics.csv |
| iter4_db_card_games_mae_s | 0.209816 | reports/within_db_schema_metrics.csv |
| iter4_db_card_games_index_coverage | 0.2857142857142857 | reports/within_db_schema_metrics.csv |
| iter4_db_codebase_community_best_model | Ridge(a=10) | reports/within_db_schema_metrics.csv |
| iter4_db_codebase_community_r2_log | -5.9499 | reports/within_db_schema_metrics.csv |
| iter4_db_codebase_community_mae_s | 0.761348 | reports/within_db_schema_metrics.csv |
| iter4_db_codebase_community_index_coverage | 0.375 | reports/within_db_schema_metrics.csv |
| iter4_db_debit_card_specializing_best_model | Ridge(a=10) | reports/within_db_schema_metrics.csv |
| iter4_db_debit_card_specializing_r2_log | 0.2777 | reports/within_db_schema_metrics.csv |
| iter4_db_debit_card_specializing_mae_s | 0.275529 | reports/within_db_schema_metrics.csv |
| iter4_db_debit_card_specializing_index_coverage | 0.6666666666666666 | reports/within_db_schema_metrics.csv |
| iter4_db_european_football_2_best_model | Ridge(a=10) | reports/within_db_schema_metrics.csv |
| iter4_db_european_football_2_r2_log | -0.6733 | reports/within_db_schema_metrics.csv |
| iter4_db_european_football_2_mae_s | 0.324076 | reports/within_db_schema_metrics.csv |
| iter4_db_european_football_2_index_coverage | 0.625 | reports/within_db_schema_metrics.csv |
| iter4_db_financial_best_model | RF | reports/within_db_schema_metrics.csv |
| iter4_db_financial_r2_log | 0.0247 | reports/within_db_schema_metrics.csv |
| iter4_db_financial_mae_s | 0.512498 | reports/within_db_schema_metrics.csv |
| iter4_db_financial_index_coverage | 0.0 | reports/within_db_schema_metrics.csv |
| iter4_db_formula_1_best_model | GBM | reports/within_db_schema_metrics.csv |
| iter4_db_formula_1_r2_log | 0.7197 | reports/within_db_schema_metrics.csv |
| iter4_db_formula_1_mae_s | 0.032734 | reports/within_db_schema_metrics.csv |
| iter4_db_formula_1_index_coverage | 0.5 | reports/within_db_schema_metrics.csv |
| iter4_db_student_club_best_model | Ridge(a=1) | reports/within_db_schema_metrics.csv |
| iter4_db_student_club_r2_log | 0.6307 | reports/within_db_schema_metrics.csv |
| iter4_db_student_club_mae_s | 5.7e-05 | reports/within_db_schema_metrics.csv |
| iter4_db_student_club_index_coverage | 0.875 | reports/within_db_schema_metrics.csv |
| iter4_db_superhero_best_model | GBM | reports/within_db_schema_metrics.csv |
| iter4_db_superhero_r2_log | 0.3756 | reports/within_db_schema_metrics.csv |
| iter4_db_superhero_mae_s | 0.000624 | reports/within_db_schema_metrics.csv |
| iter4_db_superhero_index_coverage | 0.0 | reports/within_db_schema_metrics.csv |
| iter4_db_thrombosis_prediction_best_model | Ridge(a=10) | reports/within_db_schema_metrics.csv |
| iter4_db_thrombosis_prediction_r2_log | -0.7107 | reports/within_db_schema_metrics.csv |
| iter4_db_thrombosis_prediction_mae_s | 0.002268 | reports/within_db_schema_metrics.csv |


## Cross-schema regression (bridge)

Captured log: `reports/cross_schema_regression_log.txt` (from `run_full_regression.py`).

```text
Train: 310 rows from 9 databases
Test:  64 rows  | fast=55 slow=9
Test runtime range: 0.0001s - 4.1456s  mean=0.2300s

=== REGRESSION: all seen DBs -> unseen (financial + formula_1) ===
  Linear Regression: MAE(log)=2.9160  RMSE(log)=3.8583  R2(log)=-1.2795  |  MAE(s)=36756.0998  RMSE(s)=283461.1723  R2(s)=-137347871362.8284
  Ridge (alpha=1): MAE(log)=2.8807  RMSE(log)=3.7565  R2(log)=-1.1608  |  MAE(s)=17633.8931  RMSE(s)=135028.8421  R2(s)=-31166487685.9291
  Ridge (alpha=10): MAE(log)=2.6737  RMSE(log)=3.2220  R2(log)=-0.5896  |  MAE(s)=255.5448  RMSE(s)=1865.3697  R2(s)=-5947912.1784
  Lasso: MAE(log)=2.7063  RMSE(log)=3.2940  R2(log)=-0.6615  |  MAE(s)=528.1449  RMSE(s)=3956.3163  R2(s)=-26755727.7622

=== PER-DB (Ridge alpha=1) ===
  financial: n=30 fast=26 slow=4 | runtime 0.0003s-3.0621s mean=0.2202s
    MAE(log)=2.7661  R2(log)=-1.3506  |  MAE(s)=26.1392  R2(s)=-44616.5697
  formula_1: n=34 fast=29 slow=5 | runtime 0.0001s-4.1456s mean=0.2387s
    MAE(log)=2.9818  R2(log)=-1.0616  |  MAE(s)=33170.1466  R2(s)=-48825647922.5095

=== SEEN DB baseline (within-DB split, Ridge alpha=1) for comparison ===
  Seen-DB (Ridge alpha=1): MAE(log)=3.2667  R2(log)=-0.4048  |  MAE(s)=0.8354  R2(s)=-147.5524
DONE

```

## Side-by-side unseen vs seen (geometry summary)

Use `classifier_geometries_summary.csv`: `unseen_*` aligns with held-out-DB training geometry; `seen_*` with per-DB 80/20 pooled geometry.
