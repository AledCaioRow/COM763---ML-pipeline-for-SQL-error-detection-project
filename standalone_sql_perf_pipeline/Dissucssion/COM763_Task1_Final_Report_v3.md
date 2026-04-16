# COM 763: Advanced Machine Learning (Task 1)

# SQL Query Runtime Prediction

**Author:** Aled Caio Rowlands  
**Module:** COM 763 Advanced Machine Learning

---

### Abstract

This report documents a machine learning pipeline for SQL query runtime prediction using the BIRD Mini-Dev text-to-SQL benchmark {cite}. The project began as binary classification (fast vs slow) and shifted to regression on `log(runtime_seconds)` after the classifier proved too weak.

The core finding was a clear limitation: {Data size predominatly, creating the fact that SQL features do not appear to transfer across unseen databases. Nor in single seen databases. With extra context (databse size, rows, etc, appear to provide better evience for a workign SQL prediction mode)}. Cross-schema classification stayed near-random and cross-schema regression produced negative R². When schema statistics were added and evaluation was restricted to individual databases, performance improved sharply for well-indexed schemas, **R²(log) ≈ 0.945** on `debit_card_specializing` and **≈ 0.929** on `formula_1` with Ridge (α = 10), whilst still failing on poorly indexed schemas like `financial`. However these the test data was tiny and so users of the streamlit app can test the model themselves, creating their own test dataset.

The project had **three main iterations**. Iteration 1 trained a global classifier on seen databases and tested on held-out ones (cross-schema); it was also re-evaluated using a per-database 80/20 query split that pooled queries from all schemas into a shared train/test set, same model and features, different split geometry. Iteration 2 attempted per-database classification, which proved ill-posed given label skew. Iteration 3 trained per-database regression models with schema statistics added alongside SQL features, and is the final deployed system.

## 1. Problem Definition and System Framing

SQL query performance is difficult to judge from query text alone, yet slow queries have real costs: compute time, delayed dashboards, and increased cloud spend. A lightweight model estimating runtime before execution would have practical value in a development or analyst workflow.

The central research question was:

> **Can a model trained on queries from some databases predict runtime on databases it has never seen?**

Usefulness depends on cross-schema transfer: if a model only works on schemas it has already seen, every new deployment requires retraining.

**Why BIRD Mini-Dev?** BIRD (Li et al., 2024) is a text-to-SQL benchmark covering 11 SQLite databases across finance, sport, and education domains. Each entry in BIRD pairs a natural language question with a **gold-standard SQL query** — the correct SQL answer used to evaluate LLM outputs. The project used these gold queries directly: executing them against their corresponding SQLite databases to collect runtime measurements. This provided realistic, diverse SQL across well-structured schemas, small enough to run repeatedly on local hardware.

**Why the original cross-schema question was naive:** Because the gold queries were designed for correctness against specific schemas, their runtime is determined not just by query structure but by the data volumes and indexing decisions of each database. SQL text features — join count, aggregation count, nesting depth — capture query structure, but contain no information about table sizes or whether indexes exist. Asking an SQL-text-only model to transfer across schemas was asking it to predict something it could not observe. Had the BIRD schema metadata been incorporated from the outset — even something as simple as total row count per database — cross-database generalisation would have been far more plausible.

**Project evolution:** The framing shifted from binary classification to regression on `log(runtime_seconds)`, and then further to per-database regression with schema statistics. The final system does not attempt universal cross-schema prediction; it provides calibrated within-database models for databases it has been trained on.

**Deployed product:** The Streamlit application provides five pages. The **Live Compare** page is the core output: users select one of the 11 BIRD databases from a dropdown, paste a SQL query, and the app executes it directly against that database's SQLite file using the same 3-run median timing procedure used in training, then compares the measured runtime to the model's predicted tier side-by-side. **Model Results** displays all evaluation artefacts — confusion matrices, ROC and PR curves, feature importance, and the per-database comparison table. **Predict** accepts SQL text, extracts structural features, and returns a fast/slow classification with probability. **Data Explorer** provides interactive runtime distribution plots, class balance charts, and feature correlation heatmaps.

## 2. Data Pipeline and Feature Handling

The data source was **BIRD Mini-Dev**, covering 11 SQLite databases. Each query was executed with **3 timing runs** and a **30-second timeout**, taking the **median runtime** as the label.


| Snapshot                    | Raw rows | Labelled rows | Fast | Slow |
| --------------------------- | -------- | ------------- | ---- | ---- |
| Earlier baseline extraction | 425      | 320           | 213  | 107  |
| Current expanded snapshot   | 498      | 374           | 249  | 125  |


### SQL structural features (25 features)

Features were extracted from SQL text using `sqlparse` with no access to the database at extraction time. The 25 features cover: query size (`n_tokens`, `query_length`), join complexity (`n_joins`, `n_tables_approx`), predicate structure (`n_where_predicates`), clause flags (`has_group_by`, `has_order_by`, `has_having`, `has_distinct`, `has_limit`, `has_union`), subquery indicators (`n_subqueries`, `has_subquery`, `max_nesting_depth`, `has_correlated_subquery`), aggregation counts (`n_count`, `n_sum`, `n_avg`, `n_max`, `n_min`, `n_aggregations`), and pattern flags (`has_between`, `has_in_clause`, `has_like`, `has_exists`).

### Schema statistics features (6 features, added in Iteration 3)

After SQL-only models failed, six schema-level statistics were computed directly from each SQLite database file and appended as additional features: `schema_n_tables`, `schema_total_rows`, `schema_max_table_rows`, `schema_total_indexes`, `schema_index_coverage` (the proportion of tables with at least one index defined), and `schema_log_total_rows`. Because we train per-database, these values are constant across all rows for a given schema. They act as an environmental anchor telling the model the scale and structure of the system it is working within.

### Labelling and the regression pivot

A quantile-based scheme initially labelled the bottom 50% of runtimes `fast`, the top 25% `slow`, and dropped the middle 25% to give cleaner class boundaries. This removed 124 queries (24.9%). Whilst defensible in principle, it proved problematic: several databases had extreme skew (0% slow or 90%+ slow), making binary classification meaningless for those schemas. From Iteration 2 onwards the project pivoted to **regression on log(runtime_seconds)**. The log transformation compresses the heavy runtime tail and makes the regression target approximately Gaussian.

### The EDA failure

The fundamental error was not performing EDA on per-database distributions until the pipeline, feature extraction, training loop, and Streamlit deployment were already wired to BIRD Mini-Dev. This was a fundametal error. By the time the table below was produced, switching datasets would have required many hours of re-engineering with no guarantee of finding data of comparable structure.


| Database                | Queries | Fast | Slow | % Slow |
| ----------------------- | ------- | ---- | ---- | ------ |
| superhero               | 50      | 50   | 0    | 0%     |
| card_games              | 50      | 7    | 43   | 86%    |
| student_club            | 47      | 47   | 0    | 0%     |
| european_football_2     | 45      | 8    | 37   | 82%    |
| formula_1               | 41      | 38   | 3    | 7%     |
| codebase_community      | 35      | 3    | 32   | 91%    |
| financial               | 32      | 28   | 4    | 12%    |
| thrombosis_prediction   | 27      | 27   | 0    | 0%     |
| toxicology              | 25      | 24   | 1    | 4%     |
| debit_card_specializing | 19      | 14   | 5    | 26%    |
| california_schools      | 3       | 3    | 0    | 0%     |


`california_schools` has only 3 queries because BIRD Mini-Dev contains very few benchmark questions targeting this schema, it is one of the lightest databases in the benchmark. It was excluded from per-database modelling (minimum threshold: 15 queries).

Figure 12 — Query counts per database  
*Figure 12. Per-database query counts and class balance, showing extreme skew across schemas.*

## 3. Model Implementation and Debugging

### Iteration 1 — Cross-schema classification (held-out databases)

A global classifier was trained on all databases except `financial` and `formula_1`, then tested on those two held-out schemas. Given the SQL-only feature set, containing no information about table sizes or index coverage, this was bound to fail; the features the model needs to distinguish slow from fast queries on an unseen schema simply do not exist in the input.

**Result:** XGBoost selected by CV . Test F1 (slow) = 0.186, ROC-AUC = 0.461, accuracy = 0.52 — near-random. The original deployment question is not answerable with SQL text alone.

The same classifier setup was then re-evaluated with a different split: each database was split 80/20 at the query level and the per-database training/test portions pooled globally. This is not a different model — it tests whether query-level signal exists within schemas the model has seen. **Result:** Random Forest, F1 (slow) = 0.391, ROC-AUC = 0.64. Marginally better, but still poor. The improvement reflects proximity of test queries to their training schemas, not a meaningful gain in predictive power.

### Iteration 2 — Per-database SQL-only classification

Fitting separate classifiers per database was ill-posed: several databases have 0% slow queries under the global quantile policy (the slow class is undefined), and others are 86–91% slow. Per-database classification was abandoned in favour of regression. {However along this had no real impact. }  
  
Then provide figure}

### Iteration 3 — Per-database regression with schema statistics

For each database with at least 15 queries, a regression model is trained on 80% of that database's queries using the full **31-feature vector** (25 SQL structural + 6 schema statistics) to predict `log(runtime_seconds)`.

**Why per-database and why schema statistics?** Training within a single schema means the schema statistics are directly informative: `schema_index_coverage` — the fraction of that database's tables with at least one index defined — is the single most predictive structural feature. A well-indexed database has stable, index-driven execution paths, so query structure features reliably map to runtime because the query planner's route is predictable. A database with zero index coverage (e.g. `financial`) routes every query through a full table scan, making runtime a function of unseen row counts rather than query shape — no amount of SQL features can predict that.

**Model zoo:** Ridge (α=1), Ridge (α=10), Lasso (α=0.01), Random Forest (100 trees), Gradient Boosting — all in a `StandardScaler → model` pipeline, selected by R²(log) on the 20% held-out test set.

Figure 13 — Best within-database R²  
*Figure 13. Within-database performance varies sharply by schema; index coverage is the key discriminator.*

## 4. Experimental Evaluation and Model Selection

### Cross-schema verdict

Across both classification and regression, SQL-only models do not transfer. Cross-schema Ridge regression yields R²(log) = −1.05 — worse than predicting the mean — confirming that switching to a continuous target does not fix the transfer problem. The gains in Iteration 3 come entirely from adding schema context.

Figure 10 — R² comparison  
*Figure 10. Unseen-schema regression remains below R² = 0 for SQL-only features.*

### Within-database schema-aware results


| Database                | n   | Slow % | Index coverage | Best model  | R²(log)     | MAE (s)  |
| ----------------------- | --- | ------ | -------------- | ----------- | ----------- | -------- |
| debit_card_specializing | 19  | 26%    | 0.67           | Ridge(α=10) | **0.945**   | 0.031    |
| formula_1               | 41  | 7%     | 0.50           | Ridge(α=10) | **0.929**   | 0.152    |
| student_club            | 47  | 0%     | 0.88           | Ridge(α=10) | **0.640**   | 0.000049 |
| european_football_2     | 45  | 82%    | 0.62           | Lasso       | **0.148**   | 0.120    |
| superhero               | 50  | 0%     | 0.00           | Lasso       | 0.099       | 0.000106 |
| toxicology              | 25  | 4%     | 1.00           | RF          | 0.009       | 0.001    |
| card_games              | 50  | 86%    | 0.29           | RF          | -0.329      | 0.195    |
| codebase_community      | 35  | 91%    | 0.38           | RF          | -0.390      | 0.230    |
| financial               | 32  | 12%    | 0.00           | RF          | **-10.435** | 0.007    |


Values are best R²(log) per database from `reports/within_db_schema_metrics.csv`.

**Caveat on the top two R² figures:** `debit_card_specializing` has n=19, giving only 3–4 test rows at 20%. `formula_1` has n=41, giving roughly 8 test rows. R² computed over this few points has very high variance; the figures of 0.945 and 0.929 may partly reflect a favourable random split rather than a robust generalisation result. They should be interpreted as indicative of strong within-database signal rather than precise held-out estimates.

**Index coverage as the dividing line:** Databases with index coverage ≥ 0.50 (`debit_card_specializing`, `formula_1`, `student_club`) all achieve positive R², whilst `financial` and `superhero` — both at index coverage 0.00 — perform at or below baseline. The exception is `toxicology` (coverage 1.00, R² ≈ 0.009): the raw data shows a runtime range of 0.186 ms to 703 ms, so variance does exist. The near-zero R² is more likely a small-sample artefact — with only 25 labelled rows the test set contains just 5 queries, and the single slow outlier (0.703s) almost certainly fell into the training split, leaving the test set with only sub-6ms queries and no meaningful signal to evaluate against.

Figure 14 — Schema complexity versus R²  
*Figure 14. Index coverage predicts model success more clearly than raw database size.*

## 5. Limitations

The dominant unresolved issue throughout this project is **data size**. Even the largest single databases in the labelled snapshot yield only 9–10 test queries at an 80/20 split (`superhero` and `card_games`), and the two most promising databases are worse: `debit_card_specializing` produces **4 test rows** and `formula_1` **7 test rows**. R², MAE, and F1 computed over 4–10 samples have very high variance and cannot reliably distinguish a genuinely predictive model from a lucky split. The positive results reported in Section 4 should therefore be read as suggesting promising signal rather than confirmed generalisation.

This limitation cannot be resolved by changing the model or the features — it requires more queries executed against each database. Within the scope of this project, expanding the dataset was not feasible: BIRD Mini-Dev contains a fixed number of benchmark questions per schema, and generating additional syntactically valid and semantically meaningful SQL for each database would have been a separate, substantial undertaking.

**The Live Compare page as an organic test-data mechanism:** The deployed Streamlit application partially addresses this by turning user interaction into evaluation data. When a user inputs a SQL query on the Live Compare page, the app executes it against the SQLite database, records the actual median runtime, and runs the model prediction — producing a paired (predicted, actual) observation that extends the effective test set beyond the static BIRD snapshot. Focusing this on `debit_card_specializing` and `formula_1` — the two databases where the model has shown the strongest signal — means that each real query entered by a user is a meaningful, in-context test of model correctness under conditions that the original 4–7 test rows could not provide. Over time, a log of these human-inputted queries with measured runtimes would constitute a far more reliable evaluation corpus than any fixed academic benchmark of this size.

## 6. Deployment

The Streamlit dashboard (`streamlit_app/app.py`) is the deployed artefact of this project, hosted on Streamlit Community Cloud.

- **Live Compare:** The primary page. User selects one of the 11 BIRD SQLite databases, inputs a SQL query, and the app executes it directly against the SQLite file (3-run median, 30-second timeout) whilst running the prediction pipeline in parallel. Measured runtime and predicted tier are displayed side-by-side with absolute error in milliseconds and a tier-match verdict.
- **Model Results:** All evaluation artefacts — confusion matrix, ROC and PR curves, feature importance, CV fold scores, per-database and per-difficulty breakdowns, and the full model comparison table.
- **Predict:** Accepts SQL text, extracts the 25 structural features via the same `sqlparse` extractor used in training, and returns a fast/slow classification with probability.
- **Data Explorer:** Interactive plots of runtime distribution, class balance, feature correlation heatmap, and scatter plots coloured by database or difficulty.

The deployment should be interpreted as proof of system integration. The best per-database regression results require schema statistics of the target database; the global SQL-only classifier — exposed on the Predict page — performs near-randomly on unseen schemas, which is an honest reflection of the core finding.

- **Streamlit deployment URL:** *Insert public app URL here after publishing to Streamlit Community Cloud.*
- **GitHub repository URL:** [https://github.com/AledCaioRow/COM763---ML-pipeline-for-SQL-error-detection-project](https://github.com/AledCaioRow/COM763---ML-pipeline-for-SQL-error-detection-project)

## References

Li, J., Hui, B., Qu, G., Yang, J., Li, B., Li, B., Wang, B., Qin, B., Geng, R., Huo, N., Zhou, X., Ma, C., Huang, R., Lou, Q., Chen, Z., Zhang, Z., Li, Z., Zhu, J., Cai, T., Chen, R., Chen, X., Huang, S., Liu, K. and Zhu, Y. (2024). *Can LLM Already Serve as A Database Interface? A Big Bench for Large-Scale Database Grounded Text-to-SQLs.* Advances in Neural Information Processing Systems (NeurIPS), 36. Available at: [https://arxiv.org/abs/2305.03111](https://arxiv.org/abs/2305.03111)

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. and Duchesnay, E. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, pp.2825-2830.

Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), pp.5-32.

Friedman, J.H. (2001). *Greedy function approximation: a gradient boosting machine.* Annals of Statistics, 29(5), pp.1189-1232.

Hoerl, A.E. and Kennard, R.W. (1970). *Ridge regression: biased estimation for nonorthogonal problems.* Technometrics, 12(1), pp.55-67.

Marcus, R., Negi, P., Mao, H., Zhang, C., Alizadeh, M., Kraska, T., Papaemmanouil, O. and Tatbul, N. (2019). *Neo: A Learned Query Optimizer.* Proceedings of the VLDB Endowment, 12(11), pp.1705-1718.

Streamlit Inc. (2024). *Streamlit Documentation.* Available at: [https://docs.streamlit.io](https://docs.streamlit.io)