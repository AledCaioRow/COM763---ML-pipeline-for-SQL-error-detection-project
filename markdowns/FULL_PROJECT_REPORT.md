# SQL Query Runtime Prediction: Full Project Report

**Module:** COM 763 Advanced Machine Learning
**Task:** Portfolio Task 1 — Supervised Learning Pipeline
**Dataset:** BIRD Mini-Dev benchmark (11 SQLite databases, 374 labelled queries)

---

## Overview

This report documents the complete progression of a supervised machine learning pipeline designed to predict SQL query runtime. Starting from a binary classifier asking "will this query be slow?", the project evolved through six phases as the evidence revealed fundamental problems with the data, the label design, and the original cross-schema transfer assumption. Each phase represents a genuine change in understanding — not a planned experiment sequence, but a series of findings that forced the approach to change.

**The central research question:** Can a model trained on SQL queries from some databases correctly predict runtime on databases it has never seen before?

**The answer reached after six phases:** No — not with structural SQL features alone. Runtime is driven primarily by schema-level properties (table sizes, index coverage) that the query text cannot reveal. Within a database that the model knows, prediction becomes feasible when indexes are present.

---

## The Dataset

### Collection method

Each query was **timed by actually running it** against its real SQLite database. Typically 3 timing runs per query were taken with a 30-second timeout. This is slower and more expensive than scraping a query log, but every label is a measured physical observation, not an estimate.

### Size and labelling

| Stage            | Raw rows | Labelled rows | Fast | Slow |
|------------------|----------|---------------|------|------|
| Initial pipeline | 425      | 320           | 213  | 107  |
| Data expansion   | 498      | 374           | 249  | 125  |

Labelling used **quantile-based bucketing**: top 25% of runtimes labelled slow, bottom 50% labelled fast, middle 25% dropped as ambiguous. This meant **124 queries (24.9%) were discarded** — a structural problem discussed in Phase 4.

### Database breakdown

![Figure 12 -- Query counts per database](../reports/narrative_figures/fig12_query_counts_per_db.png)
*Figure 12: Query counts per database. The dataset is heavily imbalanced across schemas.*

| Database                  | Queries | Fast | Slow | % Slow |
|---------------------------|---------|------|------|--------|
| superhero                 | 50      | 50   | 0    | 0%     |
| card\_games               | 50      | 7    | 43   | 86%    |
| student\_club             | 47      | 47   | 0    | 0%     |
| european\_football\_2     | 45      | 8    | 37   | 82%    |
| formula\_1                | 41      | 38   | 3    | 7%     |
| codebase\_community       | 35      | 3    | 32   | 91%    |
| financial                 | 32      | 28   | 4    | 12%    |
| thrombosis\_prediction    | 27      | 27   | 0    | 0%     |
| toxicology                | 25      | 24   | 1    | 4%     |
| debit\_card\_specializing | 19      | 14   | 5    | 26%    |
| california\_schools       | 3       | 3    | 0    | 0%     |

### Feature set

**25 SQL structural features** extracted from query text:

| Category      | Features |
|---------------|----------|
| Size          | n\_tokens, query\_length |
| Joins/tables  | n\_joins, n\_tables\_approx |
| Clauses       | has\_group\_by, has\_order\_by, has\_having, has\_distinct, has\_limit, has\_union |
| Subqueries    | n\_subqueries, has\_subquery, max\_nesting\_depth, has\_correlated\_subquery |
| Aggregations  | n\_count, n\_sum, n\_avg, n\_max, n\_min, n\_aggregations |
| Predicates    | n\_where\_predicates, has\_between, has\_in\_clause, has\_like, has\_exists |

An additional **9 EXPLAIN QUERY PLAN features** were tested as a "Tree+Global" variant: plan step count, full table scans, index searches, temporary B-tree sorts, correlated subqueries, co-routines, UNION nodes, subquery nodes, materialise nodes.

**6 schema statistics features** were added in Phase 6: n\_tables, total\_rows, max\_table\_rows, total\_indexes, index\_coverage, log\_total\_rows.

---

---

## Phase 1 — Can the Model Generalise to Unseen Databases?

### Why this phase exists

This is the question that defines whether the tool is actually useful. If a model can only predict runtime on databases it has already seen during training, it cannot be deployed in practice — every new production database would require a full retraining cycle. The real-world value is in a model that generalises: install it on a new schema and it works immediately. Phase 1 tests exactly that.

### What was done

The pipeline held out `financial` and `formula_1` as unseen test databases and trained on the remaining 9. Four classifier types were tested: Random Forest, XGBoost, Gradient Boosting, and Logistic Regression.

### Results

![Figure 1 -- Seen vs Unseen F1 and ROC-AUC](../reports/narrative_figures/fig1_seen_vs_unseen_commits.png)
*Figure 1: Seen-DB vs unseen-DB performance across pipeline stages. The gap between bars is the transfer problem.*

![Figure 11 -- All classifier models compared](../reports/narrative_figures/fig11_all_models_comparison.png)
*Figure 11: All classifier models at final pipeline stage. No model substantially outperforms others on unseen databases.*

![Figure 6 -- Confusion matrix on the unseen holdout](../reports/narrative_figures/fig6_confusion_matrix_unseen.png)
*Figure 6: Confusion matrix on financial + formula_1 holdout. The model misses most slow queries.*

**Unseen-DB classifier results (financial + formula_1 holdout):**

| Pipeline stage    | Best model | Unseen F1 | Unseen ROC-AUC | Unseen Accuracy |
|-------------------|------------|-----------|----------------|-----------------|
| Baseline          | XGBoost    | 0.1765    | 0.5442         | 0.4909          |
| Pipeline improved | XGBoost    | 0.2069    | 0.5148         | 0.5490          |
| Data expanded     | XGBoost    | 0.1739    | 0.4567         | 0.4795          |

**Per-database breakdown:**

| Database   | n  | Accuracy | F1 (slow) | Precision | Recall |
|------------|----|----------|-----------|-----------|--------|
| financial  | 32 | 0.4375   | 0.182     | 0.111     | 0.500  |
| formula\_1 | 41 | 0.5854   | 0.190     | 0.111     | 0.667  |

### What this showed

Unseen-DB F1 peaked at 0.21 and ROC-AUC fell to 0.46 — below the level of a coin flip — at the final stage. When the model flagged a query as slow on an unseen database, it was wrong approximately **9 times out of 10** (precision = 0.111 on both databases). The model was not transferring. This is not a tuning problem: XGBoost, Random Forest, and Gradient Boosting all produced broadly similar unseen results, suggesting the failure is in the feature set, not the model choice.

---

---

## Phase 2 — Seen-DB Performance: Apparently Good, Actually Misleading

### Why this phase exists

After Phase 1 showed near-zero unseen transfer, the natural question was whether the model had learned anything at all. The simplest check is seen-DB performance: does it work on databases it has already trained on? If it fails here too, the problem is the features. If it succeeds here but fails on unseen databases, the problem is generalisation. The unexpected finding — that performance got *worse* as data improved — is itself a key discovery that reshapes the rest of the project.

### What was done

Focus moved to a within-database evaluation: train on some queries from each schema, test on other queries from the same schema. This is a weaker test than cross-schema transfer but a necessary diagnostic.

### Results

![Figure 2 -- Seen-DB F1 declining with each pipeline improvement](../reports/narrative_figures/fig2_seen_f1_decline.png)
*Figure 2: Seen-DB F1 declining as the pipeline improved and more data was added. This is the wrong direction.*

**Seen-DB classifier results:**

| Pipeline stage    | Best model    | Seen F1 | Seen ROC-AUC | Seen Accuracy |
|-------------------|---------------|---------|--------------|---------------|
| Baseline          | Random Forest | 0.5000  | 0.6863       | 0.6765        |
| Pipeline improved | XGBoost       | 0.4444  | 0.6701       | 0.6378        |
| Data expanded     | Random Forest | 0.3913  | 0.6400       | 0.6410        |

**Tree ablation — seen-DB ROC-AUC reaching ceiling:**

| Pipeline stage    | Global seen ROC | Tree+Global seen ROC |
|-------------------|-----------------|----------------------|
| Baseline          | 0.9444          | 0.9524               |
| Pipeline improved | 0.9939          | 1.0000               |
| Data expanded     | 1.0000          | 1.0000               |

### Why these numbers are misleading

**The test sets are tiny.** The seen-DB slow-class test count is only 20-30 queries total across all databases. On test sets this small, one correct or incorrect prediction swings F1 by 0.05-0.15. An F1 of 0.50 could reflect correctly identifying 3 of 5 slow queries in one lucky split — not a model that genuinely learned to detect slow queries.

**Performance fell as more data was added — the wrong direction.** Every time the pipeline improved or more data was added, seen-DB F1 went down (0.500 → 0.444 → 0.391). A model that genuinely learned the signal should improve or stay stable with more data. Declining performance when real data is added is the signature of a model that was **overfitting to specific outliers in a small test split**, not learning a general pattern.

**ROC-AUC of 1.0 is a red flag.** On the matched tree-eligible subset, seen-DB ROC-AUC reached 1.0000. Perfect discrimination on a training-adjacent evaluation with ~10-16 slow queries in the test fold is a near-certain sign of memorisation, not learning.

**The seen-vs-unseen gap quantifies the problem:** F1 drops by approximately **58%** moving from seen to unseen databases (0.445 to 0.186). A model that genuinely understood runtime complexity would not lose more than half its predictive ability just by changing schema.

---

---

## Phase 3 — Switching from Classifiers to Regression

### Why this phase exists

The binary fast/slow classifier had a structural weakness: it could not say *how much* slower a query is. Performance metrics (F1, ROC-AUC) are highly sensitive to the specific threshold and test-set composition — on a holdout with only 3-7 slow queries, one prediction changes F1 by 0.10. Switching to regression removes the threshold problem entirely, uses the full information in the runtime measurement, and makes failures more visible: instead of "F1 = 0.18", you see "MAE = 5,553 seconds" — which is unambiguously catastrophic and more informative about what went wrong.

### What was done

Regression was run on the same financial + formula_1 holdout (301 training rows, 73 test rows). The target was `log(runtime_s)` due to the heavily right-skewed runtime distribution. Four models were tested: Linear Regression, Ridge (α=1), Ridge (α=10), and Lasso.

### Results

![Figure 7 -- Predicted vs actual runtime scatter (unseen holdout)](../reports/narrative_figures/fig7_regression_unseen_scatter.png)
*Figure 7: Predicted vs actual log(runtime) on financial + formula_1 holdout. Points should cluster on the diagonal — they don't.*

**Regression results — financial + formula_1 holdout:**

| Model             | MAE (log) | RMSE (log) | R2 (log)   | MAE (s)     |
|-------------------|-----------|------------|------------|-------------|
| Linear Regression | 2.6036    | 3.6613     | -2.0381    | 12,333.98s  |
| Ridge (a=1)       | 2.5696    | 3.5528     | -1.8607    | 5,553.84s   |
| Ridge (a=10)      | **2.3846**| **3.0053** | **-1.0469**| 61.98s      |
| Lasso             | 2.4342    | 3.1485     | -1.2467    | 231.10s     |

**For comparison — seen-DB regression baseline (within-DB split, Ridge a=1):**

| Metric    | Seen-DB | Unseen-DB |
|-----------|---------|-----------|
| MAE (log) | 2.9913  | 2.5696    |
| R2 (log)  | -0.1471 | -1.8607   |

**Per-database breakdown (Ridge a=1):**

| Database   | n  | Runtime range       | MAE (log) | R2 (log)  | MAE (s)    |
|------------|----|---------------------|-----------|-----------|------------|
| financial  | 32 | 0.0003s - 1.4276s   | 2.5737    | -1.5730   | 19.33s     |
| formula\_1 | 41 | 0.0001s - 1.5830s   | 2.5665    | -2.2532   | 9,873.45s  |

### What this showed

**All R2 values are negative.** A negative R2 means the model is worse than simply predicting the training mean for every query. The seconds-scale MAE values are enormous (up to 12,333 seconds) because the model predicts fast runtimes for queries that actually take over a second — and exponentiating a large log-scale error produces an astronomical absolute error.

This confirmed what the classifier results were masking. The classifier's F1 of 0.18-0.21 was not partial learning — it was the model accidentally catching 1-2 of the 3-7 slow queries per database by chance. Regression makes that visible: the model has no continuous understanding of runtime at all on unseen schemas. Higher regularisation (Ridge a=10) helps somewhat on log scale but the fundamental signal is absent.

---

---

## Phase 4 — The Data Was Worse Than Expected

### Why this phase exists

By Phase 3, both the classifier and regression had failed on unseen databases. Before changing the model architecture again, the data itself needed interrogating. If the labels are structurally broken — encoding schema identity rather than query complexity — no model can succeed regardless of architecture. Phase 4 is a forensic analysis of the data that explains why the labels are unreliable and justifies why the problem is fundamentally harder than it first appeared.

### What was found

![Figure 4 -- The dropped middle bracket](../reports/narrative_figures/fig4_dropped_middle_pie.png)
*Figure 4: Label distribution. Nearly a quarter of all collected data was silently discarded by the quantile bucketing method.*

![Figure 5 -- Runtime distribution (raw and log scale)](../reports/narrative_figures/fig5_runtime_distribution.png)
*Figure 5: Runtime distribution. The raw scale is heavily right-skewed; even on log scale the distribution is wide and uneven across databases.*

![Figure 3 -- Label distribution per database](../reports/narrative_figures/fig3_per_db_label_skew.png)
*Figure 3: Per-database label skew. The slow percentage varies from 0% to 91% — the labels encode schema identity, not query complexity.*

### Problem A: A quarter of the data was silently dropped

| Label         | Count | % of raw data |
|---------------|-------|---------------|
| fast          | 249   | 50.0%         |
| mid (dropped) | 124   | 24.9%         |
| slow          | 125   | 25.1%         |

**124 queries — nearly a quarter of all collected data — were thrown away.** The intent was to keep labels clean by avoiding ambiguous borderline cases. The consequence: the model never saw what a borderline query looks like, the gap between classes was artificially widened, and the runtime continuum was hidden.

The runtime percentile spread shows how compressed the "fast" label is:

| Percentile | Runtime     |
|------------|-------------|
| min        | 0.000044s   |
| 25th       | 0.000333s   |
| 50th       | 0.001451s   |
| 75th       | 0.165947s   |
| max        | 1.662923s   |

The 25th-75th percentile spans 0.000333s to 0.166s — a **500x range** all compressed into the single "fast" label.

### Problem B: Labels encode schema identity, not query complexity

The per-database slow percentages vary from 0% to 91%:

| Database               | % Slow | Structural problem                        |
|------------------------|--------|-------------------------------------------|
| codebase\_community    | 91%    | Almost everything labelled slow           |
| card\_games            | 86%    | Almost everything labelled slow           |
| european\_football\_2  | 82%    | Almost everything labelled slow           |
| student\_club          | 0%     | No slow queries — contributes nothing     |
| superhero              | 0%     | No slow queries — contributes nothing     |
| thrombosis\_prediction | 0%     | No slow queries — contributes nothing     |
| california\_schools    | 0%     | No slow queries — contributes nothing     |
| formula\_1             | 7%     | Effectively all fast                      |
| toxicology             | 4%     | Effectively all fast                      |

Quantile labels were computed **globally** across all databases together. This means a "fast" query in card\_games might be slower in absolute seconds than a "slow" query in superhero. The classifier was partly learning to identify which database a query came from, not how complex the query is — which explains why feature coefficients show counter-intuitive signs (ORDER BY predicting faster, subqueries predicting faster). These are schema-identity artefacts, not genuine complexity signals.

---

---

## Phase 5 — Focused Experiment: The Two Databases with 50 Queries

### Why this phase exists

Given the label distribution problems in Phase 4, most databases are either entirely fast or entirely slow — they cannot provide a balanced test of whether the model distinguishes between the two classes. The only databases with enough queries to form a meaningful train/test split and a genuine mix of runtimes are superhero (50 queries, all fast) and card\_games (50 queries, 86% slow). Phase 5 isolates these two to ask: even in the best-case scenario, with the most data, can the model predict runtime for these schemas?

### What was done

Superhero + card\_games were held out; the remaining 274 rows were used for training. Both **classification** and **regression** were tested, and three feature variants compared:

- **Global** — 25 SQL structural features
- **Matched Global** — same (all 374 queries pass EXPLAIN, so identical to Global here)
- **Tree+Global** — adds 9 EXPLAIN QUERY PLAN features

### Classification results

| Variant        | Best model        | F1 (slow) | ROC-AUC | Accuracy |
|----------------|-------------------|-----------|---------|----------|
| Global         | Gradient Boosting | 0.1935    | 0.3350  | 0.5000   |
| Matched Global | Gradient Boosting | 0.1935    | 0.3350  | 0.5000   |
| Tree+Global    | Random Forest     | 0.1379    | 0.3858  | 0.5000   |

**Key detail — per-database classification:**

| Database   | F1 (slow) | ROC-AUC | Accuracy | Note |
|------------|-----------|---------|----------|------|
| superhero  | 0.0000    | N/A     | 0.7800   | No slow queries exist — metric undefined |
| card\_games | 0.1633   | 0.5664  | 0.1800   | High precision (0.67) but very low recall (0.09) |

ROC-AUC of 0.34-0.39 is worse than random (0.50). Tree+Global added no improvement — actually slightly worse on F1. Both holdout databases scored worse than the financial+formula\_1 holdout because of the extreme class imbalances (0% slow and 86% slow).

### Regression results

![Figure 8 -- Predicted vs actual scatter for superhero and card_games](../reports/narrative_figures/fig8_regression_50q_scatter.png)
*Figure 8: Predicted vs actual log(runtime) for superhero (left) and card_games (right). On both, predictions are clustered in a narrow band regardless of actual runtime.*

![Figure 10 -- R2 comparison across all regression experiments](../reports/narrative_figures/fig10_r2_comparison.png)
*Figure 10: R2 (log scale) across all regression experiments. All values are negative — the model is worse than a naive mean-prediction baseline on every unseen holdout.*

**Regression results — superhero + card\_games holdout:**

| Model             | MAE (log)  | RMSE (log) | R2 (log)    | MAE (s)   |
|-------------------|------------|------------|-------------|-----------|
| Linear Regression | 3.5719     | 3.9416     | -0.2690     | 0.1563s   |
| Ridge (a=1)       | 3.5612     | 3.9259     | -0.2589     | 0.1521s   |
| Ridge (a=10)      | **3.5490** | **3.8787** | **-0.2288** | 0.1486s   |
| Lasso             | 3.5552     | 3.8909     | -0.2366     | **0.1479s** |

**Per-database breakdown (Ridge a=1, global features):**

**superhero** — 50 queries, all under 1ms (0.000044s to 0.000932s):

| Metric         | Value     |
|----------------|-----------|
| MAE (log)      | 2.8316    |
| R2 (log)       | -15.90    |
| MAE (seconds)  | 0.0058s   |
| R2 (seconds)   | -1,501    |

The model overestimates superhero runtimes by 10-30x. Trained on databases where runtimes span milliseconds to seconds, it predicts 12-34ms for queries that finish in under 0.1ms.

**card\_games** — 50 queries, runtimes 0.000381s to 0.936480s:

| Metric         | Value    |
|----------------|----------|
| MAE (log)      | 4.2909   |
| R2 (log)       | -3.55    |
| MAE (seconds)  | 0.2983s  |
| R2 (seconds)   | -1.28    |

Five worst predictions:

| Actual (s) | Predicted (s) | Error (s) | Difficulty  |
|------------|---------------|-----------|-------------|
| 0.9365     | 0.0019        | 0.9346    | challenging |
| 0.8721     | 0.0015        | 0.8706    | simple      |
| 0.7878     | 0.0031        | 0.7847    | moderate    |
| 0.7846     | 0.0008        | 0.7838    | moderate    |
| 0.7199     | 0.0038        | 0.7161    | challenging |

Queries taking 720ms-937ms are predicted at 1-4ms. Critically, one of the five worst is labelled **simple** — SQL structural features do not explain why it is slow. Its slowness is driven by the card\_games schema (large tables, poor indexing) — information the current feature set cannot see.

**Feature coefficients (Ridge a=1, global features):**

![Figure 9 -- Feature coefficients](../reports/narrative_figures/fig9_feature_coefficients.png)
*Figure 9: Feature coefficients after StandardScaler. Counter-intuitive signs (ORDER BY predicts faster, subqueries predict faster) are schema-confounding artefacts, not genuine patterns.*

| Feature               | Coefficient | Expected direction | Actual |
|-----------------------|-------------|-------------------|--------|
| query\_length          | +1.286      | Longer = slower   | Correct |
| has\_limit             | +0.961      | LIMIT = faster    | Wrong |
| n\_tokens              | -0.749      | More = slower     | Wrong |
| has\_order\_by          | -0.662      | ORDER BY = slower | Wrong |
| n\_max                 | +0.401      | Aggregation = slower | Correct |
| has\_subquery          | -0.269      | Subquery = slower | Wrong |

Multiple signs are counter-intuitive. This is the definitive sign of schema confounding: the model learned correlations that hold across training schemas but reverse or disappear on superhero and card\_games.

### What Phase 5 confirmed

Even with the most data available (50 queries per database), even using regression instead of classification, even adding EXPLAIN plan features — the cross-schema transfer failure is complete. The model has no signal. The slowness of card\_games queries is driven by something the feature set cannot see.

---

---

## Phase 6 — Adding Schema Statistics: Training Within Each Database

### Why this phase exists

Phases 1-5 consistently pointed to the same root cause: the model had no knowledge of the schema it was operating on. A "simple" card\_games query takes 0.87 seconds because card\_games has 800,000 rows in its largest table and only 29% of its tables are indexed — facts completely invisible to the SQL structural features. Phase 6 tests the natural fix: add database-level schema statistics directly to the feature set. Simultaneously, it changes the evaluation strategy from cross-database to within-database. Rather than asking "does the model generalise to new schemas?", it asks "can the model explain runtime within a schema it already knows?" This is the fairer, more practically achievable test.

**Script:** `run_schema_stats_model.py`
**Output:** `reports/within_db_schema_metrics.csv`

### New features added

Six database-level statistics extracted from each SQLite database via `PRAGMA index_list()` and `SELECT COUNT(*)`:

| Feature                  | Description                                                    |
|--------------------------|----------------------------------------------------------------|
| `schema_n_tables`        | Number of tables in the database                               |
| `schema_total_rows`      | Total row count across all tables                              |
| `schema_max_table_rows`  | Largest row count in any single table                          |
| `schema_total_indexes`   | Total number of indexes defined in the database                |
| `schema_index_coverage`  | Fraction of tables that have at least one index (0 to 1)       |
| `schema_log_total_rows`  | log(schema_total_rows + 1) — log-scaled database size          |

### Schema statistics per database

| Database                  | Total rows  | Max table rows | Indexes | Index coverage |
|---------------------------|-------------|----------------|---------|----------------|
| financial                 | 1,079,680   | 1,056,320      | 0       | **0.00**       |
| card\_games               | 803,451     | 427,907        | 2       | 0.29           |
| codebase\_community       | 740,646     | 303,155        | 3       | 0.38           |
| formula\_1                | 493,267     | 400,524        | 7       | 0.50           |
| debit\_card\_specializing | 423,051     | 383,282        | 4       | **0.67**       |
| european\_football\_2     | 222,803     | 183,978        | 6       | **0.62**       |
| toxicology                | 36,922      | 18,312         | 4       | **1.00**       |
| student\_club             | 42,511      | 41,877         | 7       | **0.88**       |
| thrombosis\_prediction    | 15,252      | 13,908         | 1       | 0.33           |
| superhero                 | 10,614      | 5,825          | 0       | 0.00           |
| california\_schools       | 29,941      | 17,686         | 3       | **1.00**       |

The pattern is immediately visible: databases with large tables and low index coverage (financial, card\_games, codebase\_community) are the slow databases. Databases with small tables or full index coverage (superhero, student\_club, toxicology) are consistently fast.

### Evaluation: within-database 80/20 split

Each database was trained and tested independently — 80% of its queries used for training, 20% held for testing. Five models were tested: Ridge (α=1), Ridge (α=10), Lasso, Random Forest, and Gradient Boosting.

### Results

![Figure 13 -- Within-database R2(log) per database](../reports/narrative_figures/fig13_within_db_r2.png)
*Figure 13: Within-database R2(log) for the best model per database. Green bars = model beats naive mean (R2 > 0); red bars = model fails.*

![Figure 14 -- Schema complexity vs model R2](../reports/narrative_figures/fig14_schema_vs_r2.png)
*Figure 14: Left — index coverage vs R2. Right — database size vs R2. Index coverage predicts model success; database size alone does not.*

**Best model per database (sorted by R2, highest to lowest):**

| Database                  | n  | slow% | total\_rows | idx\_cov | Best model  | R2 (log)   | MAE (log) | MAE (s)     |
|---------------------------|----|-------|-------------|----------|-------------|------------|-----------|-------------|
| debit\_card\_specializing | 19 | 26%   | 423,051     | 0.67     | Ridge(a=10) | **+0.945** | 0.446     | 0.031s      |
| formula\_1                | 41 | 7%    | 493,267     | 0.50     | Ridge(a=10) | **+0.929** | 0.696     | 0.152s      |
| student\_club             | 47 | 0%    | 42,511      | 0.88     | Ridge(a=10) | **+0.640** | 0.172     | 0.000049s   |
| european\_football\_2     | 45 | 82%   | 222,803     | 0.62     | Lasso       | **+0.148** | 1.133     | 0.120s      |
| superhero                 | 50 | 0%    | 10,614      | 0.00     | Lasso       | +0.099     | 0.500     | 0.000106s   |
| toxicology                | 25 | 4%    | 36,922      | 1.00     | RF          | +0.009     | 0.723     | 0.001s      |
| thrombosis\_prediction    | 27 | 0%    | 15,252      | 0.33     | GBM         | -0.141     | 0.644     | 0.001s      |
| card\_games               | 50 | 86%   | 803,451     | 0.29     | RF          | -0.329     | 1.310     | 0.195s      |
| codebase\_community       | 35 | 91%   | 740,646     | 0.38     | RF          | -0.390     | 0.363     | 0.230s      |
| financial                 | 32 | 12%   | 1,079,680   | 0.00     | RF          | **-10.435**| 1.706     | 0.007s      |

### Full model breakdown per key database

**debit\_card\_specializing** (n=19, 26% slow, 423K rows, idx\_cov=0.67):

| Model       | MAE (log) | R2 (log)   | MAE (s) | R2 (s) |
|-------------|-----------|------------|---------|--------|
| Ridge(a=1)  | 0.4551    | 0.9400     | 0.024s  | 0.794  |
| Ridge(a=10) | **0.4457**| **0.9449** | 0.031s  | 0.630  |
| Lasso       | 0.6378    | 0.9082     | 0.049s  | 0.087  |
| RF          | 1.5548    | 0.3062     | 0.058s  | -0.222 |
| GBM         | 0.5870    | 0.9213     | **0.018s** | **0.887** |

**formula\_1** (n=41, 7% slow, 493K rows, idx\_cov=0.50):

| Model       | MAE (log) | R2 (log)   | MAE (s)  | R2 (s)  |
|-------------|-----------|------------|----------|---------|
| Ridge(a=1)  | 0.9714    | 0.8111     | 1.883s   | -65.28  |
| Ridge(a=10) | **0.6961**| **0.9287** | 0.152s   | **0.493** |
| Lasso       | 1.3947    | 0.5059     | 23.763s  | -16,932 |
| RF          | 1.3047    | 0.3409     | 0.225s   | -0.197  |
| GBM         | 1.5216    | -0.0057    | 0.226s   | -0.204  |

**student\_club** (n=47, 0% slow, 43K rows, idx\_cov=0.88):

| Model       | MAE (log) | R2 (log)   | MAE (s)     | R2 (s) |
|-------------|-----------|------------|-------------|--------|
| Ridge(a=1)  | 0.1873    | 0.6341     | 0.000051s   | 0.588  |
| Ridge(a=10) | **0.1715**| **0.6404** | **0.000049s** | 0.489 |
| Lasso       | 0.1719    | 0.5967     | 0.000050s   | 0.435  |
| RF          | 0.2333    | 0.1138     | 0.000067s   | -0.107 |
| GBM         | 0.3064    | -0.5680    | 0.000098s   | -1.149 |

**card\_games** (n=50, 86% slow, 803K rows, idx\_cov=0.29):

| Model       | MAE (log) | R2 (log)   | MAE (s)  | R2 (s)  |
|-------------|-----------|------------|----------|---------|
| Ridge(a=1)  | 1.7702    | -1.2863    | 0.543s   | -19.20  |
| Ridge(a=10) | 1.6127    | -0.8922    | 0.323s   | -3.16   |
| Lasso       | 1.7288    | -0.9469    | 0.451s   | -14.49  |
| RF          | **1.3097**| **-0.3285**| **0.195s** | **-0.49** |
| GBM         | 1.6976    | -1.5856    | 0.242s   | -1.42   |

**financial** (n=32, 12% slow, 1.08M rows, idx\_cov=0.00):

| Model       | MAE (log) | R2 (log)    | MAE (s)  | R2 (s)     |
|-------------|-----------|-------------|----------|------------|
| Ridge(a=1)  | 2.0674    | -28.6697    | 0.653s   | -3,618,050 |
| Ridge(a=10) | 1.6872    | -18.5858    | 0.156s   | -203,720   |
| Lasso       | 1.9783    | -22.7126    | 0.210s   | -366,189   |
| RF          | **1.7059**| **-10.435** | **0.007s** | -95.77   |
| GBM         | 2.1714    | -33.0575    | 0.146s   | -95,673    |

### What the results show

**Three databases achieve R2 > 0.60:**

`debit_card_specializing` (R2 = 0.945): 67% of tables are indexed. SQL features meaningfully predict which queries hit those indexes and which do not. Ridge regression explains 94% of runtime variance within this database.

`formula_1` (R2 = 0.929): 50% index coverage, only 7% of queries are slow. The few slow queries stand out structurally — the model can identify them. Ridge(α=10) also achieves R2(s) = 0.49, meaning it explains 49% of raw seconds variance.

`student_club` (R2 = 0.640): All queries are fast (0% slow) but runtimes span 0.000128s-0.000568s. Even within this tiny range, linear regression captures relative variation when the database is 88% indexed.

**Three databases fail (R2 < -0.3):**

`financial` (R2 = -10.44): Zero indexes, 1,079,680 total rows. With no indexes, every query involves full table scans. Runtime is governed by I/O timing and caching effects that SQL text cannot predict. The model is nearly 10x worse than predicting the mean for every query.

`codebase_community` (R2 = -0.39): 91% of queries are slow. When almost all outcomes are identical, there is no variation to model — the 3 fast queries in the training set provide insufficient contrast.

`card_games` (R2 = -0.33): 86% slow, 803K rows, 29% index coverage. Same structural problem as codebase\_community combined with large-table data volume effects the features cannot capture.

### The index coverage signal (Figure 14)

Figure 14 (left panel) shows a clear pattern: **databases with high index coverage have positive R2; databases with zero or low index coverage fail.** This is the causal story:

- When indexes are present, query runtime is governed by whether the query uses them — a property that SQL structural features (joins, WHERE predicates, subqueries) can partially capture.
- When there are no indexes, runtime is governed by full table scan timing — dependent on data volume, hardware state, and cache effects that SQL text cannot reveal.

Figure 14 (right panel) shows no clear relationship between total database size and R2. `debit_card_specializing` has 423K rows and R2 = 0.945; `financial` has 1M rows and R2 = -10.44. **Size alone does not predict model failure — index coverage mediates the effect of size.**

---

---

## Conclusion

### Summary of findings across all phases

| Phase | Question asked | Answer | R2 / F1 |
|-------|----------------|--------|---------|
| 1 | Can the model generalise to unseen databases? | No | F1 = 0.18-0.21 (worse than random on ROC) |
| 2 | Does the model at least learn seen databases? | Apparently, but it's likely overfitting | Seen F1 fell as data grew; ROC hit 1.00 |
| 3 | Does regression confirm the classifier failure? | Yes, more clearly | R2(log) = -1.05 to -2.04 on unseen holdout |
| 4 | Is the data itself the problem? | Yes — labels encode schema identity | 4/11 DBs: 0% slow; 3/11 DBs: >80% slow |
| 5 | Do the two largest databases succeed? | No | R2(log) = -0.23, classification ROC = 0.33 |
| 6 | Does within-database training + schema stats work? | Partially — depends on index coverage | R2 = 0.94 (indexed) vs -10.44 (unindexed) |

### Root cause

**The model could not see the most important information: how large the tables are and whether they have indexes.** A query on a 1,000-row table with full index coverage will almost always be fast. The same query on a 1,000,000-row table with no indexes may take seconds. The SQL text of these two queries can be identical. Without schema statistics, no feature engineering of the SQL structure can close this gap.

### What would fix it

1. **Query-level schema features** — row counts and index availability for the specific tables each query touches (not just database-level aggregates). This requires parsing the SQL to extract table names and looking up their individual row counts.

2. **More queries per database** — the current 19-50 queries per database is too few for any within-database model to generalise reliably. A minimum of 200-500 timed queries per database would be needed.

3. **More diverse databases** — 11 databases, with 4 having zero slow queries, is not enough to learn cross-schema patterns. A benchmark of 50+ databases with varied schema complexity would give cross-schema transfer a genuine chance.

4. **Retain the middle bracket** — dropping 124 rows (24.9%) removed exactly the borderline examples that help a model learn where the fast/slow boundary lies. Regression directly on `runtime_s` avoids the need for this cut.

5. **Per-database calibration** — even a schema-specific bias term added at inference time would absorb the runtime offset between schemas without requiring additional training data.

---

## Evidence Files

| File | What it contains | Produced by |
|------|-----------------|-------------|
| `data/query_dataset_features.csv` | 374 labelled queries with 25 SQL features | Pipeline |
| `reports/within_db_schema_metrics.csv` | Phase 6 within-database results (all models, all DBs) | `run_schema_stats_model.py` |
| `reports/schema_stats.csv` | Per-database schema statistics (rows, indexes) | `run_schema_stats_model.py` |
| `reports/commit_rerun_metrics.csv` | Classification results across pipeline stages | `rerun_commit_comparison.py` |
| `reports/tree_ablation_commit_metrics.csv` | Global vs Tree+Global ablation | Pipeline |
| `reports/tree_fairness_control_metrics.csv` | Matched global vs Tree+Global fairness control | Pipeline |
| `reports/per_database_results.csv` | Per-DB breakdown on financial+formula\_1 holdout | Pipeline |
| `run_schema_stats_model.py` | Phase 6 within-database regression script | -- |
| `run_50q_analysis.py` | Phase 5 classification script | -- |
| `run_50q_regression.py` | Phase 5 regression script | -- |
| `run_full_regression.py` | Phase 3 regression script | -- |
| `run_report_graphs.py` | Figures 1-12 | -- |
| `run_phase6_figures.py` | Figures 13-14 | -- |
