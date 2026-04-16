# Results: Superhero & Card\_Games — The Two 50-Query Databases

This file tests the model when **superhero** and **card\_games** are held out as unseen databases and the model trains on the remaining 274 labelled rows. It also compares three feature variants: **Global**, **Matched Global**, and **Tree+Global**.

---

## 1) Why these two databases

These are the only two databases in the dataset with exactly **50 queries each** — the largest per-database samples. Everything else has fewer (student\_club: 47, european\_football\_2: 45, down to california\_schools: 3).

| Database   | Queries | Fast | Slow | % Slow |
|------------|---------|------|------|--------|
| superhero  | 50      | 50   | 0    | 0%     |
| card\_games | 50      | 7    | 43   | 86%    |
| **Combined** | **100** | **57** | **43** | **43%** |

**Critical observation — label skew:**

- **superhero** has **zero slow queries** in this dataset. Every single one was labelled fast by the quantile method. This means binary classification evaluation for superhero alone is degenerate: there is no slow class to detect, so F1 is undefined and ROC-AUC cannot be computed.
- **card\_games** is the opposite extreme: 86% of its queries were labelled slow. The model, trained mostly on a mix, is not calibrated for this distribution.
- This skew is exactly why cross-schema transfer is the central unsolved problem in this project.

---

## 2) Train / test split

| Split           | Training rows | Test rows | Notes                                      |
|-----------------|---------------|-----------|--------------------------------------------|
| Global          | 274           | 100       | All other 9 databases used for training    |
| Matched Global  | 274           | 100       | Same rows — all 374 queries pass EXPLAIN   |
| Tree+Global     | 274           | 100       | Same rows + 9 EXPLAIN QUERY PLAN features  |

All 374 queries in the dataset successfully parsed `EXPLAIN QUERY PLAN`, so the matched-global subset is identical to the full global subset here. This differs from the original tree fairness control (sections 5–6 of `RESULTS_ACROSS_ATTEMPTS.md`) where only a fraction of queries were tree-eligible under the older `sql_runtime_predictor` extraction method.

---

## 3) Model performance: combined (superhero + card\_games)

### Global (25 SQL structural features)

| Model               | F1 (slow) | ROC-AUC | Accuracy |
|---------------------|-----------|---------|----------|
| Gradient Boosting   | **0.1935** | 0.3350  | 0.5000   |
| Random Forest       | 0.1333    | 0.3913  | 0.4800   |
| Logistic Regression | 0.0426    | 0.3239  | 0.5500   |

### Matched Global (same training rows as tree, SQL features only)

| Model               | F1 (slow) | ROC-AUC | Accuracy |
|---------------------|-----------|---------|----------|
| Gradient Boosting   | **0.1935** | 0.3350  | 0.5000   |
| Random Forest       | 0.1333    | 0.3913  | 0.4800   |
| Logistic Regression | 0.0426    | 0.3239  | 0.5500   |

*Identical to global because all rows are tree-eligible — no subset difference.*

### Tree+Global (SQL features + 9 EXPLAIN QUERY PLAN features)

| Model               | F1 (slow) | ROC-AUC | Accuracy |
|---------------------|-----------|---------|----------|
| Random Forest       | **0.1379** | 0.3858  | 0.5000   |
| Logistic Regression | 0.0435    | 0.3341  | 0.5600   |
| Gradient Boosting   | 0.1250    | 0.3152  | 0.4400   |

### Summary comparison

| Variant         | Best model          | F1 (slow) | ROC-AUC | Accuracy |
|-----------------|---------------------|-----------|---------|----------|
| Global          | Gradient Boosting   | 0.1935    | 0.3350  | 0.5000   |
| Matched Global  | Gradient Boosting   | 0.1935    | 0.3350  | 0.5000   |
| Tree+Global     | Random Forest       | 0.1379    | 0.3858  | 0.5000   |

**What this says:**

- All three variants score **below 0.20 F1** and **below 0.40 ROC-AUC** — worse than a random classifier (ROC = 0.50).
- Adding plan features did **not** improve performance here; tree+global is actually slightly worse on F1.
- Unlike the financial+formula\_1 holdout in `RESULTS_ACROSS_ATTEMPTS.md` (where unseen ROC reached 0.46–0.57), transfer to these two databases is near-random.
- The most likely reason: **superhero has no slow queries and card\_games is 86% slow** — distributions the model has never seen during training.

---

## 4) Per-database breakdown (Global Random Forest)

### superhero — 50 queries, all fast (0 slow)

| Metric    | Value |
|-----------|-------|
| F1 (slow) | 0.0000 (undefined — no slow class) |
| ROC-AUC   | N/A   |
| Accuracy  | 0.7800 |

```
              precision  recall  f1-score  support
fast               1.00    0.78      0.88       50
slow               0.00    0.00      0.00        0
accuracy                             0.78       50
```

The model predicted 11 of 50 queries as slow when none actually are. The 0.78 accuracy is entirely from correct fast predictions. There is no meaningful signal to evaluate here.

---

### card\_games — 50 queries, 7 fast / 43 slow

| Metric    | Value  |
|-----------|--------|
| F1 (slow) | 0.1633 |
| ROC-AUC   | 0.5664 |
| Accuracy  | 0.1800 |

```
              precision  recall  f1-score  support
fast               0.11    0.71      0.20        7
slow               0.67    0.09      0.16       43
accuracy                             0.18       50
```

The model finds some slow queries (precision 0.67) but recall is very low (0.09) — it only flags a handful of the 43 actual slow queries. ROC-AUC of 0.57 is marginally better than random, meaning the underlying probability estimates have *some* ordering signal even if the hard predictions are mostly wrong.

---

## 5) Comparison with the standard financial+formula\_1 holdout

For reference, the same pipeline using **financial** and **formula\_1** as holdout (from `RESULTS_ACROSS_ATTEMPTS.md`, commit `018ac87`):

| Holdout DBs               | Best model | F1 (slow) | ROC-AUC | Accuracy |
|---------------------------|------------|-----------|---------|----------|
| financial + formula\_1     | XGBoost    | 0.1739    | 0.4567  | 0.4795   |
| **superhero + card\_games** | **Gradient Boosting** | **0.1935** | **0.3350** | **0.5000** |

F1 is marginally higher here (0.19 vs 0.17), but ROC-AUC is much lower (0.34 vs 0.46). The ROC gap means that while the model occasionally gets lucky on hard predictions, its probability estimates for superhero/card\_games are more disordered than for financial/formula\_1. This reflects the extreme class imbalance in superhero (0% slow) and card\_games (86% slow).

---

## 6) Why performance is so poor on these databases

Three compounding factors:

1. **Superhero label collapse** — all 50 queries in superhero were labelled fast, so the training data implicitly learns nothing about slow patterns in that schema. At test time the model has no reference for what slow looks like there.

2. **card\_games distribution mismatch** — 86% of card\_games queries are slow, but the training label balance is roughly 70% fast / 30% slow. The model's priors push it toward fast predictions, causing very low recall for slow queries.

3. **Schema-specific patterns dominate** — the SQL structural features (joins, aggregates, nesting depth, etc.) encode patterns that vary by schema topology as much as by query complexity. A model trained on nine other schemas does not generalise cleanly to these two.

---

## 7) EXPLAIN QUERY PLAN features used in Tree+Global

All 374 queries successfully ran `EXPLAIN QUERY PLAN` against their SQLite databases. The 9 plan features added on top of the 25 global features were:

| Feature              | What it counts                                  |
|----------------------|-------------------------------------------------|
| `plan_n_steps`       | Total number of plan steps                      |
| `plan_scan`          | Full table scans                                |
| `plan_search`        | Index-based searches                            |
| `plan_temp_b_tree`   | Temporary B-tree sorts (ORDER BY / GROUP BY)    |
| `plan_correlated`    | Correlated subquery plan nodes                  |
| `plan_co_routine`    | Co-routine / CTE materialisation nodes          |
| `plan_union`         | UNION plan nodes                                |
| `plan_subquery`      | Scalar subquery nodes                           |
| `plan_materialize`   | Materialise nodes                               |

Unlike the original tree ablation (which used the `sql_runtime_predictor` QPPNet-style node encoding), these are simple keyword-count features extracted directly from the EXPLAIN text — lighter weight but also less expressive.

---

## 8) Key takeaways

- The two largest-sample databases are also the **hardest transfer targets** because of extreme label skew.
- superhero cannot be meaningfully evaluated as a binary classification target in this dataset — it needs either more data labelled as slow, or a regression approach rather than binary classification.
- card\_games ROC (0.57) suggests *some* ranking signal exists; the problem is calibration and recall, not complete blindness.
- Tree+Global showed **no consistent advantage** over global-only on this holdout, consistent with the pattern seen in `RESULTS_ACROSS_ATTEMPTS.md`.
- The central finding holds: structural features work within seen schemas but do not reliably transfer to unseen ones with different runtime distributions.

---

## Evidence files

- `data/query_dataset_features.csv`
- `reports/per_database_results.csv`
- `reports/tree_fairness_control_metrics.csv`
- `reports/tree_ablation_commit_metrics.csv`
- `run_50q_analysis.py` — script that produced the numbers in this file
