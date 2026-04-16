# Results: Superhero & Card\_Games — Runtime Regression (not Classification)

This file repeats the superhero + card\_games holdout experiment from `RESULTS_50Q_DATABASES.md` but replaces binary classification (fast/slow label) with **direct runtime regression** — predicting the actual `runtime_s` value using Linear Regression, Ridge, and Lasso. The target is `log(runtime_s)` because the distribution is heavily right-skewed; metrics are reported in both log and original seconds scale.

> **Note on naming:** "Logistic Regression" is a classifier. What is used here is **Linear Regression** (and regularised variants Ridge/Lasso) to predict a continuous number. Logistic regression cannot predict a runtime value.

---

## 1) Why switch to regression

- The binary fast/slow label collapses all variation into two buckets, hiding whether the model understands *how slow* a query is.
- **superhero** has zero slow queries — it literally cannot be evaluated as a classifier. Regression on `runtime_s` gives it a valid target.
- **card\_games** runtimes span 0.4ms–937ms, a 2000× range. Regression exposes whether the model tracks that variation, not just whether it crosses a threshold.

---

## 2) Setup

| Item | Detail |
|------|--------|
| Target variable | `log(runtime_s)` for training; metrics also back-transformed to seconds |
| Holdout | superhero + card\_games (100 queries total) |
| Training set | All other 9 databases (274 queries) |
| Feature variants | Global (25 SQL features), Matched Global (same), Tree+Global (+9 EXPLAIN plan features) |
| Models | Linear Regression, Ridge α=1, Ridge α=10, Lasso α=0.01 |
| Scaling | StandardScaler applied to all inputs before fitting |

Runtime ranges in the test set:

| Database   | n  | Min runtime | Max runtime | Mean runtime |
|------------|----|-------------|-------------|--------------|
| superhero  | 50 | 0.000044s   | 0.000932s   | 0.0003s      |
| card\_games | 50 | 0.000381s   | 0.936480s   | 0.2999s      |
| **Combined** | **100** | — | — | — |

---

## 3) Results: Global features (25 SQL structural features)

Trained on 274 rows from other databases, tested on 100 superhero + card\_games rows.

| Model             | MAE (log) | RMSE (log) | R² (log) | MAE (s)  | RMSE (s) | R² (s)  |
|-------------------|-----------|------------|----------|----------|----------|---------|
| Linear Regression | 3.5719    | 3.9416     | −0.2690  | 0.1563   | 0.2913   | −0.4808 |
| Ridge (α=1)       | 3.5612    | 3.9259     | −0.2589  | 0.1521   | 0.2818   | −0.3852 |
| Ridge (α=10)      | **3.5490**| **3.8787** | **−0.2288** | **0.1486** | 0.2781 | −0.3495 |
| Lasso             | 3.5552    | 3.8909     | −0.2366  | 0.1479   | **0.2781** | **−0.3492** |

**Best overall: Ridge (α=10)** by MAE(log).

---

## 4) Results: Matched Global (tree-eligible subset, SQL features only)

Since all 374 queries pass `EXPLAIN QUERY PLAN`, the matched global subset is identical to the full global subset. Results are the same.

| Model             | MAE (log) | RMSE (log) | R² (log) | MAE (s)  | RMSE (s) | R² (s)  |
|-------------------|-----------|------------|----------|----------|----------|---------|
| Linear Regression | 3.5719    | 3.9416     | −0.2690  | 0.1563   | 0.2913   | −0.4808 |
| Ridge (α=1)       | 3.5612    | 3.9259     | −0.2589  | 0.1521   | 0.2818   | −0.3852 |
| Ridge (α=10)      | 3.5490    | 3.8787     | −0.2288  | 0.1486   | 0.2781   | −0.3495 |
| Lasso             | 3.5552    | 3.8909     | −0.2366  | 0.1479   | 0.2781   | −0.3492 |

---

## 5) Results: Tree+Global (SQL features + 9 EXPLAIN plan features)

| Model             | MAE (log) | RMSE (log) | R² (log) | MAE (s)  | RMSE (s) | R² (s)  |
|-------------------|-----------|------------|----------|----------|----------|---------|
| Linear Regression | 3.5949    | 3.9441     | −0.2706  | 0.1665   | 0.3377   | −0.9893 |
| Ridge (α=1)       | 3.5498    | 3.9031     | −0.2443  | 0.1500   | 0.2793   | −0.3609 |
| Ridge (α=10)      | 3.5304    | 3.8537     | −0.2130  | 0.1503   | 0.2794   | −0.3626 |
| Lasso             | **3.5265**| 3.8642     | −0.2196  | 0.1500   | 0.2792   | −0.3603 |

**Best overall: Lasso** by MAE(log). Unregularised Linear Regression with tree features badly overfits (R²(s) = −0.99).

---

## 6) Variant comparison summary

| Variant         | Best model    | MAE (log) | R² (log)  | MAE (s)  | R² (s)   |
|-----------------|---------------|-----------|-----------|----------|----------|
| Global          | Ridge α=10    | 3.5490    | −0.2288   | 0.1486   | −0.3495  |
| Matched Global  | Ridge α=10    | 3.5490    | −0.2288   | 0.1486   | −0.3495  |
| Tree+Global     | Lasso         | 3.5265    | −0.2196   | 0.1500   | −0.3603  |

**All R² values are negative.** A negative R² means the model performs *worse* than simply predicting the training-set mean for every query. This is the regression equivalent of the near-random ROC-AUC seen in the classifier version. The model has learned runtime patterns that do not transfer to these two schemas.

Tree+Global is very marginally better on log-MAE (3.527 vs 3.549) but slightly worse on MAE in seconds (0.150 vs 0.149), so there is no meaningful win from adding plan features.

---

## 7) Per-database breakdown (Ridge α=1, global features)

### superhero — 50 queries, runtimes 0.000044s–0.000932s

| Metric         | Value       |
|----------------|-------------|
| MAE (log)      | 2.8316      |
| R² (log)       | −15.90      |
| MAE (seconds)  | 0.0058s     |
| RMSE (seconds) | 0.0090s     |
| R² (seconds)   | −1501.37    |

The model consistently **overestimates** superhero runtimes by ~10–30×. All superhero queries finish in under 1ms, but the model — trained on databases with much longer runtimes — predicts values in the 1ms–30ms range. The R² of −1501 reflects this: the predictions are systematically off by orders of magnitude even though the absolute seconds error (0.006s) looks small.

**Worst 5 predictions:**

| Difficulty   | Actual (s) | Predicted (s) | Error (s) |
|--------------|------------|---------------|-----------|
| moderate     | 0.000070   | 0.033637      | 0.033568  |
| challenging  | 0.000209   | 0.025548      | 0.025339  |
| moderate     | 0.000351   | 0.024127      | 0.023777  |
| moderate     | 0.000356   | 0.020817      | 0.020462  |
| challenging  | 0.000146   | 0.012994      | 0.012848  |

**Best 5 predictions:**

| Difficulty | Actual (s) | Predicted (s) | Error (s) |
|------------|------------|---------------|-----------|
| simple     | 0.000126   | 0.001576      | 0.001450  |
| moderate   | 0.000158   | 0.001191      | 0.001033  |
| simple     | 0.000044   | 0.001039      | 0.000995  |
| moderate   | 0.000662   | 0.000955      | 0.000293  |
| moderate   | 0.000777   | 0.000672      | 0.000105  |

Even the "best" predictions are still 10× overestimates in most cases.

---

### card\_games — 50 queries, runtimes 0.000381s–0.936480s

| Metric         | Value       |
|----------------|-------------|
| MAE (log)      | 4.2909      |
| R² (log)       | −3.55       |
| MAE (seconds)  | 0.2983s     |
| RMSE (seconds) | 0.3984s     |
| R² (seconds)   | −1.28       |

The model **completely misses the slow queries**. It predicts 1–4ms for queries that actually take 700–936ms — errors of ~700ms per query. The fast queries (under 1ms) are predicted reasonably, but those only account for 7 of the 50 queries.

**Worst 5 predictions (all slow queries):**

| Difficulty   | Actual (s) | Predicted (s) | Error (s) |
|--------------|------------|---------------|-----------|
| challenging  | 0.9365     | 0.001874      | 0.9346    |
| simple       | 0.8721     | 0.001521      | 0.8706    |
| moderate     | 0.7878     | 0.003125      | 0.7847    |
| moderate     | 0.7846     | 0.000787      | 0.7838    |
| challenging  | 0.7199     | 0.003776      | 0.7161    |

**Best 5 predictions (all fast queries):**

| Difficulty | Actual (s) | Predicted (s) | Error (s) |
|------------|------------|---------------|-----------|
| moderate   | 0.000929   | 0.002069      | 0.001140  |
| moderate   | 0.000381   | 0.001271      | 0.000891  |
| moderate   | 0.000619   | 0.001465      | 0.000846  |
| moderate   | 0.000818   | 0.001551      | 0.000733  |
| simple     | 0.000933   | 0.001243      | 0.000310  |

The pattern is clear: the model has no signal for what makes a card\_games query slow. The slow queries are structurally similar to fast ones in the features it can see (SQL tokens, joins, aggregates), but they happen to be slow for reasons internal to the card\_games schema (large tables, missing indexes, data volume effects) that the feature set does not capture.

---

## 8) Feature coefficients (Ridge α=1, global features)

Signed coefficients after StandardScaler — larger absolute value = stronger influence on predicted log(runtime):

| Feature               | Coefficient | Interpretation                              |
|-----------------------|-------------|---------------------------------------------|
| query\_length          | +1.286      | Longer SQL text → predicted slower          |
| has\_limit             | +0.961      | LIMIT present → predicted slower (surprising) |
| n\_tokens              | −0.749      | More tokens → predicted faster (offset by query\_length) |
| has\_order\_by          | −0.662      | ORDER BY → predicted faster (schema-specific artefact) |
| n\_max                 | +0.401      | MAX() aggregation → predicted slower        |
| n\_where\_predicates    | −0.352      | More WHERE conditions → predicted faster    |
| has\_group\_by          | +0.351      | GROUP BY → predicted slower                 |
| n\_avg                 | +0.291      | AVG() → predicted slower                   |
| max\_nesting\_depth     | +0.286      | Deeper nesting → predicted slower           |
| has\_subquery          | −0.269      | Subquery → predicted faster (artefact)      |

Several signs are counter-intuitive (ORDER BY predicts faster, subquery predicts faster). This is a classic sign of **schema confounding**: the model has learned correlations that happen to hold across the training schemas but reverse or disappear in superhero and card\_games.

---

## 9) Comparison with classification results

| Approach       | Variant | Best model    | Primary metric | Value  |
|----------------|---------|---------------|----------------|--------|
| Classification | Global  | Gradient Boost | F1 (slow)     | 0.1935 |
| Classification | Global  | Random Forest  | ROC-AUC       | 0.3913 |
| **Regression** | **Global** | **Ridge α=10** | **R² (log)** | **−0.2288** |
| **Regression** | **Global** | **Ridge α=10** | **MAE (s)**  | **0.1486s** |

Both approaches confirm the same thing: the model fails to transfer to these two schemas. The regression version makes this more concrete — mean error of ~0.15s driven almost entirely by card\_games slow queries being predicted as fast.

---

## 10) Takeaways

1. **Regression confirms the classifier finding.** All R² are negative — the model does no better than guessing the training mean. Cross-schema transfer to these two databases is near-random for both approaches.

2. **Superhero is a degenerate case either way.** In classification, it had no slow class. In regression, its runtimes are so uniformly tiny (< 1ms) that the model — calibrated to other schemas — consistently overestimates by 10–30×.

3. **card\_games slow queries are invisible to the feature set.** The model predicts 1–4ms for queries that take 700–937ms. The SQL structure of those slow card\_games queries looks like fast queries from other schemas.

4. **Tree+Global adds no meaningful gain** here. MAE improves by ~0.02 log-units, which is negligible and inconsistent across metrics.

5. **Higher regularisation (Ridge α=10, Lasso) consistently beats unregularised regression.** On out-of-distribution schemas, more shrinkage is safer because the learned coefficients reflect training-schema artefacts rather than general patterns.

6. **Regression is the more natural next step** for this problem (as noted in `RESULTS_ACROSS_ATTEMPTS.md` section 8) because it avoids the labelling instability seen with the quantile bucketing approach. But it does not fix the fundamental cross-schema transfer problem.

---

## Evidence files

- `data/query_dataset_features.csv`
- `run_50q_regression.py` — script that produced the numbers in this file
- `markdowns/RESULTS_50Q_DATABASES.md` — classification version of the same holdout
- `markdowns/RESULTS_ACROSS_ATTEMPTS.md` — full cross-commit analysis using financial+formula\_1 holdout
