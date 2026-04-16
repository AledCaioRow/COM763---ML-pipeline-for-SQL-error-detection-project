# Results Across Attempts: Fixed Logistic Regression Classification

This companion summary keeps the classifier fixed to **Logistic Regression** across the same narrative commits, so the comparison is about data and pipeline changes rather than model switching.

Core commits compared:

- `1a537e0` (baseline)
- `061f0ff` (pipeline maturity update)
- `018ac87` (larger MiniDev snapshot)

Excluded from this comparison:

- `1d9a3b4` (`Codespace Version`)

## 1) What changed between commits

### `1a537e0` (baseline)

- First complete end-to-end classifier pipeline.
- Purpose: establish the reference point.

### `061f0ff` (pipeline maturity)

- Same dataset size as baseline, but better training/evaluation flow.
- Purpose: test whether process quality alone helps.

### `018ac87` (more data snapshot)

- Larger raw and labelled dataset.
- Purpose: test whether more data alone helps transfer.

## 2) Data progression

| Commit    | Raw rows | Labelled rows | Fast | Slow |
| --------- | -------- | ------------- | ---- | ---- |
| `1a537e0` | 425      | 320           | 213  | 107  |
| `061f0ff` | 425      | 320           | 213  | 107  |
| `018ac87` | 498      | 374           | 249  | 125  |

## 3) Fixed Logistic Regression on the full labelled dataset

Settings used:

- Same classifier in every commit: `Logistic Regression`
- Unseen-DB: hold out `financial`, `formula_1`
- Seen-DB: within-db unseen-query split
- Same seed (`42`)

Evidence: `reports/fixed_logistic_commit_metrics.csv`

| Commit    | Seen-DB test rows | Unseen-DB test rows | Seen-DB F1 / ROC / Acc        | Unseen-DB F1 / ROC / Acc      |
| --------- | ----------------- | ------------------- | ----------------------------- | ----------------------------- |
| `1a537e0` | 68                | 55                  | `0.3750 / 0.5958 / 0.7059`    | `0.4211 / 0.7177 / 0.8000`    |
| `061f0ff` | 69                | 51                  | `0.4000 / 0.6295 / 0.6957`    | `0.5333 / 0.7704 / 0.8627`    |
| `018ac87` | 78                | 73                  | `0.3429 / 0.5064 / 0.7051`    | `0.3200 / 0.6537 / 0.7671`    |

What this says:

- With model choice held constant, `061f0ff` is the strongest of the three commits.
- The pipeline-maturity change helped more than the later data increase for this classifier.
- The larger `018ac87` snapshot did **not** produce a better logistic-regression transfer result than `061f0ff`.
- Logistic Regression is useful as a fixed control, but it does not reach the stronger within-schema ceilings seen when the best model is allowed to vary by commit.

## 4) Matched tree-eligible fairness control with fixed Logistic Regression

This is the closest like-for-like comparison when you want both sides evaluated on the exact same tree-eligible subset.

Evidence: `reports/tree_fairness_fixed_lr_metrics.csv`

| Commit    | Matched rows | Fast / Slow | Matched global seen F1 / ROC | Tree+Global seen F1 / ROC | Matched global unseen F1 / ROC | Tree+Global unseen F1 / ROC |
| --------- | ------------ | ----------- | ---------------------------- | ------------------------- | ------------------------------ | --------------------------- |
| `1a537e0` | 211          | `196 / 15`  | `0.5000 / 0.7209`            | `0.8000 / 0.9524`         | `0.3636 / 0.7517`              | `0.4615 / 0.9252`           |
| `061f0ff` | 210          | `194 / 16`  | `0.5000 / 0.8333`            | `0.6667 / 1.0000`         | `0.3333 / 0.7519`              | `0.5455 / 0.9259`           |
| `018ac87` | 244          | `231 / 13`  | `0.0000 / 0.4354`            | `0.8571 / 1.0000`         | `0.3333 / 0.7576`              | `0.5000 / 0.8918`           |

What this says:

- On the matched subset, tree+global beats fixed logistic global on unseen F1 and unseen ROC-AUC in all 3 commits.
- The average unseen F1 lift is about `+0.16` (`0.3636 -> 0.4615`, `0.3333 -> 0.5455`, `0.3333 -> 0.5000`).
- So even when model-selection effects are removed, the tree signal still looks useful on the transfer split.

## 5) How to read the matched seen-DB numbers

- The same small-sample warning applies here as in the main tree ablation: the matched subset has only `13-16` slow queries total before the seen split.
- That means the seen-DB slow-class test count is probably in single digits, so large F1 swings can come from one or two predictions.
- The near-perfect seen ROC-AUC is the more stable result: on schemas already seen during training, ranking slow queries is close to ceiling.
- That makes the main failure mode clearer: the features carry strong within-schema signal, but that signal does not transfer reliably across schemas.

## 6) Why not add more runtime buckets?

- More buckets would fragment an already small minority class.
- In the largest snapshot there are only `125` slow queries total; splitting them into 3 or 5 tiers would make each rare class even smaller.
- That would make F1 less stable, not more informative.

## 7) Why regression is the cleaner next step

- You already have continuous runtime values, so binary labels discard information.
- Quantile labelling also drops the middle band, which reduces usable data (`498` raw rows to `374` labelled rows at `018ac87`).
- Predicting `log(runtime_seconds)` would use all timed queries and preserve ordering information that classification throws away.
- If you still need tiers for the report or UI, you can bucket the regression output afterwards.

## Evidence files

- `reports/commit_rerun_metrics.csv`
- `reports/fixed_logistic_commit_metrics.csv`
- `reports/tree_ablation_commit_metrics.csv`
- `reports/tree_fairness_fixed_lr_metrics.csv`
- `reports/tree_fairness_control_metrics.csv`
- `rerun_commit_comparison.py`

## Limits

- The fixed logistic-regression full-dataset values in Section 3 are stored in `reports/fixed_logistic_commit_metrics.csv` and were generated with the repo's consistent rerun helper.
- The matched fairness subset is very small and heavily imbalanced, so its seen-DB F1 values should be treated as supportive rather than definitive.
- The `Codespace Version` commit was excluded intentionally because the narrative comparison in this repo centers on the three non-codespace commits above.
