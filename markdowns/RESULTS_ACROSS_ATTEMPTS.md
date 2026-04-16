# Results Across Attempts (Explained in Plain English)

This version focuses on meaning, not just numbers: what changed between commits, why those changes were made, which model/data versions were used, and what the scores mean in practical terms.

Core commits compared:

- `1a537e0` (baseline)
- `061f0ff` (pipeline maturity update)
- `018ac87` (larger MiniDev snapshot)

## 1) What changed between commits (and why)

### `1a537e0` (baseline)

- Built the first full working pipeline end-to-end.
- Purpose: establish a clean reference point for all later comparisons.

### `061f0ff` (modeling/evaluation maturity)

- Same dataset size as baseline, but stronger training/evaluation flow.
- Purpose: improve quality by better modeling/process choices, not by adding data.
- Why important: if scores change here, it means process/model choices matter.

### `018ac87` (more data snapshot)

- Increased dataset size (raw and labelled rows grew).
- Purpose: test whether more data alone improves unseen-database transfer.
- Why important: if unseen transfer still struggles, quantity alone is not enough.

## 2) Data progression


| Commit    | Raw rows | Labelled rows | Fast | Slow |
| --------- | -------- | ------------- | ---- | ---- |
| `1a537e0` | 425      | 320           | 213  | 107  |
| `061f0ff` | 425      | 320           | 213  | 107  |
| `018ac87` | 498      | 374           | 249  | 125  |


Interpretation:

- `1a537e0 -> 061f0ff`: quality/process change, not data-size change.
- `061f0ff -> 018ac87`: data-size increase test.

## 3) Seen-DB vs Unseen-DB (main classifier)

Settings used:

- Unseen-DB: hold out `financial`, `formula_1`.
- Seen-DB: within-db unseen-query split.
- Same seed (`42`) and same model registry.


| Commit    | Unseen-DB best model + F1 / ROC / Acc | Seen-DB best model | Seen-DB F1 / ROC / Acc     |
| --------- | ------------------------------------- | ------------------ | -------------------------- |
| `1a537e0` | `XGBoost: 0.1765 / 0.5442 / 0.4909`   | Random Forest      | `0.5000 / 0.6863 / 0.6765` |
| `061f0ff` | `XGBoost: 0.2069 / 0.5148 / 0.5490`   | XGBoost            | `0.4444 / 0.6701 / 0.6377` |
| `018ac87` | `XGBoost: 0.1739 / 0.4567 / 0.4795`   | Random Forest      | `0.3913 / 0.6400 / 0.6410` |


What this says:

- Seen-db performance is always much better than unseen-db.
- Main weakness is still schema transfer (new DBs), not same-schema query variation.

## 4) SQL features: without tree vs with tree

Two SQL feature versions were tested:

- **Global-only (without tree)**: no node-tree embedding.
- **Tree+Global (with tree)**: adds plan-tree node structure to global features.

Evidence (`reports/tree_ablation_commit_metrics.csv`):


| Commit    | Extracted rows | Global-only seen F1 / unseen F1 | Tree+Global seen F1 / unseen F1 | Global-only seen ROC / unseen ROC | Tree+Global seen ROC / unseen ROC |
| --------- | -------------- | ------------------------------- | ------------------------------- | --------------------------------- | --------------------------------- |
| `1a537e0` | 211            | `0.6667 / 0.4286`               | `0.8000 / 0.4615`               | `0.9444 / 0.9320`                 | `0.9524 / 0.9252`                 |
| `061f0ff` | 210            | `0.8571 / 0.5455`               | `0.6667 / 0.5455`               | `0.9939 / 0.9370`                 | `1.0000 / 0.9259`                 |
| `018ac87` | 244            | `1.0000 / 0.4615`               | `0.8571 / 0.5000`               | `1.0000 / 0.9091`                 | `1.0000 / 0.8918`                 |


Which variant was better?

- Seen F1 on the matched extracted rows flips by commit: global-only wins in `061f0ff` and `018ac87`, while tree+global wins in `1a537e0`.
- Seen ROC-AUC is saturated for both variants in every commit (`0.9444-1.0000`), so they are effectively tied at ceiling for within-schema ranking.
- Unseen F1: tree+global is only slightly better in 2 commits and tied in 1 commit.
- Unseen ROC-AUC: tree+global is worse in all 3 commits.
- Best unseen F1 among SQL variants: `0.5455` at `061f0ff` (tie).
- Best unseen ROC-AUC among SQL variants: global-only `0.9370` at `061f0ff`.

Practical meaning:

- The seen-db F1 winner flips because the matched extracted subset is tiny: only `210-244` total rows and `13-16` slow queries before the seen split.
- That means the seen-db slow-class test count is probably in single digits, so differences like `1.00` vs `0.86` may only reflect a one-query swing.
- The near-perfect seen ROC-AUC is the more stable signal here: on schemas the model has already seen, both feature sets can rank slow queries almost perfectly.
- That makes the transfer result more interpretable: the features work within a schema context, but the signal does not generalize cleanly across schemas.
- Adding tree features did **not** give a clear, stable transfer win in this setup.

## 5) Fairness control: same query set for global and tree

This is an added control test, not a replacement for the main results above.

Why run it:

- The standalone global classifier in Section 3 used the full labelled dataset.
- The tree-capable pipeline only works on the subset of queries where `EXPLAIN QUERY PLAN` succeeds.
- So a fairer post hoc check is to rerun the global classifier on that exact same tree-eligible subset, then compare again.

Evidence (`reports/tree_fairness_control_metrics.csv`):


| Commit    | Full labelled rows | Tree-eligible matched rows | Matched global unseen best model + F1 / ROC / Acc | Tree+Global unseen F1 / ROC |
| --------- | ------------------ | -------------------------- | ------------------------------------------------- | --------------------------- |
| `1a537e0` | 320                | 211                        | `Logistic Regression: 0.3636 / 0.7517 / 0.8727`   | `0.4615 / 0.9252`           |
| `061f0ff` | 320                | 210                        | `Logistic Regression: 0.3333 / 0.7519 / 0.8431`   | `0.5455 / 0.9259`           |
| `018ac87` | 374                | 244                        | `Logistic Regression: 0.3333 / 0.7576 / 0.8904`   | `0.5000 / 0.8918`           |


What this says:

- On the matched tree-eligible subset, `tree+global` beats the matched global rerun on unseen F1 in all 3 commits.
- On the same matched subset, `tree+global` also beats the matched global rerun on unseen ROC-AUC in all 3 commits.
- That means the tree signal still looks useful after removing the extra-data advantage from the global side.

Important caution:

- This fairness-control subset is much more imbalanced than the full dataset because only a small number of slow queries survive plan extraction:
  - `1a537e0`: `196 fast / 15 slow`
  - `061f0ff`: `194 fast / 16 slow`
  - `018ac87`: `231 fast / 13 slow`
- So this test is best read as a supporting control, not the main headline benchmark.

Fixed-model check (`reports/tree_fairness_fixed_lr_metrics.csv`):

- To remove model-selection effects, a second fairness rerun fixed both sides to **Logistic Regression**.
- Result: tree+global still beats matched global on unseen F1 and ROC-AUC in all 3 commits:
  - `1a537e0`: unseen F1 `0.3636 -> 0.4615`, unseen ROC `0.7517 -> 0.9252`
  - `061f0ff`: unseen F1 `0.3333 -> 0.5455`, unseen ROC `0.7519 -> 0.9259`
  - `018ac87`: unseen F1 `0.3333 -> 0.5000`, unseen ROC `0.7576 -> 0.8918`

Quantity-only control (`reports/global_downsampled_to_tree_size_metrics.csv`):

- A separate control kept the **main global classifier unchanged** and only downsampled its dataset size to match the tree+global row count.
- This isolates the effect of **having fewer rows**, without changing the global feature family.
- Seen-DB F1 changed as follows after downsampling:
  - `1a537e0`: `0.5000 -> 0.5185`
  - `061f0ff`: `0.4444 -> 0.4000`
  - `018ac87`: `0.3913 -> 0.2759`
- So reducing the global dataset to tree size does hurt or destabilize performance in 2 of the 3 commits, but it does **not** fully explain the tree result by itself.
- The closest like-for-like tree ablation is still `reports/tree_ablation_commit_metrics.csv`, because there `global_only` and `tree+global` already use the same extracted rows.

## 6) What does 0.8 vs 0.9 mean?

These are not tiny differences.

- **F1 (slow class)**:
  - `0.90`: very reliable slow-query detection.
  - `0.80`: strong, but noticeably more misses/false alarms than `0.90`.
  - `0.50`: moderate only.
  - `0.20`: weak in practice.
- **ROC-AUC**:
  - `0.50` = random,
  - `0.60` = weak,
  - `0.70` = fair/usable,
  - `0.80+` = strong,
  - `0.90+` = excellent separation.

So yes, `0.8` vs `0.9` is a meaningful quality jump.

## 7) Summary stats with interpretation

Across commits:

- average seen-vs-unseen absolute F1 gap = `0.2595`
- average relative F1 drop when moving to unseen DB = `57.90%`

Meaning:

- about **58%** of useful slow-class performance is lost when schemas are unseen.
- this is the central technical finding.

## 8) Why not add more buckets?

- More runtime buckets would fragment an already small slow-query sample (`125` slow queries in the largest snapshot).
- Splitting those slow examples into 3 or 5 tiers would leave each rare class even smaller and make F1 estimates less stable.
- The current quantile-labelling setup also drops the middle band, which reduces usable sample size (`498 -> 374` rows at `018ac87`).
- So if the goal is to preserve more runtime information, regression on `log(runtime_seconds)` is a more natural next step than finer classification buckets.

## 9) Final takeaway

- The project result is valid and meaningful: the pipeline learns useful patterns on familiar schemas.
- The near-ceiling seen-DB ROC in the tree ablation shows there is strong within-schema signal.
- The core unsolved problem is cross-schema transfer to unseen databases.
- In other words, the features appear to encode schema-specific patterns more than schema-general ones.
- More data alone did not reliably fix this.
- Tree+global features were not consistently better than global-only features in this rerun.
- But in the new matched-subset fairness control, tree+global was stronger than the matched global rerun on unseen transfer across all 3 commits.

## Evidence files

- `reports/commit_rerun_metrics.csv`
- `reports/tree_ablation_commit_metrics.csv`
- `reports/tree_fairness_control_metrics.csv`
- `reports/tree_fairness_fixed_lr_metrics.csv`
- `reports/global_downsampled_to_tree_size_metrics.csv`
- `reports/tree_fairness_query_manifest.csv`
- `reports/model_results.txt`

## Limits

- Seen-db values come from deterministic within-db reruns and may differ from older historical scripts.
- Tree-ablation row counts are lower than full labelled counts because some SQLs fail `EXPLAIN QUERY PLAN` and are dropped.
- The matched fairness-control subset is small and heavily skewed toward fast queries, so its values should be treated as supportive rather than definitive.

