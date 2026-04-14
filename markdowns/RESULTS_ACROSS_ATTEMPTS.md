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

- Unseen F1: tree+global is only slightly better in 2 commits and tied in 1 commit.
- Unseen ROC-AUC: tree+global is worse in all 3 commits.
- Best unseen F1 among SQL variants: `0.5455` at `061f0ff` (tie).
- Best unseen ROC-AUC among SQL variants: global-only `0.9370` at `061f0ff`.

Practical meaning:

- Adding tree features did **not** give a clear, stable transfer win in this setup.

## 5) What does 0.8 vs 0.9 mean?

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

## 6) Summary stats with interpretation

Across commits:

- average seen-vs-unseen absolute F1 gap = `0.2595`
- average relative F1 drop when moving to unseen DB = `57.90%`

Meaning:

- about **58%** of useful slow-class performance is lost when schemas are unseen.
- this is the central technical finding.

## 7) Final takeaway

- The project result is valid and meaningful: the pipeline learns useful patterns on familiar schemas.
- The core unsolved problem is cross-schema transfer to unseen databases.
- More data alone did not reliably fix this.
- Tree+global features were not consistently better than global-only features in this rerun.

## Evidence files

- `reports/commit_rerun_metrics.csv`
- `reports/tree_ablation_commit_metrics.csv`
- `reports/model_results.txt`

## Limits

- Seen-db values come from deterministic within-db reruns and may differ from older historical scripts.
- Tree-ablation row counts are lower than full labelled counts because some SQLs fail `EXPLAIN QUERY PLAN` and are dropped.

