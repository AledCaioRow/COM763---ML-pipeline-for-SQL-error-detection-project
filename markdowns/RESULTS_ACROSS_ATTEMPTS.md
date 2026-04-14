# Results Across Attempts (Narrative Progression)

This is the story of how the project changed over time, with stats used as evidence rather than just decoration. I only use the core milestones here:

`1a537e0` -> `061f0ff` -> `018ac87`


## Quick snapshot of the three core checkpoints

| Commit | What stage this represents | Data size (raw / labelled) | CV winner | XGBoost held-out (F1 / ROC-AUC / Acc) | Best model on test set |
|---|---|---|---|---|---|
| `1a537e0` | First stable baseline on `main` | `425 / 320` (`fast=213`, `slow=107`) | XGBoost | `0.1765 / 0.5442 / 0.49` | Logistic Regression (`0.4211`) [recomputed] |
| `061f0ff` | Reporting + evaluation expansion checkpoint | `425 / 320` (`fast=213`, `slow=107`) | XGBoost | `0.2222 / 0.5519 / 0.59` | Logistic Regression (`0.5333`) |
| `018ac87` | MiniDev-refreshed final snapshot | `498 / 374` (`fast=249`, `slow=125`) | XGBoost | `0.1860 / 0.4610 / 0.52` | Logistic Regression (`0.3200`) |

## What changed, why we changed it, and what happened to the stats

### Baseline commit: `1a537e0` (main)

This is the point where the project became a complete working baseline: data loading, feature extraction, training, evaluation, and dashboard all existed together.

Why this matters:
- It gives a clean reference point for every later claim.
- It already uses a hard split strategy (`formula_1` and `financial` held out), so the low score is actually honest.

What the numbers say:
- XGBoost is best in CV, but held-out test is weak (`F1 0.1765`, `ROC-AUC 0.5442`).
- This already hints that the project problem is not "pick a fancier model", but "generalise across unseen schemas."

### Expansion checkpoint: `061f0ff`

This commit is important because it upgrades the evidence layer: more evaluation outputs, all-model comparisons, more artifacts saved, and better reporting depth.

Why this change was made:
- To move from "single headline metric" to a defensible experimental narrative.
- To make it easier to debug where models fail, not just whether they fail.

What changed in stats versus baseline:
- XGBoost held-out F1: `0.1765 -> 0.2222` (+0.0457)
- XGBoost ROC-AUC: `0.5442 -> 0.5519` (+0.0077)
- XGBoost Accuracy: `0.49 -> 0.59` (+0.10)
- Slow-class F1 (XGBoost): `0.18 -> 0.22`
- Best-on-test Logistic Regression also improves: `0.4211 -> 0.5333`

Interpretation:
- Data size did not change, so this looks like a quality-of-process gain (training/eval setup and rerun discipline), not a "more data fixed everything" effect.

### Final refreshed snapshot: `018ac87`

This commit brings in the larger MiniDev-based snapshot and refreshed outputs.

Why this change was made:
- To move to a larger, fresher dataset and keep the project state aligned with actual current artifacts.
- To lock in the submission-facing snapshot.

What changed in stats versus `061f0ff`:
- XGBoost held-out F1: `0.2222 -> 0.1860` (-0.0362)
- XGBoost ROC-AUC: `0.5519 -> 0.4610` (-0.0909)
- XGBoost Accuracy: `0.59 -> 0.52` (-0.07)
- Slow-class precision/recall/F1 shifts from `0.14/0.50/0.22` to `0.11/0.57/0.19`
- Best-on-test Logistic Regression drops from `0.5333` to `0.3200`

Interpretation:
- Bigger data helped coverage but did not automatically improve unseen-schema generalisation.
- The recall increase with precision drop shows the model is catching more slow queries but with more false alarms; net F1 still falls.
- This is exactly why the report should focus on trade-offs and generalisation, not just one score.

## Why the choices were sensible overall

- Keeping database-aware holdout was the right call even though it hurts metrics, because it prevents leakage and gives realistic deployment behavior.
- Expanding evaluation artifacts was necessary for scientific reporting quality.
- Refreshing to the larger MiniDev snapshot was also reasonable, even though scores dropped, because it reflects a more current and broader data state.

## Main-branch history note

`main` has only one commit (`1a537e0`) in this repo history, so progression comes from branch milestones, then mapped back into one coherent narrative.

## Limits and how they were handled

- `1a537e0` had no committed `reports/all_models_test_comparison.csv`, so that missing part was recomputed from committed data/config (same holdout and model family) to keep comparison fair.
- `reports/quick_experiments_summary.csv` is untracked in the current working tree and absent from older commits, so cross-commit Exp1/Exp2 history cannot be audited from git alone.