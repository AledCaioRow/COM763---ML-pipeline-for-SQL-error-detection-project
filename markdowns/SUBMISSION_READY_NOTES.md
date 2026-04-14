# Submission-Ready Update Notes

This file captures the high-impact updates completed now, with metrics copied from generated report files and experiment outputs.

## 1) Quick iterative experiments completed

Source: `reports/quick_experiments_summary.csv`

| Experiment | Threshold | F1 slow | Precision slow | Recall slow | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline (best_model.joblib, threshold=0.5) | 0.500 | 0.1860 | 0.1111 | 0.5714 | 0.4610 |
| Exp1 Weighted Logistic Regression (class_weight=balanced) | 0.500 | 0.2632 | 0.1613 | 0.7143 | 0.6602 |
| Exp2 Weighted Logistic + tuned threshold (0.38) | 0.380 | 0.1587 | 0.0893 | 0.7143 | 0.6602 |

Interpretation:
- Exp1 gives a clear improvement versus baseline on the target class (`slow`) F1 and recall.
- Exp2 shows threshold tuning can change class trade-offs, but in this run it reduced F1.
- This is valid iterative improvement evidence: hypothesis -> intervention -> measured outcome.

## 2) Canonical baseline metrics (use these exact numbers)

Source: `reports/model_results.txt` (current baseline run)

- Train set: `301`
- Test set: `73`
- Best model by CV F1: `XGBoost`
- Baseline test F1: `0.1860`
- Baseline ROC-AUC: `0.4610`
- Baseline accuracy: `0.52`
- Slow class precision: `0.11`
- Slow class recall: `0.57`
- Holdout databases: `financial`, `formula_1`

Use these exact values consistently in your report. Do not mix with earlier metric snapshots.

## 3) System B framing for report

Current repository evidence supports this phrasing:

- System B (`sql_runtime_predictor/`) is **designed and partially implemented** as a runtime regression pipeline.
- Core modules and configs exist, but `sql_runtime_predictor/artifacts/eval_report.json` is not currently present.
- Therefore, present System B as a future/ongoing direction unless you run it end-to-end and produce final artifacts/metrics.

Suggested sentence:

> "System B is implemented at code level as a plan-tree runtime regression pipeline, but at the submission checkpoint it remains a partially executed future direction because end-to-end training/evaluation artifacts were not finalized."

## 4) Reproducible commands (copy into report appendix)

Run baseline pipeline:

```bash
python setup_bird.py
python -u main.py
```

Run quick iterative experiments:

```bash
python rerun_quick_improvements.py
```

Run Streamlit locally:

```bash
cd streamlit_app
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## 5) Streamlit deployment URL

Fill this in before submission:

- Deployment URL: `https://<your-app-name>.streamlit.app`
- Repository URL: `<your-github-repo-link>`
- Commit/tag used for deployment: `<commit-hash-or-tag>`

