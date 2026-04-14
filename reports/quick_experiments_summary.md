# Quick Experiments (System A)

| Experiment | Threshold | F1 slow | Precision slow | Recall slow | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline (best_model.joblib, threshold=0.5) | 0.500 | 0.1860 | 0.1111 | 0.5714 | 0.4610 |
| Exp1 Weighted Logistic Regression (class_weight=balanced) | 0.500 | 0.2632 | 0.1613 | 0.7143 | 0.6602 |
| Exp2 Weighted Logistic + tuned threshold (0.38) | 0.380 | 0.1587 | 0.0893 | 0.7143 | 0.6602 |
