"""
Regression on full dataset: financial + formula_1 as unseen holdout.
Used for Phase 3 of PROJECT_NARRATIVE_REPORT - switching from classifiers to regression.
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_CSV = r"C:\Users\aled_\Downloads\COM763---ML-pipeline-for-SQL-error-detection-project\data\query_dataset_features.csv"
SEED = 42
HOLDOUT_DBS = ["financial", "formula_1"]

FEATURE_COLS = [
    "n_tokens","query_length","n_joins","n_tables_approx","n_where_predicates",
    "has_group_by","has_order_by","has_having","has_distinct","has_limit",
    "has_union","n_subqueries","has_subquery","max_nesting_depth",
    "n_count","n_sum","n_avg","n_max","n_min","n_aggregations",
    "has_between","has_in_clause","has_like","has_exists","has_correlated_subquery",
]

df = pd.read_csv(DATA_CSV)
df = df[df["runtime_s"] > 0].copy()
df["log_runtime"] = np.log(df["runtime_s"])

train = df[~df["db_id"].isin(HOLDOUT_DBS)]
test  = df[ df["db_id"].isin(HOLDOUT_DBS)]

print(f"Train: {len(train)} rows from {train['db_id'].nunique()} databases")
print(f"Test:  {len(test)} rows  | fast={sum(test['label_binary']==0)} slow={sum(test['label_binary']==1)}")
print(f"Test runtime range: {test['runtime_s'].min():.4f}s - {test['runtime_s'].max():.4f}s  mean={test['runtime_s'].mean():.4f}s")
print()

models = {
    "Linear Regression": Pipeline([("sc", StandardScaler()), ("reg", LinearRegression())]),
    "Ridge (alpha=1)":   Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
    "Ridge (alpha=10)":  Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=10.0))]),
    "Lasso":             Pipeline([("sc", StandardScaler()), ("reg", Lasso(alpha=0.01, max_iter=5000))]),
}

print("=== REGRESSION: all seen DBs -> unseen (financial + formula_1) ===")
for name, m in models.items():
    m.fit(train[FEATURE_COLS], train["log_runtime"])
    preds_log = m.predict(test[FEATURE_COLS])
    preds_s   = np.exp(preds_log)
    true_s    = np.exp(test["log_runtime"].values)

    mae_log  = mean_absolute_error(test["log_runtime"].values, preds_log)
    rmse_log = np.sqrt(mean_squared_error(test["log_runtime"].values, preds_log))
    r2_log   = r2_score(test["log_runtime"].values, preds_log)
    mae_s    = mean_absolute_error(true_s, preds_s)
    rmse_s   = np.sqrt(mean_squared_error(true_s, preds_s))
    r2_s     = r2_score(true_s, preds_s)
    print(f"  {name}: MAE(log)={mae_log:.4f}  RMSE(log)={rmse_log:.4f}  R2(log)={r2_log:.4f}  |  MAE(s)={mae_s:.4f}  RMSE(s)={rmse_s:.4f}  R2(s)={r2_s:.4f}")

print()
print("=== PER-DB (Ridge alpha=1) ===")
m = Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=1.0))])
m.fit(train[FEATURE_COLS], train["log_runtime"])
for db in HOLDOUT_DBS:
    sub = test[test["db_id"] == db]
    preds_log = m.predict(sub[FEATURE_COLS])
    preds_s   = np.exp(preds_log)
    true_s    = np.exp(sub["log_runtime"].values)
    mae_log   = mean_absolute_error(sub["log_runtime"].values, preds_log)
    r2_log    = r2_score(sub["log_runtime"].values, preds_log)
    mae_s     = mean_absolute_error(true_s, preds_s)
    r2_s      = r2_score(true_s, preds_s)
    n_fast    = sum(sub["label_binary"]==0)
    n_slow    = sum(sub["label_binary"]==1)
    print(f"  {db}: n={len(sub)} fast={n_fast} slow={n_slow} | runtime {true_s.min():.4f}s-{true_s.max():.4f}s mean={true_s.mean():.4f}s")
    print(f"    MAE(log)={mae_log:.4f}  R2(log)={r2_log:.4f}  |  MAE(s)={mae_s:.4f}  R2(s)={r2_s:.4f}")

print()
print("=== SEEN DB baseline (within-DB split, Ridge alpha=1) for comparison ===")
from sklearn.model_selection import train_test_split
seen_dbs = [d for d in df["db_id"].unique() if d not in HOLDOUT_DBS]
seen_data = df[df["db_id"].isin(seen_dbs)]
X_seen = seen_data[FEATURE_COLS]
y_seen = seen_data["log_runtime"]
X_tr, X_te, y_tr, y_te = train_test_split(X_seen, y_seen, test_size=0.2, random_state=SEED)
m_seen = Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=1.0))])
m_seen.fit(X_tr, y_tr)
preds_log = m_seen.predict(X_te)
preds_s   = np.exp(preds_log)
true_s    = np.exp(y_te.values)
mae_log = mean_absolute_error(y_te.values, preds_log)
r2_log  = r2_score(y_te.values, preds_log)
mae_s   = mean_absolute_error(true_s, preds_s)
r2_s    = r2_score(true_s, preds_s)
print(f"  Seen-DB (Ridge alpha=1): MAE(log)={mae_log:.4f}  R2(log)={r2_log:.4f}  |  MAE(s)={mae_s:.4f}  R2(s)={r2_s:.4f}")
print("DONE")
