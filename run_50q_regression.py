"""
Regression analysis: predict runtime_s on superhero + card_games holdout.
Uses Linear/Ridge/Lasso regression instead of binary classification.
Compares Global, Matched Global, Tree+Global feature variants.
"""
import sqlite3, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DB_BASE  = r"C:\Users\aled_\Downloads\COM763---ML-pipeline-for-SQL-error-detection-project\Mini Dev\MINIDEV\dev_databases"
DATA_CSV = r"C:\Users\aled_\Downloads\COM763---ML-pipeline-for-SQL-error-detection-project\data\query_dataset_features.csv"
SEED     = 42
TARGET_DBS = ["superhero", "card_games"]

FEATURE_COLS = [
    "n_tokens","query_length","n_joins","n_tables_approx","n_where_predicates",
    "has_group_by","has_order_by","has_having","has_distinct","has_limit",
    "has_union","n_subqueries","has_subquery","max_nesting_depth",
    "n_count","n_sum","n_avg","n_max","n_min","n_aggregations",
    "has_between","has_in_clause","has_like","has_exists","has_correlated_subquery",
]

PLAN_KEYWORDS = [
    "SCAN","SEARCH","TEMP B-TREE","CORRELATED","CO-ROUTINE",
    "UNION","SUBQUERY","MATERIALIZE",
]

def plan_features(sql, conn):
    try:
        rows = conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
        details = [str(r[-1]).upper() for r in rows]
        feats = {"plan_n_steps": len(details)}
        for kw in PLAN_KEYWORDS:
            key = "plan_" + kw.lower().replace(" ", "_").replace("-", "_")
            feats[key] = sum(kw in d for d in details)
        return feats, True
    except Exception:
        return {}, False

df = pd.read_csv(DATA_CSV)

# Drop the tiny number of rows with zero/negative runtime to avoid log issues
df = df[df["runtime_s"] > 0].copy()
df["log_runtime"] = np.log(df["runtime_s"])

# Extract plan features
print("Extracting EXPLAIN QUERY PLAN features...")
plan_rows = []
for _, row in df.iterrows():
    db_path = f"{DB_BASE}/{row['db_id']}/{row['db_id']}.sqlite"
    try:
        conn = sqlite3.connect(db_path)
        feats, ok = plan_features(row["sql"], conn)
        conn.close()
    except Exception:
        feats, ok = {}, False
    plan_rows.append({"question_id": row["question_id"], "plan_ok": ok, **feats})

plan_df = pd.DataFrame(plan_rows)
plan_cols = [c for c in plan_df.columns if c.startswith("plan_") and c != "plan_ok"]
merged = df.merge(plan_df, on="question_id")
tree_elig = merged[merged["plan_ok"]].copy()

TREE_COLS = FEATURE_COLS + plan_cols

# Splits
train_g = df[~df["db_id"].isin(TARGET_DBS)]
test_g  = df[ df["db_id"].isin(TARGET_DBS)]
train_t = tree_elig[~tree_elig["db_id"].isin(TARGET_DBS)]
test_t  = tree_elig[ tree_elig["db_id"].isin(TARGET_DBS)]

print(f"Global: train={len(train_g)}, test={len(test_g)}")
print(f"Tree:   train={len(train_t)}, test={len(test_t)}")


def make_models():
    return {
        "Linear Regression": Pipeline([("sc", StandardScaler()), ("reg", LinearRegression())]),
        "Ridge (alpha=1)":   Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
        "Ridge (alpha=10)":  Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=10.0))]),
        "Lasso":             Pipeline([("sc", StandardScaler()), ("reg", Lasso(alpha=0.01, max_iter=5000))]),
    }


def metrics(y_true_log, y_pred_log, label):
    """Compute metrics in both log and original scale."""
    mae_log  = mean_absolute_error(y_true_log, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    r2_log   = r2_score(y_true_log, y_pred_log)

    y_true_s = np.exp(y_true_log)
    y_pred_s = np.exp(y_pred_log)
    mae_s    = mean_absolute_error(y_true_s, y_pred_s)
    rmse_s   = np.sqrt(mean_squared_error(y_true_s, y_pred_s))
    r2_s     = r2_score(y_true_s, y_pred_s)

    return {
        "label":    label,
        "mae_log":  round(mae_log,  4),
        "rmse_log": round(rmse_log, 4),
        "r2_log":   round(r2_log,   4),
        "mae_s":    round(mae_s,    4),
        "rmse_s":   round(rmse_s,   4),
        "r2_s":     round(r2_s,     4),
    }


def evaluate(X_tr, y_tr, X_te, y_te, variant_label):
    print(f"\n[{variant_label}]")
    all_results = []
    for name, m in make_models().items():
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        r = metrics(y_te.values, preds, name)
        all_results.append(r)
        print(f"  {name}: MAE(log)={r['mae_log']}  RMSE(log)={r['rmse_log']}  R²(log)={r['r2_log']}  |  MAE(s)={r['mae_s']}  RMSE(s)={r['rmse_s']}  R²(s)={r['r2_s']}")
    best = min(all_results, key=lambda x: x["mae_log"])
    print(f"  >> Best by MAE(log): {best['label']}")
    return all_results


print("\n" + "="*70)
r_global  = evaluate(train_g[FEATURE_COLS], train_g["log_runtime"], test_g[FEATURE_COLS], test_g["log_runtime"], "GLOBAL")
r_matched = evaluate(train_t[FEATURE_COLS], train_t["log_runtime"], test_t[FEATURE_COLS], test_t["log_runtime"], "MATCHED GLOBAL")
r_tree    = evaluate(train_t[TREE_COLS],    train_t["log_runtime"], test_t[TREE_COLS],    test_t["log_runtime"], "TREE+GLOBAL")

# Per-DB breakdown
print("\n" + "="*70)
print("PER-DB BREAKDOWN (Linear Regression, log target)")
m_best = Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=1.0))])
m_best.fit(train_g[FEATURE_COLS], train_g["log_runtime"])

for db in TARGET_DBS:
    sub = test_g[test_g["db_id"] == db]
    preds_log = m_best.predict(sub[FEATURE_COLS])
    preds_s   = np.exp(preds_log)
    true_s    = np.exp(sub["log_runtime"].values)

    mae_s  = mean_absolute_error(true_s, preds_s)
    rmse_s = np.sqrt(mean_squared_error(true_s, preds_s))
    r2_s   = r2_score(true_s, preds_s)
    mae_l  = mean_absolute_error(sub["log_runtime"].values, preds_log)
    r2_l   = r2_score(sub["log_runtime"].values, preds_log)

    print(f"\n  {db} (n={len(sub)})")
    print(f"    Runtime range:  {true_s.min():.4f}s — {true_s.max():.4f}s  (mean={true_s.mean():.4f}s)")
    print(f"    MAE (log):      {mae_l:.4f}")
    print(f"    R² (log):       {r2_l:.4f}")
    print(f"    MAE (seconds):  {mae_s:.4f}s")
    print(f"    RMSE (seconds): {rmse_s:.4f}s")
    print(f"    R² (seconds):   {r2_s:.4f}")

    # Show predicted vs actual sample (worst 5 + best 5 by abs error)
    comp = pd.DataFrame({
        "difficulty": sub["difficulty"].values,
        "actual_s":   true_s,
        "pred_s":     preds_s,
        "abs_err_s":  np.abs(true_s - preds_s),
    }).sort_values("abs_err_s", ascending=False)
    print(f"    Top-5 worst predictions:")
    print(comp.head(5).to_string(index=False))
    print(f"    Top-5 best predictions:")
    print(comp.tail(5).to_string(index=False))

# Coefficient inspection for global linear model
print("\n" + "="*70)
print("FEATURE COEFFICIENTS (Ridge global model, log-runtime target)")
lr_model = Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=1.0))])
lr_model.fit(train_g[FEATURE_COLS], train_g["log_runtime"])
coefs = pd.Series(lr_model.named_steps["reg"].coef_, index=FEATURE_COLS)
coefs_sorted = coefs.abs().sort_values(ascending=False)
print(coefs.loc[coefs_sorted.index].to_string())

print("\nDONE")
