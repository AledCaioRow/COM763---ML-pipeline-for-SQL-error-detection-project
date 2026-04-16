"""
Analysis: model performance on superhero + card_games (the two 50-query databases).
Compares: Global, Matched Global, Tree+Global.
"""
import sqlite3, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DB_BASE   = r"C:\Users\aled_\Downloads\COM763---ML-pipeline-for-SQL-error-detection-project\Mini Dev\MINIDEV\dev_databases"
DATA_CSV  = r"C:\Users\aled_\Downloads\COM763---ML-pipeline-for-SQL-error-detection-project\data\query_dataset_features.csv"
SEED      = 42
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

# Extract plan features
print("Extracting EXPLAIN QUERY PLAN features...")
plan_rows = []
for _, row in df.iterrows():
    db_id = row["db_id"]
    db_path = f"{DB_BASE}/{db_id}/{db_id}.sqlite"
    try:
        conn = sqlite3.connect(db_path)
        feats, ok = plan_features(row["sql"], conn)
        conn.close()
    except Exception:
        feats, ok = {}, False
    plan_rows.append({"question_id": row["question_id"], "plan_ok": ok, **feats})

plan_df = pd.DataFrame(plan_rows)
plan_cols = [c for c in plan_df.columns if c.startswith("plan_") and c != "plan_ok"]
print(f"  Succeeded: {plan_df['plan_ok'].sum()}/{len(plan_df)}")

merged = df.merge(plan_df, on="question_id")
tree_elig = merged[merged["plan_ok"]].copy()
print(f"  Tree-eligible: {len(tree_elig)}")
print(f"  By DB: {tree_elig['db_id'].value_counts().to_dict()}")

# Splits
train_g  = df[~df["db_id"].isin(TARGET_DBS)]
test_g   = df[ df["db_id"].isin(TARGET_DBS)]
train_t  = tree_elig[~tree_elig["db_id"].isin(TARGET_DBS)]
test_t   = tree_elig[ tree_elig["db_id"].isin(TARGET_DBS)]
TREE_COLS = FEATURE_COLS + plan_cols

print(f"\nGlobal: train={len(train_g)}, test={len(test_g)}")
print(f"Tree:   train={len(train_t)}, test={len(test_t)}")
print(f"Test label dist -> fast={sum(test_g['label_binary']==0)}, slow={sum(test_g['label_binary']==1)}")
print(f"Tree test label -> fast={sum(test_t['label_binary']==0)}, slow={sum(test_t['label_binary']==1)}")

def make_models():
    return {
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=SEED),
        "Logistic Regression": Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
        "Gradient Boosting":   GradientBoostingClassifier(random_state=SEED),
    }

def evaluate(X_tr, y_tr, X_te, y_te, label=""):
    results = {}
    for name, m in make_models().items():
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        probs = m.predict_proba(X_te)[:, 1]
        f1  = f1_score(y_te, preds, zero_division=0)
        acc = accuracy_score(y_te, preds)
        try:
            roc = roc_auc_score(y_te, probs)
        except Exception:
            roc = float("nan")
        results[name] = {"f1": round(f1, 4), "roc": round(roc, 4), "acc": round(acc, 4)}
    best = max(results, key=lambda k: results[k]["f1"])
    print(f"\n[{label}]  best={best}")
    for k, v in results.items():
        print(f"  {k}: F1={v['f1']}  ROC={v['roc']}  Acc={v['acc']}")
    return results, best

print("\n" + "="*60)
r_g,  best_g  = evaluate(train_g[FEATURE_COLS], train_g["label_binary"], test_g[FEATURE_COLS], test_g["label_binary"], "GLOBAL")
r_mg, best_mg = evaluate(train_t[FEATURE_COLS], train_t["label_binary"], test_t[FEATURE_COLS], test_t["label_binary"], "MATCHED GLOBAL")
r_t,  best_t  = evaluate(train_t[TREE_COLS],    train_t["label_binary"], test_t[TREE_COLS],    test_t["label_binary"], "TREE+GLOBAL")

# Per-DB breakdown using best global model
print("\n" + "="*60)
print("PER-DB BREAKDOWN (all 3 variants, best model per variant)")
best_rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
best_rf.fit(train_g[FEATURE_COLS], train_g["label_binary"])

for db in TARGET_DBS:
    sub = test_g[test_g["db_id"] == db]
    preds = best_rf.predict(sub[FEATURE_COLS])
    probs = best_rf.predict_proba(sub[FEATURE_COLS])[:, 1]
    f1  = f1_score(sub["label_binary"], preds, zero_division=0)
    acc = accuracy_score(sub["label_binary"], preds)
    try:
        roc = roc_auc_score(sub["label_binary"], probs)
    except Exception:
        roc = float("nan")
    n_fast = int(sum(sub["label_binary"] == 0))
    n_slow = int(sum(sub["label_binary"] == 1))
    print(f"\n  {db}  (n={len(sub)}, fast={n_fast}, slow={n_slow})")
    print(f"  Global RF: F1={f1:.4f}  ROC={roc:.4f}  Acc={acc:.4f}")
    print(classification_report(sub["label_binary"], preds, target_names=["fast","slow"], zero_division=0))

print("\nALL RESULTS SUMMARY:")
print(f"Global best={best_g}: {r_g[best_g]}")
print(f"Matched Global best={best_mg}: {r_mg[best_mg]}")
print(f"Tree+Global best={best_t}: {r_t[best_t]}")
print(f"\nPlan feature columns ({len(plan_cols)}): {plan_cols}")
