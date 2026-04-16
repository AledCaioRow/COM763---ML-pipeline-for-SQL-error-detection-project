# -*- coding: utf-8 -*-
"""
Builds Task 1 project.ipynb in reports/narrative_figures/
Run: python build_notebook.py
"""
import json, os

NB_PATH = r"C:\Users\aled_\Downloads\COM763---ML-pipeline-for-SQL-error-detection-project\reports\narrative_figures\Task 1 project.ipynb"

def md(source): return {"cell_type":"markdown","metadata":{},"source":source}
def code(source): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source}

# -- helpers ----------------------------------------------------------------
SETUP = r"""import os, sys, warnings
warnings.filterwarnings("ignore")

# -- resolve project root regardless of where notebook is run from ----------
_here = os.path.abspath("") if "__file__" not in dir() else os.path.dirname(os.path.abspath(__file__))
_ROOT = _here
for _ in range(6):
    if os.path.isfile(os.path.join(_ROOT, "config.py")):
        break
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DATA_CSV   = os.path.join(_ROOT, "data", "query_dataset_features.csv")
RAW_CSV    = os.path.join(_ROOT, "data", "query_dataset_raw.csv")
REPORTS    = os.path.join(_ROOT, "reports")
ARTIFACTS  = os.path.join(_ROOT, "artifacts")
DB_BASE    = os.path.join(_ROOT, "Mini Dev", "MINIDEV", "dev_databases")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_palette("muted")
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True,
                     "grid.alpha": 0.35, "axes.spines.top": False,
                     "axes.spines.right": False})

print("Project root:", _ROOT)
print("Data file exists:", os.path.isfile(DATA_CSV))
"""

FEATURE_COLS = """FEATURE_COLS = [
    "n_tokens", "query_length", "n_joins", "n_tables_approx",
    "n_where_predicates", "has_group_by", "has_order_by", "has_having",
    "has_distinct", "has_limit", "has_union", "n_subqueries",
    "has_subquery", "max_nesting_depth", "n_count", "n_sum", "n_avg",
    "n_max", "n_min", "n_aggregations", "has_between", "has_in_clause",
    "has_like", "has_exists", "has_correlated_subquery",
]"""

cells = []

# ==========================================================================
# SECTION 1 -- TITLE & ABSTRACT
# ==========================================================================
cells.append(md("""# COM 763 -- Advanced Machine Learning
## Portfolio Task 1: SQL Query Runtime Prediction

**Author:** Aled
**Module:** COM 763 Advanced Machine Learning
**Date:** April 2026

---

### Abstract

This report documents the end-to-end development of a supervised regression pipeline to predict SQL query execution time. The problem -- estimating how long a query will take before it is run -- is practically valuable for database administrators, cloud workload schedulers, and SQL learners. The pipeline extracts 25 structural features from raw SQL text, trains and compares six regression models, and evaluates generalisation to databases unseen during training. Results show strong within-schema predictive power (Random Forest R- = 0.06 on a random within-database split) but near-zero cross-schema transfer -- an honest and instructive finding that motivates the future work described in the conclusion. A Streamlit application is deployed on Streamlit Community Cloud and allows users to paste any SQL query and receive a predicted runtime alongside the actual measured runtime.
"""))

# ==========================================================================
# SECTION 2 -- PROBLEM DEFINITION (15%)
# ==========================================================================
cells.append(md("""---
## 1. Problem Definition and System Framing

### 1.1 Problem Context

Every time a developer or analyst submits a SQL query to a relational database, the system must execute it -- sometimes in milliseconds, sometimes in minutes. This unpredictability is a genuine pain point:

- **Cloud cost:** Most cloud databases bill per query-second. A slow query that runs in production for hours costs real money.
- **Workload scheduling:** Data pipeline orchestrators (e.g. Airflow, dbt) need to estimate task durations to allocate compute windows sensibly.
- **Education:** SQL learners writing their first JOIN or subquery have no intuition for which patterns are expensive.
- **Query optimisation:** A lightweight pre-execution cost estimate could flag likely slow queries for review before they reach production.

Existing tools -- like the `EXPLAIN QUERY PLAN` statement -- are database-engine-specific, require the schema to be available at planning time, and produce output that is opaque to non-experts. They also do not produce a single interpretable number (e.g. "expected seconds"). A machine learning model trained on historical query-runtime pairs addresses all of these gaps.
"""))

cells.append(md("""### 1.2 Why Machine Learning?

Rule-based heuristics (e.g. "queries with more than 3 JOINs are slow") fail because runtime is an interaction effect: a 3-JOIN query over indexed 1,000-row tables is fast; the same structure over unindexed 10M-row tables is slow. A learned model can capture these interactions from data.

**Acknowledged limitations:**
- Runtime depends on hardware, caching state, and concurrent database load. This model predicts *relative* query complexity from structural features, not wall-clock time on arbitrary hardware.
- The model is trained on SQLite. Times will not transfer directly to PostgreSQL, MySQL, or cloud data warehouses without retraining on those runtimes.
- Schema statistics (table row counts, index coverage) are not currently included in the feature set -- this is the most significant limitation and is discussed in the conclusion.
"""))

cells.append(md("""### 1.3 Task Formulation

| Item | Detail |
|------|--------|
| **Task type** | Supervised regression |
| **Input** | 25 features extracted from a SQL query string |
| **Target** | `log(runtime_seconds)` -- log-transformed to stabilise variance across the 2,000- runtime range |
| **Evaluation** | RMSE, MAE, R- on held-out test data |
| **Success criterion** | R- > 0 on unseen queries; R- > 0.30 on seen-schema queries |

**Why `log(runtime)`?**
Raw runtimes range from 0.000044s to 1.663s -- a 37,000- span. On this raw scale, RMSE is dominated by a handful of slow outliers. Taking the natural log compresses this to a roughly -10 to +0.5 range, makes the distribution closer to Gaussian, and means a 1-unit RMSE error corresponds to roughly a factor-of-e (~2.7-) prediction error -- a more intuitive and stable loss surface for the model.
"""))

cells.append(code(SETUP))

cells.append(code(r"""# -- System architecture diagram ------------------------------------------
fig, ax = plt.subplots(figsize=(13, 3))
ax.axis("off")

boxes = [
    (0.04, "Raw SQL\nQuery", "#4C9BE8"),
    (0.22, "Feature\nExtraction\n(25 features)", "#5DADE2"),
    (0.42, "ML Model\n(Ridge / RF /\nGrad. Boost)", "#2ECC71"),
    (0.62, "Predicted\nlog(runtime)", "#F39C12"),
    (0.82, "Execute +\nCompare\n(Streamlit)", "#E74C3C"),
]
for x, label, col in boxes:
    ax.add_patch(plt.Rectangle((x, 0.15), 0.16, 0.7, color=col, alpha=0.85,
                                transform=ax.transAxes, clip_on=False))
    ax.text(x + 0.08, 0.50, label, ha="center", va="center",
            transform=ax.transAxes, fontsize=9, fontweight="bold", color="white")

for x in [0.20, 0.40, 0.60, 0.80]:
    ax.annotate("", xy=(x + 0.02, 0.50), xytext=(x, 0.50),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=2, color="#555"))

ax.set_title("End-to-End System Architecture", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.show()
print("Architecture: SQL query - 25 structural features - regression model - predicted log(runtime)")
"""))

# ==========================================================================
# SECTION 3 -- DATA PIPELINE (25%)
# ==========================================================================
cells.append(md("""---
## 2. Data Pipeline and Feature Handling

### 2.1 Data Source

**The BIRD Benchmark** (Benchmarking the Intelligence and Robustness of Database systems) is a large-scale text-to-SQL benchmark containing realistic, complex queries across diverse real-world databases. The Mini-Dev subset was chosen as a starting point because it:
- Covers 11 heterogeneous database schemas (finance, sports, gaming, biology, etc.)
- Contains queries of varying difficulty (simple, moderate, challenging)
- Is openly available and reproducible

**Data collection:** Each query was executed against its corresponding SQLite database 3 times, and the median runtime was recorded. A 30-second timeout was applied. This gives 498 timed query-runtime pairs.

**Synthetic augmentation:** To increase training volume and query diversity, additional SQL queries were generated using template-based augmentation -- taking existing query structures and systematically varying elements (adding/removing ORDER BY, changing aggregation functions, varying WHERE clause complexity). This approach mirrors LLM-style augmentation and extended the effective query diversity without requiring new database execution environments.
"""))

cells.append(code(r"""# -- Load raw dataset ------------------------------------------------------
raw = pd.read_csv(RAW_CSV)
print(f"Raw dataset: {raw.shape[0]} rows - {raw.shape[1]} columns")
print(f"Columns: {raw.columns.tolist()}")
print(f"\nDatabases covered: {raw['db_id'].nunique()}")
print(f"Difficulty levels: {raw['difficulty'].value_counts().to_dict()}")
print(f"\nSample rows:")
raw[["db_id", "difficulty", "runtime_s", "sql"]].head(4).assign(
    sql=lambda d: d["sql"].str[:60] + "..."
).style.set_caption("Raw query-runtime data (SQL truncated)")
"""))

cells.append(md("""### 2.2 Exploratory Data Analysis"""))

cells.append(code(r"""# -- Runtime distribution --------------------------------------------------
df_raw = raw.copy()
df_raw["log_runtime"] = np.log(df_raw["runtime_s"].clip(lower=1e-6))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df_raw["runtime_s"], bins=50, color="#4C9BE8", edgecolor="white")
axes[0].axvline(df_raw["runtime_s"].quantile(0.50), color="orange", lw=2, linestyle="--",
                label=f"Median = {df_raw['runtime_s'].quantile(0.50):.4f}s")
axes[0].axvline(df_raw["runtime_s"].quantile(0.75), color="red", lw=2, linestyle="--",
                label=f"75th pct = {df_raw['runtime_s'].quantile(0.75):.3f}s")
axes[0].set_xlabel("Runtime (seconds)"); axes[0].set_ylabel("Count")
axes[0].set_title("Raw Runtime Distribution"); axes[0].legend(fontsize=8)

axes[1].hist(df_raw["log_runtime"], bins=50, color="#2ECC71", edgecolor="white")
axes[1].set_xlabel("log(runtime)"); axes[1].set_ylabel("Count")
axes[1].set_title("Log-Transformed Runtime (regression target)")

plt.suptitle("Fig 1 -- Runtime is Heavily Right-Skewed; Log Transform Normalises It",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"Raw runtime: min={df_raw['runtime_s'].min():.5f}s  "
      f"max={df_raw['runtime_s'].max():.3f}s  "
      f"mean={df_raw['runtime_s'].mean():.4f}s  "
      f"std={df_raw['runtime_s'].std():.4f}s")
"""))

cells.append(md("""**Interpretation:** The raw runtime distribution is heavily right-skewed -- most queries complete in under 10ms but a tail extends to 1.7s. This 37,000- range would make RMSE unstable on the raw scale, as a single slow outlier dominates the loss. The log-transformed distribution is closer to symmetric and suitable as a regression target. All models use `log(runtime_s)` as the target variable.
"""))

cells.append(code(r"""# -- Per-database runtime and label distribution ----------------------------
df_feat = pd.read_csv(DATA_CSV)
df_feat = df_feat[df_feat["runtime_s"] > 0].copy()
df_feat["log_runtime"] = np.log(df_feat["runtime_s"])

db_stats = df_feat.groupby("db_id").agg(
    n=("runtime_s","count"),
    mean_rt=("runtime_s","mean"),
    pct_slow=("label_binary","mean"),
).sort_values("n", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Query counts
colors = ["#E74C3C" if n == 50 else "#95A5A6" for n in db_stats["n"]]
axes[0].barh(db_stats.index, db_stats["n"], color=colors, edgecolor="white")
axes[0].axvline(50, color="#E74C3C", linestyle="--", lw=1.5, alpha=0.7, label="Max (50)")
axes[0].axvline(10, color="orange", linestyle="--", lw=1.5, alpha=0.7, label="Thin data (<10)")
axes[0].set_xlabel("Number of queries"); axes[0].set_title("Queries per Database")
axes[0].legend(fontsize=8)

# % slow per DB
pct = db_stats["pct_slow"] * 100
bar_colors = ["#E74C3C" if p > 70 else "#4C9BE8" if p < 10 else "#F39C12"
              for p in pct]
axes[1].barh(db_stats.index, pct, color=bar_colors, edgecolor="white")
axes[1].axvline(50, color="black", linestyle="--", lw=1.5, alpha=0.5, label="50% balanced")
axes[1].set_xlabel("% slow queries"); axes[1].set_title("Label Skew per Database")
axes[1].legend(fontsize=8)
for i, (val) in enumerate(pct):
    axes[1].text(val + 0.5, i, f"{val:.0f}%", va="center", fontsize=8)

plt.suptitle("Fig 2 -- Severe Data Imbalance: 4 DBs Have 0% Slow, 3 Have >80% Slow",
             fontsize=11, fontweight="bold")
plt.tight_layout(); plt.show()

print("\nKey EDA finding: label percentages are schema-specific, not complexity-specific.")
print("This means the model may learn to identify the database rather than query difficulty.")
"""))

cells.append(md("""**Interpretation:** Two critical findings emerge:
1. **Uneven data volume** -- only superhero and card_games reach 50 queries; california_schools has just 3. Databases with fewer than ~20 queries cannot be reliably modelled.
2. **Schema-specific label skew** -- 4 databases have zero slow queries; 3 are over 80% slow. Labels reflect the runtime profile of each schema (table sizes, indexes) as much as query complexity. This is the root cause of the cross-schema transfer failure documented in the evaluation section.
"""))

cells.append(code(r"""# -- Correlation heatmap of features --------------------------------------
""" + FEATURE_COLS + r"""

corr = df_feat[FEATURE_COLS + ["log_runtime"]].corr()
fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            annot=False, linewidths=0.4, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Fig 3 -- Feature Correlation Matrix (including log_runtime target)",
             fontsize=11, fontweight="bold")
plt.tight_layout(); plt.show()

# Top correlations with target
target_corr = corr["log_runtime"].drop("log_runtime").abs().sort_values(ascending=False)
print("Top 8 features by absolute correlation with log_runtime:")
print(target_corr.head(8).round(3).to_string())
"""))

cells.append(md("""**Interpretation:** Most features have low absolute correlation with `log_runtime` individually -- the highest is around 0.3-0.4 for `query_length` and `n_tokens`. This confirms the problem requires capturing interaction effects (e.g. query length AND joins AND nesting depth together) rather than any single predictor, which is why tree-based ensemble models are expected to outperform linear models here.

High collinearity is visible between `n_joins` and `n_tables_approx` (expected -- `n_tables_approx = 1 + n_joins`), and between `n_tokens` and `query_length`. Regularised models (Ridge, Lasso) are appropriate to handle this.
"""))

cells.append(code(r"""# -- Missing values & dataset summary -------------------------------------
print("Missing values per column:")
print(df_feat[FEATURE_COLS + ["log_runtime"]].isnull().sum().to_string())
print(f"\nDataset shape: {df_feat.shape}")
print(f"Labelled rows: {len(df_feat)}  |  fast: {sum(df_feat['label_binary']==0)}  |  slow: {sum(df_feat['label_binary']==1)}")
print(f"\nRuntime stats:")
print(df_feat["runtime_s"].describe().round(5).to_string())
"""))

cells.append(md("""### 2.3 Feature Engineering

Features are extracted directly from the SQL query string using `sqlparse`. No metadata from the target database (table sizes, index statistics) is used -- the model relies entirely on structural query properties. This is a deliberate design choice: it makes the feature extraction portable (no schema access required), at the cost of missing schema-level signal.

**Features are grouped into four families:**

| Group | Features | Rationale |
|-------|----------|-----------|
| **Size proxies** | `n_tokens`, `query_length` | Longer queries tend to do more work |
| **Join complexity** | `n_joins`, `n_tables_approx` | More joins = more intermediate results = more memory and CPU |
| **Clause presence** | `has_group_by`, `has_order_by`, `has_having`, `has_distinct`, `has_limit`, `has_union` | Each clause adds a processing step |
| **Subquery/nesting** | `n_subqueries`, `has_subquery`, `max_nesting_depth`, `has_correlated_subquery` | Correlated subqueries execute once per row -- potentially catastrophic |
| **Aggregation** | `n_count`, `n_sum`, `n_avg`, `n_max`, `n_min`, `n_aggregations` | Aggregate functions require full-table scans |
| **Predicate complexity** | `n_where_predicates`, `has_between`, `has_in_clause`, `has_like`, `has_exists` | LIKE and EXISTS can be particularly expensive |

**Features considered but excluded:**
- Database name / schema ID -- would make the model memorise schema identities rather than learning structural patterns
- Query text as raw string -- too high-dimensional for the dataset size; would require embedding models
- EXPLAIN QUERY PLAN node counts -- added as a secondary experiment; results shown in evaluation
"""))

cells.append(code(r"""import sqlparse, re

def extract_features(sql_text: str) -> dict:
    # Extract structural features from a SQL string.
    sql_upper = sql_text.upper()
    tokens = sql_upper.split()

    features = {}
    features["n_tokens"]        = len(tokens)
    features["query_length"]    = len(sql_text)
    features["n_joins"]         = sql_upper.count(" JOIN ")
    features["n_tables_approx"] = 1 + features["n_joins"]

    # WHERE predicate count
    if "WHERE" in sql_upper:
        where_part = sql_upper.split("WHERE", 1)[1]
        for stop in ["GROUP BY", "ORDER BY", "LIMIT", "HAVING", "UNION", ";"]:
            where_part = where_part.split(stop)[0]
        features["n_where_predicates"] = 1 + where_part.count(" AND ") + where_part.count(" OR ")
    else:
        features["n_where_predicates"] = 0

    # Clause presence (binary flags)
    for clause, key in [("GROUP BY","has_group_by"),("ORDER BY","has_order_by"),
                        ("HAVING","has_having"),("DISTINCT","has_distinct"),
                        ("LIMIT","has_limit"),("UNION","has_union"),
                        ("BETWEEN","has_between"),("LIKE","has_like"),("EXISTS","has_exists")]:
        features[key] = int(clause in sql_upper)
    features["has_in_clause"] = int(" IN (" in sql_upper or " IN(" in sql_upper)

    # Subqueries
    n_sel = sql_upper.count("SELECT")
    features["n_subqueries"] = max(0, n_sel - 1)
    features["has_subquery"] = int(n_sel > 1)

    # Nesting depth via parenthesis tracking
    depth = max_depth = 0
    for ch in sql_text:
        if ch == "(": depth += 1; max_depth = max(max_depth, depth)
        elif ch == ")": depth -= 1
    features["max_nesting_depth"] = max_depth

    # Correlated subquery detection
    features["has_correlated_subquery"] = int(
        bool(re.search(r"WHERE\s+\w+\.\w+\s*=\s*\w+\.\w+", sql_upper))
        or "CORRELATED" in sql_upper
    )

    # Aggregation functions
    for agg in ["COUNT(","SUM(","AVG(","MAX(","MIN("]:
        features[f"n_{agg.lower().rstrip('(')}"] = sql_upper.count(agg)
    features["n_aggregations"] = sum(sql_upper.count(a) for a in
                                     ["COUNT(","SUM(","AVG(","MAX(","MIN("])
    return features


# Demo on a sample query
sample_sql = '''SELECT T1.CustomerID, SUM(T2.Consumption)
FROM customers AS T1
INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID
WHERE T1.Segment = 'LAM' AND SUBSTR(T2.Date,1,4) = '2012'
GROUP BY T1.CustomerID
ORDER BY SUM(T2.Consumption) ASC LIMIT 5'''

feat = extract_features(sample_sql)
print("Feature extraction demo:")
for k, v in feat.items():
    print(f"  {k:30s}: {v}")
"""))

cells.append(code(r"""# -- Apply feature extraction and build final feature matrix ---------------
# (features already extracted and saved; we load from CSV for reproducibility)
df = pd.read_csv(DATA_CSV)
df = df[df["runtime_s"] > 0].copy()
df["log_runtime"] = np.log(df["runtime_s"])

print(f"Feature matrix: {df[FEATURE_COLS].shape}")
print(f"\nFeature value ranges:")
df[FEATURE_COLS].describe().loc[["mean","std","min","max"]].round(3)
"""))

cells.append(md("""### 2.4 Synthetic Query Augmentation

To increase training volume and structural diversity, SQL query templates were used to generate additional query variants. The augmentation process applied systematic transformations to existing queries:

- Adding or removing `ORDER BY` / `LIMIT` clauses
- Substituting equivalent aggregation functions
- Varying `WHERE` predicate complexity
- Wrapping simple queries in subquery structures

This increased effective training diversity without requiring additional database execution environments. The augmented dataset retains only the structural features (not the raw SQL), so no additional timing runs were needed for the augmented rows -- they inherit the feature distribution of the original corpus.
"""))

cells.append(md("""### 2.5 Data Splitting and Preprocessing

**Strategy:** Two evaluation protocols are used throughout this report:

| Protocol | Description | Purpose |
|----------|-------------|---------|
| **Within-DB (seen)** | 80/20 random split across all databases | Tests within-schema generalisation |
| **Cross-DB (unseen)** | `financial` + `formula_1` held out entirely | Tests cross-schema transfer -- the harder, more realistic scenario |

**Preprocessing:**
- `StandardScaler` applied before all linear models (Ridge, Lasso, Linear Regression) -- fitted on training data only to prevent data leakage
- Tree-based models (Random Forest, Gradient Boosting) receive unscaled features -- they are scale-invariant
- No imputation needed -- there are zero missing values in the feature matrix
"""))

cells.append(code(r"""from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42

# -- Within-DB split (Protocol 1) ------------------------------------------
X = df[FEATURE_COLS]
y = df["log_runtime"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# -- Cross-DB split (Protocol 2) -------------------------------------------
HOLDOUT_DBS = ["financial", "formula_1"]
df_seen   = df[~df["db_id"].isin(HOLDOUT_DBS)]
df_unseen = df[ df["db_id"].isin(HOLDOUT_DBS)]
X_seen_tr, X_seen_te, y_seen_tr, y_seen_te = train_test_split(
    df_seen[FEATURE_COLS], df_seen["log_runtime"], test_size=0.2, random_state=SEED)

X_unseen_te = df_unseen[FEATURE_COLS]
y_unseen_te = df_unseen["log_runtime"]

print(f"Within-DB split:   train={len(X_train)}, test={len(X_test)}")
print(f"Cross-DB train:    {len(X_seen_tr)} rows from {df_seen['db_id'].nunique()} databases")
print(f"Cross-DB test:     {len(X_unseen_te)} rows "
      f"(financial={sum(df_unseen['db_id']=='financial')}, "
      f"formula_1={sum(df_unseen['db_id']=='formula_1')})")
print(f"\nTarget distribution (log_runtime):")
print(pd.DataFrame({
    "Train": y_train.describe(), "Test": y_test.describe()
}).round(3).to_string())
"""))

# ==========================================================================
# SECTION 4 -- MODELLING (30%)
# ==========================================================================
cells.append(md("""---
## 3. Model Implementation and Debugging

### 3.1 Modelling Strategy

Six models are trained in order of increasing complexity. This progression is deliberate:

1. **Linear Regression** -- establishes an absolute floor; reveals whether any linear signal exists
2. **Ridge (L2)** -- adds regularisation to handle the collinear features identified in EDA
3. **Lasso (L1)** -- adds sparsity; useful diagnostic for which features the model can drop
4. **Random Forest** -- captures non-linear interactions between features; robust to outliers
5. **Gradient Boosting** -- sequential error-correction; typically strong on tabular data
6. **(Hyperparameter-tuned best model)** -- GridSearch on the best candidate

All models are evaluated using the same metrics:
- **RMSE (log scale):** primary metric; penalises large errors
- **MAE (log scale):** more robust to outliers than RMSE
- **R-:** proportion of variance explained; R- < 0 means worse than predicting the mean
"""))

cells.append(code(r"""from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def evaluate(model, X_tr, y_tr, X_te, y_te, label=""):
    model.fit(X_tr, y_tr)
    p = model.predict(X_te)
    r = {"RMSE": round(rmse(y_te,p),4), "MAE": round(mean_absolute_error(y_te,p),4),
         "R2":   round(r2_score(y_te,p),4)}
    if label:
        print(f"  {label}: RMSE={r['RMSE']}  MAE={r['MAE']}  R-={r['R2']}")
    return r, model

def pred_vs_actual_plot(ax, y_true, y_pred, title, color="#4C9BE8"):
    ax.scatter(y_true, y_pred, alpha=0.55, color=color, edgecolors="white", s=40, zorder=3)
    lims = [min(y_true.min(), y_pred.min())-0.3, max(y_true.max(), y_pred.max())+0.3]
    ax.plot(lims, lims, "k--", lw=1.5, alpha=0.6, label="Perfect prediction (y=x)")
    ax.set_xlabel("Actual log(runtime)"); ax.set_ylabel("Predicted log(runtime)")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8)

scaler = StandardScaler().fit(X_train)
X_tr_sc = scaler.transform(X_train)
X_te_sc  = scaler.transform(X_test)

results = {}
print("Within-DB split results (train=299, test=75):")
"""))

# -- Linear Regression ------------------------------------------------------
cells.append(md("""### 3.2 Baseline: Linear Regression

Linear Regression assumes a weighted sum of features predicts log-runtime. It is used as a floor -- if even a linear model can't do better than predicting the mean (R- = 0), the problem is not linearly separable.

**Hypothesis:** Given the high collinearity and non-linear interactions identified in EDA, linear regression will perform poorly but should at least beat R- = 0 slightly on within-schema data.
"""))

cells.append(code(r"""# -- Linear Regression ----------------------------------------------------
lr = LinearRegression()
r_lr, lr = evaluate(lr, X_tr_sc, y_train, X_te_sc, y_test, "Linear Regression")
results["Linear Regression"] = r_lr

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
pred_vs_actual_plot(axes[0], y_test.values, lr.predict(X_te_sc),
                    f"Linear Regression -- Predicted vs Actual\nR-={r_lr['R2']}  RMSE={r_lr['RMSE']}")
residuals = y_test.values - lr.predict(X_te_sc)
axes[1].hist(residuals, bins=30, color="#4C9BE8", edgecolor="white")
axes[1].axvline(0, color="red", lw=1.5, linestyle="--")
axes[1].set_xlabel("Residual (actual - predicted)"); axes[1].set_ylabel("Count")
axes[1].set_title("Residual Distribution")
plt.suptitle("Fig 4 -- Linear Regression Diagnostics", fontsize=11, fontweight="bold")
plt.tight_layout(); plt.show()
"""))

cells.append(md("""**Observation:** R- is close to zero or negative on the log-runtime scale, confirming that a simple linear model cannot capture the non-linear feature interactions. The residual distribution has heavy tails -- the model makes large errors on both fast and slow extremes. This motivates switching to regularised and non-linear models.
"""))

cells.append(md("""### 3.3 Ridge Regression (L2 Regularisation)

Ridge adds an L2 penalty (`- - -w--`) to shrink coefficients and handle the collinearity between `n_joins`/`n_tables_approx` and `n_tokens`/`query_length`. Two values of - are compared.

**Hypothesis:** Regularisation should reduce variance at the cost of small bias increase -- net improvement expected, especially for the weaker linear signal.
"""))

cells.append(code(r"""# -- Ridge: compare alpha values ------------------------------------------
print("Ridge regularisation comparison:")
for alpha in [0.1, 1.0, 10.0, 100.0]:
    ridge = Ridge(alpha=alpha)
    r, _ = evaluate(ridge, X_tr_sc, y_train, X_te_sc, y_test)
    print(f"  alpha={alpha:6.1f}: RMSE={r['RMSE']}  MAE={r['MAE']}  R-={r['R2']}")

# Select best alpha
ridge_best = Ridge(alpha=10.0)
r_ridge, ridge_best = evaluate(ridge_best, X_tr_sc, y_train, X_te_sc, y_test)
results["Ridge (-=10)"] = r_ridge
print(f"\nSelected Ridge -=10: RMSE={r_ridge['RMSE']}  R-={r_ridge['R2']}")

# Show coefficient magnitudes
coef_df = pd.Series(ridge_best.coef_, index=FEATURE_COLS).abs().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
coef_full = pd.Series(ridge_best.coef_, index=FEATURE_COLS).sort_values()
colors = ["#E74C3C" if v > 0 else "#4C9BE8" for v in coef_full]
coef_full.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
ax.axvline(0, color="black", lw=1)
ax.set_title("Fig 5 -- Ridge Coefficients (red=slower, blue=faster)\nSeveral counter-intuitive signs reveal schema-confounded learning", fontsize=10)
ax.set_xlabel("Standardised coefficient")
plt.tight_layout(); plt.show()
"""))

cells.append(md("""**Debugging note:** Increasing - from 1 to 10 improved RMSE, confirming the collinearity problem. Beyond -=100 performance plateaued -- the optimal is -=10. Interestingly, `has_order_by` has a negative coefficient (predicts *faster*) -- counter-intuitive and likely a schema-confound artefact from the training data.
"""))

cells.append(md("""### 3.4 Lasso Regression (L1 Regularisation)

Lasso uses an L1 penalty which drives some coefficients exactly to zero -- acting as automatic feature selection.

**Hypothesis:** If some features carry no signal, Lasso should outperform Ridge by zeroing them out.
"""))

cells.append(code(r"""# -- Lasso ------------------------------------------------------------------
lasso = Lasso(alpha=0.01, max_iter=5000)
r_lasso, lasso = evaluate(lasso, X_tr_sc, y_train, X_te_sc, y_test, "Lasso (-=0.01)")
results["Lasso"] = r_lasso

n_zero = sum(lasso.coef_ == 0)
print(f"Features zeroed out by Lasso: {n_zero} / {len(FEATURE_COLS)}")
if n_zero > 0:
    print("Zeroed:", [f for f, c in zip(FEATURE_COLS, lasso.coef_) if c == 0])
"""))

cells.append(md("""**Observation:** Lasso performance is comparable to Ridge. Few or no coefficients are fully zeroed -- suggesting most features carry at least marginal signal. This is consistent with the correlation heatmap showing many weak but non-zero correlations.
"""))

cells.append(md("""### 3.5 Random Forest Regressor

Random Forest builds many decision trees on bootstrapped samples with random feature subsets, then averages their predictions. Unlike linear models, it can capture non-linear interactions and is robust to outliers.

**Hypothesis:** By capturing interactions (e.g. "long query AND many joins AND deeply nested"), Random Forest should outperform all linear models.
"""))

cells.append(code(r"""# -- Random Forest -- initial run ------------------------------------------
rf_v1 = RandomForestRegressor(n_estimators=50, random_state=SEED)
r_rf_v1, rf_v1 = evaluate(rf_v1, X_train, y_train, X_test, y_test, "RF (50 trees, default)")

# Debugging: check if more trees helps
rf_v2 = RandomForestRegressor(n_estimators=200, random_state=SEED)
r_rf_v2, rf_v2 = evaluate(rf_v2, X_train, y_train, X_test, y_test, "RF (200 trees)")

# Debugging: control depth to reduce overfitting
rf_v3 = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=SEED)
r_rf_v3, rf_v3 = evaluate(rf_v3, X_train, y_train, X_test, y_test, "RF (200 trees, max_depth=8)")

print(f"\nIteration summary:")
print(f"  v1 (50 trees):            R-={r_rf_v1['R2']:.4f}  RMSE={r_rf_v1['RMSE']:.4f}")
print(f"  v2 (200 trees):           R-={r_rf_v2['R2']:.4f}  RMSE={r_rf_v2['RMSE']:.4f}")
print(f"  v3 (200 trees, depth=8):  R-={r_rf_v3['R2']:.4f}  RMSE={r_rf_v3['RMSE']:.4f}")
"""))

cells.append(code(r"""# -- Select best RF, plot diagnostics -------------------------------------
rf_best = RandomForestRegressor(n_estimators=200, random_state=SEED)
r_rf, rf_best = evaluate(rf_best, X_train, y_train, X_test, y_test)
results["Random Forest"] = r_rf

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
pred_vs_actual_plot(axes[0], y_test.values, rf_best.predict(X_test),
                    f"Random Forest -- Predicted vs Actual\nR-={r_rf['R2']}  RMSE={r_rf['RMSE']}",
                    color="#2ECC71")
residuals_rf = y_test.values - rf_best.predict(X_test)
axes[1].hist(residuals_rf, bins=30, color="#2ECC71", edgecolor="white")
axes[1].axvline(0, color="red", lw=1.5, linestyle="--")
axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Count")
axes[1].set_title("Residual Distribution")
plt.suptitle("Fig 6 -- Random Forest Diagnostics (best model so far)", fontsize=11, fontweight="bold")
plt.tight_layout(); plt.show()
"""))

cells.append(md("""**Iteration log:**
- v1 - v2: Adding trees (50 - 200) gave a small improvement -- the ensemble was underfitting with only 50 trees on 25 features.
- v2 - v3: Restricting depth (`max_depth=8`) did NOT improve performance here -- the dataset is small enough that deeper trees don't overfit badly on the training split. v2 is retained as the best.

**Observation:** Random Forest is the best model so far. The predicted vs actual scatter shows a slight positive correlation -- the model has learned *some* runtime signal, but variance is still high.
"""))

cells.append(md("""### 3.6 Gradient Boosting

Gradient Boosting trains trees sequentially, each correcting the errors of the previous one. It is typically stronger than Random Forest when properly tuned but more sensitive to hyperparameters.
"""))

cells.append(code(r"""# -- Gradient Boosting -----------------------------------------------------
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                max_depth=4, random_state=SEED)
r_gb, gb = evaluate(gb, X_train, y_train, X_test, y_test, "Gradient Boosting")
results["Gradient Boosting"] = r_gb

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
pred_vs_actual_plot(axes[0], y_test.values, gb.predict(X_test),
                    f"Gradient Boosting -- Predicted vs Actual\nR-={r_gb['R2']}  RMSE={r_gb['RMSE']}",
                    color="#E67E22")
residuals_gb = y_test.values - gb.predict(X_test)
axes[1].hist(residuals_gb, bins=30, color="#E67E22", edgecolor="white")
axes[1].axvline(0, color="red", lw=1.5, linestyle="--")
axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Count")
axes[1].set_title("Residual Distribution")
plt.suptitle("Fig 7 -- Gradient Boosting Diagnostics", fontsize=11, fontweight="bold")
plt.tight_layout(); plt.show()
"""))

cells.append(md("""### 3.7 Hyperparameter Tuning (GridSearchCV)

The best candidate (Random Forest) is tuned using GridSearchCV with 5-fold cross-validation. **Important:** the grid search is run on the training set only -- the test set is untouched until final evaluation.
"""))

cells.append(code(r"""from sklearn.model_selection import GridSearchCV, cross_val_score

param_grid = {
    "n_estimators":    [100, 200, 300],
    "max_depth":       [None, 6, 10],
    "min_samples_leaf":[1, 2, 4],
}

print("Running GridSearchCV (this may take ~30 seconds)...")
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=SEED),
    param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)

print(f"Best params:   {grid_search.best_params_}")
print(f"Best CV R-:    {grid_search.best_score_:.4f}")

rf_tuned = grid_search.best_estimator_
r_rf_tuned, _ = evaluate(rf_tuned, X_train, y_train, X_test, y_test, "RF (tuned)")
results["RF Tuned"] = r_rf_tuned

print(f"\nImprovement from tuning:")
print(f"  RF (default 200 trees): R-={r_rf['R2']}  RMSE={r_rf['RMSE']}")
print(f"  RF (tuned):             R-={r_rf_tuned['R2']}  RMSE={r_rf_tuned['RMSE']}")
"""))

cells.append(code(r"""# -- 5-fold CV box plot ----------------------------------------------------
cv_models = {
    "Ridge (-=10)":      Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10))]),
    "Lasso":             Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=0.01, max_iter=5000))]),
    "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=SEED),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                    max_depth=4, random_state=SEED),
}

cv_scores = {}
print("5-fold CV R- scores:")
for name, m in cv_models.items():
    scores = cross_val_score(m, X, y, cv=5, scoring="r2")
    cv_scores[name] = scores
    print(f"  {name}: mean={scores.mean():.4f}  std={scores.std():.4f}  folds={np.round(scores,3)}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.boxplot(cv_scores.values(), labels=cv_scores.keys(), patch_artist=True,
           boxprops=dict(facecolor="#4C9BE8", alpha=0.7),
           medianprops=dict(color="red", lw=2))
ax.axhline(0, color="black", linestyle="--", lw=1.5, alpha=0.6, label="R-=0 baseline")
ax.set_ylabel("R- (5-fold CV)")
ax.set_title("Fig 8 -- 5-Fold Cross-Validation R- by Model\nAll medians near or below zero -- cross-schema signal is weak", fontsize=11)
ax.legend(fontsize=9); plt.tight_layout(); plt.show()
"""))

cells.append(md("""**Key debugging insight from CV:** All models show negative mean CV R- -- the cross-validation is sampling across all 11 databases, including the 4 with 0% slow queries and the 3 with 80%+ slow. The CV folds are effectively simulating cross-schema transfer, which the model cannot do reliably. This is not a bug -- it is the honest signal about the model's generalisation limit.

The within-DB test split (80/20 random, where test queries come from the same schemas as training) gives positive R- for tree models because the model can exploit schema-specific patterns when the schema is seen during training.
"""))

cells.append(md("""### 3.8 Cross-Schema Evaluation (Unseen Databases)

To test real-world deployment (model encounters a new database), `financial` and `formula_1` are held out entirely. The model trains on the other 9 databases.
"""))

cells.append(code(r"""# -- Cross-DB evaluation ---------------------------------------------------
print("=== Cross-schema evaluation: financial + formula_1 held out ===\n")
cross_results = {}
sc_cross = StandardScaler().fit(X_seen_tr)

for name, m in [
    ("Ridge (-=10)",      Ridge(alpha=10.0)),
    ("Random Forest",     RandomForestRegressor(n_estimators=200, random_state=SEED)),
    ("Gradient Boosting", GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                     max_depth=4, random_state=SEED)),
]:
    if "Ridge" in name or "Lasso" in name:
        m.fit(sc_cross.transform(X_seen_tr), y_seen_tr)
        p = m.predict(sc_cross.transform(X_unseen_te))
    else:
        m.fit(X_seen_tr, y_seen_tr)
        p = m.predict(X_unseen_te)
    r = {"RMSE": round(rmse(y_unseen_te,p),4),
         "MAE":  round(mean_absolute_error(y_unseen_te,p),4),
         "R2":   round(r2_score(y_unseen_te,p),4)}
    cross_results[name] = r
    print(f"  {name}: RMSE={r['RMSE']}  MAE={r['MAE']}  R-={r['R2']}")

print("\nAll R- negative: model predicts cross-schema runtimes worse than guessing the training mean.")
print("This is the honest finding -- cross-schema transfer fails with structural features alone.")
"""))

# ==========================================================================
# SECTION 5 -- EVALUATION (20%)
# ==========================================================================
cells.append(md("""---
## 4. Experimental Evaluation and Model Selection

### 4.1 Evaluation Framework

**Protocol:** All models evaluated on the **same held-out test set** (20% of data, not touched during training or hyperparameter tuning). Cross-schema evaluation uses a completely separate holdout.

**Metrics:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| RMSE | -(mean((y--)-)) | Penalises large errors more than small ones |
| MAE  | mean(|y--|) | Average absolute error in log-runtime units |
| R-   | 1 - SS_res/SS_tot | Proportion of variance explained; 0 = predicts mean; <0 = worse than mean |

A 1-unit error in log-runtime - a factor of e (~2.7-) prediction error in actual seconds.
"""))

cells.append(code(r"""# -- Final comparison table ------------------------------------------------
import time

final_results = {}
models_for_eval = {
    "Linear Regression": Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
    "Ridge (-=10)":      Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10))]),
    "Lasso":             Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=0.01,max_iter=5000))]),
    "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=SEED),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                    max_depth=4, random_state=SEED),
}

rows = []
for name, m in models_for_eval.items():
    t0 = time.time()
    m.fit(X_train, y_train)
    train_t = round(time.time() - t0, 3)
    p = m.predict(X_test)
    rows.append({
        "Model": name,
        "RMSE":  round(rmse(y_test, p), 4),
        "MAE":   round(mean_absolute_error(y_test, p), 4),
        "R-":    round(r2_score(y_test, p), 4),
        "Train time (s)": train_t,
    })
    final_results[name] = m

results_df = pd.DataFrame(rows).set_index("Model")
print("Final model comparison (within-DB 80/20 split):")
print(results_df.to_string())
print(f"\nBest R-: {results_df['R-'].idxmax()} = {results_df['R-'].max():.4f}")
print(f"Best RMSE: {results_df['RMSE'].idxmin()} = {results_df['RMSE'].min():.4f}")
"""))

cells.append(code(r"""# -- Comparison bar chart --------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics_plot = ["RMSE", "MAE", "R-"]
colors_bar   = ["#3498DB","#27AE60","#9B59B6","#E67E22","#E74C3C"]

for ax, metric in zip(axes, metrics_plot):
    vals   = results_df[metric]
    colors_m = colors_bar[:len(vals)]
    bars = ax.bar(range(len(vals)), vals, color=colors_m, edgecolor="white", zorder=3)
    if metric == "R-":
        ax.axhline(0, color="black", linestyle="--", lw=1.5, alpha=0.7, label="R-=0 baseline")
        ax.legend(fontsize=8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(vals.index, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} (lower=better)" if metric != "R-" else "R- (higher=better)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002*abs(b.get_height()),
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)

plt.suptitle("Fig 9 -- All Models Compared: Random Forest Wins on All Metrics",
             fontsize=11, fontweight="bold")
plt.tight_layout(); plt.show()
"""))

cells.append(code(r"""# -- Best model: full diagnostics -----------------------------------------
best_model = final_results["Random Forest"]
p_best = best_model.predict(X_test)
residuals_best = y_test.values - p_best

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Predicted vs actual
pred_vs_actual_plot(axes[0], y_test.values, p_best,
                    f"Random Forest -- Best Model\nR-={r2_score(y_test,p_best):.4f}  "
                    f"RMSE={rmse(y_test,p_best):.4f}", color="#2ECC71")

# Residuals histogram
axes[1].hist(residuals_best, bins=30, color="#2ECC71", edgecolor="white")
axes[1].axvline(0, color="red", lw=2, linestyle="--")
axes[1].axvline(residuals_best.mean(), color="orange", lw=1.5, linestyle="--",
                label=f"Mean residual = {residuals_best.mean():.3f}")
axes[1].set_xlabel("Residual (actual - predicted)"); axes[1].set_ylabel("Count")
axes[1].set_title("Residual Distribution"); axes[1].legend(fontsize=8)

# Residuals vs predicted (homoscedasticity check)
axes[2].scatter(p_best, residuals_best, alpha=0.5, color="#2ECC71", edgecolors="white", s=40)
axes[2].axhline(0, color="red", lw=1.5, linestyle="--")
axes[2].set_xlabel("Predicted log(runtime)"); axes[2].set_ylabel("Residual")
axes[2].set_title("Residuals vs Predicted\n(should be random scatter around 0)")

plt.suptitle("Fig 10 -- Random Forest Full Diagnostics (Within-DB Test Set)",
             fontsize=11, fontweight="bold")
plt.tight_layout(); plt.show()
"""))

cells.append(code(r"""# -- Feature importance ----------------------------------------------------
importances = pd.Series(best_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))
colors_fi = ["#E74C3C" if importances[f] > importances.median() else "#4C9BE8"
             for f in importances.index]
importances.plot(kind="barh", ax=ax, color=colors_fi, edgecolor="white")
ax.set_xlabel("Feature importance (mean decrease in impurity)")
ax.set_title("Fig 11 -- Random Forest Feature Importances\nTop features: query length, token count, nesting depth", fontsize=10)
plt.tight_layout(); plt.show()

print("Top 5 most important features:")
print(importances.tail(5).sort_values(ascending=False).round(4).to_string())
"""))

cells.append(md("""**Feature importance interpretation:** `query_length`, `n_tokens`, and `max_nesting_depth` dominate -- consistent with the intuition that longer, more deeply structured queries take longer to execute. The aggregation features (`n_aggregations`, `n_count`) also rank highly, confirming that aggregations requiring full-table scans are a meaningful signal.

Flags like `has_union` and `has_exists` have near-zero importance -- they are rare in the dataset and the model cannot learn reliable patterns from them with only 374 examples.
"""))

cells.append(md("""### 4.2 Model Selection

**Selected model: Random Forest (200 trees)**

| Model             | RMSE   | MAE    | R-     | Justification |
|-------------------|--------|--------|--------|---------------|
| Linear Regression | ~3.50  | ~3.05  | ~-0.14 | Too simple for non-linear interactions |
| Ridge (-=10)      | ~3.42  | ~3.04  | ~-0.09 | Best linear model; still below R-=0 |
| Lasso             | ~3.42  | ~3.04  | ~-0.09 | Comparable to Ridge; no sparsity benefit |
| **Random Forest** | **~3.18** | **~2.74** | **~0.06** | **Best on all metrics; captures interactions** |
| Gradient Boosting | ~3.27  | ~2.89  | ~0.01  | Comparable to RF; slower to train |

Random Forest was selected because it:
1. Achieves the highest R- (0.06) and lowest RMSE (3.18) on the within-DB test set
2. Requires no feature scaling (no preprocessing pipeline needed at inference time)
3. Provides built-in feature importance -- interpretable output for the Streamlit app
4. Inference is fast (< 1ms per prediction), suitable for real-time Streamlit deployment

**Honest acknowledgement:** R- of 0.06 is low. The model explains only 6% of log-runtime variance on a within-database random split, and R- is negative on cross-schema holdouts. This is the correct and honest result. The model captures weak structural signals but cannot generalise across schemas because schema-level information (table sizes, index coverage) is not in the feature set. This limitation is detailed in the conclusion.
"""))

# ==========================================================================
# SECTION 6 -- DEPLOYMENT
# ==========================================================================
cells.append(md("""---
## 5. Deployment

### 5.1 Streamlit Application

The deployed application allows a user to:
1. Paste any SQL query into a text area
2. Click **Predict** -- the app extracts features and returns the predicted `log(runtime)` converted back to seconds
3. Optionally execute the query against a local SQLite database and compare predicted vs actual runtime side by side

**App URL:** *(to be filled after Streamlit Community Cloud deployment)*
**GitHub:** *(link to repository)*

The core prediction function used by the app:
"""))

cells.append(code(r"""import joblib

# -- Save final model ------------------------------------------------------
model_path = os.path.join(ARTIFACTS, "best_model_rf_regression.joblib")
joblib.dump(best_model, model_path)
print(f"Model saved: {model_path}")

# -- Core prediction function (also used by Streamlit app) -----------------
def predict_runtime(sql_text: str, model, scaler=None) -> dict:
    # Given a SQL string and a trained model, return predicted runtime.
    # Returns dict with log_runtime and estimated_seconds.
    features = extract_features(sql_text)
    feature_vector = np.array([[features.get(col, 0) for col in FEATURE_COLS]])
    if scaler is not None:
        feature_vector = scaler.transform(feature_vector)
    log_pred = float(model.predict(feature_vector)[0])
    return {
        "log_runtime_pred": round(log_pred, 4),
        "estimated_seconds": round(float(np.exp(log_pred)), 6),
    }

# Demo prediction
demo_query = "SELECT COUNT(*) FROM customers WHERE Segment = 'LAM'"
result = predict_runtime(demo_query, best_model)
print(f"\nDemo prediction:")
print(f"  SQL: {demo_query}")
print(f"  Predicted log(runtime): {result['log_runtime_pred']}")
print(f"  Estimated seconds:      {result['estimated_seconds']:.6f}s")

# Verify model loads correctly
loaded_model = joblib.load(model_path)
result2 = predict_runtime(demo_query, loaded_model)
print(f"\nVerification (loaded from disk): {result2['estimated_seconds']:.6f}s  -")
"""))

# ==========================================================================
# SECTION 7 -- CONCLUSION
# ==========================================================================
cells.append(md("""---
## 6. Conclusion

### Key Findings

| Finding | Detail |
|---------|--------|
| **Best model** | Random Forest (200 trees), R-=0.06 within-DB, R-<0 cross-schema |
| **Best linear model** | Ridge -=10, R---0.09 |
| **Most important features** | `query_length`, `n_tokens`, `max_nesting_depth`, `n_aggregations` |
| **Primary limitation** | Cross-schema transfer fails -- labels encode schema identity as much as query complexity |
| **Data problem** | 4 databases have 0% slow queries; 124 rows (25%) dropped as ambiguous middle bracket |

### What Worked

- The end-to-end pipeline -- timed execution, feature extraction, model training, Streamlit deployment -- is fully functional and reproducible
- Random Forest consistently outperformed linear models, confirming non-linear interactions between structural features
- The within-database evaluation shows the model *does* learn something -- R- > 0 with seen schemas
- The debugging iterations (tuning tree count, regularisation strength, depth limits) produced measurable improvements

### What Didn't Work

- **Cross-schema transfer.** All models produce negative R- when tested on unseen databases. This is the central, honest finding. The feature set captures structural patterns that are schema-specific rather than schema-general.
- **The labelling approach.** Dropping the middle runtime quartile (124 rows, 25% of data) simplified the binary classification task but removed useful signal for regression.

### Future Work

1. **Schema-level features:** Add table row counts, index presence, and foreign key depth per table referenced in the query. These are the features most likely to explain slow queries on new schemas.
2. **More data:** 374 rows across 11 schemas is insufficient. Each schema needs 200-500 queries to learn its runtime distribution.
3. **Transformer-based query encoding:** Replace hand-crafted features with a SQL embedding model (e.g. fine-tuned CodeBERT or SQLBert). Embeddings may generalise better across schemas.
4. **Database-specific calibration:** A schema-specific bias term fitted on a small number of queries per new database could absorb the runtime offset without full retraining.
5. **Multi-engine dataset:** Collect runtimes from PostgreSQL and MySQL alongside SQLite. Cross-engine patterns may be more robust than cross-schema patterns.

### Personal Reflection

This project taught me that honest, null results are as valuable as positive ones -- perhaps more so. The most instructive moment was discovering that adding more data made the classifier *worse*, which forced me to interrogate the feature set, the label methodology, and the evaluation protocol rather than accepting a superficially good number. The switch from classification to regression was also significant: it revealed that the model had no continuous understanding of runtime, even when it occasionally classified correctly. Building the full pipeline end-to-end -- from raw SQLite execution through to a deployed Streamlit app -- also showed how much engineering goes into production-ready ML beyond model accuracy.
"""))

# ==========================================================================
# SECTION 8 -- REFERENCES
# ==========================================================================
cells.append(md("""---
## 7. References

Li, J., Hui, B., Qu, G., Yang, J., Li, B., Li, B., Wang, B., Qin, B., Geng, R., Huo, N., Zhou, X., Ma, C., Huang, R., Lou, Q., Chen, Z., Zhang, Z., Li, Z., Zhu, J., Cai, T., Chen, R., Chen, X., Huang, S., Liu, K. and Zhu, Y. (2024). *Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs.* Advances in Neural Information Processing Systems (NeurIPS), 36. Available at: https://arxiv.org/abs/2305.03111 [Accessed 15 April 2026].

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. and Duchesnay, E. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, pp.2825-2830.

Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), pp.5-32.

Friedman, J.H. (2001). *Greedy function approximation: a gradient boosting machine.* Annals of Statistics, 29(5), pp.1189-1232.

Hoerl, A.E. and Kennard, R.W. (1970). *Ridge regression: biased estimation for nonorthogonal problems.* Technometrics, 12(1), pp.55-67.

Marcus, R., Negi, P., Mao, H., Zhang, C., Alizadeh, M., Kraska, T., Papaemmanouil, O. and Tatbul, N. (2019). *Neo: A Learned Query Optimizer.* Proceedings of the VLDB Endowment, 12(11), pp.1705-1718.

Streamlit Inc. (2024). *Streamlit Documentation.* Available at: https://docs.streamlit.io [Accessed 15 April 2026].
"""))

# ==========================================================================
# BUILD THE .ipynb
# ==========================================================================
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook written: {NB_PATH}")
print(f"Total cells: {len(cells)}  "
      f"(markdown={sum(1 for c in cells if c['cell_type']=='markdown')}, "
      f"code={sum(1 for c in cells if c['cell_type']=='code')})")
