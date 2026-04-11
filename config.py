"""
COM 763 — SQL Query Performance Predictor
==========================================
Central configuration: paths, seeds, hyperparameters.
Edit this file to tune the pipeline without touching module code.
"""

import os

# --------------- paths ---------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# --------------- BIRD dataset paths ---------------
_BIRD_DIR_CANDIDATES = [
    os.path.join(DATA_DIR, "bird", "mini_dev_data"),
    os.path.join(BASE_DIR, "Mini Dev", "MINIDEV"),
    os.path.join(BASE_DIR, "Mini Dev", "MINIDEV", "mini_dev_data"),
]

BIRD_DATA_DIR = next(
    (p for p in _BIRD_DIR_CANDIDATES if os.path.isdir(p)),
    _BIRD_DIR_CANDIDATES[0],
)
BIRD_SQLITE_JSON = os.path.join(BIRD_DATA_DIR, "mini_dev_sqlite.json")
BIRD_MYSQL_JSON = os.path.join(BIRD_DATA_DIR, "mini_dev_mysql.json")
BIRD_DB_DIR = os.path.join(BIRD_DATA_DIR, "dev_databases")

# --------------- query timing ---------------
TIMING_RUNS = 3
QUERY_TIMEOUT_S = 30

# --------------- labelling ---------------
# "quantile" -> top-25% = slow, bottom-50% = fast, drop middle
# "median"   -> above median = slow, below = fast (keeps every row)
LABEL_METHOD = "quantile"

# --------------- train / test split ---------------
SPLIT_METHOD = "database_aware"  # Options: "random", "database_aware"
HOLDOUT_DATABASES = ["formula_1", "financial"]
TEST_SIZE = 0.20

# --------------- reproducibility ---------------
RANDOM_SEED = 42

# --------------- features (parsed from SQL text) ---------------
FEATURE_COLS = [
    "n_tokens", "query_length", "n_joins", "n_tables_approx",
    "n_where_predicates", "has_group_by", "has_order_by", "has_having",
    "has_distinct", "has_limit", "has_union", "n_subqueries",
    "has_subquery", "max_nesting_depth", "n_aggregations",
    "n_count", "n_sum", "n_avg", "n_max", "n_min",
    "has_between", "has_in_clause", "has_like", "has_exists",
    "has_correlated_subquery",
]
