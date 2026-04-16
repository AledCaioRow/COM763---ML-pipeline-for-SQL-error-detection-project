"""
Stage 4 & 5 — Feature extraction (from SQL text) and labelling.

Features are parsed directly from the SQL string — no template metadata
is used — so the same extraction works on unseen queries at inference time.
"""

import os

import numpy as np
import pandas as pd
import sqlparse

from config import DATA_DIR, LABEL_METHOD


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(sql_text):
    """Parse a single SQL string and return a dict of structural features."""
    parsed = sqlparse.parse(sql_text)
    if not parsed:
        return {}

    sql_upper = sql_text.upper()
    tokens_flat = sql_upper.split()

    features = {}

    features["n_tokens"] = len(tokens_flat)
    features["query_length"] = len(sql_text)

    features["n_joins"] = sql_upper.count(" JOIN ")
    features["n_tables_approx"] = 1 + features["n_joins"]

    # WHERE predicates
    if "WHERE" in sql_upper:
        where_part = sql_upper.split("WHERE", 1)[1]
        for stopper in ["GROUP BY", "ORDER BY", "LIMIT", "HAVING", "UNION", ";"]:
            where_part = where_part.split(stopper)[0]
        features["n_where_predicates"] = (
            1 + where_part.count(" AND ") + where_part.count(" OR ")
        )
    else:
        features["n_where_predicates"] = 0

    features["has_group_by"] = int("GROUP BY" in sql_upper)
    features["has_order_by"] = int("ORDER BY" in sql_upper)
    features["has_having"] = int("HAVING" in sql_upper)
    features["has_distinct"] = int("DISTINCT" in sql_upper)
    features["has_limit"] = int("LIMIT" in sql_upper)
    features["has_union"] = int("UNION" in sql_upper)

    n_selects = sql_upper.count("SELECT")
    features["n_subqueries"] = max(0, n_selects - 1)
    features["has_subquery"] = int(n_selects > 1)

    # Nesting depth via parenthesis tracking
    depth = max_depth = 0
    for ch in sql_text:
        if ch == '(':
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ')':
            depth -= 1
    features["max_nesting_depth"] = max_depth

    for agg in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]:
        features[f"n_{agg.lower().rstrip('(')}"] = sql_upper.count(agg)
    features["n_aggregations"] = sum(
        sql_upper.count(a) for a in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]
    )

    features["has_between"] = int("BETWEEN" in sql_upper)
    features["has_in_clause"] = int(" IN (" in sql_upper or " IN(" in sql_upper)
    features["has_like"] = int("LIKE" in sql_upper)
    features["has_exists"] = int("EXISTS" in sql_upper)
    features["has_correlated_subquery"] = int(
        features["has_subquery"]
        and ("EXISTS" in sql_upper or sql_upper.count("WHERE") > 1)
    )

    return features


# ============================================================
# DATAFRAME HELPERS
# ============================================================

def add_parsed_features(df):
    """Append sqlparse-derived feature columns to the DataFrame."""
    required_meta = {"db_id", "difficulty"}
    missing = required_meta - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required metadata columns in raw dataset: "
            f"{sorted(missing)}. These are required for database-aware split/reporting."
        )
    feat_dicts = df["sql"].apply(extract_features)
    feat_df = pd.DataFrame(feat_dicts.tolist())
    df = pd.concat([df, feat_df], axis=1)
    print(f"[STAGE 4] Extracted {len(feat_df.columns)} features from SQL text")
    print("  Preserved metadata columns: db_id, difficulty")
    return df


def add_labels(df, method=None):
    """Add binary slow / fast labels.

    method='quantile': top 25 % → slow, bottom 50 % → fast, drop the middle
    method='median':   above median → slow, below → fast (keeps every row)
    """
    if method is None:
        method = LABEL_METHOD

    if method == "quantile":
        p50 = df["runtime_s"].quantile(0.50)
        p75 = df["runtime_s"].quantile(0.75)
        df["label"] = np.where(
            df["runtime_s"] >= p75, "slow",
            np.where(df["runtime_s"] <= p50, "fast", "mid"),
        )
        n_before = len(df)
        df = df[df["label"] != "mid"].copy()
        print(f"[STAGE 5] Quantile labelling: dropped {n_before - len(df)} 'mid' queries")
    else:
        median_rt = df["runtime_s"].median()
        df["label"] = np.where(df["runtime_s"] >= median_rt, "slow", "fast")

    df["label_binary"] = (df["label"] == "slow").astype(int)

    print(f"  Label distribution: {dict(df['label'].value_counts())}")

    features_path = os.path.join(DATA_DIR, "query_dataset_features.csv")
    df.to_csv(features_path, index=False)
    print(f"  Saved to {features_path}")
    return df
