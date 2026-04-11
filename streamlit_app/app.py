"""
SQPP Streamlit dashboard — reads pipeline outputs from the parent project.

Run from project:
    cd streamlit_app && pip install -r requirements.txt && streamlit run app.py

Optional: set SQPP_PROJECT_ROOT to the folder containing data/, reports/, artifacts/.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from components import charts
from components.sidebar import render_sidebar
from utils.data_loader import (
    artifact_paths,
    check_artifacts,
    load_pipeline_config,
    parse_model_results,
    read_csv_safe,
)
from utils.model_loader import get_feature_importances, load_model
from utils.paths import get_project_root
from utils.predictor import predict_from_features

st.set_page_config(
    page_title="SQPP Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = get_project_root()
PATHS = artifact_paths(ROOT)
PRESENCE = check_artifacts(ROOT)
PIPE = load_pipeline_config(ROOT)

FEATURE_COLS: List[str] = (
    list(PIPE.FEATURE_COLS) if PIPE is not None else []
)

MISSING_LABELS = {
    "raw_csv": "data/query_dataset_raw.csv",
    "features_csv": "data/query_dataset_features.csv",
    "model_results": "reports/model_results.txt",
    "per_db": "reports/per_database_results.csv",
    "per_diff": "reports/per_difficulty_results.csv",
    "model_joblib": "artifacts/best_model.joblib",
    "config_py": "config.py",
}
missing = [MISSING_LABELS[k] for k, ok in PRESENCE.items() if not ok]

df_raw = read_csv_safe(PATHS["raw_csv"])
df_feat = read_csv_safe(PATHS["features_csv"])
parsed = parse_model_results(PATHS["model_results"])
df_per_db = read_csv_safe(PATHS["per_db"])
df_per_diff = read_csv_safe(PATHS["per_diff"])
model = load_model(PATHS["model_joblib"]) if PRESENCE["model_joblib"] else None

if not FEATURE_COLS and df_feat is not None:
    _meta = {
        "question_id",
        "db_id",
        "sql",
        "difficulty",
        "runtime_s",
        "label",
        "label_binary",
    }
    FEATURE_COLS = [c for c in df_feat.columns if c not in _meta]


def try_extract_sql_features(sql: str) -> Optional[Dict[str, float]]:
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    try:
        from src.features.extract_features import extract_features

        d = extract_features(sql or "")
        return {k: float(v) for k, v in d.items()}
    except Exception:
        return None


def default_feature_vector() -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not FEATURE_COLS:
        return out
    if df_feat is None:
        return {c: 0.0 for c in FEATURE_COLS}
    for c in FEATURE_COLS:
        if c not in df_feat.columns:
            out[c] = 0.0
            continue
        s = df_feat[c]
        if s.dtype == object:
            out[c] = float(pd.to_numeric(s, errors="coerce").fillna(0).median())
        else:
            out[c] = float(s.median())
    return out


PAGE_OPTIONS = ["Overview", "Data Explorer", "Model Results", "Predict"]
page = render_sidebar(ROOT, PIPE, missing, PAGE_OPTIONS)


def page_overview():
    st.title("Overview")
    st.markdown(
        "Interactive dashboard for the **SQL Query Performance Predictor** "
        "(BIRD Mini-Dev, SQLite timing, structural features, binary fast/slow labels)."
    )

    c1, c2, c3, c4 = st.columns(4)
    n_raw = len(df_raw) if df_raw is not None else 0
    n_feat = len(df_feat) if df_feat is not None else 0
    n_dbs = df_raw["db_id"].nunique() if df_raw is not None and "db_id" in df_raw else 0
    n_features = len(FEATURE_COLS)
    c1.metric("Timed queries (raw)", n_raw)
    c2.metric("Labeled queries", n_feat)
    c3.metric("Databases (raw)", n_dbs)
    c4.metric("Model features", n_features)

    st.subheader("Pipeline")
    steps = [
        "Load BIRD queries (SQLite JSON, else MySQL JSON + conversion)",
        "Time each query on matching `.sqlite` (median of runs, timeout)",
        "Save `data/query_dataset_raw.csv`",
        "Extract structural features → `data/query_dataset_features.csv`",
        "Label fast/slow (quantile or median)",
        "Train / select model (CV on train), save `artifacts/best_model.joblib`",
        "Evaluate hold-out test, write `reports/*.txt` and `*.csv`",
    ]
    for i, s in enumerate(steps, 1):
        st.markdown(f"{i}. {s}")

    st.subheader("Latest model summary")
    if parsed.get("best_model"):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Best model", parsed["best_model"])
        m2.metric("Test F1", f"{parsed['test_f1']:.4f}" if parsed["test_f1"] else "—")
        m3.metric(
            "Test ROC-AUC",
            f"{parsed['test_roc_auc']:.4f}" if parsed["test_roc_auc"] else "—",
        )
        m4.metric(
            "Test accuracy",
            f"{parsed['test_accuracy']:.4f}" if parsed["test_accuracy"] else "—",
        )
    else:
        st.info("Run `python -u main.py` from the project root to generate reports.")

    if df_raw is not None and not df_raw.empty:
        cleft, cright = st.columns(2)
        with cleft:
            st.plotly_chart(
                charts.fig_difficulty_bar(df_raw),
                use_container_width=True,
            )
        with cright:
            st.plotly_chart(
                charts.fig_db_bar(df_raw),
                use_container_width=True,
            )
    else:
        st.warning("Raw dataset not found or empty.")


def page_data_explorer():
    st.title("Data Explorer")
    if df_raw is None or df_feat is None:
        st.error("Need both raw and feature CSVs. Run the pipeline first.")
        return

    tab1, tab2, tab3 = st.tabs(["Runtime", "Distributions", "Features"])

    rt = df_raw["runtime_s"]
    p50, p75 = rt.quantile(0.5), rt.quantile(0.75)

    with tab1:
        log_scale = st.checkbox("Log-scale histogram", value=False)
        st.plotly_chart(
            charts.fig_runtime_hist(rt, log_scale=log_scale, p50=p50, p75=p75),
            use_container_width=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                charts.fig_box_runtime_by(df_raw, "difficulty"),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                charts.fig_box_runtime_by(df_raw, "db_id"),
                use_container_width=True,
            )

    with tab2:
        st.plotly_chart(
            charts.fig_label_pie(df_feat, "label"),
            use_container_width=True,
        )

    with tab3:
        numeric_in_feat = [
            c for c in FEATURE_COLS if c in df_feat.columns and pd.api.types.is_numeric_dtype(df_feat[c])
        ]
        default_pick = numeric_in_feat[: min(8, len(numeric_in_feat))]
        picked = st.multiselect(
            "Features for correlation heatmap",
            options=numeric_in_feat,
            default=default_pick,
        )
        if len(picked) >= 2:
            st.plotly_chart(
                charts.fig_correlation_heatmap(df_feat, picked),
                use_container_width=True,
            )
        else:
            st.caption("Select at least two numeric features.")

        st.divider()
        x_opts = [c for c in FEATURE_COLS if c in df_feat.columns]
        x_col = st.selectbox("Scatter x-axis", x_opts, index=0 if x_opts else None)
        color_opts = [c for c in ("label", "difficulty", "db_id") if c in df_feat.columns]
        color_col = st.selectbox("Color by", color_opts, index=0 if color_opts else None)
        if x_col and color_col:
            st.plotly_chart(
                charts.fig_scatter_runtime(df_feat, x_col, color_col),
                use_container_width=True,
            )


def page_model_results():
    st.title("Model results")
    if parsed.get("cv_models"):
        st.plotly_chart(
            charts.fig_cv_f1_bars(parsed["cv_models"]),
            use_container_width=True,
        )
    else:
        st.caption("Could not parse CV block from model_results.txt.")

    if parsed.get("class_metrics"):
        st.plotly_chart(
            charts.fig_class_pr_bars(parsed["class_metrics"]),
            use_container_width=True,
        )

    if df_per_diff is not None and not df_per_diff.empty and "f1" in df_per_diff.columns:
        d = df_per_diff.copy()
        d["f1"] = pd.to_numeric(d["f1"], errors="coerce")
        fig = charts.fig_importance_bar(
            d["difficulty"].astype(str),
            d["f1"],
            title="Test F1 by difficulty tier",
            top_n=len(d),
        )
        st.plotly_chart(fig, use_container_width=True)

    imp = get_feature_importances(model, FEATURE_COLS)
    if imp is not None:
        arr, names = imp
        st.plotly_chart(
            charts.fig_importance_bar(names, arr, title="Feature importance (model, top 15)"),
            use_container_width=True,
        )
    elif parsed.get("top_features"):
        names, vals = zip(*parsed["top_features"])
        st.plotly_chart(
            charts.fig_importance_bar(names, vals, title="Top features (from report file)"),
            top_n=15,
        )
    else:
        st.caption("Load `artifacts/best_model.joblib` to show importances.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Per database")
        if df_per_db is not None:
            st.dataframe(df_per_db, use_container_width=True)
        else:
            st.caption("Missing per_database_results.csv")
    with c2:
        st.subheader("Per difficulty")
        if df_per_diff is not None:
            st.dataframe(df_per_diff, use_container_width=True)
        else:
            st.caption("Missing per_difficulty_results.csv")

    with st.expander("Raw report text"):
        st.text(parsed.get("raw_text") or "(empty)")


def page_predict():
    st.title("Predict")
    if model is None:
        st.error("No saved model found. Run `python -u main.py` first.")
        return
    if not FEATURE_COLS:
        st.error("Feature list empty. Ensure config.py or features CSV is present.")
        return

    tab_manual, tab_sql = st.tabs(["Manual features", "From SQL"])

    defaults = default_feature_vector()

    with tab_manual:
        st.caption(
            "Adjust features, then submit. Defaults use dataset medians where available."
        )
        bin_cols = [c for c in FEATURE_COLS if str(c).startswith("has_")]
        num_cols = [c for c in FEATURE_COLS if c not in bin_cols]
        st.caption(f"{len(num_cols)} numeric · {len(bin_cols)} binary (has_*)")

        with st.form("predict_manual_form"):
            feats: Dict[str, float] = {}
            if bin_cols:
                st.markdown("**Binary**")
                bc1, bc2 = st.columns(2)
                for j, col in enumerate(bin_cols):
                    base = defaults.get(col, 0.0)
                    use = bc1 if j % 2 == 0 else bc2
                    feats[col] = (
                        1.0 if use.checkbox(col, value=bool(round(base))) else 0.0
                    )
            if num_cols:
                st.markdown("**Numeric**")
                for col in num_cols:
                    base = defaults.get(col, 0.0)
                    if df_feat is not None and col in df_feat.columns:
                        s = pd.to_numeric(df_feat[col], errors="coerce").fillna(0)
                        lo, hi = float(s.min()), float(s.max())
                        if lo == hi:
                            hi = lo + 1.0
                    else:
                        lo, hi = 0.0, max(1.0, abs(base) * 2 + 1.0)
                    mid = min(max(float(base), lo), hi)
                    feats[col] = float(
                        st.slider(
                            col,
                            min_value=lo,
                            max_value=hi,
                            value=mid,
                            key=f"sl_{col}",
                        )
                    )
            submitted = st.form_submit_button("Predict")

        if submitted:
            res = predict_from_features(model, feats, FEATURE_COLS)
            if res:
                st.success(f"**{res['label'].upper()}** (binary={res['label_binary']})")
                if "probability_slow" in res:
                    st.metric("P(slow)", f"{res['probability_slow']:.4f}")
                imp = get_feature_importances(model, FEATURE_COLS)
                if imp is not None:
                    arr, names = imp
                    vals = [feats.get(n, 0.0) for n in names]
                    st.plotly_chart(
                        charts.fig_contribution_proxy(names, vals, arr),
                        use_container_width=True,
                    )

    with tab_sql:
        sql = st.text_area("SQL", height=180, placeholder="SELECT ...")
        if st.button("Extract features & predict", key="pred_sql"):
            extracted = try_extract_sql_features(sql)
            if not extracted:
                st.warning("Could not import pipeline feature extractor. Is `src/` on the project root?")
            else:
                vec = {c: float(extracted.get(c, 0.0)) for c in FEATURE_COLS}
                res = predict_from_features(model, vec, FEATURE_COLS)
                if res:
                    st.success(f"**{res['label'].upper()}** (binary={res['label_binary']})")
                    if "probability_slow" in res:
                        st.metric("P(slow)", f"{res['probability_slow']:.4f}")
                    with st.expander("Extracted feature vector"):
                        st.json({k: vec[k] for k in FEATURE_COLS})


if page == "Overview":
    page_overview()
elif page == "Data Explorer":
    page_data_explorer()
elif page == "Model Results":
    page_model_results()
else:
    page_predict()
