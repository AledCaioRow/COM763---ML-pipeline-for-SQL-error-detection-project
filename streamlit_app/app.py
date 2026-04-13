"""
SQPP Streamlit dashboard — reads pipeline outputs from the parent project.

Run from project:
    cd streamlit_app && pip install -r requirements.txt && streamlit run app.py

Optional: set SQPP_PROJECT_ROOT to the folder containing data/, reports/, artifacts/.
"""

from __future__ import annotations

import os
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
from utils.model_loader import load_model
from utils.paths import ensure_project_root_on_path, get_project_root
from utils.predictor import (
    load_cutoff_artifact,
    map_runtime_to_tier,
    predict_from_features,
    predict_runtime_seconds,
)

st.set_page_config(
    page_title="SQPP Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = get_project_root()
ensure_project_root_on_path(ROOT)
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
missing = [MISSING_LABELS[k] for k, ok in PRESENCE.items() if (k in MISSING_LABELS and not ok)]
runtime_missing = []
if not PRESENCE.get("runtime_checkpoint", False):
    runtime_missing.append("sql_runtime_predictor/artifacts/runtime_predictor.pt")
if not PRESENCE.get("runtime_cutoffs", False):
    runtime_missing.append("sql_runtime_predictor/artifacts/runtime_tier_cutoffs.json")

df_raw = read_csv_safe(PATHS["raw_csv"])
df_feat = read_csv_safe(PATHS["features_csv"])
parsed = parse_model_results(PATHS["model_results"])
df_per_db = read_csv_safe(PATHS["per_db"])
df_per_diff = read_csv_safe(PATHS["per_diff"])
df_split_summary = read_csv_safe(PATHS["split_summary"])
df_model_compare = read_csv_safe(PATHS["all_models_test_comparison"])
df_class_distribution = read_csv_safe(PATHS["class_distribution_csv"])
df_error_analysis = read_csv_safe(PATHS["error_analysis_csv"])
model = load_model(PATHS["model_joblib"]) if PRESENCE["model_joblib"] else None
runtime_cutoffs = (
    load_cutoff_artifact(PATHS["runtime_cutoffs"])
    if PRESENCE.get("runtime_cutoffs", False)
    else None
)

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
    ensure_project_root_on_path(ROOT)
    try:
        from src.features.extract_features import extract_features

        d = extract_features(sql or "")
        return {k: float(v) for k, v in d.items()}
    except ModuleNotFoundError:
        return None
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


PAGE_OPTIONS = ["Overview", "Data Explorer", "Model Results", "Predict", "Live Compare"]
REPORT_MANIFEST = {
    "model_results.txt": PATHS["model_results"],
    "per_database_results.csv": PATHS["per_db"],
    "per_difficulty_results.csv": PATHS["per_diff"],
    "classification_report.csv": PATHS["classification_report"],
    "all_models_test_comparison.csv": PATHS["all_models_test_comparison"],
    "cv_fold_scores.csv": PATHS["cv_fold_scores"],
    "confusion_matrix.csv": PATHS["confusion_matrix"],
    "confusion_matrix.png": PATHS["confusion_matrix_png"],
    "confusion_matrix_normalised.png": PATHS["confusion_matrix_normalised_png"],
    "roc_curve.png": PATHS["roc_curve_png"],
    "pr_curve.png": PATHS["pr_curve_png"],
    "feature_importance.csv": PATHS["feature_importance_csv"],
    "feature_importance.png": PATHS["feature_importance_png"],
    "class_distribution.png": PATHS["class_distribution_png"],
    "runtime_distribution.png": PATHS["runtime_distribution_png"],
    "cv_boxplot.png": PATHS["cv_boxplot_png"],
    "learning_curve.png": PATHS["learning_curve_png"],
    "error_analysis.csv": PATHS["error_analysis_csv"],
    "split_summary.csv": PATHS["split_summary"],
}
artefact_status = {name: os.path.exists(path) for name, path in REPORT_MANIFEST.items()}
page = render_sidebar(ROOT, PIPE, missing, PAGE_OPTIONS, artefact_status=artefact_status)


def show_csv_or_info(path: str, message: str, as_table: bool = False):
    df = read_csv_safe(path)
    if df is None:
        st.info(message)
        return None
    if as_table:
        st.table(df)
    else:
        st.dataframe(df, use_container_width=True)
    return df


def show_image_or_info(path: str, message: str):
    if not os.path.exists(path):
        st.info(message)
        return False
    st.image(path, use_container_width=True)
    return True


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

    st.subheader("Pipeline Stats")
    best_model_name = "—"
    best_model_f1 = "—"
    total_queries = n_raw
    train_n = "—"
    test_n = "—"

    if df_model_compare is not None and not df_model_compare.empty:
        best_row = df_model_compare.sort_values("Test F1", ascending=False).iloc[0]
        best_model_name = str(best_row.get("Model", "—"))
        if "Test F1" in best_row:
            best_model_f1 = f"{float(best_row['Test F1']):.4f}"
    if df_split_summary is not None and not df_split_summary.empty:
        total_queries = int(df_split_summary["n_queries"].sum())
        train_row = df_split_summary.loc[df_split_summary["split"].astype(str).str.contains("Train", case=False)]
        test_row = df_split_summary.loc[df_split_summary["split"].astype(str).str.contains("Test", case=False)]
        if not train_row.empty:
            train_n = f"{int(train_row.iloc[0]['n_queries'])}"
        if not test_row.empty:
            test_n = f"{int(test_row.iloc[0]['n_queries'])}"

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total queries", total_queries)
    s2.metric("Train / test size", f"{train_n} / {test_n}")
    s3.metric("Best model (test F1)", f"{best_model_name} ({best_model_f1})")
    s4.metric("Features used", 25)

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
        st.info("Run the pipeline to generate this artefact.")
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

    st.subheader("Train / Test Split")
    show_csv_or_info(
        PATHS["split_summary"],
        "Run the pipeline to generate this artefact.",
        as_table=True,
    )

    st.subheader("Class Balance")
    show_image_or_info(PATHS["class_distribution_png"], "Run the pipeline to generate this artefact.")
    if df_class_distribution is not None:
        with st.expander("Class count table"):
            st.dataframe(df_class_distribution, use_container_width=True)

    st.subheader("Runtime Distribution")
    show_image_or_info(PATHS["runtime_distribution_png"], "Run the pipeline to generate this artefact.")

    st.subheader("Per-Database and Per-Difficulty Support")
    db_table = read_csv_safe(PATHS["per_db"])
    diff_table = read_csv_safe(PATHS["per_diff"])
    if db_table is None or diff_table is None:
        st.info("Run the pipeline to generate this artefact.")
    else:
        if "support" not in db_table.columns and "db_id" in df_feat.columns:
            support_map = df_feat.groupby("db_id").size()
            db_table["support"] = db_table["db_id"].map(support_map).fillna(0).astype(int)
        if "support" not in diff_table.columns and "difficulty" in df_feat.columns:
            support_map = df_feat.groupby("difficulty").size()
            diff_table["support"] = (
                diff_table["difficulty"].astype(str).map(support_map).fillna(0).astype(int)
            )
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(db_table, use_container_width=True)
        with c2:
            st.dataframe(diff_table, use_container_width=True)


def page_model_results():
    st.title("Model Results")

    st.subheader("Model Comparison")
    show_csv_or_info(PATHS["all_models_test_comparison"], "Run the pipeline to generate this artefact.")

    st.subheader("Cross-Validation Performance")
    show_image_or_info(PATHS["cv_boxplot_png"], "Run the pipeline to generate this artefact.")

    st.subheader("Confusion Matrix")
    left, right = st.columns(2)
    with left:
        show_image_or_info(PATHS["confusion_matrix_png"], "Run the pipeline to generate this artefact.")
    with right:
        show_image_or_info(
            PATHS["confusion_matrix_normalised_png"],
            "Run the pipeline to generate this artefact.",
        )
    with st.expander("Confusion matrix counts"):
        show_csv_or_info(PATHS["confusion_matrix"], "Run the pipeline to generate this artefact.")

    st.subheader("ROC Curve")
    show_image_or_info(PATHS["roc_curve_png"], "Run the pipeline to generate this artefact.")

    st.subheader("Precision-Recall Curve")
    show_image_or_info(PATHS["pr_curve_png"], "Run the pipeline to generate this artefact.")

    st.subheader("Feature Importance")
    show_image_or_info(PATHS["feature_importance_png"], "Run the pipeline to generate this artefact.")
    with st.expander("Feature importance table"):
        show_csv_or_info(PATHS["feature_importance_csv"], "Run the pipeline to generate this artefact.")

    st.subheader("Per-Class Metrics")
    show_csv_or_info(PATHS["classification_report"], "Run the pipeline to generate this artefact.")

    with st.expander("Full Text Report"):
        st.text(parsed.get("raw_text") or "(empty)")

    if os.path.exists(PATHS["learning_curve_png"]):
        st.subheader("Learning Curve")
        st.image(PATHS["learning_curve_png"], use_container_width=True)

    if df_error_analysis is not None:
        st.subheader("Error Analysis")
        st.caption("Misclassified test queries grouped by database and difficulty.")
        st.dataframe(df_error_analysis, use_container_width=True)


def page_predict():
    st.title("Predict")
    has_classifier = model is not None and bool(FEATURE_COLS)
    has_runtime = PRESENCE.get("runtime_checkpoint", False)
    if not has_classifier:
        st.info(
            "Legacy classifier artifacts are unavailable. "
            "Only runtime prediction is shown below."
        )
    if not has_runtime:
        st.warning(
            "Runtime checkpoint is missing. Run:\n"
            "`cd sql_runtime_predictor && python -m src.modeling.train && python -m src.performance.evaluate`"
        )

    tab_names: List[str] = []
    if has_classifier:
        tab_names.extend(["Manual features", "From SQL"])
    tab_names.append("Runtime (hybrid)")
    tabs = st.tabs(tab_names)

    defaults = default_feature_vector()
    tab_idx = 0
    if has_classifier:
        tab_manual = tabs[tab_idx]
        tab_idx += 1
        tab_sql = tabs[tab_idx]
        tab_idx += 1

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

    tab_runtime = tabs[tab_idx]
    with tab_runtime:
        st.caption(
            "Primary output is predicted runtime (seconds). "
            "Speed tiers are display labels derived from calibrated cutoffs."
        )
        if runtime_missing:
            st.info("Missing runtime artifacts:\n- " + "\n- ".join(runtime_missing))
        if not has_runtime:
            return

        db_options: List[str] = []
        if df_raw is not None and "db_id" in df_raw.columns:
            db_options = sorted(df_raw["db_id"].astype(str).dropna().unique().tolist())

        if db_options:
            db_id = st.selectbox("Database (db_id)", db_options, key="runtime_db_id")
        else:
            db_id = st.text_input("Database (db_id)", value="")

        runtime_sql = st.text_area(
            "SQL for runtime prediction",
            height=180,
            placeholder="SELECT ...",
            key="runtime_pred_sql",
        )
        if st.button("Predict runtime & tier", key="pred_runtime"):
            if not db_id.strip() or not runtime_sql.strip():
                st.warning("Provide both db_id and SQL.")
            else:
                with st.spinner("Running runtime predictor..."):
                    res = predict_runtime_seconds(
                        ROOT,
                        runtime_sql.strip(),
                        db_id.strip(),
                        PATHS["runtime_checkpoint"],
                    )
                if "error" in res:
                    st.error(res["error"])
                else:
                    rt_s = float(res["runtime_seconds"])
                    c1, c2 = st.columns(2)
                    c1.metric("Predicted runtime", f"{rt_s:.4f} s")
                    c2.metric("Predicted runtime (ms)", f"{rt_s * 1000.0:.2f} ms")

                    tier = (
                        map_runtime_to_tier(rt_s, runtime_cutoffs, db_id=db_id.strip())
                        if runtime_cutoffs
                        else None
                    )
                    if tier is not None:
                        st.success(
                            f"Tier: **{tier['label']}** "
                            f"(threshold source: {tier['threshold_source']})"
                        )
                        with st.expander("Tier thresholds (seconds)"):
                            st.json(
                                {
                                    "labels": tier["labels"],
                                    "thresholds_seconds": tier["thresholds_seconds"],
                                }
                            )
                    else:
                        st.caption(
                            "No tier calibration artifact loaded. "
                            "Run runtime evaluation to generate cutoff JSON."
                        )


if page == "Overview":
    page_overview()
elif page == "Data Explorer":
    page_data_explorer()
elif page == "Model Results":
    page_model_results()
elif page == "Live Compare":
    from components.live_compare import render_live_compare

    render_live_compare(ROOT)
else:
    page_predict()
