"""
SQL Query Explorer — Page 2: Model Performance
===============================================
Shows the history of all submitted queries and regression metrics that
measure how accurately the per-database model's predicted runtime tracks
the actual measured SQLite runtime.

Metrics computed over the growing history
-----------------------------------------
R²    – coefficient of determination on raw runtime_s
MAE   – mean absolute error in milliseconds
Rolling R² trend chart (appears after 3+ valid queries)
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.history_db import fetch_all, fetch_for_metrics, clear_history, init_db
from utils.retrain import merge_and_retrain

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

# ── purple & gold CSS (same palette as app.py) ────────────────────────────────
st.markdown(
    """
    <style>
    :root {
        --purple-dark:   #1E1B2E;
        --purple-mid:    #2D1B69;
        --purple-accent: #7C3AED;
        --purple-light:  #A78BFA;
        --gold:          #F59E0B;
        --gold-light:    #FCD34D;
        --text-primary:  #F3F0FF;
        --text-muted:    #C4B5FD;
        --card-bg:       #2D2040;
        --border:        #4C3880;
    }
    .stApp { background-color: var(--purple-dark); color: var(--text-primary); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--purple-mid) 0%, var(--purple-dark) 100%);
        border-right: 2px solid var(--gold);
    }
    [data-testid="stSidebar"] * { color: var(--text-primary) !important; }
    h1, h2, h3 { color: var(--gold) !important; }
    h4, h5, h6 { color: var(--purple-light) !important; }
    label { color: var(--gold-light) !important; font-weight: 600; }
    .stButton > button {
        background: linear-gradient(135deg, var(--purple-accent), var(--purple-mid));
        color: var(--gold-light); border: 1px solid var(--gold);
        border-radius: 8px; font-weight: 700;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--gold), var(--purple-accent));
        color: var(--purple-dark);
    }
    [data-testid="metric-container"] {
        background: var(--card-bg); border: 1px solid var(--border);
        border-radius: 10px; padding: 0.8rem 1rem;
    }
    [data-testid="metric-container"] label { color: var(--text-muted) !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--gold) !important; font-weight: 800;
    }
    hr { border-color: var(--gold) !important; opacity: 0.4; }
    code {
        background: var(--card-bg) !important; color: var(--purple-light) !important;
        border: 1px solid var(--border) !important; border-radius: 6px !important;
    }
    .metric-card {
        background: var(--card-bg); border: 1px solid var(--border);
        border-radius: 12px; padding: 1rem 1.5rem; margin: 0.5rem 0 1rem 0;
    }
    .r2-bar-track {
        background: #3D2B6B; border-radius: 20px; height: 22px;
        width: 100%; overflow: hidden; margin-top: 0.4rem;
    }
    .r2-bar-fill { height: 100%; border-radius: 20px; transition: width 0.4s ease; }
    .big-value { font-size: 2.2rem; font-weight: 900; color: var(--gold); }
    .sub-label { color: var(--text-muted); font-size: 0.85rem; margin-top: 0.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── metric helpers ────────────────────────────────────────────────────────────
def compute_metrics(rows: list[dict]) -> dict | None:
    """
    Compute regression metrics over history rows.
    Returns None if fewer than 2 usable rows.
    """
    if len(rows) < 2:
        return None
    y_true = np.array([r["runtime_s"]           for r in rows])
    y_pred = np.array([r["predicted_runtime_s"]  for r in rows])

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None

    mae_s  = float(np.mean(np.abs(y_true - y_pred)))
    mae_ms = mae_s * 1000

    # R² on log scale (more stable when runtimes span orders of magnitude)
    log_y_true = np.log1p(y_true)
    log_y_pred = np.log1p(np.clip(y_pred, 0, None))
    ss_res_log = np.sum((log_y_true - log_y_pred) ** 2)
    ss_tot_log = np.sum((log_y_true - log_y_true.mean()) ** 2)
    r2_log = float(1 - ss_res_log / ss_tot_log) if ss_tot_log > 0 else None

    return {"r2": r2, "r2_log": r2_log, "mae_ms": mae_ms, "n": len(rows)}


def colour_for(r2: float | None) -> str:
    if r2 is None: return "#A78BFA"
    if r2 >= 0.7:  return "#22C55E"
    if r2 >= 0.4:  return "#F59E0B"
    if r2 >= 0.0:  return "#F97316"
    return "#EF4444"


def interpret(r2: float | None) -> str:
    if r2 is None: return "Not enough data."
    if r2 >= 0.7:  return "Strong — predicted runtimes closely track actual SQLite timings."
    if r2 >= 0.4:  return "Moderate — reasonable alignment between prediction and measurement."
    if r2 >= 0.0:  return "Weak — model has limited accuracy on these queries so far."
    return "Negative — model predictions are further from reality than a naïve mean estimate."


def make_trend_chart(df_valid: pd.DataFrame) -> plt.Figure:
    """Rolling R² (on log scale) as queries accumulate."""
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#1E1B2E")
    ax.set_facecolor("#2D2040")

    n = len(df_valid)
    r2_vals = []
    for k in range(2, n + 1):
        sub       = df_valid.iloc[:k]
        y_true    = np.log1p(sub["runtime_s"].values)
        y_pred    = np.log1p(np.clip(sub["predicted_runtime_s"].values, 0, None))
        ss_res    = np.sum((y_true - y_pred) ** 2)
        ss_tot    = np.sum((y_true - y_true.mean()) ** 2)
        r2_vals.append(float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"))

    x = list(range(2, n + 1))
    ax.plot(x, r2_vals, color="#F59E0B", linewidth=2.5, marker="o", markersize=4)
    ax.axhline(0, color="#A78BFA", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(1, color="#4ADE80", linestyle=":", linewidth=1, alpha=0.4)
    ax.set_xlabel("Queries submitted", color="#C4B5FD", fontsize=9)
    ax.set_ylabel("R² (log scale)", color="#C4B5FD", fontsize=9)
    ax.set_title("Rolling R²(log) as queries accumulate", color="#F59E0B", fontsize=10)
    ax.tick_params(colors="#C4B5FD")
    for spine in ax.spines.values():
        spine.set_edgecolor("#4C3880")
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    return fig


def make_scatter_chart(df_valid: pd.DataFrame) -> plt.Figure:
    """Predicted vs actual runtime scatter plot."""
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#1E1B2E")
    ax.set_facecolor("#2D2040")

    dbs   = df_valid["database"].unique() if "database" in df_valid.columns else [""]
    cmap  = ["#F59E0B", "#A78BFA", "#4ADE80", "#F87171"]
    for i, db in enumerate(dbs):
        sub = df_valid[df_valid["database"] == db] if "database" in df_valid.columns else df_valid
        ax.scatter(
            sub["runtime_s"] * 1000,
            sub["predicted_runtime_s"] * 1000,
            color=cmap[i % len(cmap)],
            alpha=0.75, s=40, label=db,
        )

    lim = max(
        df_valid["runtime_s"].max() * 1000,
        df_valid["predicted_runtime_s"].max() * 1000,
    ) * 1.1
    ax.plot([0, lim], [0, lim], color="#6B7280", linestyle="--", linewidth=1)
    ax.set_xlabel("Actual runtime (ms)", color="#C4B5FD", fontsize=9)
    ax.set_ylabel("Predicted runtime (ms)", color="#C4B5FD", fontsize=9)
    ax.set_title("Predicted vs Actual runtime", color="#F59E0B", fontsize=10)
    ax.tick_params(colors="#C4B5FD")
    for spine in ax.spines.values():
        spine.set_edgecolor("#4C3880")
    if len(dbs) > 1:
        ax.legend(fontsize=8, labelcolor="#C4B5FD",
                  facecolor="#2D2040", edgecolor="#4C3880")
    fig.tight_layout()
    return fig


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Model Performance")
    st.markdown("---")
    st.page_link("app.py", label="🔮  Query Runner", icon="🔮")
    st.markdown("---")
    st.markdown(
        "<small style='color:#C4B5FD;'>"
        "R² and MAE are computed from the per-database <strong style='color:#F59E0B;'>"
        "regression model</strong> (GBM / Ridge / RF trained on 31 features). "
        "Actual labels use a 50 ms threshold."
        "</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    if st.button("🗑️  Clear all history", use_container_width=True):
        clear_history()
        st.success("History cleared.")
        st.rerun()

# ── header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:0'>📊 Model Performance</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#C4B5FD;margin-top:0.2rem;'>"
    "Every query you run extends the evaluation corpus. "
    "R² and MAE update automatically to show how well the regression model "
    "predicts real SQLite execution time on your inputs.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── fetch data ────────────────────────────────────────────────────────────────
all_rows    = fetch_all()
valid_rows  = fetch_for_metrics()

# ── top-level counts ─────────────────────────────────────────────────────────
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Total queries", len(all_rows))
with m2:
    n_errors = sum(1 for r in all_rows if r["had_error"])
    st.metric("Errors", n_errors)
with m3:
    st.metric("Valid (with timing)", len(valid_rows))

st.markdown("---")

# ── regression metrics ───────────────────────────────────────────────────────
st.markdown("### Regression Metrics — Model vs Reality")

metrics = compute_metrics(valid_rows)

if metrics is None:
    st.info(
        f"Submit at least **2 successful queries** to compute metrics. "
        f"({len(valid_rows)} valid {'query' if len(valid_rows) == 1 else 'queries'} so far)"
    )
else:
    r2     = metrics["r2"]
    r2_log = metrics["r2_log"]
    mae_ms = metrics["mae_ms"]
    n_used = metrics["n"]

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        r2_display = f"{r2:.4f}" if r2 is not None else "undefined"
        st.metric("R² (raw runtime)", r2_display, help="Coefficient of determination on raw runtime_s. 1=perfect, 0=no skill, <0=worse than mean.")
    with mc2:
        r2l_display = f"{r2_log:.4f}" if r2_log is not None else "undefined"
        st.metric("R² (log runtime)", r2l_display, help="R² on log1p(runtime_s) — more stable when runtimes span orders of magnitude.")
    with mc3:
        st.metric("MAE", f"{mae_ms:.2f} ms", help="Mean absolute error in milliseconds between predicted and actual runtime.")

    # R² bar (using log R² as the headline score)
    headline_r2 = r2_log if r2_log is not None else r2
    if headline_r2 is not None:
        bar_pct = max(0.0, min(1.0, headline_r2)) * 100
        col = colour_for(headline_r2)
        interp = interpret(headline_r2)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="big-value">{headline_r2:.4f}</div>
                <div class="sub-label">
                    R²(log) based on {n_used} queries &nbsp;|&nbsp; {interp}
                </div>
                <div class="r2-bar-track">
                    <div class="r2-bar-fill"
                         style="width:{bar_pct:.1f}%; background:{col};"></div>
                </div>
                <div style="display:flex;justify-content:space-between;
                            font-size:0.75rem;color:#7C3AED;margin-top:0.2rem;">
                    <span>−∞ (no skill)</span><span>0.0</span><span>1.0 (perfect)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Charts side by side
    df_valid = pd.DataFrame(valid_rows)
    # Attach database column from all_rows for scatter colouring
    all_df = pd.DataFrame(all_rows)
    if not all_df.empty and "database" in all_df.columns:
        id_to_db = dict(zip(all_df["id"], all_df["database"]))
        if "id" in df_valid.columns:
            df_valid["database"] = df_valid["id"].map(id_to_db)

    ch1, ch2 = st.columns(2)
    if len(valid_rows) >= 3:
        with ch1:
            fig_trend = make_trend_chart(df_valid)
            st.pyplot(fig_trend, use_container_width=True)
            plt.close(fig_trend)
    if len(valid_rows) >= 2:
        with ch2:
            fig_scatter = make_scatter_chart(df_valid)
            st.pyplot(fig_scatter, use_container_width=True)
            plt.close(fig_scatter)

st.markdown("---")

# ── query history table ───────────────────────────────────────────────────────
st.markdown("### Query History")

if not all_rows:
    st.info("No queries submitted yet. Head to the Query Runner page to get started.")
else:
    df = pd.DataFrame(all_rows)

    display_cols = {
        "id":                   "ID",
        "timestamp":            "Timestamp (UTC)",
        "database":             "Database",
        "sql_text":             "SQL",
        "runtime_s":            "Actual (ms)",
        "predicted_runtime_s":  "Predicted (ms)",
        "row_count":            "Rows",
        "had_error":            "Error?",
    }
    df_show = df[[c for c in display_cols if c in df.columns]].rename(columns=display_cols)

    if "Actual (ms)" in df_show.columns:
        df_show["Actual (ms)"] = df_show["Actual (ms)"].apply(
            lambda x: f"{x*1000:.2f}" if pd.notna(x) else "—"
        )
    if "Predicted (ms)" in df_show.columns:
        df_show["Predicted (ms)"] = df_show["Predicted (ms)"].apply(
            lambda x: f"{x*1000:.2f}" if pd.notna(x) else "—"
        )
    if "Error?" in df_show.columns:
        df_show["Error?"] = df_show["Error?"].apply(lambda x: "Yes" if x else "No")

    st.dataframe(df_show, use_container_width=True, height=420)

    # per-database breakdown
    st.markdown("#### Breakdown by database")
    db_grp = (
        df.groupby("database")
        .agg(
            queries=("id", "count"),
            avg_actual_ms=("runtime_s",
                           lambda x: f"{x.mean()*1000:.2f} ms" if x.notna().any() else "—"),
            avg_predicted_ms=("predicted_runtime_s",
                              lambda x: f"{x.mean()*1000:.2f} ms" if x.notna().any() else "—"),
        )
        .reset_index()
    )
    st.dataframe(db_grp, use_container_width=True)

st.markdown("---")

# ── retrain section ───────────────────────────────────────────────────────────
st.markdown("### Retrain Model")
st.markdown(
    "<p style='color:#C4B5FD;'>"
    "Each query you submit with a measured runtime is a new training observation. "
    "Click <strong style='color:#F59E0B;'>Retrain</strong> to merge those observations "
    "into the training data and refit the per-database regression models. "
    "The app will immediately use the new models for subsequent predictions.</p>",
    unsafe_allow_html=True,
)

n_trainable = sum(
    1 for r in all_rows
    if not r["had_error"] and r.get("runtime_s") is not None
)
st.markdown(
    f"<small style='color:#C4B5FD;'>"
    f"{n_trainable} valid query {'observation' if n_trainable == 1 else 'observations'} "
    f"available to merge into training data.</small>",
    unsafe_allow_html=True,
)

if st.button("🔁  Retrain Model", use_container_width=True, type="primary"):
    with st.spinner("Merging history into training CSV and retraining... this may take 30–60 seconds."):
        outcome = merge_and_retrain()

    if outcome["error"]:
        st.error(f"Retraining failed: {outcome['error']}")
        if outcome["output"]:
            with st.expander("Script output"):
                st.code(outcome["output"])
    else:
        st.success(
            f"Retraining complete. "
            f"{outcome['n_added']} new row{'s' if outcome['n_added'] != 1 else ''} merged. "
            f"Models reloaded."
        )
        with st.expander("Training script output"):
            st.code(outcome["output"])
