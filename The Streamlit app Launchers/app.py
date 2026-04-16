"""
SQL Query Explorer — Page 1: Query Runner
==========================================
Purple & gold themed Streamlit app.
• Select a database (formula_1 or financial)
• Enter or paste a SQL query
• View query results, real SQLite runtime, and ML model prediction
• 5 example queries per database
• Schema diagrams (Formula 1 and financial ERDs in assets/)
"""

import sys
from pathlib import Path

# Make utils importable regardless of cwd
_HERE = Path(__file__).resolve().parent
FORMULA_1_SCHEMA_IMG = _HERE / "assets" / "formula_1_schema.png"
FINANCIAL_SCHEMA_IMG = _HERE / "assets" / "financial_schema.png"
sys.path.insert(0, str(_HERE))

import streamlit as st
import pandas as pd

from utils.db_runner import run_query, DB_PATHS
from utils.predictor import predict_runtime
from utils.history_db import init_db, record_query

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SQL Query Explorer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── initialise history DB ────────────────────────────────────────────────────
init_db()

# ── purple & gold theme (injected CSS) ───────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---- root palette ---- */
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

    /* ---- global background ---- */
    .stApp { background-color: var(--purple-dark); color: var(--text-primary); }

    /* ---- sidebar ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--purple-mid) 0%, var(--purple-dark) 100%);
        border-right: 2px solid var(--gold);
    }
    [data-testid="stSidebar"] * { color: var(--text-primary) !important; }

    /* ---- headings ---- */
    h1, h2, h3 { color: var(--gold) !important; }
    h4, h5, h6 { color: var(--purple-light) !important; }

    /* ---- selectbox / text_area labels ---- */
    label, .stSelectbox label, .stTextArea label {
        color: var(--gold-light) !important;
        font-weight: 600;
    }

    /* ---- input widgets ---- */
    .stSelectbox > div > div,
    .stTextArea textarea {
        background-color: var(--card-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
    .stTextArea textarea:focus { border-color: var(--gold) !important; }

    /* ---- buttons ---- */
    .stButton > button {
        background: linear-gradient(135deg, var(--purple-accent), var(--purple-mid));
        color: var(--gold-light);
        border: 1px solid var(--gold);
        border-radius: 8px;
        font-weight: 700;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--gold), var(--purple-accent));
        color: var(--purple-dark);
        border-color: var(--gold-light);
    }

    /* ---- metric cards ---- */
    [data-testid="metric-container"] {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    [data-testid="metric-container"] label { color: var(--text-muted) !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--gold) !important;
        font-weight: 800;
    }

    /* ---- dataframe ---- */
    .stDataFrame { border: 1px solid var(--border); border-radius: 8px; }

    /* ---- expanders ---- */
    .streamlit-expanderHeader {
        background: var(--card-bg) !important;
        color: var(--gold-light) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    /* ---- info / success / warning / error boxes ---- */
    .stAlert { border-radius: 8px; }

    /* ---- divider ---- */
    hr { border-color: var(--gold) !important; opacity: 0.4; }

    /* ---- image placeholder box ---- */
    .schema-image-box {
        background: var(--card-bg);
        border: 2px dashed var(--gold);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        color: var(--text-muted);
        font-size: 0.95rem;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .schema-image-box .icon { font-size: 3rem; margin-bottom: 0.5rem; }
    .schema-image-box .label {
        color: var(--gold);
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.3rem;
    }

    code {
        background: var(--card-bg) !important;
        color: var(--purple-light) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        font-size: 0.8rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── example queries ───────────────────────────────────────────────────────────
EXAMPLES = {
    "formula_1": [
        (
            "Count all races",
            "SELECT COUNT(*) AS total_races FROM races;",
        ),
        (
            "Top 10 drivers by nationality",
            "SELECT forename, surname, nationality FROM drivers ORDER BY surname LIMIT 10;",
        ),
        (
            "Races with most results (JOIN)",
            (
                "SELECT r.name, COUNT(rs.resultId) AS num_results\n"
                "FROM races r\n"
                "JOIN results rs ON r.raceId = rs.raceId\n"
                "GROUP BY r.raceId\n"
                "ORDER BY num_results DESC\n"
                "LIMIT 5;"
            ),
        ),
        (
            "Drivers with most championship-leading positions",
            (
                "SELECT d.forename, d.surname, COUNT(ds.driverStandingsId) AS p1_rounds\n"
                "FROM drivers d\n"
                "JOIN driverStandings ds ON d.driverId = ds.driverId\n"
                "WHERE ds.position = 1\n"
                "GROUP BY d.driverId\n"
                "ORDER BY p1_rounds DESC\n"
                "LIMIT 10;"
            ),
        ),
        (
            "Average race duration per circuit (aggregation + JOIN)",
            (
                "SELECT c.name AS circuit, AVG(rs.milliseconds) AS avg_ms\n"
                "FROM circuits c\n"
                "JOIN races r   ON c.circuitId = r.circuitId\n"
                "JOIN results rs ON r.raceId   = rs.raceId\n"
                "WHERE rs.milliseconds IS NOT NULL\n"
                "GROUP BY c.circuitId\n"
                "ORDER BY avg_ms DESC\n"
                "LIMIT 5;"
            ),
        ),
    ],
    "financial": [
        (
            "Count all accounts",
            "SELECT COUNT(*) AS total_accounts FROM account;",
        ),
        (
            "Top 10 accounts by balance",
            "SELECT account_id, date, balance FROM account ORDER BY balance DESC LIMIT 10;",
        ),
        (
            "Accounts with most transactions (JOIN)",
            (
                "SELECT a.account_id, COUNT(t.trans_id) AS num_transactions\n"
                "FROM account a\n"
                "JOIN trans t ON a.account_id = t.account_id\n"
                "GROUP BY a.account_id\n"
                "ORDER BY num_transactions DESC\n"
                "LIMIT 5;"
            ),
        ),
        (
            "Districts with most accounts",
            (
                "SELECT d.district_id, d.A2 AS district_name,\n"
                "       COUNT(a.account_id) AS num_accounts\n"
                "FROM district d\n"
                "JOIN account a ON d.district_id = a.district_id\n"
                "GROUP BY d.district_id\n"
                "ORDER BY num_accounts DESC\n"
                "LIMIT 10;"
            ),
        ),
        (
            "High-value loans with account info",
            (
                "SELECT l.loan_id, l.amount, l.status, a.account_id\n"
                "FROM loan l\n"
                "JOIN account a ON l.account_id = a.account_id\n"
                "WHERE l.status IN ('A', 'B')\n"
                "ORDER BY l.amount DESC\n"
                "LIMIT 10;"
            ),
        ),
    ],
}

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔮 SQL Query Explorer")
    st.markdown("---")
    st.markdown(
        "<span style='color:#F59E0B;font-weight:700;'>Database</span>",
        unsafe_allow_html=True,
    )
    selected_db = st.selectbox(
        "Select database",
        options=list(DB_PATHS.keys()),
        format_func=lambda x: f"🗄️  {x.replace('_', ' ').title()}",
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#C4B5FD;'>"
        "These are the two <strong style='color:#F59E0B;'>holdout databases</strong> "
        "that achieved the best model results during training evaluation."
        "</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.page_link("pages/2_Model_Performance.py", label="📊  Model Performance", icon="📊")

# ── main layout ───────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:0'>🔮 SQL Query Explorer</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='color:#C4B5FD;margin-top:0.2rem;'>"
    f"Run live SQL against <strong style='color:#F59E0B;'>{selected_db.replace('_',' ').title()}</strong> "
    f"and get an instant ML prediction of query performance.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── two-column layout: left = input+results, right = schema image + examples ─
col_main, col_side = st.columns([3, 2], gap="large")

with col_main:
    st.markdown("#### Enter your SQL query")
    sql_input = st.text_area(
        "SQL",
        height=180,
        placeholder="SELECT * FROM ...",
        label_visibility="collapsed",
    )

    run_btn = st.button("▶  Run Query & Predict", use_container_width=True)

    # ── results ──────────────────────────────────────────────────────────────
    if run_btn:
        sql_text = sql_input.strip()
        if not sql_text:
            st.warning("Please enter a SQL query first.")
        else:
            with st.spinner("Executing query and running model..."):
                db_result = run_query(sql_text, selected_db)
                ml_result = predict_runtime(sql_text, selected_db)

            had_error = bool(db_result["error"])

            # --- store in history ---
            record_query(
                database=selected_db,
                sql_text=sql_text,
                runtime_s=db_result["runtime_s"] if not had_error else None,
                row_count=db_result["row_count"] if not had_error else None,
                predicted_runtime_s=ml_result["predicted_runtime_s"],
                predicted_label=ml_result["label"],
                had_error=had_error,
            )

            # --- metrics row ---
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                actual_ms = (
                    f"{db_result['runtime_s']*1000:.2f} ms"
                    if not had_error else "Error"
                )
                st.metric("Actual runtime", actual_ms)
            with m2:
                pred_ms = (
                    f"{ml_result['predicted_ms']:.2f} ms"
                    if ml_result["predicted_ms"] is not None else "—"
                )
                st.metric("Predicted runtime", pred_ms)
            with m3:
                if not had_error and ml_result["predicted_ms"] is not None:
                    err_ms = abs(db_result["runtime_s"]*1000 - ml_result["predicted_ms"])
                    st.metric("Absolute error", f"{err_ms:.2f} ms")
                else:
                    st.metric("Absolute error", "—")
            with m4:
                st.metric("Rows returned", db_result["row_count"] if not had_error else "—")

            if ml_result.get("error"):
                st.warning(f"Model prediction unavailable: {ml_result['error']}")

            # --- error or results table ---
            if db_result["error"]:
                st.error(f"Query error: {db_result['error']}")
            elif db_result["rows"]:
                st.markdown("**Query results**")
                df = pd.DataFrame(db_result["rows"], columns=db_result["columns"])
                st.dataframe(df, use_container_width=True, height=300)
            else:
                st.info("Query executed successfully — no rows returned.")

with col_side:
    # ── schema diagram (per selected database) ─────────────────────────────
    st.markdown("#### Schema diagram")
    if selected_db == "formula_1":
        if FORMULA_1_SCHEMA_IMG.is_file():
            st.image(
                str(FORMULA_1_SCHEMA_IMG),
                caption="Formula 1 — entity-relationship overview of tables used in example queries.",
                use_container_width=True,
            )
        else:
            st.warning(f"Expected schema image at `{FORMULA_1_SCHEMA_IMG}` — file missing.")
    elif selected_db == "financial":
        if FINANCIAL_SCHEMA_IMG.is_file():
            st.image(
                str(FINANCIAL_SCHEMA_IMG),
                caption="Financial — entity-relationship overview of tables used in example queries.",
                use_container_width=True,
            )
        else:
            st.warning(f"Expected schema image at `{FINANCIAL_SCHEMA_IMG}` — file missing.")
    else:
        st.markdown(
            """
            <div class="schema-image-box">
                <div class="icon">🖼️</div>
                <div class="label">Schema diagram</div>
                <div>No ERD configured for this database.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── example queries ──────────────────────────────────────────────────────
    st.markdown("#### Example queries")
    st.markdown(
        "<small style='color:#C4B5FD;'>"
        "Click the copy icon on any example, then paste into the query box."
        "</small>",
        unsafe_allow_html=True,
    )

    for i, (title, sql) in enumerate(EXAMPLES[selected_db], start=1):
        with st.expander(f"Example {i} — {title}"):
            st.code(sql, language="sql")
