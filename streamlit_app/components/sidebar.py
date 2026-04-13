"""Sidebar: project root, pipeline settings, navigation."""

import os
from typing import Any, List

import streamlit as st


def render_sidebar(
    project_root: str,
    pipeline_config: Any,
    missing_files: List[str],
    page_options: List[str],
    artefact_status: Any = None,
    page_key: str = "sqpp_page",
) -> str:
    st.sidebar.title("SQPP Dashboard")
    st.sidebar.caption("SQL Query Performance Predictor")

    st.sidebar.markdown("**Project root**")
    st.sidebar.code(project_root, language="text")
    env_hint = os.environ.get("SQPP_PROJECT_ROOT", "").strip()
    if env_hint:
        st.sidebar.caption("`SQPP_PROJECT_ROOT` is set.")

    if pipeline_config is not None:
        st.sidebar.markdown("**Pipeline settings**")
        st.sidebar.write(
            {
                "Split": getattr(pipeline_config, "SPLIT_METHOD", "—"),
                "Holdout DBs": getattr(
                    pipeline_config, "HOLDOUT_DATABASES", []
                ),
                "Timing runs": getattr(pipeline_config, "TIMING_RUNS", "—"),
                "Query timeout (s)": getattr(
                    pipeline_config, "QUERY_TIMEOUT_S", "—"
                ),
                "Label method": getattr(pipeline_config, "LABEL_METHOD", "—"),
            }
        )
    else:
        st.sidebar.warning("`config.py` not found at project root.")

    if missing_files:
        st.sidebar.error("Missing files:")
        for m in missing_files:
            st.sidebar.caption(f"• {m}")

    if artefact_status:
        with st.sidebar.expander("Artefact status"):
            for label, exists in artefact_status.items():
                icon = "✅" if exists else "❌"
                st.caption(f"{icon} {label}")

    st.sidebar.divider()
    page = st.sidebar.radio("Page", page_options, key=page_key)
    return page
