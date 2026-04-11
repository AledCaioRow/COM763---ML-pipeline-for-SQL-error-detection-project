"""Metric row helpers."""

from typing import Optional

import streamlit as st


def metric_row(cols, labels, values, deltas: Optional[list] = None):
    """Fill a row of st.metric in pre-created columns."""
    deltas = deltas or [None] * len(labels)
    for i, col in enumerate(cols):
        if i < len(labels):
            col.metric(labels[i], values[i], delta=deltas[i])
