"""Plotly chart builders for SQPP dashboard."""

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def fig_difficulty_bar(df: pd.DataFrame, col: str = "difficulty") -> go.Figure:
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    return px.bar(counts, x=col, y="count", title="Queries by difficulty")


def fig_db_bar(df: pd.DataFrame, col: str = "db_id") -> go.Figure:
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    counts = counts.sort_values("count", ascending=True)
    return px.bar(
        counts,
        x="count",
        y=col,
        orientation="h",
        title="Queries per database",
    )


def fig_runtime_hist(
    series: pd.Series,
    log_scale: bool = False,
    p50: Optional[float] = None,
    p75: Optional[float] = None,
) -> go.Figure:
    s = series.dropna()
    if log_scale:
        s = np.log10(s.clip(lower=1e-9))
        xlab = "log10(runtime_s)"
    else:
        xlab = "runtime_s"
    fig = px.histogram(s, nbins=40, title="Runtime distribution")
    fig.update_layout(xaxis_title=xlab, yaxis_title="count", showlegend=False)
    if p50 is not None:
        x = np.log10(p50) if log_scale else p50
        fig.add_vline(x=x, line_dash="dash", line_color="green", annotation_text="P50")
    if p75 is not None:
        x = np.log10(p75) if log_scale else p75
        fig.add_vline(x=x, line_dash="dash", line_color="red", annotation_text="P75")
    return fig


def fig_box_runtime_by(
    df: pd.DataFrame,
    group_col: str,
    runtime_col: str = "runtime_s",
) -> go.Figure:
    d = df[[group_col, runtime_col]].dropna()
    return px.box(
        d,
        x=group_col,
        y=runtime_col,
        title=f"Runtime by {group_col}",
    )


def fig_correlation_heatmap(df: pd.DataFrame, cols: Sequence[str]) -> go.Figure:
    sub = df[list(cols)].select_dtypes(include=[np.number])
    if sub.shape[1] < 2:
        fig = go.Figure()
        fig.update_layout(title="Need at least 2 numeric columns")
        return fig
    corr = sub.corr()
    return px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Feature correlation",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )


def fig_scatter_runtime(
    df: pd.DataFrame,
    x_col: str,
    color_col: str,
    runtime_col: str = "runtime_s",
) -> go.Figure:
    d = df[[x_col, runtime_col, color_col]].dropna()
    color_series = d[color_col].astype(str)
    return px.scatter(
        d,
        x=x_col,
        y=runtime_col,
        color=color_series,
        title=f"{runtime_col} vs {x_col}",
        opacity=0.65,
    )


def fig_label_pie(df: pd.DataFrame, label_col: str = "label") -> go.Figure:
    counts = df[label_col].value_counts().reset_index()
    counts.columns = [label_col, "count"]
    return px.pie(
        counts,
        names=label_col,
        values="count",
        title="Label distribution",
        hole=0.35,
    )


def fig_cv_f1_bars(
    models: List[dict],
) -> go.Figure:
    if not models:
        return go.Figure().update_layout(title="No CV results parsed")
    names = [m["name"] for m in models]
    means = [m["f1_mean"] for m in models]
    stds = [m["f1_std"] for m in models]
    fig = go.Figure(
        data=[
            go.Bar(
                x=names,
                y=means,
                error_y=dict(type="data", array=stds, visible=True),
                marker_color="#636EFA",
            )
        ]
    )
    fig.update_layout(
        title="Cross-validation F1 (train set)",
        yaxis_title="F1",
        xaxis_title="Model",
    )
    return fig


def fig_class_pr_bars(class_metrics: dict) -> go.Figure:
    rows = []
    for cls, m in class_metrics.items():
        rows.append({"class": cls, "precision": m["precision"], "recall": m["recall"]})
    if not rows:
        return go.Figure().update_layout(title="No per-class metrics")
    df = pd.DataFrame(rows)
    fig = go.Figure(
        data=[
            go.Bar(name="precision", x=df["class"], y=df["precision"]),
            go.Bar(name="recall", x=df["class"], y=df["recall"]),
        ]
    )
    fig.update_layout(
        barmode="group",
        title="Test set: precision & recall by class",
        yaxis_title="score",
    )
    return fig


def fig_importance_bar(
    names: Sequence[str],
    values: Sequence[float],
    title: str = "Feature importance (top 15)",
    top_n: int = 15,
) -> go.Figure:
    df = pd.DataFrame({"feature": names, "importance": values})
    df = df.sort_values("importance", ascending=True).tail(top_n)
    return px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title=title,
    )


def fig_contribution_proxy(
    feature_names: Sequence[str],
    feature_values: Sequence[float],
    importances: Sequence[float],
) -> go.Figure:
    """Bar chart of value * importance (heuristic, not SHAP)."""
    arr_v = np.asarray(feature_values, dtype=float)
    arr_i = np.asarray(importances, dtype=float)
    contrib = arr_v * arr_i
    df = pd.DataFrame({"feature": feature_names, "value × importance": contrib})
    df = df.reindex(df["value × importance"].abs().sort_values(ascending=True).index)
    df = df.tail(15)
    return px.bar(
        df,
        x="value × importance",
        y="feature",
        orientation="h",
        title="Heuristic contribution (value × importance), top 15",
    )
