"""Resolve SQPP project root and artifact paths."""

import os

_ENV_ROOT = "SQPP_PROJECT_ROOT"


def get_project_root() -> str:
    """Parent of streamlit_app/, or SQPP_PROJECT_ROOT if set."""
    explicit = os.environ.get(_ENV_ROOT, "").strip()
    if explicit and os.path.isdir(explicit):
        return os.path.abspath(explicit)
    here = os.path.dirname(os.path.abspath(__file__))
    # streamlit_app/utils -> streamlit_app -> project root
    streamlit_dir = os.path.dirname(here)
    root = os.path.dirname(streamlit_dir)
    return root


def project_path(*parts: str) -> str:
    return os.path.join(get_project_root(), *parts)
