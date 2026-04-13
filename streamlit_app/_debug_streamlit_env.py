# region agent log
"""One-shot env check for streamlit CLI; writes NDJSON to workspace debug-b0d619.log."""
import json
import os
import shutil
import sys
import time

_LOG = os.path.join(os.path.dirname(__file__), "..", "debug-b0d619.log")


def _line(payload: dict) -> None:
    payload.setdefault("sessionId", "b0d619")
    payload.setdefault("timestamp", int(time.time() * 1000))
    with open(_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


_line(
    {
        "hypothesisId": "H4",
        "location": "_debug_streamlit_env.py:sys",
        "message": "python executable and version",
        "data": {
            "sys_executable": sys.executable,
            "sys_version": sys.version.split()[0],
        },
    }
)
_line(
    {
        "hypothesisId": "H3",
        "location": "_debug_streamlit_env.py:venv",
        "message": "virtual env markers",
        "data": {
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
            "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV"),
        },
    }
)
which_streamlit = shutil.which("streamlit")
_line(
    {
        "hypothesisId": "H2",
        "location": "_debug_streamlit_env.py:which",
        "message": "streamlit on PATH",
        "data": {"shutil_which_streamlit": which_streamlit},
    }
)
try:
    import streamlit as st  # noqa: F401

    _ver = getattr(st, "__version__", "unknown")
    _line(
        {
            "hypothesisId": "H1",
            "location": "_debug_streamlit_env.py:import",
            "message": "streamlit import ok",
            "data": {"streamlit_version": _ver},
        }
    )
except Exception as e:
    _line(
        {
            "hypothesisId": "H1",
            "location": "_debug_streamlit_env.py:import",
            "message": "streamlit import failed",
            "data": {"error_type": type(e).__name__, "error": str(e)},
        }
    )
# endregion agent log
