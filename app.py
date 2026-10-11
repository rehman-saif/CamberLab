"""
CamberLab — Streamlit front end for the AirfRANS coefficient surrogate.

Two pages share the sidebar inputs: CamberLab (predictions) and Model accuracy
(app_pages/prediction.py, app_pages/accuracy.py).

Usage:
    uv run streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app_pages.common import sidebar_query  # noqa: E402

st.set_page_config(page_title="CamberLab", layout="wide")

page = st.navigation([st.Page("app_pages/prediction.py", title="CamberLab", default=True),
                      st.Page("app_pages/accuracy.py", title="Model accuracy")], position="top")

query = sidebar_query(accuracy_page=page.title == "Model accuracy")
if query is None:
    st.info("Enter a valid NACA 4-digit (e.g. 0012, 2412) or 5-digit (e.g. 23012, 23112) code.")
    st.stop()
st.session_state["query"] = query
page.run()

st.divider()
st.caption("Data: AirfRANS (Bonnet et al., NeurIPS 2022 Datasets & Benchmarks, arXiv:2212.07564), "
           "via PLAID-datasets/AirfRANS_clipped, © Safran, licensed under the Open Database "
           "License (ODbL 1.0).")
