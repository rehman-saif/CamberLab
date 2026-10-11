"""
Page: CamberLab — Lift, drag, polar and wall-curve predictions for the sidebar query.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_pages.common import (
    POLAR,
    TASK,
    add_95_band,
    curve_figure,
    style_plot,
    surface_figure,
)
from scripts.surrogate.data import SPLITS_DIR, load_dataset
from scripts.surrogate.inference import curve_quantities, has_curve_models, predict_curve, predict_surface
from scripts.surrogate.models import FAMILY_LABELS

# "Nearby CFD rows": training cases within these bands of the query (as in the
# original regime app, which used ±8% Re and ±0.01 t/c; camber is added because
# AirfRANS sections are cambered).
NEAR_RE_FRAC = 0.08
NEAR_DT = 0.01
NEAR_DM = 0.01


@st.cache_data(show_spinner=False)
def load_training() -> pd.DataFrame:
    """Training rows of the task only — the test split is never read here."""
    idx = np.load(SPLITS_DIR / f"airfrans_{TASK}_train_idx.npy")
    return load_dataset().iloc[idx].reset_index(drop=True)


@st.cache_data(show_spinner=False)


def nearby_cfd(train: pd.DataFrame, re: float, geo: dict) -> pd.DataFrame:
    """Training cases at a similar Re and section (any α)."""
    return train[(train.Re.sub(re).abs() <= max(NEAR_RE_FRAC * re, 75_000.0))
                 & (train.t_max.sub(geo["t_max"]).abs() <= NEAR_DT)
                 & (train.m_max.sub(geo["m_max"]).abs() <= NEAR_DM)]


def polar_figure(df: pd.DataFrame, family: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df.Cd, y=df.Cl, mode="lines+markers", line=dict(color=POLAR, width=3),
                               marker=dict(size=6), name=f"{FAMILY_LABELS[family]} surrogate",
                               customdata=df.alpha_deg,
                               hovertemplate="α %{customdata:.2f}°<br>Cd %{x:.5f}<br>Cl %{y:.5f}<extra></extra>"))
    return style_plot(fig, "Cd", "Cl", "Drag polar")


query = st.session_state["query"]
code, geo, re, family = query.code, query.geo, query.re, query.family
train = load_training()

st.title("CamberLab")
st.caption("Surrogate prediction of Cl, Cd and wall Cp, Cf for NACA 4- and 5-digit sections, trained on "
           "800 steady RANS (k-ω SST) solutions from AirfRANS.")

alphas = np.linspace(query.alpha_min, query.alpha_max, query.n_points)
df = add_95_band(predict_curve(alphas, re, code, family, TASK))
warnings = df.attrs["warnings"]
outside = [w for w in warnings if "outside" in w]
if outside:
    st.warning("**Query outside the training domain: extrapolated predictions.**\n\n"
               + "\n".join(f"- {w}" for w in outside))
for w in warnings:
    if w not in outside:
        st.info(w)

best = df.loc[df.L_over_D.idxmax()]
kpi = st.columns(5)
kpi[0].metric("(L/D) max", f"{best.L_over_D:.1f}")
kpi[1].metric("α at (L/D) max", f"{best.alpha_deg:.1f}°")
kpi[2].metric("Cl max (sweep)", f"{df.Cl.max():.3f}")
kpi[3].metric("Cd min (sweep)", f"{df.Cd.min():.5f}")
zero = np.interp(0.0, df.Cl, df.alpha_deg) if df.Cl.min() < 0 < df.Cl.max() else np.nan
kpi[4].metric("Zero-lift α₀", "—" if np.isnan(zero) else f"{zero:.2f}°")

near = nearby_cfd(train, re, geo)
c1, c2 = st.columns(2)
c1.plotly_chart(curve_figure(df, "Cl", "Lift curve", "Cl", family, near), width="stretch")
c2.plotly_chart(curve_figure(df, "Cd", "Drag curve", "Cd", family, near), width="stretch")
c3, c4 = st.columns(2)
c3.plotly_chart(curve_figure(df, "L_over_D", "Efficiency curve", "L/D", family), width="stretch")
c4.plotly_chart(polar_figure(df, family), width="stretch")
plural = "" if len(near) == 1 else "s"
notes = [f"Nearby CFD: {len(near)} training solution{plural} within ±{100 * NEAR_RE_FRAC:.0f}% Re, "
         f"±{NEAR_DT:g} t/c and ±{NEAR_DM:g} camber (any α); neighbouring sections, shown for context only."]
notes.append("Shaded bands: 95% prediction interval." if family in ("gp", "krg")
             else f"No prediction interval: {FAMILY_LABELS[family]} provides no calibrated predictive variance.")
st.caption(" ".join(notes))
quantities = curve_quantities(TASK)
if quantities and not has_curve_models(family, TASK):
    st.caption(f"Wall-distribution models for {FAMILY_LABELS[family]} are not bundled; train them with "
               "`scripts/09_train_surrogates.py --outputs curves`.")
elif quantities:
    surf = predict_surface(query.surface_alpha, re, code, family, TASK)
    cols = st.columns(len(quantities))
    for col, q in zip(cols, quantities):
        col.plotly_chart(surface_figure(surf, q, query.surface_alpha), width="stretch")
    if "Cf" not in quantities:
        st.caption("Cf omitted: the reconstructed wall shear stress failed the drag-consistency gate.")

st.subheader("Tabulated predictions")
cols = [c for c in ["alpha_deg", "Cl", "Cl_std", "Cd", "Cd_std", "L_over_D", "L_over_D_std"] if c in df]
table = df[cols].astype(float).round(6)
st.dataframe(table, width="stretch", hide_index=True)
st.download_button("Download predictions CSV", data=table.to_csv(index=False).encode("utf-8"),
                   file_name=f"naca{code}_Re{re:.2e}_{family}.csv", mime="text/csv")
