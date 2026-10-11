"""
Module: app_pages/common.py
Purpose: Constants, plot styling and the shared sidebar for the CamberLab
         Streamlit pages (app.py is the entrypoint).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scripts.airfrans.geometry import naca_coordinates, parse_naca
from scripts.surrogate.inference import naca_features
from scripts.surrogate.models import FAMILIES, FAMILY_LABELS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TASK = "full"
REF_NACA = "0012"              # external-validation case shown on the Model accuracy page
REF_RE = 6.0e6                 # NASA TMR and Ladson reference condition
DEFAULTS = {"naca": "0012", "re": 2.0e6}   # CamberLab inputs replaced by fixed values on the accuracy page
Z95 = 1.96                     # two-sided 95% normal quantile; the inference API returns ±2σ

SERIES = "#1f77b4"
SERIES_BAND = "rgba(31, 119, 180, 0.20)"
POLAR = "#7c3aed"
TRAINING = "#f59e0b"
LOWER = "#eb6834"


@dataclass(frozen=True)
class Query:
    """The sidebar inputs, shared by both pages."""
    code: str
    geo: dict
    re: float
    family: str
    alpha_min: float
    alpha_max: float
    n_points: int
    surface_alpha: float


def sidebar_query(accuracy_page: bool = False) -> Query | None:
    """Draw the input sidebar; None if the NACA code is invalid (the error is shown in the sidebar).

    On the accuracy page only the model family is editable: the section and Re
    show the fixed validation case (NACA 0012, Re 6×10⁶) and the sweep sliders
    are greyed out. The CamberLab NACA and Re values are kept in session state
    so they reappear unchanged when the user switches back.
    """
    for k, v in DEFAULTS.items():
        # Streamlit drops the state of widgets that are not drawn; re-assigning keeps it across pages.
        st.session_state[k] = st.session_state.get(k, v)
    fixed = accuracy_page
    fixed_help = "Fixed on this page: external validation uses NACA 0012 at Re = 6×10⁶."
    with st.sidebar:
        st.header("Inputs")
        if fixed:
            naca = st.text_input("NACA code (4 or 5 digits)", value=REF_NACA, disabled=True, key="naca_fixed",
                                 help=fixed_help)
            re = st.slider("Reynolds number", 2.0e6, 6.0e6, REF_RE, format="%.1e", disabled=True,
                           key="re_fixed", help=fixed_help)
        else:
            naca = st.text_input("NACA code (4 or 5 digits)", max_chars=9, key="naca").strip()
            re = st.slider("Reynolds number", min_value=2.0e6, max_value=6.0e6, step=1.0e5, format="%.1e",
                           key="re")
        alpha_min, alpha_max = st.slider("AoA sweep (deg)", -5.0, 15.0, (0.0, 8.0), step=0.5,
                                         disabled=fixed)
        n_points = st.slider("Sweep points", 21, 161, 61, step=10, disabled=fixed)
        surface_alpha = st.slider("AoA for surface curves (deg)", -5.0, 15.0, 4.0, step=0.5,
                                  disabled=fixed)
        family = st.selectbox("Model family", FAMILIES, index=FAMILIES.index("gp"),
                              format_func=lambda f: {"gp": "GP — Gaussian process", "rf": "RF — random forest",
                                                     "mlp": "MLP — neural network",
                                                     "krg": "KRG — Kriging (SMT)"}[f])
        try:
            code = parse_naca(naca)["code"]
            geo = naca_features(code)
        except ValueError as exc:
            st.error(str(exc))
            return None
        st.subheader(f"NACA {code}")
        st.plotly_chart(airfoil_figure(code), width="stretch", config={"displayModeBar": False})
        st.caption(f"t/c {geo['t_max']:.3f} at {geo['x_tmax']:.2f}c · camber {geo['m_max']:.4f} "
                   f"at {geo['x_m']:.2f}c (geometric chord)")
    return Query(code, geo, re, family, alpha_min, alpha_max, n_points, surface_alpha)


def airfoil_figure(naca: str) -> go.Figure:
    xy = naca_coordinates(naca, n=301)
    xy = np.vstack([xy, xy[:1]])
    fig = go.Figure(go.Scatter(x=xy[:, 0], y=xy[:, 1], fill="toself", mode="lines",
                               line=dict(color=SERIES, width=2), fillcolor=SERIES_BAND,
                               hoverinfo="skip", showlegend=False))
    fig.add_hline(y=0, line_dash="dot", line_color="#8a8985", line_width=1)
    fig.update_layout(height=180, margin=dict(l=8, r=8, t=8, b=8),
                      xaxis=dict(visible=False, range=[-0.03, 1.03], scaleanchor="y", scaleratio=1),
                      yaxis=dict(visible=False, range=[-0.18, 0.18]), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def style_plot(fig: go.Figure, x_title: str, y_title: str, title: str, height: int = 430) -> go.Figure:
    """Same figure size, margins and legend placement as the original regime app."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color="#0f172a"), x=0.02, xanchor="left"),
        height=height, margin=dict(l=64, r=28, t=54, b=104),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5,
                    bgcolor="#ffffff", bordercolor="#cbd5e1", borderwidth=1,
                    font=dict(size=12, color="#1e293b"), itemwidth=30),
        font=dict(family="Arial", size=13, color="#0f172a"),
    )
    for upd, text in ((fig.update_xaxes, x_title), (fig.update_yaxes, y_title)):
        upd(title=dict(text=text, font=dict(size=13, color="#1e293b")),
            tickfont=dict(size=12, color="#334155"), showgrid=True, gridcolor="#dbe3ef",
            linecolor="#94a3b8", zerolinecolor="#94a3b8")
    return fig


def add_95_band(df: pd.DataFrame) -> pd.DataFrame:
    """Add {Cl,Cd,L_over_D}_{lo95,hi95} where a posterior σ exists.

    Cd is modelled as log Cd, so its band is exp(log Cd ± 1.96σ) with
    σ = Cd_std / Cd (inverting the delta method). L/D uses first-order error
    propagation from Cl and Cd, as the original app did.
    """
    df = df.copy()
    if df.Cl_std.isna().all():
        return df
    df["Cl_lo95"], df["Cl_hi95"] = df.Cl - Z95 * df.Cl_std, df.Cl + Z95 * df.Cl_std
    s_log = df.Cd_std / df.Cd
    df["Cd_lo95"], df["Cd_hi95"] = df.Cd * np.exp(-Z95 * s_log), df.Cd * np.exp(Z95 * s_log)
    rel = np.sqrt((df.Cl_std / df.Cl.where(df.Cl != 0)) ** 2 + s_log ** 2)
    df["L_over_D_std"] = (df.L_over_D.abs() * rel).fillna(0.0)
    df["L_over_D_lo95"] = df.L_over_D - Z95 * df.L_over_D_std
    df["L_over_D_hi95"] = df.L_over_D + Z95 * df.L_over_D_std
    return df


def curve_figure(df: pd.DataFrame, y: str, title: str, y_title: str, family: str,
                 near: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.alpha_deg, y=df[y], mode="lines+markers",
                             line=dict(color=SERIES, width=3), marker=dict(size=6),
                             name=f"{FAMILY_LABELS[family]} surrogate",
                             hovertemplate=f"α %{{x:.2f}}°<br>{y_title} %{{y:.5g}}<extra></extra>"))
    lo, hi = f"{y}_lo95", f"{y}_hi95"
    if lo in df:
        fig.add_trace(go.Scatter(x=pd.concat([df.alpha_deg, df.alpha_deg[::-1]]),
                                 y=pd.concat([df[hi], df[lo][::-1]]), fill="toself",
                                 fillcolor=SERIES_BAND, line=dict(color="rgba(255,255,255,0)"),
                                 hoverinfo="skip", name="95% uncertainty band"))
    if near is not None and len(near):
        fig.add_trace(go.Scatter(
            x=near.alpha_deg, y=near[y], mode="markers",
            marker=dict(size=7, color=TRAINING, symbol="circle-open", line=dict(width=2)),
            name="nearby CFD rows",
            customdata=np.column_stack([near.sample_id, near.Re / 1e6, near.t_max, near.m_max]),
            hovertemplate=("sample %{customdata[0]:.0f}<br>α %{x:.2f}°, Re %{customdata[1]:.2f}e6<br>"
                           "t/c %{customdata[2]:.3f}, m %{customdata[3]:.4f}<br>" + y_title +
                           " %{y:.5g}<extra></extra>")))
    return style_plot(fig, "Angle of attack (deg)", y_title, title)


def surface_figure(surf: pd.DataFrame, q: str, alpha: float) -> go.Figure:
    """Cp or Cf against x/c, upper solid and lower dashed, with 95% bands where available."""
    scale = 1e3 if q == "Cf" else 1.0
    fig = go.Figure()
    for side, dash in (("upper", "solid"), ("lower", "dash")):
        s = surf[surf.surface == side]
        if f"{q}_lo" in s:
            half = Z95 * (s[f"{q}_hi"] - s[f"{q}_lo"]) / 4    # API band is ±2σ
            fig.add_trace(go.Scatter(x=pd.concat([s.x_c, s.x_c[::-1]]),
                                     y=pd.concat([s[q] + half, (s[q] - half)[::-1]]) * scale,
                                     fill="toself", fillcolor=SERIES_BAND, line=dict(width=0),
                                     hoverinfo="skip", name="95% uncertainty band",
                                     showlegend=side == "upper"))
        fig.add_trace(go.Scatter(x=s.x_c, y=s[q] * scale, mode="lines", name=f"{side} surface",
                                 line=dict(color=SERIES if side == "upper" else LOWER, width=3, dash=dash),
                                 hovertemplate=f"x/c %{{x:.3f}}<br>{q} %{{y:.4g}}<extra>{side}</extra>"))
    if q == "Cp":
        fig.update_yaxes(autorange="reversed")
        title, y_title = f"Pressure distribution, α = {alpha:g}°", "Cp"
    else:
        fig.add_hline(y=0, line_color="#8a8985", line_width=1)
        title, y_title = f"Skin friction, α = {alpha:g}° (Cf < 0: reversed flow)", "Cf × 10³"
    return style_plot(fig, "x/c", y_title, title)
