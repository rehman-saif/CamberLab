"""
Page: Model accuracy — the surrogate against published NACA 0012 references
and on the 200 held-out test cases of the 800/200 AirfRANS split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_pages.common import (
    RESULTS_DIR,
    TASK,
    Z95,
    add_95_band,
    curve_figure,
    style_plot,
    surface_figure,
)
from scripts.airfrans.reference import REF_DIR, read_tecplot_zones
from scripts.surrogate.inference import (
    curve_quantities,
    has_curve_models,
    load_envelope,
    predict_curve,
    predict_surface,
)
from scripts.surrogate.models import FAMILIES, FAMILY_LABELS

CURVE_METRICS_PATH = RESULTS_DIR / "airfrans_curve_metrics.csv"
TEST_PRED_PATH = RESULTS_DIR / "airfrans_test_predictions.csv"
REF_RE = 6.0e6                 # NASA TMR and Ladson reference condition
REF_ALPHAS = (0, 10, 15)       # α with TMR / Ladson / Gregory surface data
REF_ALPHA_MAX = 16.0           # Ladson / Gregory run to 19° (post-stall); plot only up to the envelope edge
REF_TMR = "#dc2626"
REF_LADSON = "#0f172a"
REF_GREGORY = "#16a34a"


@st.cache_data(show_spinner=False)
def load_references() -> dict:
    """NACA 0012 reference data shipped in validation_data/common (see scripts/airfrans/reference.py)."""
    z = lambda name: read_tecplot_zones(REF_DIR / name)  # noqa: E731
    return {"ladson": z("CLCD_Ladson_expdata.dat"),
            "gregory_cl": z("CL_Gregory_expdata.dat")["Gregory digitized data"],
            "tmr": z("n0012clcd_cfl3d_sst.dat")["default"],
            "tmr_cp": z("n0012cp_cfl3d_sst.dat"), "tmr_cf": z("n0012cf_cfl3d_sst.dat"),
            "ladson_cp": z("CP_Ladson.dat"), "gregory_cp": z("CP_Gregory_expdata.dat")}


@st.cache_data(show_spinner=False)
def load_test_predictions() -> pd.DataFrame:
    """Per-case test predictions written by 10_global_validation.py (the app never reads test indices)."""
    if not TEST_PRED_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(TEST_PRED_PATH)


@st.cache_data(show_spinner=False)
def load_curve_metrics() -> pd.DataFrame:
    return pd.read_csv(CURVE_METRICS_PATH) if CURVE_METRICS_PATH.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def reference_sweep(family: str) -> pd.DataFrame:
    return add_95_band(predict_curve(np.linspace(-5.0, 15.0, 81), REF_RE, "0012", family, TASK))


@st.cache_data(show_spinner=False)
def reference_surface(family: str, alpha: float) -> pd.DataFrame:
    return predict_surface(alpha, REF_RE, "0012", family, TASK)


def ladson_points(refs: dict) -> pd.DataFrame:
    """All three Ladson grit zones (Re 6×10⁶, tripped) as one table: alpha_deg, Cl, Cd, grit."""
    return pd.concat([pd.DataFrame(v, columns=["alpha_deg", "Cl", "Cd"]).assign(grit=k)
                      for k, v in refs["ladson"].items()], ignore_index=True)


def reference_coeff_figure(pred: pd.DataFrame, refs: dict, y: str, family: str) -> go.Figure:
    """Surrogate Cl or Cd for NACA 0012 at Re 6×10⁶ against experiment and NASA TMR SST."""
    fig = curve_figure(pred, y, "", y, family)
    lad = ladson_points(refs).query("alpha_deg <= @REF_ALPHA_MAX")
    fig.add_trace(go.Scatter(x=lad.alpha_deg, y=lad[y], mode="markers", name="Ladson exp. (Re 6×10⁶, tripped)",
                             marker=dict(size=8, color=REF_LADSON, symbol="square-open", line=dict(width=2)),
                             customdata=lad.grit,
                             hovertemplate=f"Ladson %{{customdata}}<br>α %{{x:.2f}}°<br>{y} %{{y:.5g}}<extra></extra>"))
    if y == "Cl":
        g = refs["gregory_cl"]
        g = g[g[:, 0] <= REF_ALPHA_MAX]
        fig.add_trace(go.Scatter(x=g[:, 0], y=g[:, 1], mode="markers", name="Gregory exp. (Re 2.9×10⁶)",
                                 marker=dict(size=8, color=REF_GREGORY, symbol="triangle-up-open",
                                             line=dict(width=2)),
                                 hovertemplate="Gregory<br>α %{x:.2f}°<br>Cl %{y:.4f}<extra></extra>"))
    tmr = refs["tmr"]
    fig.add_trace(go.Scatter(x=tmr[:, 0], y=tmr[:, 1 if y == "Cl" else 2], mode="markers",
                             name="NASA TMR CFL3D SST (Re 6×10⁶)",
                             marker=dict(size=12, color=REF_TMR, symbol="diamond", line=dict(width=1, color="#fff")),
                             hovertemplate=f"TMR SST<br>α %{{x:.0f}}°<br>{y} %{{y:.5g}}<extra></extra>"))
    return style_plot(fig, "Angle of attack (deg)", y, "Lift curve" if y == "Cl" else "Drag curve")


def reference_surface_figure(surf: pd.DataFrame, refs: dict, q: str, alpha: int) -> go.Figure:
    fig = surface_figure(surf, q, alpha)
    if q == "Cp":
        tmr = refs["tmr_cp"][f"alpha={alpha}"]
        fig.add_trace(go.Scatter(x=tmr[:, 0], y=tmr[:, 1], mode="lines", name="NASA TMR CFL3D SST",
                                 line=dict(color=REF_TMR, width=1.5), hoverinfo="skip"))
        lad = next(v for k, v in refs["ladson_cp"].items()
                   if k.startswith("Re=6 million") and abs(float(k.split("alpha=")[1].split(",")[0]) - alpha) < 0.1)
        fig.add_trace(go.Scatter(x=lad[:, 0], y=lad[:, 1], mode="markers", name="Ladson exp. (Re 6×10⁶)",
                                 marker=dict(size=7, color=REF_LADSON, symbol="square-open", line=dict(width=1.5)),
                                 hovertemplate="Ladson<br>x/c %{x:.3f}<br>Cp %{y:.3f}<extra></extra>"))
        greg = refs["gregory_cp"].get(f"alpha={alpha}")
        if greg is not None:
            fig.add_trace(go.Scatter(x=greg[:, 0], y=greg[:, 1], mode="markers", name="Gregory exp. (Re 2.9×10⁶, upper)",
                                     marker=dict(size=8, color=REF_GREGORY, symbol="triangle-up-open",
                                                 line=dict(width=1.5)),
                                     hovertemplate="Gregory<br>x/c %{x:.3f}<br>Cp %{y:.3f}<extra></extra>"))
        title = f"Pressure distribution, α = {alpha}°"
    else:
        tmr = refs["tmr_cf"][f"alpha={alpha}, upper surface"]
        fig.add_trace(go.Scatter(x=tmr[:, 0], y=1e3 * tmr[:, 1], mode="lines", name="NASA TMR CFL3D SST (upper)",
                                 line=dict(color=REF_TMR, width=1.5), hoverinfo="skip"))
        fig.update_yaxes(range=[-2, 12])
        title = f"Skin friction, upper surface, α = {alpha}°"
    fig.update_layout(title_text=title)
    return fig


def reference_agreement(pred: pd.DataFrame, refs: dict) -> pd.DataFrame:
    """Surrogate minus reference at the reference α (interpolated along the 0.25° sweep)."""
    rows = []
    for _, r in ladson_points(refs).query("-5 <= alpha_deg <= 15").iterrows():
        rows.append(("Ladson experiment", r.alpha_deg, r.Cl, r.Cd))
    for a, cl, cd in refs["tmr"]:
        rows.append(("NASA TMR CFL3D SST", a, cl, cd))
    out = pd.DataFrame(rows, columns=["reference", "alpha_deg", "Cl_ref", "Cd_ref"])
    out["Cl_pred"] = np.interp(out.alpha_deg, pred.alpha_deg, pred.Cl)
    out["Cd_pred"] = np.interp(out.alpha_deg, pred.alpha_deg, pred.Cd)
    out["dCl"] = out.Cl_pred - out.Cl_ref
    out["dCd_pct"] = 100 * (out.Cd_pred - out.Cd_ref) / out.Cd_ref
    return out


def test_band(t: pd.DataFrame, y: str) -> tuple[np.ndarray, np.ndarray] | None:
    """95% interval of each test prediction (Cd from log Cd ± 1.96σ), or None without σ."""
    if y == "Cl":
        s = t.Cl_std.to_numpy()
        lo, hi = t.Cl_pred - Z95 * s, t.Cl_pred + Z95 * s
    else:
        s = t.logCd_std.to_numpy()
        lo, hi = t.Cd_pred * np.exp(-Z95 * s), t.Cd_pred * np.exp(Z95 * s)
    return None if np.isnan(s).all() else (np.asarray(lo), np.asarray(hi))


def _case_hover(t: pd.DataFrame) -> np.ndarray:
    return np.column_stack([t.sample_id, t.alpha_deg, t.Re / 1e6, t.t_max, t.m_max])


CASE_HOVER = ("sample %{customdata[0]:.0f}: α %{customdata[1]:.2f}°, Re %{customdata[2]:.2f}e6<br>"
              "t/c %{customdata[3]:.3f}, camber %{customdata[4]:.4f}<br>")


def parity_figure(test: pd.DataFrame, y: str, family: str) -> go.Figure:
    """CFD vs prediction for every test case, coloured by α (stall cases sit at the high end)."""
    fig = go.Figure()
    lo, hi = test[[f"{y}_true", f"{y}_pred"]].min().min(), test[[f"{y}_true", f"{y}_pred"]].max().max()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#94a3b8", width=1.5),
                             name="perfect prediction", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=test[f"{y}_true"], y=test[f"{y}_pred"], mode="markers",
                             marker=dict(size=7, color=test.alpha_deg, colorscale="Viridis", showscale=True,
                                         colorbar=dict(title="α (deg)", thickness=12), opacity=0.85),
                             name=f"{len(test)} test cases", customdata=_case_hover(test),
                             hovertemplate=CASE_HOVER + f"CFD {y} %{{x:.5g}}<br>predicted %{{y:.5g}}<extra></extra>"))
    if y == "Cd":
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    return style_plot(fig, f"{y}, CFD (AirfRANS)", f"{y}, predicted ({FAMILY_LABELS[family]})", f"Parity: {y}")


def case_errors(test: pd.DataFrame) -> pd.DataFrame:
    """Cl error (absolute) and Cd error (percent) for each test case."""
    return test.assign(dCl=test.Cl_pred - test.Cl_true,
                       dCd_pct=100 * (test.Cd_pred - test.Cd_true) / test.Cd_true)


def family_summary(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per family on the same test cases."""
    rows = []
    for fam in FAMILIES:
        t = preds[preds.family == fam]
        if t.empty:
            continue
        e = case_errors(t)
        cov = [c for c in (coverage95(t, "Cl"), coverage95(t, "Cd")) if c is not None]
        rows.append({"Model": FAMILY_LABELS[fam], "R² Cl": r_squared(t.Cl_true, t.Cl_pred),
                     "Median |ΔCl|": e.dCl.abs().median(), "R² Cd": r_squared(t.Cd_true, t.Cd_pred),
                     "Median |ΔCd|/Cd (%)": e.dCd_pct.abs().median(),
                     "P90 |ΔCd|/Cd (%)": e.dCd_pct.abs().quantile(0.9),
                     "95% PI coverage": np.mean(cov) if cov else np.nan})
    return pd.DataFrame(rows)


def r_squared(true: pd.Series, pred: pd.Series) -> float:
    return float(1 - ((pred - true) ** 2).sum() / ((true - true.mean()) ** 2).sum())


def error_vs_alpha_figure(test: pd.DataFrame, y: str) -> go.Figure:
    err = test[f"{y}_pred"] - test[f"{y}_true"] if y == "Cl" else \
        100 * (test.Cd_pred - test.Cd_true) / test.Cd_true
    y_title = "ΔCl (predicted − CFD)" if y == "Cl" else "ΔCd / Cd (%)"
    fig = go.Figure(go.Scatter(
        x=test.alpha_deg, y=err, mode="markers", name="test case",
        marker=dict(size=7, color=test.Re / 1e6, colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Re ×10⁶", thickness=12)),
        customdata=_case_hover(test), hovertemplate=CASE_HOVER + f"{y_title} %{{y:.4g}}<extra></extra>"))
    fig.add_hline(y=0, line_color="#94a3b8", line_width=1)
    return style_plot(fig, "Angle of attack (deg)", y_title, f"{'Cl residual' if y == 'Cl' else 'Cd relative error'} vs α").update_layout(
        showlegend=False)


def coverage95(test: pd.DataFrame, y: str) -> float | None:
    band = test_band(test, y)
    if band is None:
        return None
    t = test[f"{y}_true"].to_numpy()
    return float(np.mean((band[0] <= t) & (t <= band[1])))


query = st.session_state["query"]
family = query.family
envelope = load_envelope(TASK)["envelope"]
quantities = curve_quantities(TASK)

st.title("Model accuracy")
st.caption(f"Predictive accuracy of the {FAMILY_LABELS[family]} surrogate on data excluded from training. "
           f"Applicability domain: α {envelope['alpha_deg'][0]:.1f}° to {envelope['alpha_deg'][1]:.1f}°, "
           f"Re {envelope['Re'][0] / 1e6:.1f}–{envelope['Re'][1] / 1e6:.1f}×10⁶, "
           f"t/c {envelope['t_max'][0]:.2f}–{envelope['t_max'][1]:.2f}, camber ≤ {envelope['m_max'][1]:.3f}; "
           "fully turbulent k-ω SST, no transition modelling.")

tab_ref, tab_test = st.tabs(["External validation: NACA 0012", "Hold-out evaluation: AirfRANS test set"])
with tab_ref:
    refs = load_references()
    ref_pred = reference_sweep(family)
    st.markdown("NACA 0012 at Re = 6×10⁶. **Experiment:** Ladson (NASA TM 4074, tripped transition) and "
                "Gregory & O'Reilly (NASA R&M 3726, Re = 2.9×10⁶). **CFD:** NASA TMR CFL3D, k-ω SST — the "
                "turbulence model of the training data.")
    agree = reference_agreement(ref_pred, refs)
    lad, tmr = agree[agree.reference.str.startswith("Ladson")], agree[agree.reference.str.startswith("NASA")]
    k = st.columns(4)
    k[0].metric("Median |ΔCl|, experiment", f"{lad.dCl.abs().median():.3f}")
    k[1].metric("Median ΔCd/Cd, experiment", f"{lad.dCd_pct.median():+.1f}%")
    k[2].metric("Mean |ΔCl|, SST CFD", f"{tmr.dCl.abs().mean():.3f}")
    k[3].metric("Mean |ΔCd|/Cd, SST CFD", f"{tmr.dCd_pct.abs().mean():.1f}%")
    r1, r2 = st.columns(2)
    r1.plotly_chart(reference_coeff_figure(ref_pred, refs, "Cl", family), width="stretch")
    r2.plotly_chart(reference_coeff_figure(ref_pred, refs, "Cd", family), width="stretch")
    if quantities and has_curve_models(family, TASK):
        ref_alpha = st.radio("Wall distributions at α (deg)", REF_ALPHAS, index=1, horizontal=True,
                             key="ref_alpha")
        ref_surf = reference_surface(family, float(ref_alpha))
        rcols = st.columns(len(quantities))
        for col, q in zip(rcols, quantities):
            col.plotly_chart(reference_surface_figure(ref_surf, refs, q, ref_alpha), width="stretch")
    st.caption("Deviation from the SST solution measures surrogate error; the remaining offset from experiment "
               "(Cd over-predicted by fully turbulent SST) is inherited from the training CFD. "
               f"Post-stall measurements (α > {REF_ALPHA_MAX:g}°) are omitted.")
    with st.expander("Point-wise comparison"):
        st.dataframe(agree.round({"alpha_deg": 2, "Cl_ref": 4, "Cd_ref": 5, "Cl_pred": 4, "Cd_pred": 5,
                                  "dCl": 4, "dCd_pct": 2}), width="stretch", hide_index=True)

with tab_test:
    preds_all = load_test_predictions()
    preds_all = preds_all[preds_all.task == TASK] if not preds_all.empty else preds_all
    if preds_all.empty:
        st.info("No test-set predictions found. Run `uv run python scripts/10_global_validation.py`.")
    else:
        test = preds_all[preds_all.family == family]
        err = case_errors(test)
        st.markdown(f"{len(test)} AirfRANS cases withheld from training and hyper-parameter selection "
                    "(official 800/200 split). Errors are relative to the AirfRANS k-ω SST solutions.")
        k = st.columns(5)
        k[0].metric("R² Cl", f"{r_squared(test.Cl_true, test.Cl_pred):.4f}")
        k[1].metric("Median |ΔCl|", f"{err.dCl.abs().median():.4f}")
        k[2].metric("R² Cd", f"{r_squared(test.Cd_true, test.Cd_pred):.4f}")
        k[3].metric("Median |ΔCd|/Cd", f"{err.dCd_pct.abs().median():.2f}%")
        cov = [c for c in (coverage95(test, "Cl"), coverage95(test, "Cd")) if c is not None]
        k[4].metric("95% PI coverage", "—" if not cov else f"{100 * np.mean(cov):.0f}%",
                    help="Fraction of test cases inside the 95% prediction interval (mean of Cl and Cd); "
                         "nominal value 95%. GP and KRG only.")
        p1, p2 = st.columns(2)
        p1.plotly_chart(parity_figure(test, "Cl", family), width="stretch")
        p2.plotly_chart(parity_figure(test, "Cd", family), width="stretch")
        e1, e2 = st.columns(2)
        e1.plotly_chart(error_vs_alpha_figure(test, "Cl"), width="stretch")
        e2.plotly_chart(error_vs_alpha_figure(test, "Cd"), width="stretch")

        st.subheader("Comparison of surrogate families")
        summary = family_summary(preds_all)
        st.dataframe(summary.style.format({c: "{:.4f}" for c in summary.columns if c != "Model"} |
                                          {"Median |ΔCd|/Cd (%)": "{:.2f}", "P90 |ΔCd|/Cd (%)": "{:.2f}",
                                           "95% PI coverage": lambda v: "—" if pd.isna(v) else f"{100 * v:.0f}%"})
                     .apply(lambda r: ["font-weight: bold" if r.Model == FAMILY_LABELS[family] else ""] * len(r),
                            axis=1), width="stretch", hide_index=True)

        cm = load_curve_metrics()
        cm = cm[cm.task == TASK] if not cm.empty else cm
        if not cm.empty:
            with st.expander("Wall distributions (Cp, Cf): test-set metrics"):
                st.dataframe(cm.assign(family=cm.family.map(FAMILY_LABELS))[
                    ["family", "quantity", "RMSE", "R2_station_mean", "coverage_2sigma",
                     "separation_detect_accuracy"]].rename(columns={
                        "family": "Model", "quantity": "Quantity", "R2_station_mean": "R²",
                        "coverage_2sigma": "±2σ coverage", "separation_detect_accuracy": "Separation detected"})
                    .round(4), width="stretch", hide_index=True)
