"""The Streamlit app runs top to bottom headlessly without raising."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (PROJECT_ROOT / "models" / "full" / "envelope.json").exists(),
                                reason="run scripts/09_train_surrogates.py first")


def _run(naca: str | None = None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=120)
    at.run()
    if naca is not None:
        at.sidebar.text_input[0].set_value(naca).run()
    return at


def test_app_runs_default():
    at = _run()
    assert not at.exception, at.exception
    assert any("ODbL" in c.value for c in at.caption)
    assert [m.label for m in at.metric] == ["(L/D) max", "α at (L/D) max", "Cl max (sweep)",
                                            "Cd min (sweep)", "Zero-lift α₀"]


def test_app_accuracy_page_all_families():
    at = _run("4415")
    at.switch_page("app_pages/accuracy.py").run()
    assert not at.exception, at.exception
    # only the model family is editable; the section is fixed to the validation case
    assert at.sidebar.text_input[0].value == "0012" and at.sidebar.text_input[0].disabled
    assert all(s.disabled for s in at.sidebar.slider)
    assert not at.sidebar.selectbox[0].disabled
    assert any(t.value == "Model accuracy" for t in at.title)
    labels = [m.label for m in at.metric]
    assert "Median |ΔCl|, experiment" in labels
    assert "Median |ΔCd|/Cd" in labels
    for family in ("rf", "mlp", "krg"):
        at.sidebar.selectbox[0].set_value(family).run()
        assert not at.exception, (family, at.exception)
    at.switch_page("app_pages/prediction.py").run()
    assert at.sidebar.text_input[0].value == "4415", "CamberLab inputs must survive the page switch"


def test_app_five_digit_and_invalid_code():
    at = _run("23012")
    assert not at.exception, at.exception
    at = _run("99")
    assert not at.exception, at.exception
    assert at.error, "invalid NACA code should show an error"
