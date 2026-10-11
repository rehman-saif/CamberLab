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
    assert [m.label for m in at.metric] == ["Best L/D", "AoA at best L/D", "Max Cl in sweep",
                                            "Min Cd in sweep", "Zero-lift α"]


def test_app_accuracy_page_all_families():
    at = _run()
    at.switch_page("app_pages/accuracy.py").run()
    assert not at.exception, at.exception
    assert any(t.value == "Model accuracy" for t in at.title)
    labels = [m.label for m in at.metric]
    assert "Median |ΔCl| vs Ladson" in labels
    assert "Cd median |error|" in labels
    for family in ("rf", "mlp", "krg"):
        at.sidebar.selectbox[0].set_value(family).run()
        assert not at.exception, (family, at.exception)


def test_app_five_digit_and_invalid_code():
    at = _run("23012")
    assert not at.exception, at.exception
    at = _run("99")
    assert not at.exception, at.exception
    assert at.error, "invalid NACA code should show an error"
