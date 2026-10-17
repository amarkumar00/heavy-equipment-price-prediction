from __future__ import annotations

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_streamlit_app_starts_without_uncaught_exception() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    assert not app.exception
    assert any("EquipValue" in title.value for title in app.title)
    estimate_buttons = [
        button for button in app.button if button.label == "Estimate selling price"
    ]
    if estimate_buttons:
        estimate_buttons[0].click()
        app.run(timeout=60)
        assert not app.exception
        assert any(metric.label == "Estimated selling price" for metric in app.metric)
