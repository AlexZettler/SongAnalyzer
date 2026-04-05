"""FastAPI app wiring (optional ``.[api]`` extra)."""

from __future__ import annotations

import pytest


def test_fastapi_app_import() -> None:
    pytest.importorskip("fastapi")
    from song_analyzer.api.app import app

    assert app.title
    routes = {r.path for r in app.routes}
    assert any("/analyze" in p for p in routes)
