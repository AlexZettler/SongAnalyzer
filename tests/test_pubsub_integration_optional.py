"""Opt-in Pub/Sub emulator checks (not run in default CI).

Set ``RUN_PUBSUB_INTEGRATION=1`` and ``PUBSUB_EMULATOR_HOST`` after ``messaging setup-pubsub``.
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_PUBSUB_INTEGRATION") != "1",
    reason="set RUN_PUBSUB_INTEGRATION=1 and PUBSUB_EMULATOR_HOST for live Pub/Sub checks",
)
def test_pubsub_client_import_and_project() -> None:
    pytest.importorskip("google.cloud.pubsub_v1")
    assert os.environ.get("PUBSUB_EMULATOR_HOST"), "PUBSUB_EMULATOR_HOST must be set with the emulator"
