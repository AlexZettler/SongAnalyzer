"""Logging and memory-sampling helpers for NSynth / Beam prepare paths."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from song_analyzer.instruments.nsynth_train_loop import _rss_sampling_log_context
from song_analyzer.instruments.nsynth_logging import configure_nsynth_logging
from song_analyzer.instruments.train_nsynth import configure_train_logging


@pytest.fixture(autouse=True)
def _reset_library_loggers() -> Iterator[None]:
    for name in ("apache_beam", "tensorflow", "tensorflow_datasets"):
        logging.getLogger(name).setLevel(logging.NOTSET)
    yield


def test_configure_train_debug_sets_beam_tf_tfds_loggers() -> None:
    configure_train_logging(logging.DEBUG, no_log_file=True)
    assert logging.getLogger().level == logging.DEBUG
    for name in ("apache_beam", "tensorflow", "tensorflow_datasets"):
        assert logging.getLogger(name).level == logging.DEBUG


def test_configure_train_info_does_not_force_debug_on_libraries() -> None:
    configure_train_logging(logging.INFO, no_log_file=True)
    assert logging.getLogger().level == logging.INFO
    for name in ("apache_beam", "tensorflow", "tensorflow_datasets"):
        assert logging.getLogger(name).level == logging.NOTSET


def test_configure_nsynth_writes_log_file(tmp_path) -> None:
    path = tmp_path / "run.log"
    configure_nsynth_logging(
        logging.INFO,
        profile="prepare",
        log_file=path,
        no_log_file=False,
    )
    logging.getLogger("test_nsynth_log").info("marker_line_for_file_test")
    text = path.read_text(encoding="utf-8")
    assert "marker_line_for_file_test" in text
    assert "INFO" in text


def test_rss_context_noop_when_interval_none() -> None:
    with _rss_sampling_log_context(None):
        pass


def test_rss_context_logs_at_least_once(caplog: pytest.LogCaptureFixture) -> None:
    pytest.importorskip("psutil", reason="psutil is part of the [train] extra")
    caplog.set_level(logging.INFO)
    with _rss_sampling_log_context(0.05):
        pass
    assert any("process_memory rss_mb=" in r.message for r in caplog.records)


def test_rss_context_logs_with_mocked_psutil(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    mock_mi = MagicMock()
    mock_mi.rss = 10 * 1024**2
    mock_mi.vms = 20 * 1024**2
    mock_proc = MagicMock()
    mock_proc.memory_info.return_value = mock_mi
    mock_psutil = MagicMock()
    mock_psutil.Process.return_value = mock_proc

    with patch.dict(sys.modules, {"psutil": mock_psutil}):
        with _rss_sampling_log_context(0.05):
            pass
    assert any("process_memory rss_mb=" in r.message for r in caplog.records)
