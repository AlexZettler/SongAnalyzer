"""Colored console + optional file logging for NSynth prepare/train CLIs."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Literal

_NSYNTH_HANDLER_ATTR = "_song_analyzer_nsynth"

# When root is DEBUG, these loggers are often left at WARNING by libraries unless set explicitly.
_BEAM_TF_TDS_DEBUG_LOGGERS = (
    "apache_beam",
    "tensorflow",
    "tensorflow_datasets",
)

Profile = Literal["prepare", "train"]


def parse_train_log_level(name: str) -> int:
    levels = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING}
    key = name.strip().upper()
    if key not in levels:
        raise ValueError(f"Invalid log level {name!r}; expected DEBUG, INFO, or WARNING")
    return levels[key]


def default_log_path(profile: Profile) -> Path:
    """Directory: Windows LOCALAPPDATA\\song_analyzer\\logs, else ~/.cache/song_analyzer/logs."""
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        log_dir = base / "song_analyzer" / "logs"
    else:
        log_dir = Path.home() / ".cache" / "song_analyzer" / "logs"
    name = "nsynth_prepare.log" if profile == "prepare" else "nsynth_train.log"
    return log_dir / name


def _remove_tagged_root_handlers() -> None:
    root = logging.getLogger()
    to_drop = [h for h in root.handlers if getattr(h, _NSYNTH_HANDLER_ATTR, False)]
    for h in to_drop:
        root.removeHandler(h)
        h.close()


def configure_nsynth_logging(
    level: int = logging.INFO,
    *,
    profile: Profile,
    log_file: Path | None = None,
    no_log_file: bool = False,
) -> None:
    """Attach colored stderr + optional file handlers to the root logger; replace prior NSynth handlers."""
    _remove_tagged_root_handlers()

    root = logging.getLogger()
    root.setLevel(level)

    stream_fmt_plain = "%(levelname)s %(name)s: %(message)s"
    if sys.stderr.isatty():
        try:
            import colorlog as _colorlog

            stream_handler: logging.Handler = _colorlog.StreamHandler(sys.stderr)
            stream_handler.setFormatter(
                _colorlog.ColoredFormatter(
                    "%(log_color)s%(levelname)s%(reset)s %(name)s: %(message)s",
                    log_colors={
                        "DEBUG": "cyan",
                        "INFO": "green",
                        "WARNING": "yellow",
                        "ERROR": "red",
                        "CRITICAL": "bold_red",
                    },
                )
            )
        except ModuleNotFoundError:
            stream_handler = logging.StreamHandler(sys.stderr)
            stream_handler.setFormatter(logging.Formatter(stream_fmt_plain))
    else:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(logging.Formatter(stream_fmt_plain))
    setattr(stream_handler, _NSYNTH_HANDLER_ATTR, True)
    stream_handler.setLevel(level)
    root.addHandler(stream_handler)

    if not no_log_file:
        path = Path(log_file) if log_file is not None else default_log_path(profile)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        setattr(file_handler, _NSYNTH_HANDLER_ATTR, True)
        file_handler.setLevel(level)
        root.addHandler(file_handler)

    if level <= logging.DEBUG:
        for name in _BEAM_TF_TDS_DEBUG_LOGGERS:
            logging.getLogger(name).setLevel(logging.DEBUG)
    else:
        for name in _BEAM_TF_TDS_DEBUG_LOGGERS:
            logging.getLogger(name).setLevel(logging.NOTSET)
