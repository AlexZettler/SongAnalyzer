"""Beam streaming helpers.

Pipeline entrypoints live in ``song_analyzer.beam.pipelines`` (requires ``apache-beam``).
"""

from song_analyzer.beam.handlers import (
    handle_hpo_bytes,
    handle_song_bytes,
    handle_train_bytes,
)

__all__ = [
    "handle_hpo_bytes",
    "handle_song_bytes",
    "handle_train_bytes",
]
