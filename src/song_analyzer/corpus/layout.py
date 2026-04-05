from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CORPUS_DB_FILENAME = "corpus.sqlite3"
AUDIO_SUBDIR = "audio"
PSEUDO_STEMS_SUBDIR = "pseudo_stems"


@dataclass(frozen=True)
class CorpusLayout:
    """Standard directory layout under a corpus root."""

    root: Path

    @property
    def db_path(self) -> Path:
        return self.root / CORPUS_DB_FILENAME

    @property
    def audio_dir(self) -> Path:
        return self.root / AUDIO_SUBDIR

    @property
    def pseudo_stems_dir(self) -> Path:
        return self.root / PSEUDO_STEMS_SUBDIR

    def audio_file(self, track_id: str, suffix: str) -> Path:
        ext = suffix if suffix.startswith(".") else f".{suffix}"
        return self.audio_dir / f"{track_id}{ext}"
