from __future__ import annotations

import sqlite3
from pathlib import Path

from song_analyzer.corpus.layout import CORPUS_DB_FILENAME, CorpusLayout

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tracks (
    track_id TEXT PRIMARY KEY,
    mbid TEXT,
    isrc TEXT,
    title TEXT,
    artist TEXT,
    source TEXT,
    source_id TEXT,
    audio_relpath TEXT,
    duration_seconds REAL,
    file_checksum TEXT,
    fingerprint TEXT,
    lyrics_text TEXT,
    lyrics_source TEXT,
    raw_metadata_json TEXT,
    fetched_at TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tracks_mbid ON tracks(mbid);
CREATE INDEX IF NOT EXISTS idx_tracks_checksum ON tracks(file_checksum);

CREATE TABLE IF NOT EXISTS training_manifest_rows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id TEXT NOT NULL,
    stem_name TEXT NOT NULL,
    audio_relpath TEXT NOT NULL,
    family_id INTEGER NOT NULL,
    family_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    demucs_model TEXT NOT NULL,
    teacher_checkpoint TEXT,
    built_at TEXT NOT NULL,
    FOREIGN KEY (track_id) REFERENCES tracks(track_id)
);

CREATE INDEX IF NOT EXISTS idx_manifest_track ON training_manifest_rows(track_id);
"""


def init_corpus(root: str | Path) -> CorpusLayout:
    """Create corpus root, subdirectories, and SQLite database if missing."""
    layout = CorpusLayout(Path(root).resolve())
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.audio_dir.mkdir(parents=True, exist_ok=True)
    layout.pseudo_stems_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(layout.db_path)
    try:
        conn.executescript(DDL)
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        conn.commit()
    finally:
        conn.close()
    return layout


def open_corpus_db(root: str | Path) -> sqlite3.Connection:
    """Open the corpus SQLite database (must exist)."""
    r = Path(root).resolve()
    db = r / CORPUS_DB_FILENAME
    if not db.is_file():
        raise FileNotFoundError(
            f"No corpus database at {db}. Run: song-analyzer corpus init --root {r}"
        )
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def corpus_layout_from_root(root: str | Path) -> CorpusLayout:
    return CorpusLayout(Path(root).resolve())


def ensure_corpus_dirs(layout: CorpusLayout) -> None:
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.audio_dir.mkdir(parents=True, exist_ok=True)
    layout.pseudo_stems_dir.mkdir(parents=True, exist_ok=True)
