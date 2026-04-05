from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

from song_analyzer.corpus.db import open_corpus_db
from song_analyzer.corpus.layout import CorpusLayout
from song_analyzer.corpus.types import ManifestRow, TrackRecord, utc_now_iso


class TrackStore:
    def __init__(self, conn: sqlite3.Connection, layout: CorpusLayout) -> None:
        self._conn = conn
        self._layout = layout

    @classmethod
    def open(cls, root: str | Path) -> TrackStore:
        root_p = Path(root).resolve()
        conn = open_corpus_db(root_p)
        return cls(conn, CorpusLayout(root_p))

    def close(self) -> None:
        self._conn.close()

    def insert_track(self, rec: TrackRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO tracks (
                track_id, mbid, isrc, title, artist, source, source_id,
                audio_relpath, duration_seconds, file_checksum, fingerprint,
                lyrics_text, lyrics_source, raw_metadata_json, fetched_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.track_id,
                rec.mbid,
                rec.isrc,
                rec.title,
                rec.artist,
                rec.source,
                rec.source_id,
                rec.audio_relpath,
                rec.duration_seconds,
                rec.file_checksum,
                rec.fingerprint,
                rec.lyrics_text,
                rec.lyrics_source,
                rec.raw_metadata_json,
                rec.fetched_at,
                rec.created_at,
            ),
        )
        self._conn.commit()

    def get_track(self, track_id: str) -> TrackRecord | None:
        cur = self._conn.execute("SELECT * FROM tracks WHERE track_id = ?", (track_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return TrackRecord.from_row(row)

    def iter_tracks_with_audio(self) -> Iterable[TrackRecord]:
        cur = self._conn.execute(
            "SELECT * FROM tracks WHERE audio_relpath IS NOT NULL AND audio_relpath != ''"
        )
        for row in cur:
            yield TrackRecord.from_row(row)

    def update_lyrics(
        self,
        track_id: str,
        text: str,
        *,
        lyrics_source: str,
    ) -> None:
        self._conn.execute(
            """
            UPDATE tracks SET lyrics_text = ?, lyrics_source = ?, fetched_at = ?
            WHERE track_id = ?
            """,
            (text, lyrics_source, utc_now_iso(), track_id),
        )
        self._conn.commit()

    def update_metadata(
        self,
        track_id: str,
        *,
        title: str | None = None,
        artist: str | None = None,
        raw_metadata_json: str | None = None,
    ) -> None:
        parts: list[str] = []
        vals: list[object] = []
        if title is not None:
            parts.append("title = ?")
            vals.append(title)
        if artist is not None:
            parts.append("artist = ?")
            vals.append(artist)
        if raw_metadata_json is not None:
            parts.append("raw_metadata_json = ?")
            vals.append(raw_metadata_json)
        if not parts:
            return
        parts.append("fetched_at = ?")
        vals.append(utc_now_iso())
        vals.append(track_id)
        sql = "UPDATE tracks SET " + ", ".join(parts) + " WHERE track_id = ?"
        self._conn.execute(sql, vals)
        self._conn.commit()

    def clear_manifest(self) -> None:
        self._conn.execute("DELETE FROM training_manifest_rows")
        self._conn.commit()

    def insert_manifest_rows(
        self,
        rows: Sequence[ManifestRow],
        *,
        demucs_model: str,
        teacher_checkpoint: str | None,
        built_at: str,
    ) -> None:
        for r in rows:
            self._conn.execute(
                """
                INSERT INTO training_manifest_rows (
                    track_id, stem_name, audio_relpath, family_id, family_name,
                    confidence, demucs_model, teacher_checkpoint, built_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.track_id,
                    r.stem_name,
                    r.audio_relpath,
                    r.family_id,
                    r.family_name,
                    r.confidence,
                    demucs_model,
                    teacher_checkpoint,
                    built_at,
                ),
            )
        self._conn.commit()

    def resolve_audio_path(self, relpath: str) -> Path:
        p = Path(relpath)
        if p.is_absolute():
            return p
        return (self._layout.root / p).resolve()


def write_manifest_csv(path: Path, rows: Sequence[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "audio_path",
                "stem_name",
                "family_id",
                "family_name",
                "confidence",
                "track_id",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.audio_relpath,
                    r.stem_name,
                    r.family_id,
                    r.family_name,
                    r.confidence,
                    r.track_id,
                ]
            )


def read_manifest_csv(path: Path) -> list[ManifestRow]:
    out: list[ManifestRow] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(
                ManifestRow(
                    track_id=row["track_id"],
                    stem_name=row["stem_name"],
                    audio_relpath=row["audio_path"],
                    family_id=int(row["family_id"]),
                    family_name=row["family_name"],
                    confidence=float(row["confidence"]),
                )
            )
    return out
