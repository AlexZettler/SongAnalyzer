from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class TrackRecord:
    track_id: str
    created_at: str
    mbid: str | None = None
    isrc: str | None = None
    title: str | None = None
    artist: str | None = None
    source: str | None = None
    source_id: str | None = None
    audio_relpath: str | None = None
    duration_seconds: float | None = None
    file_checksum: str | None = None
    fingerprint: str | None = None
    lyrics_text: str | None = None
    lyrics_source: str | None = None
    raw_metadata_json: str | None = None
    fetched_at: str | None = None

    @classmethod
    def from_row(cls, row: Any) -> TrackRecord:
        return cls(
            track_id=str(row["track_id"]),
            mbid=row["mbid"],
            isrc=row["isrc"],
            title=row["title"],
            artist=row["artist"],
            source=row["source"],
            source_id=row["source_id"],
            audio_relpath=row["audio_relpath"],
            duration_seconds=row["duration_seconds"],
            file_checksum=row["file_checksum"],
            fingerprint=row["fingerprint"],
            lyrics_text=row["lyrics_text"],
            lyrics_source=row["lyrics_source"],
            raw_metadata_json=row["raw_metadata_json"],
            fetched_at=row["fetched_at"],
            created_at=str(row["created_at"]),
        )


@dataclass(frozen=True)
class ManifestRow:
    track_id: str
    stem_name: str
    audio_relpath: str
    family_id: int
    family_name: str
    confidence: float
