"""Process song expansion requests (metadata, lyrics, optional local audio)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from song_analyzer.corpus.db import init_corpus
from song_analyzer.corpus.ingest import import_audio_file
from song_analyzer.corpus.store import TrackStore
from song_analyzer.corpus.types import TrackRecord, utc_now_iso
from song_analyzer.messaging.payloads import SongCompletePayload, SongRequestPayload


def process_song_request(req: SongRequestPayload) -> SongCompletePayload:
    """
    Register or enrich a track: optional MusicBrainz, optional lyrics connector, optional audio file.

    Lyrics connectors may raise NotImplementedError (e.g. in-tree genius stub).
    """
    root = Path(req.corpus_root).resolve()
    if not root.exists():
        init_corpus(root)

    lyrics_fetched = False
    try:
        if req.local_audio_path:
            rec = import_audio_file(
                root,
                req.local_audio_path,
                copy=req.copy_audio,
                mbid=req.mbid,
                title=req.title,
                artist=req.artist,
                source=req.source,
                source_id=req.source_id,
                musicbrainz_enrich=req.musicbrainz_enrich and bool(req.mbid),
                musicbrainz_user_agent=req.musicbrainz_user_agent
                or "SongAnalyzer/0.1 (pubsub song.request)",
            )
            track_id = rec.track_id
            audio_relpath = rec.audio_relpath
        else:
            raw_meta: dict[str, Any] | None
            if req.mbid:
                from song_analyzer.corpus.connectors.musicbrainz import MusicBrainzClient

                client = MusicBrainzClient(
                    user_agent=req.musicbrainz_user_agent
                    or "SongAnalyzer/0.1 (pubsub song.request)"
                )
                mb_rec = client.fetch_recording(req.mbid)
                meta_title: str | None = req.title or mb_rec.title
                meta_artist: str | None = req.artist or mb_rec.artist
                raw_meta = mb_rec.raw
            else:
                meta_title = req.title
                meta_artist = req.artist
                raw_meta = None

            track_id = str(uuid.uuid4())
            created = utc_now_iso()
            rec = TrackRecord(
                track_id=track_id,
                mbid=req.mbid,
                isrc=None,
                title=meta_title,
                artist=meta_artist,
                source=req.source,
                source_id=req.source_id or req.request_id,
                audio_relpath=None,
                duration_seconds=None,
                file_checksum=None,
                fingerprint=None,
                lyrics_text=None,
                lyrics_source=None,
                raw_metadata_json=json.dumps(raw_meta) if raw_meta else None,
                fetched_at=utc_now_iso() if raw_meta else None,
                created_at=created,
            )
            store = TrackStore.open(root)
            try:
                store.insert_track(rec)
            finally:
                store.close()
            audio_relpath = None

        store = TrackStore.open(root)
        try:
            tr = store.get_track(track_id)
            if tr is None:
                return SongCompletePayload(
                    request_id=req.request_id,
                    status="error",
                    error=f"track_id {track_id} missing after insert",
                )
            from song_analyzer.corpus.connectors.stub import get_lyrics_connector

            conn = get_lyrics_connector(req.lyrics_connector)
            try:
                result = conn.fetch_lyrics(
                    title=tr.title, artist=tr.artist, source_id=tr.source_id
                )
                store.update_lyrics(
                    track_id,
                    result.text,
                    lyrics_source=f"{req.lyrics_connector}:{result.attribution}",
                )
                lyrics_fetched = True
            except NotImplementedError:
                pass
        finally:
            store.close()

        return SongCompletePayload(
            request_id=req.request_id,
            status="ok",
            track_id=track_id,
            audio_relpath=audio_relpath,
            lyrics_fetched=lyrics_fetched,
        )
    except Exception as e:  # noqa: BLE001 — surface to completion topic
        return SongCompletePayload(
            request_id=req.request_id,
            status="error",
            error=str(e),
        )
