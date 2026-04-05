from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path

import soundfile as sf

from song_analyzer.corpus.db import init_corpus
from song_analyzer.corpus.layout import AUDIO_SUBDIR, CorpusLayout
from song_analyzer.corpus.store import TrackStore
from song_analyzer.corpus.types import TrackRecord, utc_now_iso


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def import_audio_file(
    corpus_root: str | Path,
    source_path: str | Path,
    *,
    copy: bool = True,
    mbid: str | None = None,
    title: str | None = None,
    artist: str | None = None,
    source: str = "import",
    source_id: str | None = None,
    musicbrainz_enrich: bool = False,
    musicbrainz_user_agent: str | None = None,
) -> TrackRecord:
    """
    Register an audio file in the corpus (copies into ``audio/`` by default).

    If ``musicbrainz_enrich`` and ``mbid`` are set, fetches title/artist from MusicBrainz
    (requires ``pip install -e ".[corpus]"``).
    """
    layout = CorpusLayout(Path(corpus_root).resolve())
    if not layout.db_path.is_file():
        init_corpus(layout.root)

    src = Path(source_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(src)

    track_id = str(uuid.uuid4())
    suffix = src.suffix.lower() or ".wav"
    if suffix not in {".wav", ".flac", ".ogg", ".mp3", ".m4a"}:
        suffix = ".wav"

    raw_meta: dict[str, object] | None = None
    t, a = title, artist

    if musicbrainz_enrich and mbid:
        from song_analyzer.corpus.connectors.musicbrainz import MusicBrainzClient

        client = MusicBrainzClient(
            user_agent=musicbrainz_user_agent
            or "SongAnalyzerCorpus/0.1 (local corpus tool)"
        )
        mb_rec = client.fetch_recording(mbid)
        t = t or mb_rec.title
        a = a or mb_rec.artist
        raw_meta = mb_rec.raw

    if copy:
        dest_name = f"{track_id}{suffix}"
        dest = layout.audio_dir / dest_name
        layout.audio_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        audio_path = dest
        rel = f"{AUDIO_SUBDIR}/{dest_name}"
    else:
        audio_path = src
        rel = str(audio_path)

    checksum = _sha256_file(audio_path)
    duration = float(sf.info(str(audio_path)).duration)

    created = utc_now_iso()
    rec = TrackRecord(
        track_id=track_id,
        mbid=mbid,
        isrc=None,
        title=t,
        artist=a,
        source=source,
        source_id=source_id or src.name,
        audio_relpath=rel,
        duration_seconds=duration,
        file_checksum=checksum,
        fingerprint=None,
        lyrics_text=None,
        lyrics_source=None,
        raw_metadata_json=json.dumps(raw_meta) if raw_meta else None,
        fetched_at=utc_now_iso() if raw_meta else None,
        created_at=created,
    )

    store = TrackStore.open(layout.root)
    try:
        store.insert_track(rec)
    finally:
        store.close()

    return rec
