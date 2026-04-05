"""Corpus expansion from ``SongRequestPayload`` (local audio path)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from song_analyzer.corpus.song_request import process_song_request
from song_analyzer.messaging.payloads import SongRequestPayload


def test_process_song_request_local_audio(tmp_path: Path) -> None:
    wav = tmp_path / "in.wav"
    sr = 8000
    y = np.zeros(sr, dtype=np.float32)
    sf.write(str(wav), y, sr)

    req = SongRequestPayload(
        request_id="req-1",
        corpus_root=str(tmp_path / "corpus"),
        local_audio_path=str(wav),
        title="T",
        artist="A",
    )
    done = process_song_request(req)
    assert done.status == "ok"
    assert done.track_id is not None
    assert done.audio_relpath is not None
