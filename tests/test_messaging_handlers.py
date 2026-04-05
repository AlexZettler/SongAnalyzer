"""Beam handler entrypoints (lightweight mocks)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from song_analyzer.beam.handlers import handle_hpo_bytes, handle_song_bytes
from song_analyzer.messaging.payloads import HpoRequestPayload, SongRequestPayload


def test_handle_song_bytes_minimal(tmp_path: Path) -> None:
    import numpy as np
    import soundfile as sf

    wav = tmp_path / "x.wav"
    sf.write(str(wav), np.zeros(4000, dtype=np.float32), 8000)
    req = SongRequestPayload(
        request_id="r",
        corpus_root=str(tmp_path / "c"),
        local_audio_path=str(wav),
    )
    out = handle_song_bytes(req.model_dump_json().encode())
    assert b"ok" in out


@patch("song_analyzer.instruments.tune_nsynth.run_nsynth_hpo_job")
def test_handle_hpo_bytes_mock(mock_job: MagicMock, tmp_path: Path) -> None:
    from song_analyzer.instruments.tune_nsynth import HpoJobResult

    mock_job.return_value = HpoJobResult(
        exploration_id="e1",
        study_name="nsynth_family_abcd__e1",
        best_params={"lr": 0.001},
        best_value=0.42,
        archived_db_path=None,
        trials_export_csv=tmp_path / "t.csv",
        out_checkpoint=None,
    )
    req = HpoRequestPayload(
        exploration_id="e1",
        n_trials=1,
        skip_final_train=True,
        archive_tune_db_before=False,
    )
    out = handle_hpo_bytes(req.model_dump_json().encode())
    assert b"ok" in out
    assert b"nsynth_family" in out
