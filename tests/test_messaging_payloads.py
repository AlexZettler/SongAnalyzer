"""Round-trip JSON for Pub/Sub payloads."""

from __future__ import annotations

from song_analyzer.messaging.payloads import (
    HpoCompletePayload,
    HpoRequestPayload,
    SongCompletePayload,
    SongRequestPayload,
    TrainRequestPayload,
)


def test_song_request_roundtrip() -> None:
    m = SongRequestPayload(request_id="a", corpus_root="/tmp/c")
    raw = m.model_dump_json()
    assert SongRequestPayload.model_validate_json(raw).request_id == "a"


def test_hpo_complete_error_roundtrip() -> None:
    m = HpoCompletePayload(exploration_id="e1", status="error", error="boom")
    assert HpoCompletePayload.model_validate_json(m.model_dump_json()).error == "boom"


def test_train_request_roundtrip() -> None:
    m = TrainRequestPayload(
        train_job_id="j1",
        study_name="nsynth_family_x__exp1",
        out="/tmp/m.pt",
    )
    assert TrainRequestPayload.model_validate_json(m.model_dump_json()).study_name == m.study_name


def test_song_complete_ok() -> None:
    m = SongCompletePayload(
        request_id="r",
        status="ok",
        track_id="t",
        lyrics_fetched=False,
    )
    assert SongCompletePayload.model_validate_json(m.model_dump_json()).track_id == "t"
