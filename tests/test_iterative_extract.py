import numpy as np

from song_analyzer.pitch import iterative_extract as ie
from song_analyzer.schema import NoteEvent, SoloSegment


def test_iterative_extract_stops_after_peel(monkeypatch):
    sr = 8000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
    y = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    calls = {"n": 0}

    def fake_transcribe(_audio, _sr, stem_id, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return [
                NoteEvent(
                    start_time_s=0.1,
                    end_time_s=0.35,
                    midi_pitch=69,
                    velocity=0.9,
                    stem_id=stem_id,
                )
            ], "fake"
        return [], "fake"

    monkeypatch.setattr(ie, "transcribe_stem", fake_transcribe)
    notes, meta = ie.extract_notes_iteratively_for_stem(y, sr, "other", max_iterations=8)
    assert len(notes) == 1
    assert meta["stopped_reason"] == "no_notes"
    assert notes[0].midi_pitch == 69


def test_iterative_solo_filter_can_empty_pool(monkeypatch):
    sr = 8000
    y = np.random.randn(int(sr * 0.2)).astype(np.float32) * 0.01

    def fake_transcribe(_audio, _sr, stem_id, **kwargs):
        return [
            NoteEvent(
                start_time_s=0.0,
                end_time_s=0.1,
                midi_pitch=60,
                velocity=0.5,
                stem_id=stem_id,
            )
        ], "fake"

    monkeypatch.setattr(ie, "transcribe_stem", fake_transcribe)
    solo = [SoloSegment(start_time_s=0.5, end_time_s=0.8, stem_id="other", dominance=0.9)]
    notes, meta = ie.extract_notes_iteratively_for_stem(
        y,
        sr,
        "other",
        restrict_to_solo=True,
        solo_segments=solo,
    )
    assert notes == []
    assert meta["stopped_reason"] == "solo_filter_empty"
