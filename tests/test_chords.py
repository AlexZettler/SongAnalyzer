from song_analyzer.chords.detect import build_chord_timeline, chord_label_for_pcs
from song_analyzer.schema import NoteEvent


def test_chord_label_major_triad():
    label = chord_label_for_pcs([0, 4, 7], bass_pc=0)
    assert "C" in label


def test_build_chord_timeline_merges():
    notes = [
        NoteEvent(start_time_s=0.0, end_time_s=0.5, midi_pitch=60, velocity=None, stem_id="other"),
        NoteEvent(start_time_s=0.0, end_time_s=0.5, midi_pitch=64, velocity=None, stem_id="other"),
        NoteEvent(start_time_s=0.0, end_time_s=0.5, midi_pitch=67, velocity=None, stem_id="other"),
    ]
    segs = build_chord_timeline(notes, stem_id="other", hop_s=0.05, duration_s=0.6)
    assert len(segs) >= 1
    assert any("C" in s.chord_label for s in segs)
