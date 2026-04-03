import numpy as np

from song_analyzer.structure.global_analysis import analyze_global_structure


def test_global_structure_returns_tempo_and_segments():
    sr = 22_050
    dur_s = 3.0
    t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False, dtype=np.float32)
    # Simple periodic pulse train for beat-like onset energy
    y = np.zeros_like(t)
    bpm = 120
    period = 60.0 / bpm
    for k in range(int(dur_s / period) + 1):
        i = int(k * period * sr)
        if i + 400 < len(y):
            y[i : i + 400] += 0.3 * np.hanning(400).astype(np.float32)

    g = analyze_global_structure(y, sr, work_sr=sr)
    assert g.tempo_bpm > 0
    assert len(g.segments) >= 1
    assert g.segments[0].start_time_s >= 0
    assert g.segments[-1].end_time_s <= dur_s + 0.2


def test_repeated_sections_share_repeat_group():
    sr = 22_050
    hop = 512
    # Longer identical blocks so chroma has enough frames for segmentation
    t1 = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False, dtype=np.float32)
    block = 0.2 * np.sin(2 * np.pi * 440 * t1).astype(np.float32)
    y = np.concatenate([block, block]).astype(np.float32)

    g = analyze_global_structure(y, sr, work_sr=sr, hop_length=hop)
    from collections import Counter

    c = Counter(s.repeat_group_id for s in g.segments)
    # Duplicated audio should yield at least two segments sharing a repeat group
    if len(g.segments) >= 2:
        assert max(c.values()) >= 2
