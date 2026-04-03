import numpy as np

from song_analyzer.solo.detect import detect_solo_segments


def test_solo_detect_finds_dominant_stem():
    sr = 8000
    n = int(sr * 0.5)
    loud = 0.4 * np.sin(2 * np.pi * 300 * np.arange(n, dtype=np.float32) / sr).astype(np.float32)
    quiet = 0.01 * np.random.randn(n).astype(np.float32)
    stems = {"other": loud, "drums": quiet, "bass": quiet, "vocals": quiet}
    solos = detect_solo_segments(
        stems,
        sr,
        win_s=0.08,
        hop_s=0.04,
        dominance_threshold=0.5,
        min_segment_s=0.1,
    )
    assert solos
    assert all(s.stem_id == "other" for s in solos)
