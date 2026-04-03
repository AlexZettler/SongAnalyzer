import json
from pathlib import Path

import numpy as np
import soundfile as sf

import song_analyzer.pipeline as pipeline


def test_analyze_mix_mocked_separation(tmp_path, monkeypatch):
    sr = 8000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
    mix = 0.05 * np.sin(2 * np.pi * 220 * t).astype(np.float32)

    inp = tmp_path / "in.wav"
    sf.write(inp, mix, sr, subtype="FLOAT")

    def fake_sep(m, s, **kwargs):
        assert len(m) == len(t)
        z = np.zeros_like(m, dtype=np.float32)
        return {"other": m.copy(), "bass": z, "drums": z, "vocals": z}, s

    monkeypatch.setattr(pipeline, "separate_to_dict", fake_sep)

    out_dir = tmp_path / "out"
    result = pipeline.analyze_mix(
        inp,
        out_dir,
        device="cpu",
        write_stem_wavs=True,
        demucs_shifts=0,
    )

    assert (out_dir / "analysis.json").is_file()
    data = json.loads((out_dir / "analysis.json").read_text(encoding="utf-8"))
    assert data["duration_s"] > 0
    assert len(data["stems"]) == 4
    assert len(data["notes"]) >= 0
    assert result.meta["pitch_transcription"] in ("basic_pitch", "librosa_pyin")
