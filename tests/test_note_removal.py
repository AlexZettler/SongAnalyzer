import numpy as np

from song_analyzer.editing.note_removal import attenuate_note_harmonics


def test_attenuate_reduces_energy_near_harmonics():
    sr = 22050
    t = np.arange(sr * 0.5, dtype=np.float32) / sr
    f0 = 440.0
    y = 0.3 * np.sin(2 * np.pi * f0 * t).astype(np.float32)
    y2 = attenuate_note_harmonics(
        y,
        sr,
        midi_pitch=69,
        start_s=0.0,
        end_s=0.5,
        n_fft=2048,
        hop_length=512,
        attenuation=0.01,
    )
    assert y2.shape == y.shape
    assert float(np.sqrt(np.mean(y2**2))) < float(np.sqrt(np.mean(y**2)))
