import numpy as np

from song_analyzer.editing.note_removal import attenuate_note_harmonics, expander_noise_gate_region


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


def test_noise_gate_reduces_quiet_tail_in_region():
    sr = 8000
    n = sr
    y = np.zeros(n, dtype=np.float32)
    y[: sr // 2] = 0.5 * np.sin(2 * np.pi * 200 * np.arange(sr // 2, dtype=np.float32) / sr)
    y[sr // 2 :] = 0.02
    y2 = expander_noise_gate_region(
        y,
        sr,
        0.4,
        0.95,
        pad_s=0.0,
        frame_ms=20.0,
        threshold_db_below_peak=12.0,
        below_gain=0.01,
    )
    e_before = float(np.sqrt(np.mean(y[sr // 2 :] ** 2)))
    e_after = float(np.sqrt(np.mean(y2[sr // 2 :] ** 2)))
    assert e_after < e_before
