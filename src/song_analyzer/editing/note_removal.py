from __future__ import annotations

import numpy as np
import librosa

from song_analyzer.audio_io import save_wav


def _hz_series_for_note(
    midi_pitch: int,
    n_frames: int,
    sr: int,
    hop: int,
    t_start: float,
    t_end: float,
    times: np.ndarray,
) -> np.ndarray:
    """Piecewise-constant fundamental Hz per STFT frame (note held)."""
    f0 = float(librosa.midi_to_hz(midi_pitch))
    hz = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        t = float(times[i])
        if t_start <= t < t_end:
            hz[i] = f0
    return hz


def attenuate_note_harmonics(
    audio_mono: np.ndarray,
    sr: int,
    *,
    midi_pitch: int,
    start_s: float,
    end_s: float,
    n_fft: int = 4096,
    hop_length: int = 1024,
    num_harmonics: int = 12,
    bandwidth_hz: float = 80.0,
    attenuation: float = 0.08,
) -> np.ndarray:
    """
    Attenuate magnitude of STFT bins near harmonics of ``midi_pitch`` between start_s and end_s.
    ``attenuation`` scales magnitude (0 = silence, 1 = unchanged). Phase is kept from the original.
    """
    y = np.asarray(audio_mono, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=0)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(D)
    phase = np.angle(D)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=hop_length)
    n_frames = D.shape[1]

    hz_series = _hz_series_for_note(midi_pitch, n_frames, sr, hop_length, start_s, end_s, times)

    mask = np.ones_like(mag)
    for i in range(n_frames):
        f0 = hz_series[i]
        if f0 <= 0:
            continue
        for h in range(1, num_harmonics + 1):
            fh = f0 * h
            if fh >= sr / 2:
                break
            dist = np.abs(freqs - fh)
            idx = np.where(dist < bandwidth_hz)[0]
            if idx.size:
                mask[idx, i] = np.minimum(mask[idx, i], attenuation)

    D2 = (mag * mask) * np.exp(1j * phase)
    y_out = librosa.istft(D2, hop_length=hop_length, length=len(y))
    return y_out.astype(np.float32)


def expander_noise_gate_region(
    audio_mono: np.ndarray,
    sr: int,
    region_start_s: float,
    region_end_s: float,
    *,
    pad_s: float = 0.05,
    frame_ms: float = 10.0,
    threshold_db_below_peak: float = 22.0,
    below_gain: float = 0.08,
    attack_ms: float = 3.0,
    release_ms: float = 50.0,
) -> np.ndarray:
    """
    Downward expander in ``[region_start_s, region_end_s]`` (expanded by ``pad_s``):
    frames whose RMS falls far below the local peak gain are scaled toward ``below_gain``.
    """
    y = np.asarray(audio_mono, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    n = len(y)
    if n == 0:
        return y

    rs = max(0.0, region_start_s - pad_s)
    re = min(float(n / sr), region_end_s + pad_s)
    i0 = int(rs * sr)
    i1 = int(re * sr)
    i0 = max(0, min(i0, n - 1))
    i1 = max(i0 + 1, min(i1, n))
    region = y[i0:i1].copy()
    if region.size < 8:
        return y

    frame = max(8, int(sr * frame_ms / 1000.0))
    hop = max(1, frame // 4)
    rms_list: list[float] = []
    for j in range(0, len(region) - frame + 1, hop):
        sl = region[j : j + frame]
        rms_list.append(float(np.sqrt(np.mean(sl**2) + 1e-18)))
    if not rms_list:
        return y

    peak = max(rms_list)
    thresh = peak * (10.0 ** (-threshold_db_below_peak / 20.0))
    target = np.array([1.0 if r >= thresh else below_gain for r in rms_list], dtype=np.float32)

    att = float(np.exp(-hop / max(1e-6, attack_ms * 1e-3 * sr)))
    rel = float(np.exp(-hop / max(1e-6, release_ms * 1e-3 * sr)))
    smoothed = np.zeros_like(target)
    g = 1.0
    for k in range(len(target)):
        tg = float(target[k])
        if tg < g:
            g = tg + (g - tg) * att
        else:
            g = tg + (g - tg) * rel
        smoothed[k] = g

    gain_full = np.ones(len(region), dtype=np.float32)
    for k, gv in enumerate(smoothed):
        s = k * hop
        e = min(len(region), s + frame)
        gain_full[s:e] = np.minimum(gain_full[s:e], gv)

    y_out = y.copy()
    y_out[i0:i1] = region * gain_full
    return y_out.astype(np.float32)


def remix_stems(
    stems: dict[str, np.ndarray],
    *,
    out_path: str,
    sr: int,
) -> None:
    """Sum mono stems to a stereo-identical mono mix and save."""
    if not stems:
        raise ValueError("no stems")
    acc = None
    for _name, x in stems.items():
        a = np.asarray(x, dtype=np.float32)
        if a.ndim > 1:
            a = a.mean(axis=0)
        acc = a.copy() if acc is None else acc + a
    assert acc is not None
    peak = float(np.max(np.abs(acc))) + 1e-9
    if peak > 1.0:
        acc = acc / peak
    save_wav(out_path, acc, sr)
