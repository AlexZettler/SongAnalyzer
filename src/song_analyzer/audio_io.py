from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: str | Path, target_sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load audio as float32. If mono, shape (samples,); else (channels, samples)."""
    path = Path(path)
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    # data shape: (frames, channels)
    data = data.T
    if mono:
        if data.shape[0] > 1:
            data = np.mean(data, axis=0, keepdims=True)
        data = data[0]
    else:
        pass
    if target_sr is not None and target_sr != sr:
        import librosa

        if mono:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        else:
            data = np.stack(
                [librosa.resample(data[i], orig_sr=sr, target_sr=target_sr) for i in range(data.shape[0])],
                axis=0,
            )
        sr = target_sr
    return np.asarray(data, dtype=np.float32), int(sr)


def save_wav(path: str | Path, audio: np.ndarray, sr: int) -> None:
    """Save float32 audio. Mono (n,) or stereo (2, n)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        sf.write(path, x, sr, subtype="FLOAT")
    elif x.ndim == 2:
        sf.write(path, x.T, sr, subtype="FLOAT")
    else:
        raise ValueError("audio must be 1D or 2D")
