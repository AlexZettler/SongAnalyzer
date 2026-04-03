from __future__ import annotations

import numpy as np
import torch
import torchaudio

from song_analyzer.instruments.constants import NUM_NSYNTH_FAMILIES
from song_analyzer.instruments.model import FamilyClassifier

SAMPLE_RATE = 16_000
N_MELS = 64
N_FFT = 1024
HOP = 160  # 100 fps at 16k


def waveform_to_log_mel(wave: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    """wave: 1D float32 numpy or tensor, 16 kHz mono."""
    if isinstance(wave, np.ndarray):
        w = torch.from_numpy(wave.astype(np.float32, copy=False)).to(device)
    else:
        w = wave.to(device)
    if w.dim() > 1:
        w = w.mean(dim=0)
    w = w.clamp(-1.0, 1.0).unsqueeze(0)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=2.0,
    ).to(device)
    spec = mel(w)
    log_spec = torch.log(spec + 1e-6)
    return log_spec.unsqueeze(0)  # (1, 1, n_mels, time)


def build_model(device: torch.device) -> FamilyClassifier:
    m = FamilyClassifier(NUM_NSYNTH_FAMILIES, n_mels=N_MELS)
    return m.to(device)
