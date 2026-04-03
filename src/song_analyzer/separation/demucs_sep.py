from __future__ import annotations

"""
Stem separation via Demucs.

The set of stem keys in the returned dict always matches ``model.sources`` for the
chosen pretrained (do not assume fixed names in application code). Common cases:

- ``htdemucs``: ``drums``, ``bass``, ``other``, ``vocals``
- ``htdemucs_6s``: above plus ``guitar``, ``piano``

Mono input is duplicated to stereo before ``apply_model``; each stem is reduced to
mono by averaging channels. Sample rate is ``model.samplerate``.
"""

from pathlib import Path

import numpy as np
import torch
from demucs.apply import apply_model
from demucs.audio import convert_audio
from demucs.pretrained import get_model


def separate_to_dict(
    mix_mono: np.ndarray,
    sr: int,
    *,
    model_name: str = "htdemucs",
    device: str = "cpu",
    shifts: int = 0,
    split: bool = True,
    segment: float | None = None,
    progress: bool = False,
) -> tuple[dict[str, np.ndarray], int]:
    """
    Separate a mono mixture into stems. Returns ({stem_name: mono_float32}, model_sample_rate).

    Demucs expects stereo for htdemucs; we duplicate mono to two channels.
    """
    model = get_model(model_name)
    model.to(device)
    model.eval()
    target_sr = model.samplerate
    ch = model.audio_channels

    wav = torch.from_numpy(mix_mono.astype(np.float32)).view(1, -1)
    wav = wav.repeat(ch, 1)

    wav = convert_audio(wav, sr, target_sr, ch)

    ref = wav.mean(0)
    wav = wav - ref.mean()
    wav = wav / ref.std().clamp(1e-7)

    with torch.no_grad():
        sources = apply_model(
            model,
            wav[None],
            device=device,
            shifts=shifts,
            split=split,
            progress=progress,
            segment=segment,
        )

    # sources: [1, n_sources, channels, time]
    sources = sources[0]
    sources = sources * ref.std().clamp(1e-7) + ref.mean()
    names: list[str] = list(model.sources)
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        stem = sources[i]
        if stem.shape[0] > 1:
            stem = stem.mean(0)
        else:
            stem = stem[0]
        out[str(name)] = stem.cpu().numpy().astype(np.float32)

    return out, int(target_sr)


def save_stems(stems: dict[str, np.ndarray], sr: int, out_dir: Path) -> dict[str, str]:
    """Write WAV files; returns stem_id -> path."""
    from song_analyzer.audio_io import save_wav

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, audio in stems.items():
        p = out_dir / f"{name}.wav"
        save_wav(p, audio, sr)
        paths[name] = str(p.resolve())
    return paths
