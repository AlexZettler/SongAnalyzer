from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from song_analyzer.instruments.constants import NSYNTH_FAMILIES, NUM_NSYNTH_FAMILIES
from song_analyzer.instruments.mel import SAMPLE_RATE, build_model, waveform_to_log_mel
from song_analyzer.instruments.model import FamilyClassifier

_DEFAULT_CHECKPOINT_ENV = "SONGANALYZER_NSYNTH_CHECKPOINT"


def load_classifier(
    checkpoint_path: str | Path | None,
    device: str | torch.device,
) -> tuple[FamilyClassifier | None, str | None]:
    """
    Load trained weights. Returns (model, None) or (None, warning_message).
    """
    path = checkpoint_path or os.environ.get(_DEFAULT_CHECKPOINT_ENV)
    if not path:
        return None, (
            "No NSynth checkpoint: set --nsynth-checkpoint or SONGANALYZER_NSYNTH_CHECKPOINT. "
            "Using uniform family prior."
        )
    p = Path(path)
    if not p.is_file():
        return None, f"NSynth checkpoint missing at {p}; using uniform family prior."

    dev = torch.device(device)
    model = build_model(dev)
    try:
        state = torch.load(p, map_location=dev, weights_only=True)
    except TypeError:
        state = torch.load(p, map_location=dev)
    model.load_state_dict(state)
    model.eval()
    return model, None


def predict_stem_family(
    audio_mono_16k: np.ndarray,
    model: FamilyClassifier | None,
    device: str,
) -> tuple[str, float, dict[str, float]]:
    """
    Classify one stem. If model is None, returns uniform logits over families.
    """
    dev = torch.device(device)
    if model is None:
        unif = 1.0 / NUM_NSYNTH_FAMILIES
        logits = {f: unif for f in NSYNTH_FAMILIES}
        return "unknown", 0.0, logits

    # Center crop / pad to ~1.6s for stable context
    n = int(SAMPLE_RATE * 1.6)
    x = audio_mono_16k.astype(np.float32, copy=False)
    if len(x) > n:
        start = (len(x) - n) // 2
        x = x[start : start + n]
    elif len(x) < n:
        x = np.pad(x, (0, n - len(x)))

    with torch.no_grad():
        mel = waveform_to_log_mel(x, dev)
        out = model(mel)
        prob = F.softmax(out, dim=-1).squeeze(0).cpu().numpy()
    idx = int(prob.argmax())
    fam = NSYNTH_FAMILIES[idx]
    conf = float(prob[idx])
    logits = {NSYNTH_FAMILIES[i]: float(prob[i]) for i in range(len(NSYNTH_FAMILIES))}
    return fam, conf, logits
