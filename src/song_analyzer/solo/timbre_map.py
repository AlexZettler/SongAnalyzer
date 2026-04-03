from __future__ import annotations

import numpy as np
import librosa

from song_analyzer.instruments.infer import predict_stem_family
from song_analyzer.instruments.mel import SAMPLE_RATE as NSYNTH_SR
from song_analyzer.instruments.model import FamilyClassifier
from song_analyzer.schema import SoloSegment, TimbreSample


def build_timbre_samples(
    stems: dict[str, np.ndarray],
    stem_sr: int,
    solo_segments: list[SoloSegment],
    model: FamilyClassifier | None,
    device: str,
    *,
    dominance_min: float = 0.58,
    window_s: float = 1.6,
    hop_s: float = 0.5,
) -> list[TimbreSample]:
    """
    Classify timbre on short windows inside high-confidence solo segments for each stem.
    """
    out: list[TimbreSample] = []
    win = int(window_s * stem_sr)
    hop = max(1, int(hop_s * stem_sr))

    for seg in solo_segments:
        if seg.dominance < dominance_min:
            continue
        if seg.stem_id not in stems:
            continue
        audio = np.asarray(stems[seg.stem_id], dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        start = int(seg.start_time_s * stem_sr)
        end = int(seg.end_time_s * stem_sr)
        start = max(0, min(start, len(audio) - 1))
        end = max(start + 1, min(end, len(audio)))

        t0 = start
        while t0 + win <= end:
            chunk = audio[t0 : t0 + win]
            t_center = (t0 + win / 2) / stem_sr
            a16 = librosa.resample(chunk, orig_sr=stem_sr, target_sr=NSYNTH_SR)
            fam, conf, logits = predict_stem_family(a16, model, device)
            out.append(
                TimbreSample(
                    time_center_s=float(t_center),
                    stem_id=seg.stem_id,
                    family=fam,
                    confidence=conf,
                    family_logits=logits,
                )
            )
            t0 += hop
        if t0 < end and end - start >= int(0.2 * stem_sr):
            chunk = audio[max(start, end - win) : end]
            if len(chunk) >= int(0.2 * stem_sr):
                t_center = (max(start, end - win) + len(chunk) / 2) / stem_sr
                a16 = librosa.resample(chunk, orig_sr=stem_sr, target_sr=NSYNTH_SR)
                fam, conf, logits = predict_stem_family(a16, model, device)
                out.append(
                    TimbreSample(
                        time_center_s=float(t_center),
                        stem_id=seg.stem_id,
                        family=fam,
                        confidence=conf,
                        family_logits=logits,
                    )
                )

    return out
