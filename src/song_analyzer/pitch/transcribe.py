from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from song_analyzer.schema import NoteEvent


def basic_pitch_available() -> bool:
    try:
        import basic_pitch.inference  # noqa: F401

        return True
    except ImportError:
        return False


def _transcribe_basic_pitch(audio: np.ndarray, sr: int, stem_id: str) -> list[NoteEvent]:
    from basic_pitch.inference import predict

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tpath = Path(tmp.name)
    try:
        sf.write(tpath, audio.astype(np.float32), sr, subtype="FLOAT")
        _, _, note_events = predict(str(tpath))
    finally:
        tpath.unlink(missing_ok=True)

    out: list[NoteEvent] = []
    for start_time, end_time, midi, amplitude, _pitch_bend in note_events:
        out.append(
            NoteEvent(
                start_time_s=float(start_time),
                end_time_s=float(end_time),
                midi_pitch=int(midi),
                velocity=float(amplitude) if amplitude is not None else None,
                stem_id=stem_id,
            )
        )
    return out


def _transcribe_piptrack_fallback(audio: np.ndarray, sr: int, stem_id: str) -> list[NoteEvent]:
    """Monophonic-oriented fallback using piptrack + simple note grouping."""
    import librosa

    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    harmonic = librosa.effects.harmonic(y)
    f0, voiced_flag, _ = librosa.pyin(
        harmonic,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sr,
        frame_length=2048,
        hop_length=512,
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=512)
    midi = np.array([librosa.hz_to_midi(h) if h == h else np.nan for h in f0])
    notes: list[NoteEvent] = []
    i = 0
    n = len(midi)
    while i < n:
        if np.isnan(midi[i]) or not voiced_flag[i]:
            i += 1
            continue
        j = i
        acc: list[float] = []
        while j < n and voiced_flag[j] and not np.isnan(midi[j]):
            acc.append(float(midi[j]))
            j += 1
        if acc:
            m = int(round(float(np.median(acc))))
            m = max(0, min(127, m))
            notes.append(
                NoteEvent(
                    start_time_s=float(times[i]),
                    end_time_s=float(times[j - 1]) + 512 / sr,
                    midi_pitch=m,
                    velocity=None,
                    stem_id=stem_id,
                )
            )
        i = max(j, i + 1)
    return notes


def transcribe_stem(
    audio: np.ndarray,
    sr: int,
    stem_id: str,
    *,
    prefer_basic_pitch: bool = True,
) -> tuple[list[NoteEvent], str]:
    """
    Returns (notes, backend_name) where backend is 'basic_pitch' or 'librosa_pyin'.
    """
    if prefer_basic_pitch and basic_pitch_available():
        try:
            return _transcribe_basic_pitch(audio, sr, stem_id), "basic_pitch"
        except Exception:
            pass
    return _transcribe_piptrack_fallback(audio, sr, stem_id), "librosa_pyin"
