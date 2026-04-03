from __future__ import annotations

import json
import warnings
from pathlib import Path

import librosa
import numpy as np

from song_analyzer.audio_io import load_audio
from song_analyzer.chords.detect import build_chord_timeline
from song_analyzer.editing.note_removal import attenuate_note_harmonics, remix_stems
from song_analyzer.instruments.infer import load_classifier, predict_stem_family
from song_analyzer.instruments.mel import SAMPLE_RATE as NSYNTH_SR
from song_analyzer.pitch.transcribe import transcribe_stem
from song_analyzer.schema import AnalysisResult, ChordSegment, InstrumentPrediction, NoteEvent, StemAudioRef
from song_analyzer.separation.demucs_sep import save_stems, separate_to_dict


def analyze_mix(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    demucs_model: str = "htdemucs",
    demucs_shifts: int = 0,
    demucs_segment: float | None = None,
    nsynth_checkpoint: str | Path | None = None,
    write_stem_wavs: bool = True,
    chord_hop_s: float = 0.05,
) -> AnalysisResult:
    """
    Full pipeline: separate stems, classify timbre (NSynth checkpoint optional), transcribe, chords.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mix, sr = load_audio(input_path, target_sr=None, mono=True)
    duration_s = float(len(mix) / sr)

    stems, stem_sr = separate_to_dict(
        mix,
        sr,
        model_name=demucs_model,
        device=device,
        shifts=demucs_shifts,
        split=True,
        segment=demucs_segment,
        progress=True,
    )

    stem_paths: dict[str, str] = {}
    if write_stem_wavs:
        stem_paths = save_stems(stems, stem_sr, output_dir / "stems")

    clf, clf_warn = load_classifier(nsynth_checkpoint, device)
    if clf_warn:
        warnings.warn(clf_warn, stacklevel=1)

    instruments: list[InstrumentPrediction] = []
    notes: list[NoteEvent] = []
    pitch_backend = "unknown"

    for name, audio in stems.items():
        a16 = librosa.resample(audio.astype(np.float32), orig_sr=stem_sr, target_sr=NSYNTH_SR)
        fam, conf, logits = predict_stem_family(a16, clf, device)
        instruments.append(
            InstrumentPrediction(
                stem_id=name,
                family=fam,
                confidence=conf,
                family_logits=logits,
            )
        )
        stem_notes, backend = transcribe_stem(audio, stem_sr, name, prefer_basic_pitch=True)
        pitch_backend = backend
        notes.extend(stem_notes)

    stem_chords: list[ChordSegment] = []
    for name in stems:
        stem_chords.extend(
            build_chord_timeline(
                notes,
                stem_id=name,
                hop_s=chord_hop_s,
                duration_s=duration_s,
            )
        )
    mix_chords = build_chord_timeline(
        notes,
        stem_id=None,
        hop_s=chord_hop_s,
        duration_s=duration_s,
    )

    result = AnalysisResult(
        source_path=str(Path(input_path).resolve()),
        sample_rate=sr,
        duration_s=duration_s,
        stems=[
            StemAudioRef(stem_id=k, path=stem_paths.get(k)) for k in sorted(stems.keys())
        ],
        instruments=instruments,
        notes=sorted(notes, key=lambda n: (n.stem_id, n.start_time_s)),
        chords=sorted(stem_chords + mix_chords, key=lambda c: (c.stem_id or "", c.start_time_s)),
        meta={
            "demucs_model": demucs_model,
            "stem_sample_rate": stem_sr,
            "pitch_transcription": pitch_backend,
            "nsynth_checkpoint_used": clf is not None,
        },
    )

    out_json = output_dir / "analysis.json"
    out_json.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def remove_note_from_mix(
    input_path: str | Path,
    output_wav: str | Path,
    *,
    stem: str,
    midi_pitch: int,
    start_s: float,
    end_s: float,
    device: str = "cpu",
    demucs_model: str = "htdemucs",
    demucs_shifts: int = 0,
    stems_dir: str | Path | None = None,
) -> None:
    """
    Separate (or load stems from ``stems_dir``), attenuate one note on ``stem``, remix to ``output_wav``.
    """
    if stems_dir is not None:
        sd = Path(stems_dir)
        wavs = sorted(sd.glob("*.wav"))
        if not wavs:
            raise ValueError(f"no WAV files in {sd}")
        stems = {}
        stem_sr: int | None = None
        for p in wavs:
            data, sr_one = load_audio(p, target_sr=None, mono=True)
            if stem_sr is None:
                stem_sr = sr_one
            elif sr_one != stem_sr:
                raise ValueError("all stem WAVs must share the same sample rate")
            stems[p.stem] = data
        assert stem_sr is not None
    else:
        mix, sr = load_audio(input_path, target_sr=None, mono=True)
        stems, stem_sr = separate_to_dict(
            mix,
            sr,
            model_name=demucs_model,
            device=device,
            shifts=demucs_shifts,
            split=True,
            progress=True,
        )

    if stem not in stems:
        raise ValueError(f"unknown stem {stem!r}; available: {sorted(stems)}")

    edited = dict(stems)
    edited[stem] = attenuate_note_harmonics(
        stems[stem],
        stem_sr,
        midi_pitch=midi_pitch,
        start_s=start_s,
        end_s=end_s,
    )
    remix_stems(edited, out_path=str(output_wav), sr=stem_sr)


def write_analysis_json(result: AnalysisResult, path: str | Path) -> None:
    Path(path).write_text(result.model_dump_json(indent=2), encoding="utf-8")


def analysis_from_json(path: str | Path) -> AnalysisResult:
    return AnalysisResult.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
