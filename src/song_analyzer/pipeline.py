from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from song_analyzer.audio_io import load_audio
from song_analyzer.chords.detect import build_chord_timeline
from song_analyzer.editing.note_removal import attenuate_note_harmonics, remix_stems
from song_analyzer.instruments.infer import load_classifier, predict_stem_family
from song_analyzer.instruments.mel import SAMPLE_RATE as NSYNTH_SR
from song_analyzer.pitch.iterative_extract import extract_notes_iteratively_all_stems
from song_analyzer.pitch.transcribe import transcribe_stem
from song_analyzer.schema import (
    AnalysisResult,
    ChordSegment,
    InstrumentPrediction,
    NoteEvent,
    SoloSegment,
    StemAudioRef,
    TimbreSample,
)
from song_analyzer.separation.demucs_sep import save_stems, separate_to_dict
from song_analyzer.solo.detect import detect_solo_segments
from song_analyzer.solo.timbre_map import build_timbre_samples
from song_analyzer.structure.global_analysis import analyze_global_structure


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
    use_staged: bool = False,
    restrict_iterative_to_solo: bool = False,
    write_pass_json: bool = True,
    max_iterative_notes_per_stem: int = 512,
) -> AnalysisResult:
    """
    Full pipeline: separate stems, classify timbre (NSynth checkpoint optional), transcribe, chords.

    With ``use_staged=True``: pass 1 = global tempo/structure on the mix; pass 2 = Demucs,
    solo windows, timbre samples on solo crops; pass 3 = iterative note peel per stem.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mix, sr = load_audio(input_path, target_sr=None, mono=True)
    duration_s = float(len(mix) / sr)

    global_structure = None
    solo_segments: list[SoloSegment] = []
    timbre_samples: list[TimbreSample] = []

    if use_staged:
        global_structure = analyze_global_structure(np.asarray(mix, dtype=np.float32), sr)
        if write_pass_json:
            (output_dir / "pass1.json").write_text(
                global_structure.model_dump_json(indent=2),
                encoding="utf-8",
            )

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
    iterative_meta: dict | None = None

    if use_staged:
        solo_segments = detect_solo_segments(stems, stem_sr)
        timbre_samples = build_timbre_samples(
            stems,
            stem_sr,
            solo_segments,
            clf,
            device,
        )
        if write_pass_json:
            pass2 = {
                "solo_segments": [s.model_dump() for s in solo_segments],
                "timbre_sample_count": len(timbre_samples),
                "demucs_sources": sorted(stems.keys()),
            }
            (output_dir / "pass2.json").write_text(
                json.dumps(pass2, indent=2),
                encoding="utf-8",
            )

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

    if use_staged:
        notes, iterative_meta = extract_notes_iteratively_all_stems(
            stems,
            stem_sr,
            prefer_basic_pitch=True,
            max_iterations_per_stem=max_iterative_notes_per_stem,
            restrict_to_solo=restrict_iterative_to_solo,
            solo_segments=solo_segments,
        )
        pitch_backend = "iterative_peel"
        if iterative_meta and iterative_meta.get("per_stem"):
            first: dict[str, Any] = next(iter(iterative_meta["per_stem"].values()), {})
            pitch_backend = str(first.get("backend_hint", pitch_backend))
    else:
        for name, audio in stems.items():
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

    meta = {
        "demucs_model": demucs_model,
        "stem_sample_rate": stem_sr,
        "pitch_transcription": pitch_backend,
        "nsynth_checkpoint_used": clf is not None,
        "use_staged": use_staged,
        "restrict_iterative_to_solo": restrict_iterative_to_solo,
    }
    if iterative_meta is not None:
        meta["iterative_extraction"] = iterative_meta

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
        global_structure=global_structure,
        solo_segments=solo_segments,
        timbre_samples=timbre_samples,
        meta=meta,
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
