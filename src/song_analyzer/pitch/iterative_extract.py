from __future__ import annotations

from typing import Any

import numpy as np

from song_analyzer.editing.note_removal import attenuate_note_harmonics, expander_noise_gate_region
from song_analyzer.pitch.transcribe import transcribe_stem
from song_analyzer.schema import NoteEvent, SoloSegment


def _note_overlaps_solo(n: NoteEvent, solo: list[SoloSegment]) -> bool:
    mid = 0.5 * (n.start_time_s + n.end_time_s)
    for s in solo:
        if s.stem_id != n.stem_id:
            continue
        if s.start_time_s <= mid <= s.end_time_s:
            return True
    return False


def _same_note_event(a: NoteEvent, b: NoteEvent, *, tol_s: float = 0.03) -> bool:
    return (
        a.midi_pitch == b.midi_pitch
        and abs(a.start_time_s - b.start_time_s) < tol_s
        and abs(a.end_time_s - b.end_time_s) < tol_s
    )


def extract_notes_iteratively_for_stem(
    audio: np.ndarray,
    stem_sr: int,
    stem_id: str,
    *,
    prefer_basic_pitch: bool = True,
    max_iterations: int = 512,
    min_note_duration_s: float = 0.04,
    gate_pad_s: float = 0.04,
    restrict_to_solo: bool = False,
    solo_segments: list[SoloSegment] | None = None,
) -> tuple[list[NoteEvent], dict[str, Any]]:
    """
    Repeatedly transcribe, take the strongest note, harmonic-attenuate it, and noise-gate
    the same time region until no notes remain or limits hit.
    """
    working = np.asarray(audio, dtype=np.float32).copy()
    if working.ndim > 1:
        working = np.mean(working, axis=0)

    extracted: list[NoteEvent] = []
    solo = solo_segments or []
    meta: dict[str, Any] = {"stopped_reason": "max_iterations"}
    backend = "unknown"

    orig_rms = float(np.sqrt(np.mean(working**2) + 1e-18))

    for _it in range(max_iterations):
        notes, backend = transcribe_stem(
            working,
            stem_sr,
            stem_id,
            prefer_basic_pitch=prefer_basic_pitch,
        )
        if not notes:
            meta["stopped_reason"] = "no_notes"
            break

        pool = list(notes)
        if restrict_to_solo and solo:
            pool = [n for n in pool if _note_overlaps_solo(n, solo)]
        if not pool:
            meta["stopped_reason"] = "solo_filter_empty"
            break

        def score(n: NoteEvent) -> float:
            dur = max(0.0, n.end_time_s - n.start_time_s)
            v = n.velocity if n.velocity is not None else 0.5
            return dur * float(v)

        best = max(pool, key=score)
        if best.end_time_s - best.start_time_s < min_note_duration_s:
            meta["stopped_reason"] = "note_too_short"
            break

        if extracted and _same_note_event(best, extracted[-1]):
            meta["stopped_reason"] = "no_progress"
            break

        extracted.append(best)
        working = attenuate_note_harmonics(
            working,
            stem_sr,
            midi_pitch=best.midi_pitch,
            start_s=best.start_time_s,
            end_s=best.end_time_s,
        )
        working = expander_noise_gate_region(
            working,
            stem_sr,
            best.start_time_s,
            best.end_time_s,
            pad_s=gate_pad_s,
        )

        rms = float(np.sqrt(np.mean(working**2) + 1e-18))
        if rms < orig_rms * 0.02:
            meta["stopped_reason"] = "residual_energy"
            break

    meta["iterations"] = len(extracted)
    meta["backend_hint"] = backend
    return extracted, meta


def extract_notes_iteratively_all_stems(
    stems: dict[str, np.ndarray],
    stem_sr: int,
    *,
    prefer_basic_pitch: bool = True,
    max_iterations_per_stem: int = 512,
    restrict_to_solo: bool = False,
    solo_segments: list[SoloSegment] | None = None,
) -> tuple[list[NoteEvent], dict[str, Any]]:
    all_notes: list[NoteEvent] = []
    per_stem: dict[str, Any] = {}
    for name in sorted(stems.keys()):
        notes, m = extract_notes_iteratively_for_stem(
            stems[name],
            stem_sr,
            name,
            prefer_basic_pitch=prefer_basic_pitch,
            max_iterations=max_iterations_per_stem,
            restrict_to_solo=restrict_to_solo,
            solo_segments=solo_segments,
        )
        all_notes.extend(notes)
        per_stem[name] = m
    return all_notes, {"per_stem": per_stem}
