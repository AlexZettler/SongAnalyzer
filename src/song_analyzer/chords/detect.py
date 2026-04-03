from __future__ import annotations

from music21 import chord as m21_chord
from music21 import pitch as m21_pitch

from song_analyzer.schema import ChordSegment, NoteEvent


def chord_label_for_pcs(pitch_classes: list[int], bass_pc: int | None = None) -> str:
    """Map unique pitch classes (0–11) to a chord symbol; bass_pc refines slash bass when distinct."""
    pcs = sorted({int(p) % 12 for p in pitch_classes})
    if not pcs:
        return "N.C."
    if len(pcs) == 1:
        return m21_pitch.Pitch(midi=60 + pcs[0]).name

    pitches = [m21_pitch.Pitch(midi=60 + p) for p in pcs]
    c = m21_chord.Chord(pitches)
    try:
        name = str(c.pitchedCommonName) if c.pitchedCommonName else str(c.commonName)
    except Exception:
        name = c.figure
    if not name or name in ("Chord", "None"):
        name = "-".join(p.name for p in pitches)

    if bass_pc is not None:
        b = int(bass_pc) % 12
        root_pc = c.root().pitchClass if c.root() is not None else None
        if root_pc is None or b != root_pc:
            bass_name = m21_pitch.Pitch(midi=48 + b).name
            if "/" not in name:
                return f"{name}/{bass_name}"
    return name


def build_chord_timeline(
    notes: list[NoteEvent],
    *,
    stem_id: str | None,
    hop_s: float = 0.05,
    duration_s: float | None = None,
) -> list[ChordSegment]:
    """
    Quantize active pitch classes on a grid and emit merged chord segments.
    If stem_id is set, only notes with that stem_id are used; if None, all notes.
    """
    filtered = [n for n in notes if stem_id is None or n.stem_id == stem_id]
    if not filtered:
        return []

    t_end = duration_s
    if t_end is None:
        t_end = max(n.end_time_s for n in filtered) + hop_s

    grid = []
    t = 0.0
    while t < t_end:
        grid.append(t)
        t += hop_s

    frames: list[tuple[float, list[int], int | None]] = []
    for t0 in grid:
        active = [n for n in filtered if n.start_time_s <= t0 < n.end_time_s]
        if not active:
            pcs = []
            bass = None
        else:
            pcs = [n.midi_pitch % 12 for n in active]
            bass_note = min(active, key=lambda n: n.midi_pitch)
            bass = bass_note.midi_pitch % 12
        frames.append((t0, pcs, bass))

    segments: list[ChordSegment] = []
    if not frames:
        return segments

    def key_fn(x: tuple[float, list[int], int | None]) -> tuple[tuple[int, ...], int | None]:
        pcs_u = tuple(sorted({p % 12 for p in x[1]}))
        return pcs_u, x[2]

    cur_start = frames[0][0]
    cur_key = key_fn(frames[0])
    cur_pcs = sorted({p % 12 for p in frames[0][1]})
    cur_bass = frames[0][2]

    for i in range(1, len(frames)):
        t0, pcs_list, bass = frames[i]
        k = key_fn((t0, pcs_list, bass))
        if k != cur_key:
            end_t = t0
            label = chord_label_for_pcs(list(cur_pcs), cur_bass)
            segments.append(
                ChordSegment(
                    start_time_s=cur_start,
                    end_time_s=end_t,
                    chord_label=label,
                    pitch_classes=list(cur_pcs),
                    stem_id=stem_id,
                )
            )
            cur_start = t0
            cur_key = k
            cur_pcs = sorted({p % 12 for p in pcs_list})
            cur_bass = bass

    label = chord_label_for_pcs(list(cur_pcs), cur_bass)
    segments.append(
        ChordSegment(
            start_time_s=cur_start,
            end_time_s=t_end,
            chord_label=label,
            pitch_classes=list(cur_pcs),
            stem_id=stem_id,
        )
    )
    return segments
