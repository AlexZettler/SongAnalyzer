"""Build dense polyphonic clips from NSynth (TFDS) with JSON sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from song_analyzer.eval.family_bucket import DEMUCS_FOUR_STEMS, demucs_bucket_for_family
from song_analyzer.eval.sidecar import (
    SCHEMA_VERSION,
    DensityParams,
    DenseEvalSidecar,
    GroundTruthNote,
)
from song_analyzer.instruments.constants import NSYNTH_FAMILIES
from song_analyzer.instruments.mel import SAMPLE_RATE as NSYNTH_SR


def import_tfds_for_dense_eval():
    """Lazy import of TensorFlow / TFDS (same constraints as NSynth training)."""
    from song_analyzer.instruments.train_nsynth import import_tfds_for_nsynth

    return import_tfds_for_nsynth()


def _family_idx_to_name(idx: int) -> str:
    i = int(idx)
    if i < 0 or i >= len(NSYNTH_FAMILIES):
        raise ValueError(f"family index out of range: {idx}")
    return NSYNTH_FAMILIES[i]


def _as_float32_mono(audio: Any) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(x))) + 1e-9
    return x / peak


def _apply_detune(wav: np.ndarray, sr: int, cents: float) -> np.ndarray:
    if abs(cents) < 1e-6:
        return wav
    import librosa

    return librosa.effects.pitch_shift(
        y=np.asarray(wav, dtype=np.float32),
        sr=sr,
        n_steps=cents / 100.0,
    ).astype(np.float32)


def iter_nsynth_examples(
    *,
    split: str,
    data_dir: str | None,
    shuffle_buffer: int,
    seed: int,
) -> Iterator[dict[str, Any]]:
    """Yield decoded NSynth examples as numpy-friendly dicts (one note each)."""
    _, tfds = import_tfds_for_dense_eval()
    load_kw: dict[str, Any] = {}
    if data_dir is not None:
        load_kw["data_dir"] = data_dir
    ds = tfds.load(
        "nsynth/full",
        split=split,
        shuffle_files=True,
        **load_kw,
    )
    ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=False)
    for ex in tfds.as_numpy(ds):
        yield ex


def collect_notes_for_clip(
    *,
    split: str,
    data_dir: str | None,
    n_notes: int,
    seed: int,
    same_family_stack: bool,
    shuffle_buffer: int = 10_000,
    max_scan: int = 50_000,
) -> list[dict[str, Any]]:
    """
    Scan a shuffled NSynth split until ``n_notes`` examples are collected.

    If ``same_family_stack``, all examples share the same instrument family (rejection sampling).
    """
    rng = np.random.default_rng(seed)
    target_family: str | None = None
    if same_family_stack:
        target_family = str(rng.choice(NSYNTH_FAMILIES))

    out: list[dict[str, Any]] = []
    scanned = 0
    for ex in iter_nsynth_examples(
        split=split,
        data_dir=data_dir,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    ):
        scanned += 1
        if scanned > max_scan:
            raise RuntimeError(
                f"collect_notes_for_clip: exceeded max_scan={max_scan} before finding "
                f"{n_notes} notes (same_family_stack={same_family_stack})"
            )
        fam_idx = int(ex["instrument"]["family"])
        fam = _family_idx_to_name(fam_idx)
        if same_family_stack and fam != target_family:
            continue
        out.append(ex)
        if len(out) >= n_notes:
            break
    if len(out) < n_notes:
        raise RuntimeError(
            f"only collected {len(out)} notes after scanning {scanned} examples "
            f"(need {n_notes}); try same_family_stack=False or increase max_scan"
        )
    return out


def render_oracle_pseudo_stem(
    examples: list[dict[str, Any]],
    rng: np.random.Generator,
    *,
    clip_duration_s: float,
    detune_max_cents: float,
    level_jitter_db: float,
    sr: int = NSYNTH_SR,
) -> tuple[np.ndarray, list[GroundTruthNote], list[str]]:
    """
    Sum NSynth notes onto a timeline with random placement, gain, and optional detune.

    Returns (mono waveform, ground truth rows, source ids).
    """
    n = int(np.ceil(clip_duration_s * sr))
    mix = np.zeros(n, dtype=np.float32)
    gt: list[GroundTruthNote] = []
    ids: list[str] = []

    note_len = NSYNTH_SR * 4  # NSynth fixed 4s
    max_start = max(0.0, clip_duration_s - 4.0)
    if max_start <= 0:
        raise ValueError("clip_duration_s must exceed 4s (NSynth note length)")

    for ex in examples:
        raw_id = ex["id"]
        if isinstance(raw_id, bytes):
            sid = raw_id.decode("utf-8")
        else:
            sid = str(raw_id)
        ids.append(sid)

        wav = _as_float32_mono(ex["audio"])
        if len(wav) > note_len:
            wav = wav[:note_len]
        elif len(wav) < note_len:
            wav = np.pad(wav, (0, note_len - len(wav)))

        cents = float(rng.uniform(-detune_max_cents, detune_max_cents)) if detune_max_cents > 0 else 0.0
        wav = _apply_detune(wav, sr, cents)

        db_j = float(rng.uniform(-level_jitter_db, level_jitter_db)) if level_jitter_db > 0 else 0.0
        gain_linear = float(10.0 ** (db_j / 20.0))

        start_s = float(rng.uniform(0.0, max_start))
        start_i = int(round(start_s * sr))
        end_i = min(start_i + len(wav), n)
        sl = wav[: end_i - start_i] * gain_linear
        mix[start_i:end_i] += sl

        fam_idx = int(ex["instrument"]["family"])
        fam = _family_idx_to_name(fam_idx)
        pitch = int(ex["pitch"])
        vel = int(ex["velocity"]) if "velocity" in ex else None
        actual_end_s = end_i / sr

        gt.append(
            GroundTruthNote(
                source_note_id=sid,
                start_time_s=start_s,
                end_time_s=actual_end_s,
                midi_pitch=pitch,
                instrument_family=fam,
                demucs_bucket=demucs_bucket_for_family(fam),
                velocity=vel,
                gain_linear=gain_linear,
            )
        )

    peak = float(np.max(np.abs(mix))) + 1e-9
    mix = (mix / peak).astype(np.float32)
    # Scale gt gains for reporting only (audio normalized)
    return mix, gt, ids


def render_four_bucket_mix(
    examples: list[dict[str, Any]],
    rng: np.random.Generator,
    *,
    clip_duration_s: float,
    detune_max_cents: float,
    level_jitter_db: float,
    sr: int = NSYNTH_SR,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[GroundTruthNote], list[str]]:
    """
    Place the same notes as ``render_oracle_pseudo_stem`` but route each into a Demucs bucket.

    Returns (mixture mono, per-bucket mono stems, ground_truth, source_ids).
    """
    n = int(np.ceil(clip_duration_s * sr))
    buckets: dict[str, np.ndarray] = {k: np.zeros(n, dtype=np.float32) for k in DEMUCS_FOUR_STEMS}
    gt: list[GroundTruthNote] = []
    ids: list[str] = []

    note_len = NSYNTH_SR * 4
    max_start = max(0.0, clip_duration_s - 4.0)
    if max_start <= 0:
        raise ValueError("clip_duration_s must exceed 4s (NSynth note length)")

    for ex in examples:
        raw_id = ex["id"]
        if isinstance(raw_id, bytes):
            sid = raw_id.decode("utf-8")
        else:
            sid = str(raw_id)
        ids.append(sid)

        wav = _as_float32_mono(ex["audio"])
        if len(wav) > note_len:
            wav = wav[:note_len]
        elif len(wav) < note_len:
            wav = np.pad(wav, (0, note_len - len(wav)))

        cents = float(rng.uniform(-detune_max_cents, detune_max_cents)) if detune_max_cents > 0 else 0.0
        wav = _apply_detune(wav, sr, cents)

        db_j = float(rng.uniform(-level_jitter_db, level_jitter_db)) if level_jitter_db > 0 else 0.0
        gain_linear = float(10.0 ** (db_j / 20.0))

        start_s = float(rng.uniform(0.0, max_start))
        start_i = int(round(start_s * sr))
        end_i = min(start_i + len(wav), n)
        sl = wav[: end_i - start_i] * gain_linear

        fam_idx = int(ex["instrument"]["family"])
        fam = _family_idx_to_name(fam_idx)
        bucket = demucs_bucket_for_family(fam)
        buckets[bucket][start_i:end_i] += sl

        pitch = int(ex["pitch"])
        vel = int(ex["velocity"]) if "velocity" in ex else None
        actual_end_s = end_i / sr
        gt.append(
            GroundTruthNote(
                source_note_id=sid,
                start_time_s=start_s,
                end_time_s=actual_end_s,
                midi_pitch=pitch,
                instrument_family=fam,
                demucs_bucket=bucket,
                velocity=vel,
                gain_linear=gain_linear,
            )
        )

    mix = sum(buckets.values())
    peak = float(np.max(np.abs(mix))) + 1e-9
    mix = (mix / peak).astype(np.float32)
    for k in buckets:
        buckets[k] = (buckets[k] / peak).astype(np.float32)
    return mix, buckets, gt, ids


def write_clip_bundle(
    out_dir: Path,
    *,
    clip_id: str,
    seed: int,
    split: str,
    mix: np.ndarray,
    sidecar: DenseEvalSidecar,
    sr: int = NSYNTH_SR,
    oracle_relpath: str = "oracle_stem.wav",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    from song_analyzer.audio_io import save_wav

    save_wav(out_dir / oracle_relpath, mix, sr)
    (out_dir / "dense_eval.json").write_text(
        sidecar.model_dump_json(indent=2),
        encoding="utf-8",
    )


def write_full_mix_bundle(
    out_dir: Path,
    *,
    clip_id: str,
    mixture: np.ndarray,
    buckets: dict[str, np.ndarray],
    sidecar: DenseEvalSidecar,
    sr: int = NSYNTH_SR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    from song_analyzer.audio_io import save_wav

    mix_path = "mixture.wav"
    save_wav(out_dir / mix_path, mixture, sr)
    rels: dict[str, str] = {}
    for name in DEMUCS_FOUR_STEMS:
        fn = f"stem_{name}.wav"
        rels[name] = fn
        save_wav(out_dir / fn, buckets[name], sr)

    sidecar.demucs_mix_relpath = mix_path
    sidecar.per_bucket_wav_relpaths = rels
    (out_dir / "dense_eval.json").write_text(
        sidecar.model_dump_json(indent=2),
        encoding="utf-8",
    )


def build_sidecar(
    *,
    clip_id: str,
    seed: int,
    split: str,
    clip_duration_s: float,
    density: DensityParams,
    gt: list[GroundTruthNote],
    ids: list[str],
    oracle_stem_id: str = "oracle",
) -> DenseEvalSidecar:
    return DenseEvalSidecar(
        schema_version=SCHEMA_VERSION,
        clip_id=clip_id,
        seed=seed,
        tfds_split=split,
        oracle_stem_id=oracle_stem_id,
        sample_rate=NSYNTH_SR,
        clip_duration_s=clip_duration_s,
        density=density,
        nsynth_source_ids=ids,
        ground_truth_notes=gt,
    )


def load_sidecar(path: Path) -> DenseEvalSidecar:
    return DenseEvalSidecar.model_validate(json.loads(path.read_text(encoding="utf-8")))
