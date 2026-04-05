"""JSON sidecar schema for dense synthetic evaluation clips (ground truth + metadata)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


SCHEMA_VERSION = 1


class GroundTruthNote(BaseModel):
    """One contributing NSynth note in a synthetic clip."""

    source_note_id: str = Field(..., description="NSynth example id (TFDS `id`)")
    start_time_s: float = Field(..., ge=0)
    end_time_s: float = Field(..., ge=0)
    midi_pitch: int = Field(..., ge=0, le=127)
    instrument_family: str = Field(..., description="NSynth family name, e.g. guitar")
    demucs_bucket: str = Field(
        ...,
        description="Target htdemucs-style stem bucket used when building a full mix",
    )
    velocity: int | None = Field(None, ge=0, le=127)
    gain_linear: float = Field(1.0, gt=0, description="Applied RMS scale before summing")


class DensityParams(BaseModel):
    """Parameters used when rendering the clip (for reproducibility and reporting)."""

    n_notes: int = Field(..., ge=1)
    clip_duration_s: float = Field(..., gt=0)
    same_family_stack: bool = Field(False, description="If true, all notes share one family")
    detune_max_cents: float = Field(0.0, ge=0)
    level_jitter_db: float = Field(0.0, ge=0, description="Max half-spread of per-note gain in dB")
    note_audio_duration_s: float = Field(
        4.0,
        gt=0,
        description="NSynth clip length placed on the timeline (typically 4s at 16 kHz)",
    )


class DenseEvalSidecar(BaseModel):
    """
    Sidecar written next to generated WAV files (e.g. ``dense_eval.json``).

    Use **valid** or **test** TFDS splits only for evaluation to avoid train leakage.
    """

    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    clip_id: str
    seed: int
    tfds_split: str = Field(..., description="e.g. valid")
    oracle_stem_id: str = Field("oracle", description="stem_id used in predicted NoteEvents")
    sample_rate: int = Field(16000, gt=0)
    clip_duration_s: float = Field(..., gt=0)
    density: DensityParams
    nsynth_source_ids: list[str] = Field(default_factory=list, description="All contributing ids, order matches generation")
    ground_truth_notes: list[GroundTruthNote] = Field(default_factory=list)
    # Full-mix mode (optional): relative paths from sidecar directory
    demucs_mix_relpath: str | None = Field(
        None,
        description="If set, path to mixture WAV under full-stack evaluation",
    )
    per_bucket_wav_relpaths: dict[str, str] | None = Field(
        None,
        description="Reference stem WAVs per Demucs name (drums, bass, other, vocals)",
    )
    extra: dict[str, Any] = Field(default_factory=dict, description="Free-form metadata")
