from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NoteEvent(BaseModel):
    """A single note on a stem."""

    start_time_s: float = Field(..., ge=0, description="Onset in seconds")
    end_time_s: float = Field(..., ge=0, description="Offset in seconds")
    midi_pitch: int = Field(..., ge=0, le=127)
    velocity: float | None = Field(None, ge=0, le=1, description="Optional 0–1 confidence from transcriber")
    stem_id: str


class InstrumentPrediction(BaseModel):
    stem_id: str
    family: str
    confidence: float = Field(..., ge=0, le=1)
    family_logits: dict[str, float] | None = None


class ChordSegment(BaseModel):
    start_time_s: float
    end_time_s: float
    chord_label: str
    pitch_classes: list[int] = Field(default_factory=list, description="0=C … 11=B")
    stem_id: str | None = Field(None, description="None = mix-level aggregation")


class StemAudioRef(BaseModel):
    stem_id: str
    path: str | None = Field(None, description="Written WAV path when stems are exported")


class AnalysisResult(BaseModel):
    source_path: str
    sample_rate: int
    duration_s: float
    stems: list[StemAudioRef] = Field(default_factory=list)
    instruments: list[InstrumentPrediction] = Field(default_factory=list)
    notes: list[NoteEvent] = Field(default_factory=list)
    chords: list[ChordSegment] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    def model_dump_json_pretty(self) -> str:
        return self.model_dump_json(indent=2)
