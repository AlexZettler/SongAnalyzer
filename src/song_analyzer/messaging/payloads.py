"""JSON payloads published to Pub/Sub (Pydantic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class SongRequestPayload(BaseModel):
    """Expand corpus: metadata, optional lyrics, optional local audio copy."""

    request_id: str
    corpus_root: str = Field(..., description="Filesystem path to corpus root")
    mbid: str | None = None
    title: str | None = None
    artist: str | None = None
    source: str = "pubsub"
    source_id: str | None = None
    local_audio_path: str | None = Field(
        None,
        description="If set, copy or reference this file into the corpus (licensed/local use only)",
    )
    copy_audio: bool = True
    lyrics_connector: str = "genius"
    musicbrainz_user_agent: str | None = None
    musicbrainz_enrich: bool = Field(
        True,
        description="When mbid is set, enrich title/artist from MusicBrainz",
    )


class SongCompletePayload(BaseModel):
    request_id: str
    status: Literal["ok", "error"]
    track_id: str | None = None
    audio_relpath: str | None = None
    lyrics_fetched: bool = False
    error: str | None = None


class HpoRequestPayload(BaseModel):
    exploration_id: str
    n_trials: int = 20
    device: str = "cuda"
    tune_cache_dir: str | None = None
    tfds_data_dir: str | None = None
    max_val_steps: int = 200
    final_epochs: int = 3
    final_max_steps_per_epoch: int = 500
    no_tune_cache: bool = False
    tune_fresh: bool = False
    archive_tune_db_before: bool = True
    skip_final_train: bool = Field(
        True,
        description="If true, only run Optuna; final full training is a separate train.request",
    )
    out_checkpoint: str | None = Field(
        None,
        description="When skip_final_train is false, write final .pt here",
    )
    log_level: str = "INFO"


class HpoCompletePayload(BaseModel):
    exploration_id: str
    status: Literal["ok", "error"]
    study_name: str | None = None
    best_params: dict[str, Any] | None = None
    best_value: float | None = None
    archived_db_path: str | None = None
    trials_csv_path: str | None = None
    out_checkpoint: str | None = None
    error: str | None = None


class TrainRequestPayload(BaseModel):
    train_job_id: str
    study_name: str
    tune_cache_dir: str | None = None
    out: str = Field(..., description="Output .pt path")
    epochs: int = 3
    max_steps_per_epoch: int = 500
    max_val_steps: int | None = None
    device: str = "cuda"
    tfds_data_dir: str | None = None
    log_level: str = "INFO"


class TrainCompletePayload(BaseModel):
    train_job_id: str
    status: Literal["ok", "error"]
    checkpoint_path: str | None = None
    best_params_used: dict[str, Any] | None = None
    error: str | None = None


def payload_json(model: BaseModel) -> bytes:
    return model.model_dump_json().encode("utf-8")


def corpus_root_path(payload: SongRequestPayload) -> Path:
    return Path(payload.corpus_root).resolve()
