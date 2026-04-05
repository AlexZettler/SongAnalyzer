"""Registry of Pub/Sub workflows: topic names, subscriptions, and payload models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

from song_analyzer.messaging.payloads import (
    HpoCompletePayload,
    HpoRequestPayload,
    SongCompletePayload,
    SongRequestPayload,
    TrainCompletePayload,
    TrainRequestPayload,
)
from song_analyzer.messaging.topics import (
    SUBSCRIPTION_HPO_REQUESTS,
    SUBSCRIPTION_SONG_REQUESTS,
    SUBSCRIPTION_TRAIN_REQUESTS,
    TOPIC_HPO_COMPLETE,
    TOPIC_HPO_REQUESTS,
    TOPIC_SONG_COMPLETE,
    TOPIC_SONG_REQUESTS,
    TOPIC_TRAIN_COMPLETE,
    TOPIC_TRAIN_REQUESTS,
)


@dataclass(frozen=True)
class MessagingWorkflow:
    """One request topic → worker subscription → completion topic pair."""

    name: str
    request_topic: str
    complete_topic: str
    subscription_id: str | None
    request_model: type[BaseModel]
    complete_model: type[BaseModel]
    sample_request: Callable[[], BaseModel]
    sample_complete: Callable[[], BaseModel]


def _song_request() -> SongRequestPayload:
    return SongRequestPayload(request_id="r1", corpus_root="D:/corpus")


def _song_complete() -> SongCompletePayload:
    return SongCompletePayload(
        request_id="r1",
        status="ok",
        track_id="t1",
        audio_relpath="audio/t1.wav",
        lyrics_fetched=False,
    )


def _hpo_request() -> HpoRequestPayload:
    return HpoRequestPayload(
        exploration_id="exp1",
        n_trials=2,
        skip_final_train=True,
        archive_tune_db_before=False,
    )


def _hpo_complete() -> HpoCompletePayload:
    return HpoCompletePayload(
        exploration_id="exp1",
        status="ok",
        study_name="nsynth_family_abc__exp1",
        best_params={"lr": 0.001},
        best_value=0.42,
        trials_csv_path="D:/tune/exports/exp1_trials.csv",
    )


def _train_request() -> TrainRequestPayload:
    return TrainRequestPayload(
        train_job_id="t1",
        study_name="nsynth_family_abc__exp1",
        out="D:/out/model.pt",
    )


def _train_complete() -> TrainCompletePayload:
    return TrainCompletePayload(
        train_job_id="t1",
        status="ok",
        checkpoint_path="D:/out/model.pt",
        best_params_used={"lr": 0.001},
    )


WORKFLOWS: tuple[MessagingWorkflow, ...] = (
    MessagingWorkflow(
        name="song",
        request_topic=TOPIC_SONG_REQUESTS,
        complete_topic=TOPIC_SONG_COMPLETE,
        subscription_id=SUBSCRIPTION_SONG_REQUESTS,
        request_model=SongRequestPayload,
        complete_model=SongCompletePayload,
        sample_request=_song_request,
        sample_complete=_song_complete,
    ),
    MessagingWorkflow(
        name="hpo",
        request_topic=TOPIC_HPO_REQUESTS,
        complete_topic=TOPIC_HPO_COMPLETE,
        subscription_id=SUBSCRIPTION_HPO_REQUESTS,
        request_model=HpoRequestPayload,
        complete_model=HpoCompletePayload,
        sample_request=_hpo_request,
        sample_complete=_hpo_complete,
    ),
    MessagingWorkflow(
        name="train",
        request_topic=TOPIC_TRAIN_REQUESTS,
        complete_topic=TOPIC_TRAIN_COMPLETE,
        subscription_id=SUBSCRIPTION_TRAIN_REQUESTS,
        request_model=TrainRequestPayload,
        complete_model=TrainCompletePayload,
        sample_request=_train_request,
        sample_complete=_train_complete,
    ),
)
