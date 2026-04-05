"""Publish JSON payloads to Google Pub/Sub (works with PUBSUB_EMULATOR_HOST)."""

from __future__ import annotations

from pydantic import BaseModel

from song_analyzer.messaging.topics import (
    TOPIC_HPO_REQUESTS,
    TOPIC_SONG_REQUESTS,
    TOPIC_TRAIN_REQUESTS,
)


def publish_json(
    project: str,
    topic: str,
    message: BaseModel,
    *,
    ordering_key: str | None = None,
) -> str:
    """Publish model_dump_json() to topic. Returns Pub/Sub message id when available."""
    from google.cloud import pubsub_v1

    client = pubsub_v1.PublisherClient()
    path = client.topic_path(project, topic)
    data = message.model_dump_json().encode("utf-8")
    kwargs: dict[str, str] = {}
    if ordering_key is not None:
        kwargs["ordering_key"] = ordering_key
    future = client.publish(path, data, **kwargs)
    return future.result()


def publish_song_request(project: str, message: BaseModel) -> str:
    return publish_json(project, TOPIC_SONG_REQUESTS, message)


def publish_hpo_request(project: str, message: BaseModel) -> str:
    return publish_json(project, TOPIC_HPO_REQUESTS, message)


def publish_train_request(project: str, message: BaseModel) -> str:
    return publish_json(project, TOPIC_TRAIN_REQUESTS, message)
