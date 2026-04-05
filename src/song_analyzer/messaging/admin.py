"""Create Pub/Sub topics and pull subscriptions (emulator or GCP)."""

from __future__ import annotations

from google.api_core.exceptions import AlreadyExists
from google.cloud import pubsub_v1

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


def ensure_topic(publisher: pubsub_v1.PublisherClient, project: str, topic_id: str) -> str:
    path = publisher.topic_path(project, topic_id)
    try:
        publisher.create_topic(name=path)
    except AlreadyExists:
        pass
    return path


def ensure_pull_subscription(
    publisher: pubsub_v1.PublisherClient,
    subscriber: pubsub_v1.SubscriberClient,
    project: str,
    topic_id: str,
    subscription_id: str,
) -> str:
    sub_path = subscriber.subscription_path(project, subscription_id)
    topic_path = publisher.topic_path(project, topic_id)
    try:
        subscriber.create_subscription(name=sub_path, topic=topic_path)
    except AlreadyExists:
        pass
    return sub_path


def ensure_all_topics_and_subscriptions(project: str) -> None:
    """Create all workflow topics and worker pull subscriptions."""
    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()
    for tid in (
        TOPIC_SONG_REQUESTS,
        TOPIC_SONG_COMPLETE,
        TOPIC_HPO_REQUESTS,
        TOPIC_HPO_COMPLETE,
        TOPIC_TRAIN_REQUESTS,
        TOPIC_TRAIN_COMPLETE,
    ):
        ensure_topic(publisher, project, tid)
    ensure_pull_subscription(
        publisher,
        subscriber,
        project,
        TOPIC_SONG_REQUESTS,
        SUBSCRIPTION_SONG_REQUESTS,
    )
    ensure_pull_subscription(
        publisher,
        subscriber,
        project,
        TOPIC_HPO_REQUESTS,
        SUBSCRIPTION_HPO_REQUESTS,
    )
    ensure_pull_subscription(
        publisher,
        subscriber,
        project,
        TOPIC_TRAIN_REQUESTS,
        SUBSCRIPTION_TRAIN_REQUESTS,
    )
