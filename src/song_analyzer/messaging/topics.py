"""Google Pub/Sub topic names (local emulator or GCP)."""

from __future__ import annotations

# Request / completion pairs for each workflow
TOPIC_SONG_REQUESTS = "song.requests"
TOPIC_SONG_COMPLETE = "song.complete"

TOPIC_HPO_REQUESTS = "hpo.requests"
TOPIC_HPO_COMPLETE = "hpo.complete"

TOPIC_TRAIN_REQUESTS = "train.requests"
TOPIC_TRAIN_COMPLETE = "train.complete"

ALL_TOPICS: tuple[str, ...] = (
    TOPIC_SONG_REQUESTS,
    TOPIC_SONG_COMPLETE,
    TOPIC_HPO_REQUESTS,
    TOPIC_HPO_COMPLETE,
    TOPIC_TRAIN_REQUESTS,
    TOPIC_TRAIN_COMPLETE,
)

# Pull subscriptions created by ``messaging setup-pubsub`` (must match admin.py).
SUBSCRIPTION_SONG_REQUESTS = "song-requests-worker"
SUBSCRIPTION_HPO_REQUESTS = "hpo-requests-worker"
SUBSCRIPTION_TRAIN_REQUESTS = "train-requests-worker"
