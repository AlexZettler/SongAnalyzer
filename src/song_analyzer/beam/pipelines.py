"""Apache Beam streaming pipelines: Pub/Sub request -> completion topic."""

from __future__ import annotations

import apache_beam as beam
from apache_beam.io.gcp.pubsub import ReadFromPubSub, WriteToPubSub
from apache_beam.options.pipeline_options import GoogleCloudOptions, PipelineOptions, StandardOptions

from song_analyzer.beam.handlers import handle_hpo_bytes, handle_song_bytes, handle_train_bytes
from song_analyzer.messaging.topics import (
    TOPIC_HPO_COMPLETE,
    TOPIC_SONG_COMPLETE,
    TOPIC_TRAIN_COMPLETE,
)


def _pipeline_options(*, project: str, runner: str = "DirectRunner") -> PipelineOptions:
    opts = PipelineOptions(
        [
            f"--runner={runner}",
            "--streaming",
            "--project",
            project,
        ]
    )
    std = opts.view_as(StandardOptions)
    std.streaming = True
    g = opts.view_as(GoogleCloudOptions)
    g.project = project
    return opts


def _subscription_path(project: str, subscription_id: str) -> str:
    return f"projects/{project}/subscriptions/{subscription_id}"


def _topic_path(project: str, topic_id: str) -> str:
    return f"projects/{project}/topics/{topic_id}"


def run_song_ingest_pipeline(
    *,
    project: str,
    subscription_id: str,
    runner: str = "DirectRunner",
) -> None:
    """Read ``song.requests`` subscription; write ``song.complete`` messages."""
    opts = _pipeline_options(project=project, runner=runner)
    sub = _subscription_path(project, subscription_id)
    top = _topic_path(project, TOPIC_SONG_COMPLETE)
    with beam.Pipeline(options=opts) as p:
        (
            p
            | "read_song_requests" >> ReadFromPubSub(subscription=sub)
            | "process_song" >> beam.Map(handle_song_bytes)
            | "write_song_complete" >> WriteToPubSub(topic=top)
        )


def run_hpo_pipeline(
    *,
    project: str,
    subscription_id: str,
    runner: str = "DirectRunner",
) -> None:
    opts = _pipeline_options(project=project, runner=runner)
    sub = _subscription_path(project, subscription_id)
    top = _topic_path(project, TOPIC_HPO_COMPLETE)
    with beam.Pipeline(options=opts) as p:
        (
            p
            | "read_hpo_requests" >> ReadFromPubSub(subscription=sub)
            | "process_hpo" >> beam.Map(handle_hpo_bytes)
            | "write_hpo_complete" >> WriteToPubSub(topic=top)
        )


def run_train_pipeline(
    *,
    project: str,
    subscription_id: str,
    runner: str = "DirectRunner",
) -> None:
    opts = _pipeline_options(project=project, runner=runner)
    sub = _subscription_path(project, subscription_id)
    top = _topic_path(project, TOPIC_TRAIN_COMPLETE)
    with beam.Pipeline(options=opts) as p:
        (
            p
            | "read_train_requests" >> ReadFromPubSub(subscription=sub)
            | "process_train" >> beam.Map(handle_train_bytes)
            | "write_train_complete" >> WriteToPubSub(topic=top)
        )
