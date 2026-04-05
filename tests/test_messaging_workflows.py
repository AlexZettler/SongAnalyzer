"""Registry topics and models stay aligned with topics.py and payloads."""

from __future__ import annotations

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
from song_analyzer.messaging.workflows import WORKFLOWS


def test_workflows_match_topic_constants() -> None:
    expected = (
        (TOPIC_SONG_REQUESTS, TOPIC_SONG_COMPLETE, SUBSCRIPTION_SONG_REQUESTS),
        (TOPIC_HPO_REQUESTS, TOPIC_HPO_COMPLETE, SUBSCRIPTION_HPO_REQUESTS),
        (TOPIC_TRAIN_REQUESTS, TOPIC_TRAIN_COMPLETE, SUBSCRIPTION_TRAIN_REQUESTS),
    )
    assert len(WORKFLOWS) == len(expected)
    for wf, (req_t, comp_t, sub) in zip(WORKFLOWS, expected, strict=True):
        assert wf.request_topic == req_t
        assert wf.complete_topic == comp_t
        assert wf.subscription_id == sub


def test_workflow_samples_roundtrip() -> None:
    for wf in WORKFLOWS:
        req = wf.sample_request()
        comp = wf.sample_complete()
        req_raw = req.model_dump_json()
        comp_raw = comp.model_dump_json()
        assert wf.request_model.model_validate_json(req_raw) == req
        assert wf.complete_model.model_validate_json(comp_raw) == comp
