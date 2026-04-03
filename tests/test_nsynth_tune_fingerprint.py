"""Unit tests for NSynth HPO fingerprint / study naming (no TensorFlow)."""

from __future__ import annotations

import os

import pytest

from song_analyzer.instruments.nsynth_tune_fingerprint import (
    code_digest_for_test_files,
    fingerprint_payload,
    nsynth_study_name,
    study_suffix_from_payload,
)


def test_study_suffix_stable_for_same_payload() -> None:
    p = fingerprint_payload(
        dataset_name="nsynth/full",
        dataset_version="1.2.3",
        code_digest="a" * 64,
    )
    a = study_suffix_from_payload(p)
    b = study_suffix_from_payload(p)
    assert a == b
    assert len(a) == 12
    assert nsynth_study_name(a) == f"nsynth_family_{a}"


def test_study_suffix_changes_with_dataset_version() -> None:
    base = dict(
        dataset_name="nsynth/full",
        code_digest="b" * 64,
    )
    p1 = fingerprint_payload(dataset_version="1.0.0", **base)
    p2 = fingerprint_payload(dataset_version="2.0.0", **base)
    assert study_suffix_from_payload(p1) != study_suffix_from_payload(p2)


def test_study_suffix_changes_with_tf_versions() -> None:
    p1 = {
        "dataset_name": "nsynth/full",
        "dataset_version": "1",
        "code_digest": "c" * 64,
        "tensorflow": "2.14.0",
        "tensorflow_datasets": "4.9.0",
    }
    p2 = {**p1, "tensorflow": "2.15.0"}
    assert study_suffix_from_payload(p1) != study_suffix_from_payload(p2)


def test_code_digest_test_helper_ordering() -> None:
    d1 = code_digest_for_test_files({"a.py": b"x", "b.py": b"y"})
    d2 = code_digest_for_test_files({"b.py": b"y", "a.py": b"x"})
    assert d1 == d2


def test_code_digest_changes_with_content() -> None:
    d1 = code_digest_for_test_files({"train.py": b"v1"})
    d2 = code_digest_for_test_files({"train.py": b"v2"})
    assert d1 != d2


@pytest.mark.skipif(os.environ.get("RUN_NSYNTH_TUNE_SMOKE") != "1", reason="set RUN_NSYNTH_TUNE_SMOKE=1 to run")
def test_tune_one_trial_smoke(tmp_path) -> None:
    pytest.importorskip("optuna")
    pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflow_datasets")

    from song_analyzer.instruments.tune_nsynth import tune_nsynth_main

    out = tmp_path / "model.pt"
    tune_nsynth_main(
        out=out,
        device="cpu",
        n_trials=1,
        tune_cache_dir=tmp_path / "cache",
        no_tune_cache=False,
        tune_fresh=True,
        max_val_steps=1,
        final_epochs=1,
        final_max_steps_per_epoch=1,
    )
    assert out.is_file()
