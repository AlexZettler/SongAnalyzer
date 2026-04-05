from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from song_analyzer.explore.history import RunRecord, append_record, load_history
from song_analyzer.explore.param_space import preset_dense_eval, preset_nsynth_tune
from song_analyzer.explore.proposer import propose_next
from song_analyzer.explore.runner import exploration_step


def test_nsynth_encode_decode_roundtrip() -> None:
    space = preset_nsynth_tune()
    p = {
        "lr": 3e-4,
        "batch_size": 32.0,
        "epochs": 3,
        "max_steps_per_epoch": 400,
        "weight_decay": 0.01,
    }
    u = space.encode(p)
    q = space.decode(u)
    assert q["batch_size"] == 32.0
    assert q["epochs"] == 3
    assert abs(float(q["lr"]) - 3e-4) < 1e-8 * max(1.0, 3e-4)
    assert q["max_steps_per_epoch"] == 400
    assert abs(float(q["weight_decay"]) - 0.01) < 1e-9


def test_dense_eval_bool_roundtrip() -> None:
    space = preset_dense_eval()
    p = {
        "n_notes": 5,
        "clip_duration_s": 12.0,
        "same_family_stack": True,
        "detune_max_cents": 10.0,
        "level_jitter_db": 3.0,
        "note_audio_duration_s": 3.8,
    }
    u = space.encode(p)
    q = space.decode(u)
    assert q["same_family_stack"] is True
    assert q["n_notes"] == 5


def test_jsonl_history_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "h.jsonl"
    append_record(
        path,
        RunRecord(
            step=0,
            preset="x",
            mode="novelty",
            params={"a": 1},
            vector=[0.1, 0.2],
            metric=None,
            min_dist_to_history=float("inf"),
        ),
    )
    rows = load_history(path)
    assert len(rows) == 1
    assert rows[0].params == {"a": 1}
    assert rows[0].vector == [0.1, 0.2]


def test_dry_run_session_archive_enables_local(tmp_path: Path) -> None:
    space = preset_nsynth_tune()
    rng = np.random.default_rng(42)
    session: list[np.ndarray] = []
    modes: list[str] = []
    for _ in range(24):
        _, prop, _ = exploration_step(
            space=space,
            state_path=tmp_path / "missing.jsonl",
            preset_name="nsynth-tune",
            step_index=len(modes),
            rng=rng,
            explore_probability=0.0,
            n_novelty_candidates=64,
            local_sigma=0.2,
            weights=None,
            on_propose=None,
            persist=False,
            transient_archive=session,
        )
        modes.append(prop.mode)
    assert modes[0] == "novelty"
    assert any(m == "local" for m in modes[1:])


def test_propose_next_respects_explore_probability() -> None:
    space = preset_nsynth_tune()
    rng = np.random.default_rng(0)
    archive = [np.zeros(space.ndim)]
    novelty = sum(
        1
        for _ in range(200)
        if propose_next(
            space,
            archive,
            rng=rng,
            explore_probability=1.0,
            n_novelty_candidates=32,
            local_sigma=0.1,
        ).mode
        == "novelty"
    )
    assert novelty == 200
