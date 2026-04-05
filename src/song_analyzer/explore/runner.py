"""Step and loop: propose, optional callback, persist."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from song_analyzer.explore.history import RunRecord, append_record, load_history
from song_analyzer.explore.param_space import ParamSpace
from song_analyzer.explore.proposer import Proposal, propose_next

ProposalCallback = Callable[[dict[str, Any], Proposal, int], float | None]


def _archive_from_history(path: Path, ndim: int) -> list[np.ndarray]:
    rows = load_history(path)
    out: list[np.ndarray] = []
    for r in rows:
        v = np.asarray(r.vector, dtype=np.float64)
        if v.shape == (ndim,):
            out.append(v)
    return out


def exploration_step(
    *,
    space: ParamSpace,
    state_path: Path,
    preset_name: str,
    step_index: int,
    rng: np.random.Generator,
    explore_probability: float,
    n_novelty_candidates: int,
    local_sigma: float,
    weights: np.ndarray | None,
    on_propose: ProposalCallback | None,
    persist: bool,
    transient_archive: list[np.ndarray] | None = None,
) -> tuple[dict[str, Any], Proposal, float | None]:
    archive = list(_archive_from_history(state_path, space.ndim))
    if transient_archive:
        archive.extend(transient_archive)
    prop = propose_next(
        space,
        archive,
        rng=rng,
        explore_probability=explore_probability,
        n_novelty_candidates=n_novelty_candidates,
        local_sigma=local_sigma,
        weights=weights,
    )
    params = space.decode(prop.vector)
    metric: float | None = None
    if on_propose is not None:
        metric = on_propose(params, prop, step_index)
    if persist:
        w_list = None if weights is None else [float(x) for x in weights.tolist()]
        append_record(
            state_path,
            RunRecord(
                step=step_index,
                preset=preset_name,
                mode=prop.mode,
                params=params,
                vector=[float(x) for x in prop.vector.tolist()],
                metric=metric,
                min_dist_to_history=prop.min_dist_to_history,
                weights=w_list,
            ),
        )
    elif transient_archive is not None:
        transient_archive.append(np.copy(prop.vector))
    return params, prop, metric


def run_exploration_loop(
    *,
    space: ParamSpace,
    state_path: Path,
    preset_name: str,
    steps: int,
    seed: int,
    explore_probability: float,
    n_novelty_candidates: int,
    local_sigma: float,
    weights: np.ndarray | None,
    on_propose: ProposalCallback | None,
    persist: bool,
    start_step: int | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    existing = load_history(state_path)
    if start_step is not None:
        base_step = start_step
    elif existing:
        base_step = max(r.step for r in existing) + 1
    else:
        base_step = 0
    session: list[np.ndarray] = [] if not persist else []
    for k in range(steps):
        exploration_step(
            space=space,
            state_path=state_path,
            preset_name=preset_name,
            step_index=base_step + k,
            rng=rng,
            explore_probability=explore_probability,
            n_novelty_candidates=n_novelty_candidates,
            local_sigma=local_sigma,
            weights=weights,
            on_propose=on_propose,
            persist=persist,
            transient_archive=session if not persist else None,
        )
