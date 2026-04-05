"""Novelty search + local mutation in normalized parameter space."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from song_analyzer.explore.param_space import ParamSpace

Mode = Literal["novelty", "local"]


@dataclass(frozen=True)
class Proposal:
    mode: Mode
    vector: np.ndarray
    # None when there is no prior archive (first step) or distance is undefined.
    min_dist_to_history: float | None


def _weighted_distance(
    a: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray,
) -> float:
    d = a - b
    return float(np.sqrt(np.sum(weights * d * d)))


def min_dist_to_archive(
    u: np.ndarray,
    archive: list[np.ndarray],
    weights: np.ndarray,
) -> float | None:
    if not archive:
        return None
    return min(_weighted_distance(u, v, weights) for v in archive)


def propose_novelty(
    space: ParamSpace,
    archive: list[np.ndarray],
    *,
    rng: np.random.Generator,
    n_candidates: int,
    weights: np.ndarray | None = None,
) -> Proposal:
    w = (
        np.ones(space.ndim, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64).reshape(space.ndim)
    )
    best_u: np.ndarray | None = None
    best_score = -1.0
    best_min: float | None = None
    for _ in range(n_candidates):
        u = rng.random(space.ndim)
        md = min_dist_to_archive(u, archive, w)
        score = md if md is not None else float("inf")
        if score > best_score:
            best_score = score
            best_u = u
            best_min = md
    assert best_u is not None
    if best_min is not None and math.isinf(best_min):
        best_min = None
    return Proposal(mode="novelty", vector=best_u, min_dist_to_history=best_min)


def propose_local(
    space: ParamSpace,
    archive: list[np.ndarray],
    *,
    rng: np.random.Generator,
    sigma: float,
    weights: np.ndarray | None = None,
) -> Proposal:
    w = (
        np.ones(space.ndim, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64).reshape(space.ndim)
    )
    if not archive:
        u = rng.random(space.ndim)
        return Proposal(mode="local", vector=u, min_dist_to_history=None)

    base = archive[int(rng.integers(0, len(archive)))]
    noise = rng.normal(0.0, sigma, size=space.ndim)
    u = np.clip(base + noise, 0.0, 1.0)
    md = min_dist_to_archive(u, archive, w)
    if md is not None and math.isinf(md):
        md = None
    return Proposal(mode="local", vector=u, min_dist_to_history=md)


def propose_next(
    space: ParamSpace,
    archive: list[np.ndarray],
    *,
    rng: np.random.Generator,
    explore_probability: float,
    n_novelty_candidates: int,
    local_sigma: float,
    weights: np.ndarray | None = None,
) -> Proposal:
    if rng.random() < explore_probability or not archive:
        return propose_novelty(
            space,
            archive,
            rng=rng,
            n_candidates=n_novelty_candidates,
            weights=weights,
        )
    return propose_local(
        space,
        archive,
        rng=rng,
        sigma=local_sigma,
        weights=weights,
    )
