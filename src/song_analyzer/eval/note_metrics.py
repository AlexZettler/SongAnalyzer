"""Tolerant one-to-one matching between predicted and reference note lists (F1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from song_analyzer.schema import NoteEvent


BIG = 1.0e9


@dataclass(frozen=True)
class NoteMatchConfig:
    onset_tol_s: float = 0.05
    offset_tol_s: float = 0.05
    pitch_tol_semi: int = 0


def _note_arrays(
    notes: list[NoteEvent],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Onset, offset, pitch, index."""
    if not notes:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    on = np.array([n.start_time_s for n in notes], dtype=np.float64)
    off = np.array([n.end_time_s for n in notes], dtype=np.float64)
    pitch = np.array([n.midi_pitch for n in notes], dtype=np.int64)
    idx = np.arange(len(notes), dtype=np.int64)
    return on, off, pitch, idx


def _matchable(
    p_on: float,
    p_off: float,
    p_midi: int,
    r_on: float,
    r_off: float,
    r_midi: int,
    cfg: NoteMatchConfig,
) -> bool:
    if abs(p_midi - r_midi) > cfg.pitch_tol_semi:
        return False
    if abs(p_on - r_on) > cfg.onset_tol_s:
        return False
    if abs(p_off - r_off) > cfg.offset_tol_s:
        return False
    return True


def _cost(
    p_on: float,
    p_off: float,
    p_midi: int,
    r_on: float,
    r_off: float,
    r_midi: int,
    cfg: NoteMatchConfig,
) -> float:
    if not _matchable(p_on, p_off, p_midi, r_on, r_off, r_midi, cfg):
        return BIG
    # Normalized squared error (0 = perfect)
    c_on = ((p_on - r_on) / max(cfg.onset_tol_s, 1e-9)) ** 2
    c_off = ((p_off - r_off) / max(cfg.offset_tol_s, 1e-9)) ** 2
    c_p = float((p_midi - r_midi) ** 2) / max(float(cfg.pitch_tol_semi + 1), 1.0) ** 2
    return c_on + c_off + c_p


def match_notes(
    predicted: list[NoteEvent],
    reference: list[NoteEvent],
    cfg: NoteMatchConfig | None = None,
) -> dict[str, float | int | list[tuple[int, int]]]:
    """
    One-to-one minimum-cost matching (Hungarian) over notes that fall within tolerances.

    Unmatched predictions count as false positives; unmatched references as false negatives.
    Returns precision, recall, f1, match count, and list of (pred_index, ref_index) pairs.
    """
    from scipy.optimize import linear_sum_assignment

    cfg = cfg or NoteMatchConfig()

    n_p, n_r = len(predicted), len(reference)
    if n_p == 0 and n_r == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "matches": 0,
            "pairs": [],
        }
    if n_p == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matches": 0, "pairs": []}
    if n_r == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matches": 0, "pairs": []}

    n = max(n_p, n_r)
    cost = np.full((n, n), BIG, dtype=np.float64)
    p_on, p_off, p_midi, _ = _note_arrays(predicted)
    r_on, r_off, r_midi, _ = _note_arrays(reference)

    for i in range(n_p):
        for j in range(n_r):
            cost[i, j] = _cost(
                float(p_on[i]),
                float(p_off[i]),
                int(p_midi[i]),
                float(r_on[j]),
                float(r_off[j]),
                int(r_midi[j]),
                cfg,
            )
    # Dummy rows/cols already BIG

    row_ind, col_ind = linear_sum_assignment(cost)
    pairs: list[tuple[int, int]] = []
    for i, j in zip(row_ind, col_ind, strict=True):
        if i < n_p and j < n_r and cost[i, j] < BIG:
            pairs.append((int(i), int(j)))

    matches = len(pairs)
    precision = matches / n_p
    recall = matches / n_r
    if precision + recall <= 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "matches": matches,
        "pairs": pairs,
    }
