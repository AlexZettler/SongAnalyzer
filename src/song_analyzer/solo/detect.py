from __future__ import annotations

import numpy as np

from song_analyzer.schema import SoloSegment


def _rms(x: np.ndarray, start: int, end: int) -> float:
    sl = x[start:end]
    if sl.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(sl**2) + 1e-18))


def detect_solo_segments(
    stems: dict[str, np.ndarray],
    sr: int,
    *,
    win_s: float = 0.1,
    hop_s: float = 0.05,
    dominance_threshold: float = 0.55,
    min_segment_s: float = 0.15,
    vocals_downweight: float = 0.85,
) -> list[SoloSegment]:
    """
    Mark intervals where one Demucs stem dominates total energy (RMS) in short windows.
    ``vocals_downweight`` scales vocals RMS when computing shares so instrumental solos
    are easier to detect against sung mixes.
    """
    if not stems:
        return []

    names = sorted(stems.keys())
    lens = {n: len(stems[n]) for n in names}
    n_samples = min(lens.values())
    if n_samples < 16:
        return []

    win = max(1, int(win_s * sr))
    hop = max(1, int(hop_s * sr))
    duration_s = n_samples / sr

    raw_windows: list[tuple[float, str, float]] = []
    pos = 0
    while pos + win <= n_samples:
        end = pos + win
        t_center = (pos + end) / (2.0 * sr)
        rms_by: dict[str, float] = {}
        for n in names:
            x = np.asarray(stems[n][:n_samples], dtype=np.float32)
            if x.ndim > 1:
                x = np.mean(x, axis=0)
            r = _rms(x, pos, end)
            if n == "vocals":
                r *= float(vocals_downweight)
            rms_by[n] = r
        total = sum(rms_by.values()) + 1e-18
        winner = max(rms_by, key=rms_by.get)  # type: ignore[arg-type]
        dom = rms_by[winner] / total
        if dom >= dominance_threshold:
            raw_windows.append((t_center, winner, dom))
        pos += hop

    if not raw_windows:
        return []

    merged: list[SoloSegment] = []
    cur_start_t = raw_windows[0][0] - win_s / 2
    cur_end_t = raw_windows[0][0] + win_s / 2
    cur_stem = raw_windows[0][1]
    cur_dom = raw_windows[0][2]

    def flush() -> None:
        nonlocal cur_start_t, cur_end_t, cur_stem, cur_dom
        if cur_end_t - cur_start_t >= min_segment_s:
            merged.append(
                SoloSegment(
                    start_time_s=float(max(0.0, cur_start_t)),
                    end_time_s=float(min(duration_s, cur_end_t)),
                    stem_id=cur_stem,
                    dominance=float(cur_dom),
                )
            )

    for t_c, st, dom in raw_windows[1:]:
        if st == cur_stem and t_c <= cur_end_t + hop_s * 1.5:
            cur_end_t = t_c + win_s / 2
            cur_dom = max(cur_dom, dom)
        else:
            flush()
            cur_start_t = t_c - win_s / 2
            cur_end_t = t_c + win_s / 2
            cur_stem = st
            cur_dom = dom
    flush()
    return merged
