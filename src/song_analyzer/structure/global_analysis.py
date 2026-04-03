from __future__ import annotations

import numpy as np
import librosa

from song_analyzer.schema import GlobalStructureResult, StructuralSegment


def _segment_edges_from_boundaries(boundaries: np.ndarray, n_frames: int) -> np.ndarray:
    b = np.asarray(boundaries, dtype=int).reshape(-1)
    b = b[(b >= 0) & (b < n_frames)]
    edges = np.unique(np.concatenate([[0], b, [n_frames]]))
    return edges


def _repeat_group_ids(chroma: np.ndarray, edges: np.ndarray, sim_threshold: float) -> list[int]:
    """Cosine similarity of mean chroma per segment; union-find for transitive groups."""
    pairs: list[tuple[int, np.ndarray]] = []
    for idx in range(len(edges) - 1):
        s, e = int(edges[idx]), int(edges[idx + 1])
        if e <= s:
            v = np.zeros(chroma.shape[0], dtype=np.float64)
        else:
            v = np.mean(chroma[:, s:e], axis=1)
        nrm = float(np.linalg.norm(v)) + 1e-9
        pairs.append((idx, (v / nrm).astype(np.float64)))

    n_seg = len(pairs)
    parent = list(range(n_seg))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    vecs = [p[1] for p in pairs]
    for i in range(n_seg):
        for j in range(i + 1, n_seg):
            if float(np.dot(vecs[i], vecs[j])) >= sim_threshold:
                union(i, j)

    roots: dict[int, int] = {}
    out: list[int] = []
    next_id = 0
    for i in range(n_seg):
        r = find(i)
        if r not in roots:
            roots[r] = next_id
            next_id += 1
        out.append(roots[r])
    return out


def analyze_global_structure(
    y: np.ndarray,
    sr: int,
    *,
    work_sr: int = 22_050,
    hop_length: int = 512,
    repeat_similarity: float = 0.88,
) -> GlobalStructureResult:
    """
    Mix-level tempo / beats and structural segmentation from chroma (agglomerative).
    Repeated sections share ``repeat_group_id`` when mean-chroma cosine similarity is high.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    if sr != work_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=work_sr).astype(np.float32)
        sr = work_sr

    duration_s = float(len(y) / sr)
    if duration_s < 1e-3:
        return GlobalStructureResult(tempo_bpm=120.0, beat_times_s=[], segments=[])

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    bpm = float(np.asarray(tempo).reshape(-1)[0])
    if bpm <= 0 or not np.isfinite(bpm):
        bpm = 120.0
    beat_times_s = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length).tolist()

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    n_frames = int(chroma.shape[1])
    if n_frames < 2:
        return GlobalStructureResult(
            tempo_bpm=bpm,
            beat_times_s=beat_times_s,
            segments=[
                StructuralSegment(
                    start_time_s=0.0,
                    end_time_s=duration_s,
                    structure_label=0,
                    repeat_group_id=0,
                )
            ],
        )

    k = max(2, min(12, int(duration_s / 15.0) + 2))
    if n_frames < k * 2:
        k = max(2, min(k, max(2, n_frames // 4)))

    try:
        boundaries = librosa.segment.agglomerative(chroma, k)
    except Exception:
        boundaries = np.array([0], dtype=int)

    edges = _segment_edges_from_boundaries(boundaries, n_frames)
    if len(edges) < 2:
        edges = np.array([0, n_frames], dtype=int)

    group_ids = _repeat_group_ids(chroma, edges, repeat_similarity)

    segments: list[StructuralSegment] = []
    struct_i = 0
    for label_idx in range(len(edges) - 1):
        s_fr, e_fr = int(edges[label_idx]), int(edges[label_idx + 1])
        if e_fr <= s_fr:
            continue
        t0 = float(librosa.frames_to_time(s_fr, sr=sr, hop_length=hop_length))
        t1 = float(librosa.frames_to_time(e_fr, sr=sr, hop_length=hop_length))
        t1 = min(t1, duration_s)
        gid = group_ids[label_idx] if label_idx < len(group_ids) else None
        segments.append(
            StructuralSegment(
                start_time_s=t0,
                end_time_s=t1,
                structure_label=struct_i,
                repeat_group_id=gid,
            )
        )
        struct_i += 1

    if not segments:
        segments.append(
            StructuralSegment(
                start_time_s=0.0,
                end_time_s=duration_s,
                structure_label=0,
                repeat_group_id=0,
            )
        )

    return GlobalStructureResult(tempo_bpm=bpm, beat_times_s=beat_times_s, segments=segments)
