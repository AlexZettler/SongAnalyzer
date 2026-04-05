from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
import numpy as np

from song_analyzer.audio_io import load_audio, save_wav
from song_analyzer.corpus.layout import PSEUDO_STEMS_SUBDIR, CorpusLayout
from song_analyzer.corpus.store import TrackStore, write_manifest_csv
from song_analyzer.corpus.types import ManifestRow, utc_now_iso
from song_analyzer.instruments.constants import NSYNTH_FAMILIES
from song_analyzer.instruments.mel import SAMPLE_RATE
from song_analyzer.separation.demucs_sep import separate_to_dict

logger = logging.getLogger(__name__)


def _resample_mono(wave: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wave.astype(np.float32, copy=False)
    import librosa

    return librosa.resample(wave.astype(np.float32, copy=False), orig_sr=sr, target_sr=target_sr)


def build_training_manifest(
    corpus_root: str | Path,
    *,
    out_csv: Path | None,
    demucs_model: str = "htdemucs",
    device: str = "cpu",
    nsynth_checkpoint: Path | None = None,
    min_confidence: float = 0.0,
    demucs_shifts: int = 0,
    demucs_segment: float | None = None,
    progress: bool = False,
    skip_drums: bool = False,
    separate_fn: Callable[..., tuple[dict[str, np.ndarray], int]] | None = None,
    predict_fn: Callable[..., tuple[str, float, dict[str, float]]] | None = None,
    store_sqlite: bool = True,
) -> list[ManifestRow]:
    """
    For each track with audio: Demucs separation, teacher pseudo-labels per stem, save 16 kHz
    stem WAVs under ``pseudo_stems/``, optional CSV and SQLite ``training_manifest_rows``.
    """
    layout = CorpusLayout(Path(corpus_root).resolve())
    store = TrackStore.open(layout.root)
    sep = separate_fn or separate_to_dict

    if predict_fn is None:
        from song_analyzer.instruments.infer import load_classifier, predict_stem_family

        teacher, _warn = load_classifier(nsynth_checkpoint, device)
        if teacher is None:
            logger.warning(
                "No teacher checkpoint: pseudo-labels will be low-quality uniform guesses. "
                "Set --nsynth-checkpoint or SONGANALYZER_NSYNTH_CHECKPOINT."
            )

        def _pred(audio: np.ndarray, dev: str) -> tuple[str, float, dict[str, float]]:
            return predict_stem_family(audio, teacher, dev)

        predict_fn = _pred

    rows_out: list[ManifestRow] = []
    built_at = utc_now_iso()

    try:
        if store_sqlite:
            store.clear_manifest()

        for rec in store.iter_tracks_with_audio():
            mix_path = store.resolve_audio_path(rec.audio_relpath or "")
            if not mix_path.is_file():
                logger.warning("missing audio for track %s: %s", rec.track_id, mix_path)
                continue
            try:
                wave, sr = load_audio(mix_path, target_sr=None, mono=True)
            except Exception as e:
                logger.warning("failed to load %s: %s", mix_path, e)
                continue

            stems, stem_sr = sep(
                wave,
                sr,
                model_name=demucs_model,
                device=device,
                shifts=demucs_shifts,
                segment=demucs_segment,
                progress=progress,
            )

            for stem_name, stem_audio in stems.items():
                if skip_drums and stem_name == "drums":
                    continue
                mono = stem_audio.astype(np.float32, copy=False)
                mono_16k = _resample_mono(mono, stem_sr, SAMPLE_RATE)
                fam, conf, _logits = predict_fn(mono_16k, device)
                if conf < min_confidence:
                    continue
                if fam not in NSYNTH_FAMILIES:
                    continue
                family_id = NSYNTH_FAMILIES.index(fam)

                safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem_name)
                rel_name = f"{PSEUDO_STEMS_SUBDIR}/{rec.track_id}_{safe_stem}.wav"
                out_wav = layout.root / rel_name
                save_wav(out_wav, mono_16k, SAMPLE_RATE)

                rows_out.append(
                    ManifestRow(
                        track_id=rec.track_id,
                        stem_name=stem_name,
                        audio_relpath=rel_name,
                        family_id=family_id,
                        family_name=fam,
                        confidence=float(conf),
                    )
                )

        if out_csv is not None:
            write_manifest_csv(out_csv, rows_out)

        if store_sqlite:
            ck = str(nsynth_checkpoint) if nsynth_checkpoint else None
            store.insert_manifest_rows(
                rows_out,
                demucs_model=demucs_model,
                teacher_checkpoint=ck,
                built_at=built_at,
            )
    finally:
        store.close()

    return rows_out


def iter_manifest_rows_from_csv(
    manifest_csv: Path,
    corpus_root: str | Path | None,
) -> Iterator[tuple[Path, int, str]]:
    """Yield (absolute_wav_path, family_id, track_id) for training."""
    from song_analyzer.corpus.store import read_manifest_csv

    root = Path(corpus_root).resolve() if corpus_root else None
    for row in read_manifest_csv(manifest_csv):
        p = Path(row.audio_relpath)
        if not p.is_absolute():
            if root is None:
                raise ValueError("corpus_root required when manifest paths are relative")
            p = (root / p).resolve()
        yield p, row.family_id, row.track_id
