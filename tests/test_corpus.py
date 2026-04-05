from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from song_analyzer.corpus.db import init_corpus
from song_analyzer.corpus.manifest import build_training_manifest
from song_analyzer.corpus.store import TrackStore, read_manifest_csv, write_manifest_csv
from song_analyzer.corpus.types import ManifestRow, TrackRecord, utc_now_iso
from song_analyzer.instruments.mel import SAMPLE_RATE, build_model
from song_analyzer.instruments.train_corpus_finetune import train_corpus_finetune_run


def test_init_corpus_creates_db_and_dirs(tmp_path: Path) -> None:
    layout = init_corpus(tmp_path)
    assert layout.db_path.is_file()
    assert layout.audio_dir.is_dir()
    assert layout.pseudo_stems_dir.is_dir()


def test_track_roundtrip(tmp_path: Path) -> None:
    init_corpus(tmp_path)
    store = TrackStore.open(tmp_path)
    try:
        rec = TrackRecord(
            track_id="t1",
            created_at=utc_now_iso(),
            title="A",
            artist="B",
            audio_relpath="audio/t1.wav",
            duration_seconds=1.0,
            file_checksum="abc",
        )
        store.insert_track(rec)
        got = store.get_track("t1")
        assert got is not None
        assert got.title == "A"
        assert got.artist == "B"
    finally:
        store.close()


def test_manifest_csv_roundtrip(tmp_path: Path) -> None:
    rows = [
        ManifestRow(
            track_id="t1",
            stem_name="vocals",
            audio_relpath="pseudo_stems/t1_vocals.wav",
            family_id=10,
            family_name="vocal",
            confidence=0.9,
        )
    ]
    p = tmp_path / "m.csv"
    write_manifest_csv(p, rows)
    back = read_manifest_csv(p)
    assert len(back) == 1
    assert back[0].track_id == "t1"
    assert back[0].family_id == 10
    assert back[0].audio_relpath == "pseudo_stems/t1_vocals.wav"


def test_build_training_manifest_mocked(tmp_path: Path) -> None:
    init_corpus(tmp_path)
    wav = tmp_path / "in.wav"
    sf.write(wav, np.zeros(8000, dtype=np.float32), 8000)

    store = TrackStore.open(tmp_path)
    try:
        store.insert_track(
            TrackRecord(
                track_id="trk",
                created_at=utc_now_iso(),
                audio_relpath=str(wav),
                duration_seconds=1.0,
                file_checksum="x",
            )
        )
    finally:
        store.close()

    def fake_sep(
        mix_mono: np.ndarray,
        sr: int,
        **kwargs: object,
    ) -> tuple[dict[str, np.ndarray], int]:
        return {"vocals": np.zeros(SAMPLE_RATE, dtype=np.float32)}, SAMPLE_RATE

    def fake_pred(
        audio: np.ndarray,
        dev: str,
    ) -> tuple[str, float, dict[str, float]]:
        return "vocal", 0.95, {}

    csv_path = tmp_path / "manifest.csv"
    rows = build_training_manifest(
        tmp_path,
        out_csv=csv_path,
        demucs_model="htdemucs",
        device="cpu",
        nsynth_checkpoint=None,
        separate_fn=fake_sep,
        predict_fn=fake_pred,
        store_sqlite=True,
    )
    assert len(rows) == 1
    assert rows[0].family_name == "vocal"
    assert csv_path.is_file()
    stem_file = tmp_path / rows[0].audio_relpath
    assert stem_file.is_file()

    store = TrackStore.open(tmp_path)
    try:
        cur = store._conn.execute("SELECT COUNT(*) AS c FROM training_manifest_rows")
        assert int(cur.fetchone()["c"]) == 1
    finally:
        store.close()


def test_train_corpus_finetune_smoke(tmp_path: Path) -> None:
    root = tmp_path / "corp"
    init_corpus(root)
    stem_dir = root / "pseudo_stems"
    stem_dir.mkdir(parents=True, exist_ok=True)
    stem_path = stem_dir / "x_v.wav"
    sf.write(stem_path, np.random.randn(SAMPLE_RATE * 2).astype(np.float32) * 0.01, SAMPLE_RATE)

    manifest = tmp_path / "m.csv"
    write_manifest_csv(
        manifest,
        [
            ManifestRow(
                track_id="x",
                stem_name="vocals",
                audio_relpath="pseudo_stems/x_v.wav",
                family_id=0,
                family_name="bass",
                confidence=0.5,
            )
        ],
    )

    ckpt = tmp_path / "init.pt"
    m = build_model(torch.device("cpu"))
    torch.save(m.state_dict(), ckpt)
    out = tmp_path / "out.pt"

    train_corpus_finetune_run(
        manifest_csv=manifest,
        corpus_root=root,
        init_checkpoint=ckpt,
        out=out,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        lr=1e-4,
        weight_decay=0.01,
        crop_seconds=0.5,
        corpus_steps_per_epoch=2,
        nsynth_steps_per_epoch=0,
        nsynth_batch_size=8,
        tfds_data_dir=None,
        max_val_steps=None,
        seed=42,
    )
    assert out.is_file()


def test_training_manifest_rows_foreign_key(tmp_path: Path) -> None:
    init_corpus(tmp_path)
    store = TrackStore.open(tmp_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            store.insert_manifest_rows(
                [
                    ManifestRow(
                        track_id="missing",
                        stem_name="a",
                        audio_relpath="p.wav",
                        family_id=0,
                        family_name="bass",
                        confidence=1.0,
                    )
                ],
                demucs_model="m",
                teacher_checkpoint=None,
                built_at=utc_now_iso(),
            )
    finally:
        store.close()
