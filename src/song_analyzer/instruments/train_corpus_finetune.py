"""Fine-tune FamilyClassifier from pseudo-labeled corpus manifest (optional NSynth mix)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from song_analyzer.corpus.store import read_manifest_csv
from song_analyzer.instruments.mel import SAMPLE_RATE, build_model, waveform_to_log_mel
from song_analyzer.instruments.train_nsynth import import_tfds_for_nsynth, resolve_tfds_data_dir

logger = logging.getLogger(__name__)


class CorpusFamilyDataset(Dataset[tuple[np.ndarray, int]]):
    """Random fixed-length crops from pseudo-labeled stem WAVs (16 kHz mono)."""

    def __init__(
        self,
        manifest_csv: Path,
        corpus_root: Path,
        *,
        crop_samples: int,
        seed: int = 0,
    ) -> None:
        self._rows = read_manifest_csv(manifest_csv)
        self._root = corpus_root.resolve()
        self._crop = crop_samples
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        from song_analyzer.audio_io import load_audio

        row = self._rows[i]
        p = Path(row.audio_relpath)
        if not p.is_absolute():
            p = (self._root / p).resolve()
        x, sr = load_audio(p, target_sr=SAMPLE_RATE, mono=True)
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"expected {SAMPLE_RATE} Hz manifest audio, got {sr} for {p}")
        n = self._crop
        if len(x) > n:
            start = int(self._rng.integers(0, len(x) - n + 1))
            x = x[start : start + n].copy()
        elif len(x) < n:
            x = np.pad(x.astype(np.float32, copy=False), (0, n - len(x)))
        return x.astype(np.float32, copy=False), row.family_id


def _collate_identity(
    batch: list[tuple[np.ndarray, int]],
) -> list[tuple[np.ndarray, int]]:
    return batch


def run_corpus_training_steps(
    loader: DataLoader[tuple[np.ndarray, int]],
    *,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    train: bool,
    phase: str,
) -> tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    step = 0
    with ctx:
        for batch in loader:
            if step >= max_steps:
                break
            if train:
                optimizer.zero_grad()
            loss_mean = 0.0
            bsz = len(batch)
            for wave, y_int in batch:
                mel = waveform_to_log_mel(wave, device)
                logits = model(mel)
                target = torch.tensor([y_int], device=device, dtype=torch.long)
                loss = F.cross_entropy(logits, target)
                if train:
                    loss.backward()
                loss_mean += float(loss.detach().cpu())
                pred = int(logits.argmax(dim=-1).item())
                if pred == int(y_int):
                    correct += 1
                total += 1
            if train:
                optimizer.step()
            losses.append(loss_mean / max(bsz, 1))
            step += 1
    if not losses:
        return 0.0, 0.0
    return float(np.mean(losses)), correct / max(total, 1)


def train_corpus_finetune_run(
    *,
    manifest_csv: Path,
    corpus_root: Path,
    init_checkpoint: Path,
    out: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    crop_seconds: float,
    corpus_steps_per_epoch: int,
    nsynth_steps_per_epoch: int,
    nsynth_batch_size: int,
    tfds_data_dir: str | Path | None,
    max_val_steps: int | None,
    seed: int,
) -> torch.nn.Module:
    from song_analyzer.instruments.nsynth_train_loop import run_nsynth_split

    crop_samples = max(int(crop_seconds * SAMPLE_RATE), int(0.5 * SAMPLE_RATE))
    ds = CorpusFamilyDataset(
        manifest_csv,
        corpus_root,
        crop_samples=crop_samples,
        seed=seed,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_identity,
        drop_last=False,
        num_workers=0,
    )

    model = build_model(device)
    state = torch.load(init_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    tfds = None
    data_dir = None
    if nsynth_steps_per_epoch > 0:
        _, tfds = import_tfds_for_nsynth()
        data_dir = resolve_tfds_data_dir(tfds_data_dir)

    for epoch in range(epochs):
        if nsynth_steps_per_epoch > 0 and tfds is not None:
            tr_loss, tr_acc = run_nsynth_split(
                tfds,
                split="train",
                batch_size=nsynth_batch_size,
                max_steps=nsynth_steps_per_epoch,
                device=device,
                model=model,
                optimizer=opt,
                train=True,
                shuffle_files=True,
                shuffle_buffer=10_000,
                data_dir=data_dir,
            )
            logger.info(
                "epoch %s nsynth mix: loss %.4f acc %.4f",
                epoch + 1,
                tr_loss,
                tr_acc,
            )

        c_loss, c_acc = run_corpus_training_steps(
            loader,
            device=device,
            model=model,
            optimizer=opt,
            max_steps=corpus_steps_per_epoch,
            train=True,
            phase="corpus",
        )
        logger.info(
            "epoch %s corpus: loss %.4f acc %.4f",
            epoch + 1,
            c_loss,
            c_acc,
        )

        if max_val_steps is not None and max_val_steps > 0 and tfds is not None:
            va_loss, va_acc = run_nsynth_split(
                tfds,
                split="valid",
                batch_size=nsynth_batch_size,
                max_steps=max_val_steps,
                device=device,
                model=model,
                optimizer=None,
                train=False,
                shuffle_files=False,
                shuffle_buffer=0,
                data_dir=data_dir,
            )
            logger.info(
                "epoch %s nsynth valid: loss %.4f acc %.4f",
                epoch + 1,
                va_loss,
                va_acc,
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    logger.info("saved %s", out)
    return model


def train_corpus_main(argv: list[str] | None = None) -> None:
    from song_analyzer.instruments.nsynth_logging import (
        configure_nsynth_logging,
        parse_train_log_level,
    )

    p = argparse.ArgumentParser(description="Fine-tune FamilyClassifier from corpus manifest")
    p.add_argument("--manifest-csv", type=Path, required=True)
    p.add_argument("--corpus-root", type=Path, required=True)
    p.add_argument("--init-checkpoint", type=Path, required=True, help="NSynth-trained .pt")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--crop-seconds", type=float, default=1.6)
    p.add_argument(
        "--corpus-steps-per-epoch",
        type=int,
        default=200,
        help="Max DataLoader batches from manifest per epoch",
    )
    p.add_argument(
        "--nsynth-steps-per-epoch",
        type=int,
        default=0,
        help="If >0, mix NSynth train batches each epoch (requires [train] extra)",
    )
    p.add_argument("--nsynth-batch-size", type=int, default=32)
    p.add_argument("--max-val-steps", type=int, default=None)
    p.add_argument("--tfds-data-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    args = p.parse_args(argv)

    configure_nsynth_logging(parse_train_log_level(args.log_level), profile="train")

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_corpus_finetune_run(
        manifest_csv=args.manifest_csv,
        corpus_root=args.corpus_root,
        init_checkpoint=args.init_checkpoint,
        out=args.out,
        device=dev,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        crop_seconds=args.crop_seconds,
        corpus_steps_per_epoch=args.corpus_steps_per_epoch,
        nsynth_steps_per_epoch=args.nsynth_steps_per_epoch,
        nsynth_batch_size=args.nsynth_batch_size,
        tfds_data_dir=args.tfds_data_dir,
        max_val_steps=args.max_val_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        train_corpus_main()
    except ImportError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
