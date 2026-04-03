"""Train family classifier on NSynth via TensorFlow Datasets (optional [train] extra)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from song_analyzer.instruments.mel import build_model
from song_analyzer.instruments.nsynth_train_loop import run_nsynth_split


def _import_tfds():
    if sys.version_info >= (3, 14):
        raise ImportError(
            "NSynth training needs TensorFlow and tensorflow-datasets, which are not "
            "available for Python 3.14+ on PyPI yet. Use Python 3.13 in a venv, "
            'then: pip install -e ".[train]"'
        )
    try:
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        import tensorflow_datasets as tfds
    except ImportError as e:
        raise ImportError(
            'Training requires TensorFlow. Install with: pip install -e ".[train]" '
            "(needs a Python version TensorFlow publishes wheels for; this project targets 3.13)."
        ) from e
    return tf, tfds


def train_nsynth_run(
    tfds,
    *,
    out: Path | None,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.01,
    max_steps_per_epoch: int = 500,
    max_val_steps: int | None = None,
    model: torch.nn.Module | None = None,
    save: bool = True,
    verbose: bool = True,
) -> torch.nn.Module:
    """Train on nsynth/full train split; optional val metrics each epoch. Saves state_dict if out and save."""
    model = model or build_model(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        tr_loss, tr_acc = run_nsynth_split(
            tfds,
            split="train",
            batch_size=batch_size,
            max_steps=max_steps_per_epoch,
            device=device,
            model=model,
            optimizer=opt,
            train=True,
            shuffle_files=True,
            shuffle_buffer=10_000,
        )
        va_loss: float | None = None
        va_acc: float | None = None
        if max_val_steps is not None:
            va_loss, va_acc = run_nsynth_split(
                tfds,
                split="valid",
                batch_size=batch_size,
                max_steps=max_val_steps,
                device=device,
                model=model,
                optimizer=None,
                train=False,
                shuffle_files=False,
                shuffle_buffer=0,
            )
        if verbose:
            msg = f"epoch {epoch + 1} train loss {tr_loss:.4f} acc {tr_acc:.4f}"
            if va_loss is not None:
                msg += f" | val loss {va_loss:.4f} acc {va_acc:.4f}"
            print(msg)

    if save and out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out)
        if verbose:
            print(f"saved {out}")
    return model


def train_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train NSynth instrument-family classifier")
    p.add_argument("--out", type=Path, required=True, help="Output .pt state_dict path")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-steps-per-epoch", type=int, default=500, help="Cap steps for quick runs")
    p.add_argument(
        "--max-val-steps",
        type=int,
        default=None,
        help="If set, run validation on nsynth valid split each epoch (cap steps)",
    )
    args = p.parse_args(argv)

    _, tfds = _import_tfds()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_nsynth_run(
        tfds,
        out=args.out,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_steps_per_epoch=args.max_steps_per_epoch,
        max_val_steps=args.max_val_steps,
        save=True,
        verbose=True,
    )


if __name__ == "__main__":
    train_main()
