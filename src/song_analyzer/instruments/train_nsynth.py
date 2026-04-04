"""Train family classifier on NSynth via TensorFlow Datasets (optional [train] extra)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

from song_analyzer.instruments.mel import build_model
from song_analyzer.instruments.nsynth_logging import (
    configure_nsynth_logging,
    parse_train_log_level,
)
from song_analyzer.instruments.nsynth_train_loop import run_nsynth_split

logger = logging.getLogger(__name__)


def configure_train_logging(
    level: int = logging.INFO,
    *,
    log_file: Path | None = None,
    no_log_file: bool = False,
) -> None:
    """Configure root logging for training CLIs (colored console + optional file)."""
    configure_nsynth_logging(
        level,
        profile="train",
        log_file=log_file,
        no_log_file=no_log_file,
    )


def import_tfds_for_nsynth():
    if sys.version_info >= (3, 14):
        raise ImportError(
            "NSynth training needs TensorFlow and tensorflow-datasets, which are not "
            "available for Python 3.14+ on PyPI yet. Use Python 3.13 in a venv, "
            'then: pip install -e ".[train]"'
        )
    logger.info("loading TensorFlow and tensorflow-datasets (may take a moment)")
    try:
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        import tensorflow_datasets as tfds
    except ImportError as e:
        raise ImportError(
            'Training requires TensorFlow. Install with: pip install -e ".[train]" '
            "(needs a Python version TensorFlow publishes wheels for; this project targets 3.13)."
        ) from e
    try:
        import importlib_resources  # noqa: F401 — tfds.load() imports this lazily
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "tensorflow-datasets requires the PyPI package 'importlib-resources' "
            "(Python import name: importlib_resources). "
            "Install with: pip install 'importlib-resources>=6' "
            'or reinstall the project: pip install -e ".[train]"'
        ) from e
    try:
        import apache_beam  # noqa: F401 — NSynth (tfds) imports beam at dataset load
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "NSynth via tensorflow-datasets requires apache-beam. "
            'Install with: pip install "apache-beam>=2.54" '
            'or reinstall: pip install -e ".[train]"'
        ) from e
    try:
        import dill  # noqa: F401 — from apache-beam[dill]; used when Beam pickler is dill (NSynth prepare)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "NSynth preparation needs the 'dill' package (install apache-beam with the dill extra). "
            'Reinstall with: pip install -e ".[train]" '
            '(ensure dependency is apache-beam[dill], not apache-beam alone).'
        ) from e
    return tf, tfds


def resolve_tfds_data_dir(explicit: str | os.PathLike[str] | None = None) -> str | None:
    """TFDS cache: explicit arg wins, else ``TFDS_DATA_DIR`` env (TensorFlow Datasets default)."""
    if explicit is not None:
        return os.fspath(explicit)
    return os.environ.get("TFDS_DATA_DIR")


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
    tfds_data_dir: str | os.PathLike[str] | None = None,
) -> torch.nn.Module:
    """Train on nsynth/full train split; optional val metrics each epoch. Saves state_dict if out and save."""
    data_dir = resolve_tfds_data_dir(tfds_data_dir)
    cuda_ok = torch.cuda.is_available()
    logger.info(
        "starting training device=%s cuda_available=%s epochs=%s batch_size=%s lr=%s "
        "weight_decay=%s max_steps_per_epoch=%s max_val_steps=%s tfds_data_dir=%s",
        device,
        cuda_ok,
        epochs,
        batch_size,
        lr,
        weight_decay,
        max_steps_per_epoch,
        max_val_steps,
        data_dir if data_dir is not None else "(default ~/tensorflow_datasets or TFDS_DATA_DIR)",
    )
    progress_train = max(1, max_steps_per_epoch // 10) if verbose else None
    progress_val = (
        max(1, max_val_steps // 10) if verbose and max_val_steps is not None else None
    )
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
            data_dir=data_dir,
            progress_log_interval=progress_train,
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
                data_dir=data_dir,
                progress_log_interval=progress_val,
            )
        if verbose:
            msg = f"epoch {epoch + 1} train loss {tr_loss:.4f} acc {tr_acc:.4f}"
            if va_loss is not None:
                msg += f" | val loss {va_loss:.4f} acc {va_acc:.4f}"
            logger.info(msg)

    if save and out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out)
        if verbose:
            logger.info("saved %s", out)
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
    p.add_argument(
        "--tfds-data-dir",
        type=Path,
        default=None,
        help="TensorFlow Datasets root (default: TFDS_DATA_DIR env or ~/tensorflow_datasets)",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING"],
        default="INFO",
        help="Python logging level for training messages",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Append logs to this file (default: platform log dir / nsynth_train.log)",
    )
    p.add_argument(
        "--no-log-file",
        action="store_true",
        help="Do not write a log file (console only)",
    )
    args = p.parse_args(argv)

    configure_train_logging(
        parse_train_log_level(args.log_level),
        log_file=args.log_file,
        no_log_file=args.no_log_file,
    )
    _, tfds = import_tfds_for_nsynth()

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
        tfds_data_dir=args.tfds_data_dir,
    )


if __name__ == "__main__":
    train_main()
