"""Train NSynth classifier using best trial from a persisted Optuna study."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from song_analyzer.instruments.nsynth_tune_fingerprint import default_tune_cache_dir
from song_analyzer.instruments.train_nsynth import (
    configure_train_logging,
    import_tfds_for_nsynth,
    parse_train_log_level,
    resolve_tfds_data_dir,
    train_nsynth_run,
)

logger = logging.getLogger(__name__)


def _sqlite_url(path: Path) -> str:
    return "sqlite:///" + path.resolve().as_posix()


def train_nsynth_from_study(
    *,
    study_name: str,
    out: Path,
    tune_cache_dir: Path | None = None,
    epochs: int = 3,
    max_steps_per_epoch: int = 500,
    max_val_steps: int | None = None,
    device: str = "cuda",
    tfds_data_dir: Path | None = None,
    log_level: str = "INFO",
    log_file: Path | None = None,
    no_log_file: bool = False,
) -> dict[str, Any]:
    """
    Load ``study_name`` from ``<tune_cache_dir>/nsynth_tune.db`` and run full training
    with best-trial hyperparameters (lr, batch_size, weight_decay).
    """
    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            'Loading studies requires Optuna. Install with: pip install -e ".[train]"'
        ) from e

    configure_train_logging(
        parse_train_log_level(log_level),
        log_file=log_file,
        no_log_file=no_log_file,
    )
    cache_root = tune_cache_dir or default_tune_cache_dir()
    db_path = cache_root / "nsynth_tune.db"
    if not db_path.is_file():
        raise FileNotFoundError(f"Optuna database not found: {db_path}")

    storage = optuna.storages.RDBStorage(_sqlite_url(db_path))
    study = optuna.load_study(study_name=study_name, storage=storage)
    best = study.best_params
    logger.info("loaded study %s best_value=%s params=%s", study_name, study.best_value, best)

    _, tfds = import_tfds_for_nsynth()
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    data_dir = resolve_tfds_data_dir(tfds_data_dir)

    train_nsynth_run(
        tfds,
        out=out,
        device=device_t,
        epochs=epochs,
        batch_size=int(best["batch_size"]),
        lr=float(best["lr"]),
        weight_decay=float(best["weight_decay"]),
        max_steps_per_epoch=max_steps_per_epoch,
        max_val_steps=max_val_steps,
        save=True,
        verbose=True,
        tfds_data_dir=data_dir,
    )
    return dict(best)
