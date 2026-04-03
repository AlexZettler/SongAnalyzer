"""Hyperparameter search for NSynth family training (Optuna + optional SQLite cache)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from song_analyzer.instruments.mel import build_model
from song_analyzer.instruments.nsynth_train_loop import run_nsynth_split
from song_analyzer.instruments.nsynth_tune_fingerprint import (
    code_digest_for_instrument_sources,
    default_tune_cache_dir,
    fingerprint_payload,
    nsynth_study_name,
    study_suffix_from_payload,
)
from song_analyzer.instruments.train_nsynth import _import_tfds, train_nsynth_run


def _nsynth_dataset_version(tfds_module) -> str:
    b = tfds_module.builder("nsynth", config="full")
    return str(b.info.version)


def _sqlite_url(path: Path) -> str:
    return "sqlite:///" + path.resolve().as_posix()


def tune_nsynth_main(
    *,
    out: Path,
    device: str,
    n_trials: int,
    tune_cache_dir: Path | None,
    no_tune_cache: bool,
    tune_fresh: bool,
    max_val_steps: int,
    final_epochs: int,
    final_max_steps_per_epoch: int,
) -> None:
    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            'Tuning requires Optuna. Install with: pip install -e ".[train]"'
        ) from e

    _, tfds = _import_tfds()
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")

    code_digest = code_digest_for_instrument_sources()
    ds_ver = _nsynth_dataset_version(tfds)
    payload = fingerprint_payload(
        dataset_name="nsynth/full",
        dataset_version=ds_ver,
        code_digest=code_digest,
    )
    suffix = study_suffix_from_payload(payload)
    study_name = nsynth_study_name(suffix)

    cache_root = tune_cache_dir or default_tune_cache_dir()
    if not no_tune_cache:
        cache_root.mkdir(parents=True, exist_ok=True)
        db_path = cache_root / "nsynth_tune.db"
        storage = optuna.storages.RDBStorage(_sqlite_url(db_path))
        if tune_fresh:
            try:
                optuna.delete_study(study_name=study_name, storage=storage)
            except KeyError:
                pass
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=not tune_fresh,
        )
    else:
        study = optuna.create_study(direction="minimize")

    print(f"Optuna study: {study_name} (nsynth full v{ds_ver}, fingerprint suffix {suffix})")

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        trial_epochs = trial.suggest_int("epochs", 1, 5)
        max_train_steps = trial.suggest_int("max_steps_per_epoch", 100, 800)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 0.2, log=True)

        model = build_model(device_t)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        last_val = 0.0
        for epoch in range(trial_epochs):
            run_nsynth_split(
                tfds,
                split="train",
                batch_size=batch_size,
                max_steps=max_train_steps,
                device=device_t,
                model=model,
                optimizer=opt,
                train=True,
                shuffle_files=True,
                shuffle_buffer=10_000,
            )
            last_val, _ = run_nsynth_split(
                tfds,
                split="valid",
                batch_size=batch_size,
                max_steps=max_val_steps,
                device=device_t,
                model=model,
                optimizer=None,
                train=False,
                shuffle_files=False,
                shuffle_buffer=0,
            )
            trial.report(last_val, epoch)

        trial.set_user_attr(
            "hparams",
            {
                "lr": lr,
                "batch_size": batch_size,
                "epochs": trial_epochs,
                "max_steps_per_epoch": max_train_steps,
                "weight_decay": weight_decay,
            },
        )
        return last_val

    study.optimize(objective, n_trials=n_trials, show_progress_bar=sys.stderr.isatty())

    best = study.best_params
    print(f"best validation loss (last epoch): {study.best_value:.4f}")
    print(f"best params: {best}")

    train_nsynth_run(
        tfds,
        out=out,
        device=device_t,
        epochs=final_epochs,
        batch_size=int(best["batch_size"]),
        lr=float(best["lr"]),
        weight_decay=float(best["weight_decay"]),
        max_steps_per_epoch=final_max_steps_per_epoch,
        max_val_steps=None,
        save=True,
        verbose=True,
    )
