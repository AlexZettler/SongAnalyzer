"""Hyperparameter search for NSynth family training (Optuna + optional SQLite cache)."""

from __future__ import annotations

import logging
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
from song_analyzer.instruments.train_nsynth import (
    configure_train_logging,
    import_tfds_for_nsynth,
    parse_train_log_level,
    resolve_tfds_data_dir,
    train_nsynth_run,
)

logger = logging.getLogger(__name__)


def _nsynth_dataset_version(tfds_module, *, data_dir: str | None = None) -> str:
    kw: dict = {}
    if data_dir is not None:
        kw["data_dir"] = data_dir
    b = tfds_module.builder("nsynth", config="full", **kw)
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
    tfds_data_dir: Path | None = None,
    log_level: str = "INFO",
    log_file: Path | None = None,
    no_log_file: bool = False,
) -> None:
    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            'Tuning requires Optuna. Install with: pip install -e ".[train]"'
        ) from e

    configure_train_logging(
        parse_train_log_level(log_level),
        log_file=log_file,
        no_log_file=no_log_file,
    )
    logger.info(
        "starting Optuna tuning n_trials=%s device=%s cuda_available=%s no_tune_cache=%s "
        "tune_fresh=%s max_val_steps=%s final_epochs=%s final_max_steps_per_epoch=%s tfds_data_dir=%s",
        n_trials,
        device,
        torch.cuda.is_available(),
        no_tune_cache,
        tune_fresh,
        max_val_steps,
        final_epochs,
        final_max_steps_per_epoch,
        resolve_tfds_data_dir(tfds_data_dir)
        if tfds_data_dir is not None
        else "(default ~/tensorflow_datasets or TFDS_DATA_DIR)",
    )
    _, tfds = import_tfds_for_nsynth()
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    data_dir = resolve_tfds_data_dir(tfds_data_dir)

    code_digest = code_digest_for_instrument_sources()
    ds_ver = _nsynth_dataset_version(tfds, data_dir=data_dir)
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

    logger.info(
        "Optuna study: %s (nsynth full v%s, fingerprint suffix %s)",
        study_name,
        ds_ver,
        suffix,
    )

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        trial_epochs = trial.suggest_int("epochs", 1, 5)
        max_train_steps = trial.suggest_int("max_steps_per_epoch", 100, 800)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 0.2, log=True)

        debug_batches = logger.isEnabledFor(logging.DEBUG)
        prog_train = max(1, max_train_steps // 10) if debug_batches else None
        prog_val = max(1, max_val_steps // 10) if debug_batches else None

        logger.info(
            "trial %s hparams lr=%s batch_size=%s trial_epochs=%s max_steps_per_epoch=%s weight_decay=%s",
            trial.number,
            lr,
            batch_size,
            trial_epochs,
            max_train_steps,
            weight_decay,
        )

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
                data_dir=data_dir,
                progress_log_interval=prog_train,
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
                data_dir=data_dir,
                progress_log_interval=prog_val,
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
    logger.info("best validation loss (last epoch): %.4f", study.best_value)
    logger.info("best params: %s", best)

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
        tfds_data_dir=data_dir,
    )
