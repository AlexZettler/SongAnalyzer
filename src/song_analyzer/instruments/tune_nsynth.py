"""Hyperparameter search for NSynth family training (Optuna + optional SQLite cache)."""

from __future__ import annotations

import logging
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def _sanitize_exploration_id(raw: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw.strip())
    return s[:120] or "explore"


@dataclass(frozen=True)
class HpoJobResult:
    exploration_id: str
    study_name: str
    best_params: dict[str, Any]
    best_value: float
    archived_db_path: Path | None
    trials_export_csv: Path | None
    out_checkpoint: Path | None


def _archive_tune_database(db_path: Path, exploration_id: str, archive_root: Path) -> Path:
    archive_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = archive_root / f"{_sanitize_exploration_id(exploration_id)}_{ts}.db"
    shutil.copy2(db_path, dest)
    return dest


def _export_trials_csv(study: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = study.trials_dataframe()
    df.to_csv(path, index=False)
    return path


def run_nsynth_hpo_job(
    *,
    out: Path | None,
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
    exploration_id: str | None = None,
    archive_tune_db_before: bool = False,
    skip_final_train: bool = False,
) -> HpoJobResult:
    """
    Run Optuna HPO; optionally archive SQLite DB, export trials CSV, and skip final full train.

    When ``exploration_id`` is set, the Optuna study name includes it so prior studies in the
    same RDB file remain available for ``train_nsynth_from_study``.
    """
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
    expl = exploration_id or ""
    logger.info(
        "starting Optuna tuning n_trials=%s device=%s cuda_available=%s no_tune_cache=%s "
        "tune_fresh=%s max_val_steps=%s final_epochs=%s final_max_steps_per_epoch=%s "
        "tfds_data_dir=%s exploration_id=%s skip_final_train=%s",
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
        expl or "(default study name)",
        skip_final_train,
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
    base_study = nsynth_study_name(suffix)
    study_name = (
        f"{base_study}__{_sanitize_exploration_id(exploration_id)}"
        if exploration_id
        else base_study
    )

    cache_root = tune_cache_dir or default_tune_cache_dir()
    archived: Path | None = None
    export_csv: Path | None = None
    db_path: Path | None = None

    if not no_tune_cache:
        cache_root.mkdir(parents=True, exist_ok=True)
        db_path = cache_root / "nsynth_tune.db"
        if archive_tune_db_before and exploration_id and db_path.is_file():
            arch_dir = cache_root / "tune_archive"
            archived = _archive_tune_database(db_path, exploration_id, arch_dir)
            logger.info("archived tune DB to %s", archived)

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

    best = dict(study.best_params)
    best_val = float(study.best_value)
    logger.info("best validation loss (last epoch): %.4f", best_val)
    logger.info("best params: %s", best)

    key = _sanitize_exploration_id(exploration_id) if exploration_id else "default"
    if not no_tune_cache:
        export_csv = _export_trials_csv(
            study, cache_root / "exports" / f"{key}_trials.csv"
        )
        logger.info("exported trials to %s", export_csv)

    out_checkpoint: Path | None = None
    if not skip_final_train:
        if out is None:
            raise ValueError("final training requires out= when skip_final_train is False")
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
        out_checkpoint = out

    return HpoJobResult(
        exploration_id=expl or "default",
        study_name=study_name,
        best_params=best,
        best_value=best_val,
        archived_db_path=archived,
        trials_export_csv=export_csv,
        out_checkpoint=out_checkpoint,
    )


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
    """CLI path: tune then train with best hyperparameters (unchanged behavior)."""
    run_nsynth_hpo_job(
        out=out,
        device=device,
        n_trials=n_trials,
        tune_cache_dir=tune_cache_dir,
        no_tune_cache=no_tune_cache,
        tune_fresh=tune_fresh,
        max_val_steps=max_val_steps,
        final_epochs=final_epochs,
        final_max_steps_per_epoch=final_max_steps_per_epoch,
        tfds_data_dir=tfds_data_dir,
        log_level=log_level,
        log_file=log_file,
        no_log_file=no_log_file,
        exploration_id=None,
        archive_tune_db_before=False,
        skip_final_train=False,
    )
