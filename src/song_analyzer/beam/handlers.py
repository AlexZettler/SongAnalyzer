"""Pub/Sub message bodies: decode JSON, run workflow, encode completion JSON."""

from __future__ import annotations

from pathlib import Path

from song_analyzer.corpus.song_request import process_song_request
from song_analyzer.messaging.payloads import (
    HpoCompletePayload,
    HpoRequestPayload,
    SongRequestPayload,
    TrainCompletePayload,
    TrainRequestPayload,
)


def handle_song_bytes(data: bytes) -> bytes:
    req = SongRequestPayload.model_validate_json(data)
    done = process_song_request(req)
    return done.model_dump_json().encode("utf-8")


def handle_hpo_bytes(data: bytes) -> bytes:
    from song_analyzer.instruments.tune_nsynth import run_nsynth_hpo_job

    req = HpoRequestPayload.model_validate_json(data)
    try:
        cache = Path(req.tune_cache_dir) if req.tune_cache_dir else None
        tfds = Path(req.tfds_data_dir) if req.tfds_data_dir else None
        out_pt = Path(req.out_checkpoint) if req.out_checkpoint else None
        result = run_nsynth_hpo_job(
            out=out_pt,
            device=req.device,
            n_trials=req.n_trials,
            tune_cache_dir=cache,
            no_tune_cache=req.no_tune_cache,
            tune_fresh=req.tune_fresh,
            max_val_steps=req.max_val_steps,
            final_epochs=req.final_epochs,
            final_max_steps_per_epoch=req.final_max_steps_per_epoch,
            tfds_data_dir=tfds,
            log_level=req.log_level,
            exploration_id=req.exploration_id,
            archive_tune_db_before=req.archive_tune_db_before,
            skip_final_train=req.skip_final_train,
        )
        return HpoCompletePayload(
            exploration_id=req.exploration_id,
            status="ok",
            study_name=result.study_name,
            best_params=result.best_params,
            best_value=result.best_value,
            archived_db_path=str(result.archived_db_path)
            if result.archived_db_path
            else None,
            trials_csv_path=str(result.trials_export_csv)
            if result.trials_export_csv
            else None,
            out_checkpoint=str(result.out_checkpoint)
            if result.out_checkpoint
            else None,
        ).model_dump_json().encode("utf-8")
    except Exception as e:  # noqa: BLE001
        return HpoCompletePayload(
            exploration_id=req.exploration_id,
            status="error",
            error=str(e),
        ).model_dump_json().encode("utf-8")


def handle_train_bytes(data: bytes) -> bytes:
    from song_analyzer.instruments.train_from_study import train_nsynth_from_study

    req = TrainRequestPayload.model_validate_json(data)
    try:
        cache = Path(req.tune_cache_dir) if req.tune_cache_dir else None
        tfds = Path(req.tfds_data_dir) if req.tfds_data_dir else None
        best = train_nsynth_from_study(
            study_name=req.study_name,
            out=Path(req.out),
            tune_cache_dir=cache,
            epochs=req.epochs,
            max_steps_per_epoch=req.max_steps_per_epoch,
            max_val_steps=req.max_val_steps,
            device=req.device,
            tfds_data_dir=tfds,
            log_level=req.log_level,
        )
        return TrainCompletePayload(
            train_job_id=req.train_job_id,
            status="ok",
            checkpoint_path=req.out,
            best_params_used=best,
        ).model_dump_json().encode("utf-8")
    except Exception as e:  # noqa: BLE001
        return TrainCompletePayload(
            train_job_id=req.train_job_id,
            status="error",
            error=str(e),
        ).model_dump_json().encode("utf-8")
