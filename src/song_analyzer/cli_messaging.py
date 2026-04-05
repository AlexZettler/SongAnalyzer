"""CLI: Pub/Sub publish, local handlers, Beam streaming workers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from song_analyzer.messaging.topics import (
    SUBSCRIPTION_HPO_REQUESTS,
    SUBSCRIPTION_SONG_REQUESTS,
    SUBSCRIPTION_TRAIN_REQUESTS,
)

mess_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Pub/Sub messaging and Beam streaming workers (install .[stream]).",
)


def _require_pubsub() -> None:
    try:
        import google.cloud.pubsub_v1  # noqa: F401
    except ImportError as e:
        raise typer.BadParameter(
            'Missing google-cloud-pubsub. Install with: pip install -e ".[stream]"'
        ) from e


def _require_beam() -> None:
    _require_pubsub()
    try:
        import apache_beam  # noqa: F401
    except ImportError as e:
        raise typer.BadParameter(
            'Missing apache-beam. Install with: pip install -e ".[stream]"'
        ) from e


def register(parent: typer.Typer) -> None:
    parent.add_typer(mess_app, name="messaging")


@mess_app.command("setup-pubsub")
def setup_pubsub_cmd(
    project: str = typer.Option(
        ...,
        "--project",
        "-p",
        envvar="PUBSUB_PROJECT_ID",
        help="GCP project id (any string with the emulator)",
    ),
) -> None:
    """Create topics and pull subscriptions for all workflows."""
    _require_pubsub()
    from song_analyzer.messaging.admin import ensure_all_topics_and_subscriptions

    ensure_all_topics_and_subscriptions(project)
    typer.echo(f"Ensured topics and subscriptions for project {project!r}")


@mess_app.command("publish-song-request")
def publish_song_request_cmd(
    project: str = typer.Option(..., "--project", "-p", envvar="PUBSUB_PROJECT_ID"),
    request_id: str = typer.Option(..., "--request-id"),
    corpus_root: Path = typer.Option(..., "--corpus-root", exists=False),
    mbid: Optional[str] = typer.Option(None, "--mbid"),
    title: Optional[str] = typer.Option(None, "--title"),
    artist: Optional[str] = typer.Option(None, "--artist"),
    local_audio: Optional[Path] = typer.Option(
        None,
        "--local-audio",
        exists=True,
        dir_okay=False,
        help="Optional file to copy into corpus",
    ),
    no_copy_audio: bool = typer.Option(False, "--no-copy-audio"),
) -> None:
    _require_pubsub()
    from song_analyzer.messaging.payloads import SongRequestPayload
    from song_analyzer.messaging.publish import publish_song_request

    msg = SongRequestPayload(
        request_id=request_id,
        corpus_root=str(corpus_root.resolve()),
        mbid=mbid,
        title=title,
        artist=artist,
        local_audio_path=str(local_audio.resolve()) if local_audio else None,
        copy_audio=not no_copy_audio,
    )
    mid = publish_song_request(project, msg)
    typer.echo(f"published message_id={mid}")


@mess_app.command("publish-hpo-request")
def publish_hpo_request_cmd(
    project: str = typer.Option(..., "--project", "-p", envvar="PUBSUB_PROJECT_ID"),
    exploration_id: str = typer.Option(..., "--exploration-id"),
    n_trials: int = typer.Option(20, "--n-trials"),
    device: str = typer.Option("cuda", "--device"),
    tune_cache_dir: Optional[Path] = typer.Option(None, "--tune-cache-dir"),
    tfds_data_dir: Optional[Path] = typer.Option(None, "--tfds-data-dir"),
    skip_final_train: bool = typer.Option(
        True,
        "--skip-final-train/--final-train",
        help="If final train, set --out-checkpoint",
    ),
    out_checkpoint: Optional[Path] = typer.Option(
        None,
        "--out-checkpoint",
        help="Required when --final-train",
    ),
    no_archive: bool = typer.Option(False, "--no-archive"),
    tune_fresh: bool = typer.Option(False, "--tune-fresh"),
) -> None:
    _require_pubsub()
    if not skip_final_train and out_checkpoint is None:
        raise typer.BadParameter("--out-checkpoint required with --final-train")
    from song_analyzer.messaging.payloads import HpoRequestPayload
    from song_analyzer.messaging.publish import publish_hpo_request

    msg = HpoRequestPayload(
        exploration_id=exploration_id,
        n_trials=n_trials,
        device=device,
        tune_cache_dir=str(tune_cache_dir.resolve()) if tune_cache_dir else None,
        tfds_data_dir=str(tfds_data_dir.resolve()) if tfds_data_dir else None,
        skip_final_train=skip_final_train,
        out_checkpoint=str(out_checkpoint.resolve()) if out_checkpoint else None,
        archive_tune_db_before=not no_archive,
        tune_fresh=tune_fresh,
    )
    mid = publish_hpo_request(project, msg)
    typer.echo(f"published message_id={mid}")


@mess_app.command("publish-train-request")
def publish_train_request_cmd(
    project: str = typer.Option(..., "--project", "-p", envvar="PUBSUB_PROJECT_ID"),
    train_job_id: str = typer.Option(..., "--train-job-id"),
    study_name: str = typer.Option(..., "--study-name"),
    out: Path = typer.Option(..., "--out"),
    tune_cache_dir: Optional[Path] = typer.Option(None, "--tune-cache-dir"),
    tfds_data_dir: Optional[Path] = typer.Option(None, "--tfds-data-dir"),
    epochs: int = typer.Option(3, "--epochs"),
    max_steps: int = typer.Option(500, "--max-steps-per-epoch"),
    device: str = typer.Option("cuda", "--device"),
) -> None:
    _require_pubsub()
    from song_analyzer.messaging.payloads import TrainRequestPayload
    from song_analyzer.messaging.publish import publish_train_request

    msg = TrainRequestPayload(
        train_job_id=train_job_id,
        study_name=study_name,
        out=str(out.resolve()),
        tune_cache_dir=str(tune_cache_dir.resolve()) if tune_cache_dir else None,
        tfds_data_dir=str(tfds_data_dir.resolve()) if tfds_data_dir else None,
        epochs=epochs,
        max_steps_per_epoch=max_steps,
        device=device,
    )
    mid = publish_train_request(project, msg)
    typer.echo(f"published message_id={mid}")


def _read_json_payload(payload_file: Optional[Path]) -> bytes:
    if payload_file is not None:
        return payload_file.read_bytes()
    return sys.stdin.buffer.read()


@mess_app.command("run-local-song")
def run_local_song_cmd(
    payload_file: Optional[Path] = typer.Option(
        None,
        "--payload-file",
        "-f",
        help="JSON file; default: stdin",
    ),
) -> None:
    """Process one song.request JSON without Pub/Sub (debug)."""
    from song_analyzer.beam.handlers import handle_song_bytes

    out = handle_song_bytes(_read_json_payload(payload_file))
    typer.echo(out.decode("utf-8"))


@mess_app.command("run-local-hpo")
def run_local_hpo_cmd(
    payload_file: Optional[Path] = typer.Option(None, "--payload-file", "-f"),
) -> None:
    """Process one hpo.request JSON (requires .[train])."""
    from song_analyzer.beam.handlers import handle_hpo_bytes

    out = handle_hpo_bytes(_read_json_payload(payload_file))
    typer.echo(out.decode("utf-8"))


@mess_app.command("run-local-train")
def run_local_train_cmd(
    payload_file: Optional[Path] = typer.Option(None, "--payload-file", "-f"),
) -> None:
    """Process one train.request JSON (requires .[train])."""
    from song_analyzer.beam.handlers import handle_train_bytes

    out = handle_train_bytes(_read_json_payload(payload_file))
    typer.echo(out.decode("utf-8"))


@mess_app.command("beam-song-worker")
def beam_song_worker_cmd(
    project: str = typer.Option(..., "--project", "-p", envvar="PUBSUB_PROJECT_ID"),
    subscription: str = typer.Option(
        SUBSCRIPTION_SONG_REQUESTS,
        "--subscription",
        help="Pull subscription for song.requests (see messaging setup-pubsub)",
    ),
    runner: str = typer.Option("DirectRunner", "--runner"),
) -> None:
    """Run Beam streaming: song.requests -> song.complete."""
    _require_beam()
    from song_analyzer.beam.pipelines import run_song_ingest_pipeline

    run_song_ingest_pipeline(project=project, subscription_id=subscription, runner=runner)


@mess_app.command("beam-hpo-worker")
def beam_hpo_worker_cmd(
    project: str = typer.Option(..., "--project", "-p", envvar="PUBSUB_PROJECT_ID"),
    subscription: str = typer.Option(SUBSCRIPTION_HPO_REQUESTS, "--subscription"),
    runner: str = typer.Option("DirectRunner", "--runner"),
) -> None:
    """Run Beam streaming: hpo.requests -> hpo.complete."""
    _require_beam()
    from song_analyzer.beam.pipelines import run_hpo_pipeline

    run_hpo_pipeline(project=project, subscription_id=subscription, runner=runner)


@mess_app.command("beam-train-worker")
def beam_train_worker_cmd(
    project: str = typer.Option(..., "--project", "-p", envvar="PUBSUB_PROJECT_ID"),
    subscription: str = typer.Option(SUBSCRIPTION_TRAIN_REQUESTS, "--subscription"),
    runner: str = typer.Option("DirectRunner", "--runner"),
) -> None:
    """Run Beam streaming: train.requests -> train.complete."""
    _require_beam()
    from song_analyzer.beam.pipelines import run_train_pipeline

    run_train_pipeline(project=project, subscription_id=subscription, runner=runner)


@mess_app.command("print-sample-payloads")
def print_sample_payloads_cmd() -> None:
    """Print example JSON for each workflow request and completion (see messaging.workflows)."""
    from song_analyzer.messaging.workflows import WORKFLOWS

    for wf in WORKFLOWS:
        typer.echo(f"--- {wf.name} request ---")
        typer.echo(wf.sample_request().model_dump_json())
        typer.echo(f"--- {wf.name} complete ---")
        typer.echo(wf.sample_complete().model_dump_json())
