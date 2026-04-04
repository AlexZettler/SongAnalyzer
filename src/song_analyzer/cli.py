from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import typer

from song_analyzer.pipeline import analyze_mix, remove_note_from_mix

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("analyze")
def analyze_cmd(
    audio: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Directory for stems + analysis.json"),
    device: str = typer.Option("cpu", help="torch device for Demucs / classifier (e.g. cuda)"),
    demucs_model: str = typer.Option("htdemucs", help="Demucs pretrained name"),
    demucs_shifts: int = typer.Option(0, help="Random shift passes (0 = faster, 1 = better)"),
    demucs_segment: Optional[float] = typer.Option(None, help="Optional segment length override for Demucs"),
    nsynth_checkpoint: Optional[Path] = typer.Option(
        None,
        "--nsynth-checkpoint",
        help="Trained FamilyClassifier state dict (.pt); else env SONGANALYZER_NSYNTH_CHECKPOINT",
    ),
    no_stem_wavs: bool = typer.Option(False, help="Do not write per-stem WAV files"),
    chord_hop: float = typer.Option(0.05, help="Chord analysis hop in seconds"),
    staged: bool = typer.Option(
        False,
        "--staged",
        help="3-pass pipeline: global structure, solo/timbre, iterative note peeling",
    ),
    restrict_iterative_solo: bool = typer.Option(
        False,
        "--restrict-iterative-solo",
        help="Staged only: peel notes only when their midpoint falls in a solo window",
    ),
    no_pass_json: bool = typer.Option(
        False,
        help="Staged only: do not write pass1.json / pass2.json debug files",
    ),
    max_iterative_notes: int = typer.Option(
        512,
        help="Staged only: max peel iterations per stem",
    ),
) -> None:
    """Separate a full mix, classify stems, transcribe notes, and estimate chords."""
    warnings.filterwarnings("default", category=UserWarning)
    analyze_mix(
        audio,
        output_dir,
        device=device,
        demucs_model=demucs_model,
        demucs_shifts=demucs_shifts,
        demucs_segment=demucs_segment,
        nsynth_checkpoint=nsynth_checkpoint,
        write_stem_wavs=not no_stem_wavs,
        chord_hop_s=chord_hop,
        use_staged=staged,
        restrict_iterative_to_solo=restrict_iterative_solo,
        write_pass_json=not no_pass_json,
        max_iterative_notes_per_stem=max_iterative_notes,
    )
    typer.echo(f"Wrote {output_dir / 'analysis.json'}")


@app.command("train-nsynth")
def train_nsynth_cmd(
    out: Path = typer.Option(..., "--out", help="Output .pt path"),
    epochs: int = typer.Option(3),
    batch_size: int = typer.Option(32, "--batch-size"),
    lr: float = typer.Option(1e-3, "--lr"),
    weight_decay: float = typer.Option(0.01, "--weight-decay"),
    device: str = typer.Option("cuda", "--device"),
    max_steps: int = typer.Option(500, "--max-steps-per-epoch"),
    max_val_steps: Optional[int] = typer.Option(
        None,
        "--max-val-steps",
        help="Cap validation batches on nsynth valid split (each epoch when set); when --tune, defaults to 200 if omitted",
    ),
    tune: bool = typer.Option(False, "--tune", help="Run Optuna HPO then train with best hyperparameters"),
    tune_trials: int = typer.Option(20, "--tune-trials"),
    tune_cache_dir: Optional[Path] = typer.Option(
        None,
        "--tune-cache-dir",
        help="Directory for SQLite study cache (default: user cache song_analyzer/tune)",
    ),
    no_tune_cache: bool = typer.Option(
        False,
        "--no-tune-cache",
        help="In-memory Optuna only (no resume)",
    ),
    tune_fresh: bool = typer.Option(
        False,
        "--tune-fresh",
        help="Delete existing study for this fingerprint and start over",
    ),
    tfds_data_dir: Optional[Path] = typer.Option(
        None,
        "--tfds-data-dir",
        help="TensorFlow Datasets root (default: TFDS_DATA_DIR env or ~/tensorflow_datasets)",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Python logging level: DEBUG, INFO, or WARNING",
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        help="Append logs to this file (default: platform log dir / nsynth_train.log)",
    ),
    no_log_file: bool = typer.Option(
        False,
        "--no-log-file",
        help="Do not write a log file (console only)",
    ),
) -> None:
    """Train instrument-family classifier on NSynth (requires pip install -e '.[train]')."""
    if tune:
        from song_analyzer.instruments.tune_nsynth import tune_nsynth_main

        val_cap = max_val_steps if max_val_steps is not None else 200
        tune_nsynth_main(
            out=out,
            device=device,
            n_trials=tune_trials,
            tune_cache_dir=tune_cache_dir,
            no_tune_cache=no_tune_cache,
            tune_fresh=tune_fresh,
            max_val_steps=val_cap,
            final_epochs=epochs,
            final_max_steps_per_epoch=max_steps,
            tfds_data_dir=tfds_data_dir,
            log_level=log_level,
            log_file=log_file,
            no_log_file=no_log_file,
        )
        return

    from song_analyzer.instruments.train_nsynth import train_main

    argv = [
        "--out",
        str(out),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
        "--device",
        device,
        "--max-steps-per-epoch",
        str(max_steps),
    ]
    if max_val_steps is not None:
        argv.extend(["--max-val-steps", str(max_val_steps)])
    if tfds_data_dir is not None:
        argv.extend(["--tfds-data-dir", str(tfds_data_dir)])
    argv.extend(["--log-level", log_level])
    if log_file is not None:
        argv.extend(["--log-file", str(log_file)])
    if no_log_file:
        argv.append("--no-log-file")
    train_main(argv)


@app.command("prepare-nsynth")
def prepare_nsynth_cmd(
    tfds_data_dir: Optional[Path] = typer.Option(
        None,
        "--tfds-data-dir",
        help="TensorFlow Datasets root (default: TFDS_DATA_DIR env or ~/tensorflow_datasets)",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Python logging level: DEBUG, INFO, or WARNING",
    ),
    log_rss_interval_seconds: Optional[float] = typer.Option(
        None,
        "--log-rss-interval-seconds",
        help="During prepare, log process RSS every N seconds (needs psutil from .[train])",
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        help="Append logs to this file (default: platform log dir / nsynth_prepare.log)",
    ),
    no_log_file: bool = typer.Option(
        False,
        "--no-log-file",
        help="Do not write a log file (console only)",
    ),
) -> None:
    """Download and prepare NSynth TFRecords (requires pip install -e '.[train]')."""
    from song_analyzer.instruments.prepare_nsynth import prepare_main

    argv: list[str] = []
    if tfds_data_dir is not None:
        argv.extend(["--tfds-data-dir", str(tfds_data_dir)])
    argv.extend(["--log-level", log_level])
    if log_rss_interval_seconds is not None:
        argv.extend(
            ["--log-rss-interval-seconds", str(log_rss_interval_seconds)]
        )
    if log_file is not None:
        argv.extend(["--log-file", str(log_file)])
    if no_log_file:
        argv.append("--no-log-file")
    prepare_main(argv)


@app.command("remove-note")
def remove_note_cmd(
    audio: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output: Path = typer.Option(..., "--output", "-o", help="Output mixed WAV path"),
    stem: str = typer.Option(..., help="Stem name: drums, bass, other, vocals"),
    midi_pitch: int = typer.Option(..., help="MIDI note number to attenuate"),
    start: float = typer.Option(..., help="Start time in seconds"),
    end: float = typer.Option(..., help="End time in seconds"),
    device: str = typer.Option("cpu", "--device"),
    demucs_model: str = typer.Option("htdemucs"),
    stems_dir: Optional[Path] = typer.Option(
        None,
        "--stems-dir",
        help="Use pre-separated stems (*.wav) instead of running Demucs",
    ),
) -> None:
    """Attenuate one note on a stem and remix (approximate)."""
    remove_note_from_mix(
        audio,
        output,
        stem=stem,
        midi_pitch=midi_pitch,
        start_s=start,
        end_s=end,
        device=device,
        demucs_model=demucs_model,
        stems_dir=stems_dir,
    )
    typer.echo(f"Wrote {output}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
