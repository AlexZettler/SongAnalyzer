from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from song_analyzer.pipeline import analyze_mix, remove_note_from_mix

app = typer.Typer(no_args_is_help=True, add_completion=False)

corpus_app = typer.Typer(no_args_is_help=True, add_completion=False, help="Song corpus, lyrics metadata, pseudo-labels.")
app.add_typer(corpus_app, name="corpus")


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


@app.command("explore-run")
def explore_run_cmd(
    preset: str = typer.Option(
        ...,
        "--preset",
        help="Parameter space: nsynth-tune (training hparams) or dense-eval (synthetic clip density)",
    ),
    state: Path = typer.Option(
        ...,
        "--state",
        help="Append-only JSONL path recording every proposed point (created if missing)",
    ),
    steps: int = typer.Option(1, "--steps", min=1, help="How many proposals to emit"),
    seed: int = typer.Option(0, "--seed", help="RNG seed (reproducible proposals)"),
    explore_ratio: float = typer.Option(
        0.5,
        "--explore-ratio",
        min=0.0,
        max=1.0,
        help="Fraction of steps that maximize distance to prior points (novelty); rest mutate a past point",
    ),
    novelty_candidates: int = typer.Option(
        2048,
        "--novelty-candidates",
        min=1,
        help="Random candidates per novelty step; larger explores the box more thoroughly",
    ),
    local_sigma: float = typer.Option(
        0.12,
        "--local-sigma",
        min=1e-6,
        help="Gaussian noise in normalized [0,1] space for local iterations",
    ),
    weights: Optional[str] = typer.Option(
        None,
        "--weights",
        help="Optional comma-separated positive weights per dimension for distance (same order as preset)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print proposals only; do not append to the state file",
    ),
) -> None:
    """
    Long-run parameter exploration: alternate between **novelty** (far from all prior runs)
    and **local** steps (perturb a previous point). Use with your own training/eval driver
    by reading each printed JSON line, or chain ``--steps`` in a shell loop.

    Unlike single-objective Optuna, this targets **coverage** of the feasible box so weak
    regions of the space are visited over time.
    """
    from song_analyzer.explore.param_space import (
        preset_dense_eval,
        preset_nsynth_tune,
    )
    from song_analyzer.explore.runner import exploration_step

    key = preset.strip().lower().replace("_", "-")
    if key == "nsynth-tune":
        space = preset_nsynth_tune()
        preset_name = "nsynth-tune"
    elif key == "dense-eval":
        space = preset_dense_eval()
        preset_name = "dense-eval"
    else:
        raise typer.BadParameter("use --preset nsynth-tune or dense-eval", param_hint="--preset")

    w_arr: np.ndarray | None = None
    if weights is not None:
        parts = [p.strip() for p in weights.split(",") if p.strip()]
        if len(parts) != space.ndim:
            raise typer.BadParameter(
                f"expected {space.ndim} weights for this preset, got {len(parts)}",
                param_hint="--weights",
            )
        w_arr = np.array([float(x) for x in parts], dtype=np.float64)
        if np.any(w_arr <= 0):
            raise typer.BadParameter("all weights must be positive", param_hint="--weights")

    def _echo(params: dict[str, object], prop: object, step_index: int) -> None:
        md = prop.min_dist_to_history
        if md is not None and not math.isfinite(md):
            md = None
        line = {
            "step": step_index,
            "preset": preset_name,
            "mode": prop.mode,
            "params": params,
            "vector": [float(x) for x in prop.vector.tolist()],
            "min_dist_to_history": md,
        }
        typer.echo(json.dumps(line, sort_keys=True))

    rng = np.random.default_rng(seed)
    existing_max = -1
    if state.is_file():
        from song_analyzer.explore.history import load_history

        hist = load_history(state)
        if hist:
            existing_max = max(r.step for r in hist)
    base = existing_max + 1
    session: list[np.ndarray] = [] if dry_run else []

    for k in range(steps):
        params, prop, _ = exploration_step(
            space=space,
            state_path=state,
            preset_name=preset_name,
            step_index=base + k,
            rng=rng,
            explore_probability=explore_ratio,
            n_novelty_candidates=novelty_candidates,
            local_sigma=local_sigma,
            weights=w_arr,
            on_propose=None,
            persist=not dry_run,
            transient_archive=session if dry_run else None,
        )
        _echo(params, prop, base + k)


@corpus_app.command("init")
def corpus_init_cmd(
    root: Path = typer.Option(..., "--root", help="Corpus root directory (creates DB and subfolders)"),
) -> None:
    """Create corpus.sqlite3, audio/, and pseudo_stems/ under --root."""
    from song_analyzer.corpus.db import init_corpus

    layout = init_corpus(root.resolve())
    typer.echo(f"Initialized corpus at {layout.root}")


@corpus_app.command("import-audio")
def corpus_import_audio_cmd(
    root: Path = typer.Option(..., "--root", help="Corpus root"),
    path: Path = typer.Option(..., "--path", help="Audio file to register"),
    mbid: Optional[str] = typer.Option(None, help="MusicBrainz recording MBID"),
    title: Optional[str] = typer.Option(None),
    artist: Optional[str] = typer.Option(None),
    source: str = typer.Option("import", help="Provenance label for this row"),
    no_copy: bool = typer.Option(
        False,
        "--no-copy",
        help="Store absolute path to file instead of copying into corpus audio/",
    ),
    musicbrainz_enrich: bool = typer.Option(
        False,
        "--musicbrainz-enrich",
        help="Fetch title/artist from MusicBrainz when --mbid is set (needs pip install -e '.[corpus]')",
    ),
    user_agent: Optional[str] = typer.Option(
        None,
        "--user-agent",
        help="MusicBrainz User-Agent header (include contact URL per MB policy)",
    ),
) -> None:
    """Copy (default) or reference an audio file and insert a track row."""
    from song_analyzer.corpus.ingest import import_audio_file

    rec = import_audio_file(
        root,
        path,
        copy=not no_copy,
        mbid=mbid,
        title=title,
        artist=artist,
        source=source,
        musicbrainz_enrich=musicbrainz_enrich,
        musicbrainz_user_agent=user_agent,
    )
    typer.echo(f"track_id={rec.track_id} audio={rec.audio_relpath}")


@corpus_app.command("fetch-lyrics")
def corpus_fetch_lyrics_cmd(
    root: Path = typer.Option(..., "--root"),
    track_id: str = typer.Option(..., "--track-id"),
    connector: str = typer.Option(
        "genius",
        "--connector",
        help="Lyrics connector name (in-tree: genius stub only — implement your own subclass)",
    ),
) -> None:
    """Fetch lyrics via a registered connector and store on the track row."""
    from song_analyzer.corpus.connectors.stub import get_lyrics_connector
    from song_analyzer.corpus.store import TrackStore

    store = TrackStore.open(root.resolve())
    try:
        tr = store.get_track(track_id)
        if tr is None:
            raise typer.BadParameter(f"unknown track_id: {track_id}", param_hint="--track-id")
        conn = get_lyrics_connector(connector)
        try:
            result = conn.fetch_lyrics(title=tr.title, artist=tr.artist, source_id=tr.source_id)
        except NotImplementedError as e:
            typer.echo(f"{connector} connector: {e}", err=True)
            raise typer.Exit(code=1) from e
        store.update_lyrics(track_id, result.text, lyrics_source=f"{connector}:{result.attribution}")
    finally:
        store.close()
    typer.echo(f"Lyrics stored for {track_id} (source={connector})")


@corpus_app.command("build-training-manifest")
def corpus_build_manifest_cmd(
    root: Path = typer.Option(..., "--root", help="Corpus root"),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help="Write manifest CSV path (default: <root>/training_manifest.csv)",
    ),
    device: str = typer.Option("cpu", help="torch device for Demucs and teacher"),
    demucs_model: str = typer.Option("htdemucs"),
    demucs_shifts: int = typer.Option(0),
    nsynth_checkpoint: Optional[Path] = typer.Option(
        None,
        "--nsynth-checkpoint",
        help="Teacher FamilyClassifier .pt; else env SONGANALYZER_NSYNTH_CHECKPOINT",
    ),
    min_confidence: float = typer.Option(
        0.0,
        "--min-confidence",
        help="Skip stem rows below this teacher confidence",
    ),
    skip_drums: bool = typer.Option(False, "--skip-drums", help="Omit drums stem from manifest"),
    no_csv: bool = typer.Option(False, "--no-csv", help="Only update SQLite manifest table"),
    no_sqlite: bool = typer.Option(False, "--no-sqlite", help="Only write CSV"),
) -> None:
    """Demucs each track, pseudo-label stems, save 16 kHz WAVs under pseudo_stems/."""
    if no_csv and no_sqlite:
        raise typer.BadParameter("use at most one of --no-csv / --no-sqlite", param_hint="--no-csv")

    from song_analyzer.corpus.manifest import build_training_manifest

    csv_path = None if no_csv else (out or (root.resolve() / "training_manifest.csv"))
    build_training_manifest(
        root.resolve(),
        out_csv=csv_path,
        demucs_model=demucs_model,
        device=device,
        nsynth_checkpoint=nsynth_checkpoint,
        min_confidence=min_confidence,
        demucs_shifts=demucs_shifts,
        progress=True,
        skip_drums=skip_drums,
        store_sqlite=not no_sqlite,
    )
    if csv_path is not None:
        typer.echo(f"Wrote {csv_path}")
    typer.echo("Manifest rows stored in corpus DB (unless --no-sqlite).")


@app.command("train-corpus-finetune")
def train_corpus_finetune_cmd(
    manifest_csv: Path = typer.Option(..., "--manifest-csv", exists=True, dir_okay=False),
    corpus_root: Path = typer.Option(..., "--corpus-root", file_okay=False),
    init_checkpoint: Path = typer.Option(
        ...,
        "--init-checkpoint",
        exists=True,
        dir_okay=False,
        help="Starting FamilyClassifier weights (.pt)",
    ),
    out: Path = typer.Option(..., "--out", help="Output fine-tuned .pt"),
    epochs: int = typer.Option(3),
    batch_size: int = typer.Option(16, "--batch-size"),
    lr: float = typer.Option(1e-4, "--lr"),
    weight_decay: float = typer.Option(0.01, "--weight-decay"),
    crop_seconds: float = typer.Option(1.6, "--crop-seconds"),
    corpus_steps: int = typer.Option(200, "--corpus-steps-per-epoch"),
    nsynth_steps: int = typer.Option(0, "--nsynth-steps-per-epoch"),
    nsynth_batch_size: int = typer.Option(32, "--nsynth-batch-size"),
    max_val_steps: Optional[int] = typer.Option(None, "--max-val-steps"),
    tfds_data_dir: Optional[Path] = typer.Option(None, "--tfds-data-dir"),
    device: str = typer.Option("cuda", "--device"),
    seed: int = typer.Option(0, "--seed"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Fine-tune classifier on pseudo-labeled stems; optional NSynth mix (requires '.[train]')."""
    import torch

    from song_analyzer.instruments.nsynth_logging import (
        configure_nsynth_logging,
        parse_train_log_level,
    )
    from song_analyzer.instruments.train_corpus_finetune import train_corpus_finetune_run

    configure_nsynth_logging(parse_train_log_level(log_level), profile="train")
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    train_corpus_finetune_run(
        manifest_csv=manifest_csv,
        corpus_root=corpus_root,
        init_checkpoint=init_checkpoint,
        out=out,
        device=dev,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        crop_seconds=crop_seconds,
        corpus_steps_per_epoch=corpus_steps,
        nsynth_steps_per_epoch=nsynth_steps,
        nsynth_batch_size=nsynth_batch_size,
        tfds_data_dir=tfds_data_dir,
        max_val_steps=max_val_steps,
        seed=seed,
    )
    typer.echo(f"Wrote {out}")


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
