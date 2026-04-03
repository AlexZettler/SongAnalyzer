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
    )
    typer.echo(f"Wrote {output_dir / 'analysis.json'}")


@app.command("train-nsynth")
def train_nsynth_cmd(
    out: Path = typer.Option(..., "--out", help="Output .pt path"),
    epochs: int = typer.Option(3),
    batch_size: int = typer.Option(32, "--batch-size"),
    lr: float = typer.Option(1e-3, "--lr"),
    device: str = typer.Option("cuda", "--device"),
    max_steps: int = typer.Option(500, "--max-steps-per-epoch"),
) -> None:
    """Train instrument-family classifier on NSynth (requires pip install -e '.[train]')."""
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
        "--device",
        device,
        "--max-steps-per-epoch",
        str(max_steps),
    ]
    train_main(argv)


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
