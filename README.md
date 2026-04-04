# Song Analyzer

Offline pipeline for full-mix audio: stem separation (Demucs), instrument-family classification (train on NSynth), note transcription (Basic Pitch), chord naming, and approximate per-note removal on stems.

## Install

Requires **Python 3.13+** (use 3.13 for TensorFlow-based extras until 3.14 wheels ship).

```bash
pip install -e ".[dev]"
```

For **Spotify Basic Pitch** transcription (TensorFlow; use Python 3.13 — wheels not on PyPI for 3.14+ yet):

```bash
pip install -e ".[dev,basicpitch]"
```

Without `basicpitch`, the CLI uses a **librosa piptrack** fallback (weaker on polyphonic stems).

Training the instrument classifier from NSynth (optional extra):

```bash
pip install -e ".[dev,train]"
```

## Usage

```bash
song-analyzer analyze path/to/song.wav --output-dir ./out --device cuda
song-analyzer remove-note path/to/song.wav --output ./out/mix_minus.wav --stem other --midi-pitch 60 --start 1.0 --end 2.0
```

Set `SONGANALYZER_NSYNTH_CHECKPOINT` or pass `--nsynth-checkpoint` to a trained `.pt` file for instrument labels; otherwise stems are labeled with a uniform fallback and a warning.

For a full description of the analysis stages, output files (`analysis.json`, optional `pass1.json` / `pass2.json`), and how they relate to NSynth checkpoints and training caches, see [docs/PIPELINE_AND_TRAINING.md](docs/PIPELINE_AND_TRAINING.md).

## Train NSynth classifier

```bash
song-analyzer train-nsynth --out checkpoints/nsynth.pt --epochs 5 --device cuda
```

Requires `[train]` and TensorFlow Datasets download of `nsynth` on first run. Training checkpoints are PyTorch **state dicts** only; Optuna tuning (with `--tune`) stores studies under the user cache — details in [docs/PIPELINE_AND_TRAINING.md](docs/PIPELINE_AND_TRAINING.md).
