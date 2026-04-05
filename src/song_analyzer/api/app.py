"""FastAPI app: upload audio, return ``AnalysisResult`` JSON from ``analyze_mix``."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from song_analyzer.pipeline import analyze_mix

app = FastAPI(
    title="Song Analyzer",
    description="Full-mix analysis: separation, instrument families, notes, chords.",
)


def _env_path(key: str) -> Path | None:
    v = os.environ.get(key)
    return Path(v).resolve() if v else None


@app.post("/analyze")
async def analyze_upload(
    file: UploadFile = File(..., description="WAV, FLAC, MP3, or OGG"),
) -> dict[str, Any]:
    """Run the same pipeline as ``song-analyzer analyze`` on an uploaded file."""
    name = (file.filename or "").lower()
    if not any(name.endswith(ext) for ext in (".wav", ".flac", ".mp3", ".ogg", ".m4a")):
        raise HTTPException(
            status_code=400,
            detail="Expected an audio filename ending in .wav, .flac, .mp3, .ogg, or .m4a",
        )
    device = os.environ.get("SONGANALYZER_API_DEVICE", "cpu")
    demucs_model = os.environ.get("SONGANALYZER_API_DEMUCS_MODEL", "htdemucs")
    nsynth_checkpoint = _env_path("SONGANALYZER_NSYNTH_CHECKPOINT")
    staged = os.environ.get("SONGANALYZER_API_STAGED", "").lower() in ("1", "true", "yes")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    filename = file.filename or "upload.wav"

    def run_pipeline() -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="song_analyzer_api_") as td:
            tdir = Path(td)
            suffix = Path(filename).suffix or ".wav"
            in_path = tdir / f"input{suffix}"
            out_dir = tdir / "out"
            in_path.write_bytes(raw)
            result = analyze_mix(
                in_path,
                out_dir,
                device=device,
                demucs_model=demucs_model,
                nsynth_checkpoint=nsynth_checkpoint,
                use_staged=staged,
                write_stem_wavs=False,
            )
            return result.model_dump()

    return await run_in_threadpool(run_pipeline)
