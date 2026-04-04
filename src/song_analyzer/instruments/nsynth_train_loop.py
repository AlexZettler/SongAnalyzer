"""Shared NSynth train/val steps for the family classifier.

Loads TensorFlow Datasets ``nsynth/full`` (train/valid), handles first-time prepare (Beam +
dill; Windows in-process runner). Used by ``train_nsynth``, ``tune_nsynth``, and ``prepare_nsynth``. Dataset cache
lives under ``TFDS_DATA_DIR``, ``--tfds-data-dir``, or the default TFDS home directory.
"""

from __future__ import annotations

import logging
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from song_analyzer.instruments.mel import waveform_to_log_mel

logger = logging.getLogger(__name__)


@contextmanager
def _rss_sampling_log_context(interval_sec: float | None) -> Iterator[None]:
    """Log this process RSS (and VMS) on an interval while the block runs (DirectRunner = one process)."""
    if interval_sec is None or interval_sec <= 0:
        yield
        return
    try:
        import psutil
    except ModuleNotFoundError:
        logger.warning(
            "log_rss_interval_seconds=%s but psutil is not installed; "
            'install the [train] extra: pip install -e ".[train]"',
            interval_sec,
        )
        yield
        return

    stop = threading.Event()

    def _loop() -> None:
        proc = psutil.Process()
        while not stop.is_set():
            mi = proc.memory_info()
            logger.info(
                "process_memory rss_mb=%.1f vms_mb=%.1f",
                mi.rss / (1024**2),
                mi.vms / (1024**2),
            )
            if stop.wait(timeout=interval_sec):
                break

    thread = threading.Thread(target=_loop, name="rss-sampler", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=interval_sec + 2.0)


def _tfds_download_and_prepare_kwargs_for_beam(tfds_module) -> dict:
    """Tell TFDS/Beam to use dill for pickling during beam-based dataset preparation.

    Beam 2.65+ defaults to cloudpickle; serializing NSynth's pipeline can recurse
    through etils lazy imports (see tensorflow/datasets#11055). Dill avoids that path.

    On Windows, ``DirectRunner`` is ``SwitchingDirectRunner``, which often selects
    ``PrismRunner`` (a ``PortableRunner``) and hits gRPC DEADLINE_EXCEEDED. TFDS
    accepts ``DownloadConfig.beam_runner``; we pass ``BundleBasedDirectRunner`` so
    execution stays in-process without the portable stack.
    """
    import apache_beam as beam

    opts = beam.options.pipeline_options.PipelineOptions(["--pickle_library=dill"])
    if sys.platform == "win32":
        from apache_beam.runners.direct.direct_runner import BundleBasedDirectRunner

        dl = tfds_module.download.DownloadConfig(
            beam_options=opts,
            beam_runner=BundleBasedDirectRunner(),
        )
    else:
        dl = tfds_module.download.DownloadConfig(beam_options=opts)
    return {"download_config": dl}


def prepare_nsynth_tfrecords(
    tfds_module,
    *,
    data_dir: str | None = None,
    log_rss_interval_seconds: float | None = None,
) -> Path:
    """Run TFDS ``download_and_prepare`` for ``nsynth/full`` (materializes TFRecords under the builder data dir).

    Uses the same Beam ``DownloadConfig`` as training (dill; Windows in-process runner).
    ``log_rss_interval_seconds`` logs process RSS periodically during prepare (DirectRunner uses one process).
    """
    load_kw: dict = {}
    if data_dir is not None:
        load_kw["data_dir"] = data_dir
    builder = tfds_module.builder("nsynth/full", **load_kw)
    variant = Path(builder.data_dir)
    logger.info("TFDS builder data_dir=%s", variant)
    if variant.is_dir() and not builder.is_prepared():
        raise RuntimeError(
            "NSynth cache looks incomplete (folder exists but the dataset is not "
            f"readable): {variant}\n"
            "Delete that folder (or the whole nsynth/full version directory) and run "
            "again. First-time preparation is slow and CPU-heavy; DirectRunner is "
            "used automatically on Windows."
        )
    if builder.is_prepared():
        logger.info("nsynth/full already prepared at %s", variant)
        return variant
    logger.warning(
        "first-time download_and_prepare for NSynth can take a long time and is CPU-heavy"
    )
    dap_kw = _tfds_download_and_prepare_kwargs_for_beam(tfds_module)
    with _rss_sampling_log_context(log_rss_interval_seconds):
        builder.download_and_prepare(**dap_kw)
    logger.info("nsynth/full prepared; TFRecords under %s", variant)
    return variant


def _run_batches(
    tfds,
    ds,
    *,
    max_steps: int,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    train: bool,
    phase: str,
    progress_log_interval: int | None,
) -> tuple[float, float]:
    """Iterate tfds batches; return (mean_loss, accuracy)."""
    losses: list[float] = []
    correct = 0
    total = 0
    if train:
        model.train()
    else:
        model.eval()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for step, batch in enumerate(tfds.as_numpy(ds)):
            if step >= max_steps:
                break
            audio = batch["audio"]
            family_idx = batch["instrument"]["family"]
            bsz = int(audio.shape[0])
            loss_mean = 0.0
            if train:
                assert optimizer is not None
                optimizer.zero_grad()
            for i in range(bsz):
                x = np.asarray(audio[i], dtype=np.float32)
                mel = waveform_to_log_mel(x, device)
                logits = model(mel)
                target = torch.tensor([int(family_idx[i])], device=device, dtype=torch.long)
                loss = F.cross_entropy(logits, target)
                if train:
                    loss.backward()
                loss_mean += float(loss.detach().cpu())
                pred = int(logits.argmax(dim=-1).item())
                if pred == int(family_idx[i]):
                    correct += 1
                total += 1
            if train:
                assert optimizer is not None
                optimizer.step()
            batch_loss = loss_mean / max(bsz, 1)
            losses.append(batch_loss)
            if (
                progress_log_interval is not None
                and progress_log_interval > 0
                and logger.isEnabledFor(logging.INFO)
            ):
                n_done = step + 1
                if n_done % progress_log_interval == 0 or n_done >= max_steps:
                    run_mean = float(np.mean(losses))
                    run_acc = correct / max(total, 1)
                    logger.info(
                        "%s step %s/%s batch_loss %.4f running_mean_loss %.4f running_acc %.4f",
                        phase,
                        n_done,
                        max_steps,
                        batch_loss,
                        run_mean,
                        run_acc,
                    )
    if not losses:
        return 0.0, 0.0
    acc = correct / max(total, 1)
    return float(np.mean(losses)), acc


def run_nsynth_split(
    tfds_module,
    *,
    split: str,
    batch_size: int,
    max_steps: int,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    train: bool,
    shuffle_files: bool = True,
    shuffle_buffer: int = 10_000,
    data_dir: str | None = None,
    progress_log_interval: int | None = None,
) -> tuple[float, float]:
    """Load nsynth/full split, optionally shuffle, batch, run up to max_steps. Returns (mean_loss, acc)."""
    phase = "train" if train else "valid"
    logger.info(
        "loading nsynth/full split=%s batch_size=%s max_steps=%s mode=%s",
        split,
        batch_size,
        max_steps,
        phase,
    )
    load_kw: dict = {}
    if data_dir is not None:
        load_kw["data_dir"] = data_dir
    builder = tfds_module.builder("nsynth/full", **load_kw)
    variant = Path(builder.data_dir)
    logger.info("TFDS builder data_dir=%s", variant)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "dataset version=%s shuffle_files=%s shuffle_buffer=%s",
            builder.info.version,
            shuffle_files,
            shuffle_buffer if train else 0,
        )
    if variant.is_dir() and not builder.is_prepared():
        raise RuntimeError(
            "NSynth cache looks incomplete (folder exists but the dataset is not "
            f"readable): {variant}\n"
            "Delete that folder (or the whole nsynth/full version directory) and run "
            "again. First-time preparation is slow and CPU-heavy; DirectRunner is "
            "used automatically on Windows."
        )
    if builder.is_prepared():
        logger.info("dataset already prepared; loading split")
    else:
        logger.warning(
            "first-time download_and_prepare for NSynth can take a long time and is CPU-heavy"
        )
    dap_kw = (
        None
        if builder.is_prepared()
        else _tfds_download_and_prepare_kwargs_for_beam(tfds_module)
    )
    ds = tfds_module.load(
        "nsynth/full",
        split=split,
        shuffle_files=shuffle_files,
        download_and_prepare_kwargs=dap_kw,
        **load_kw,
    )
    if shuffle_buffer > 0 and train:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.batch(batch_size)
    return _run_batches(
        tfds_module,
        ds,
        max_steps=max_steps,
        device=device,
        model=model,
        optimizer=optimizer,
        train=train,
        phase=phase,
        progress_log_interval=progress_log_interval,
    )
