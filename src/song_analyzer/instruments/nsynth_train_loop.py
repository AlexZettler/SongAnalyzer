"""Shared NSynth train/val steps for family classifier (used by train_nsynth and tune_nsynth)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from song_analyzer.instruments.mel import waveform_to_log_mel


def _run_batches(
    tfds,
    ds,
    *,
    max_steps: int,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    train: bool,
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
            losses.append(loss_mean / max(bsz, 1))
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
) -> tuple[float, float]:
    """Load nsynth/full split, optionally shuffle, batch, run up to max_steps. Returns (mean_loss, acc)."""
    ds = tfds_module.load("nsynth/full", split=split, shuffle_files=shuffle_files)
    if shuffle_buffer > 0 and train:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.batch(batch_size)
    return _run_batches(tfds_module, ds, max_steps=max_steps, device=device, model=model, optimizer=optimizer, train=train)
