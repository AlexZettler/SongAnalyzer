"""Train family classifier on NSynth via TensorFlow Datasets (optional [train] extra)."""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from song_analyzer.instruments.mel import build_model, waveform_to_log_mel


def _import_tfds():
    try:
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        import tensorflow_datasets as tfds
    except ImportError as e:
        raise ImportError(
            'Training requires the [train] extra: pip install -e ".[train]"'
        ) from e
    return tf, tfds


def train_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train NSynth instrument-family classifier")
    p.add_argument("--out", type=Path, required=True, help="Output .pt state_dict path")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-steps-per-epoch", type=int, default=500, help="Cap steps for quick runs")
    args = p.parse_args(argv)

    _, tfds = _import_tfds()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        ds = tfds.load("nsynth/full", split="train", shuffle_files=True)
        ds = ds.shuffle(10_000).batch(args.batch_size)
        losses: list[float] = []
        for step, batch in enumerate(tfds.as_numpy(ds)):
            if step >= args.max_steps_per_epoch:
                break
            audio = batch["audio"]
            family_idx = batch["instrument"]["family"]
            bsz = int(audio.shape[0])
            loss_mean = 0.0
            opt.zero_grad()
            for i in range(bsz):
                x = np.asarray(audio[i], dtype=np.float32)
                mel = waveform_to_log_mel(x, device)
                logits = model(mel)
                target = torch.tensor([int(family_idx[i])], device=device, dtype=torch.long)
                loss = F.cross_entropy(logits, target)
                loss.backward()
                loss_mean += float(loss.detach().cpu())
            opt.step()
            losses.append(loss_mean / max(bsz, 1))
        if losses:
            print(f"epoch {epoch + 1} mean loss {float(np.mean(losses)):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    train_main()
