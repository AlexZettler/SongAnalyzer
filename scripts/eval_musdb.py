#!/usr/bin/env python3
"""
Optional separation metrics on MUSDB18 (SI-SDR, etc.).

Requires a local MUSDB18 folder, `museval`, and `stempeg` or soundfile-capable stems.

Example (after installing museval and preparing paths):

  pip install museval
  python scripts/eval_musdb.py --musdb-root "D:/data/musdb18" --demucs-model htdemucs

This repository does not vendor MUSDB18; wire your own loader to compare Demucs outputs
to reference stems and call ``museval.metrics.bss_eval``.
"""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description="MUSDB18 / museval hook (stub)")
    p.add_argument("--musdb-root", type=str, required=True)
    p.add_argument("--demucs-model", type=str, default="htdemucs")
    args = p.parse_args()
    try:
        import museval  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "Install museval in your environment to run full evaluation: pip install museval"
        ) from e
    print(
        "Stub: point this script at your MUSDB18 hierarchy under",
        args.musdb_root,
        "and compare Demucs-separated stems to mixture/reference using museval.metrics.bss_eval.",
    )
    print("Demucs model name:", args.demucs_model)


if __name__ == "__main__":
    main()
