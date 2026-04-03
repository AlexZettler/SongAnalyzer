#!/usr/bin/env python3
"""
Optional note-level evaluation on MAPS (or similar) piano transcriptions.

Compare predicted note lists (from ``song_analyzer.pipeline.analyze_mix`` or Basic Pitch)
to reference MIDI using mir_eval.transcription metrics once you add a loader for your
MAPS folder layout.
"""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description="MAPS / mir_eval hook (stub)")
    p.add_argument("--maps-root", type=str, required=True)
    args = p.parse_args()
    try:
        import mir_eval  # noqa: F401
    except ImportError as e:
        raise SystemExit("Install mir_eval: pip install mir_eval") from e
    print("Stub: load MAPS audio+MIDI from", args.maps_root, "and run mir_eval.transcription.*")


if __name__ == "__main__":
    main()
