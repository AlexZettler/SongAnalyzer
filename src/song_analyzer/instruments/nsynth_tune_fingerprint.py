"""Fingerprint for NSynth HPO cache: invalidates Optuna study when data or training code changes."""

from __future__ import annotations

import hashlib
import json
import os
from importlib import metadata as importlib_metadata
from pathlib import Path

# Bump when you change something not covered by hashed training files (e.g. external data prep).
FINGERPRINT_SALT = "1"

_INSTRUMENTS_DIR = Path(__file__).resolve().parent

# Files whose contents affect training semantics (keep in sync with tune/train stack).
FINGERPRINT_SOURCE_NAMES: tuple[str, ...] = (
    "train_nsynth.py",
    "nsynth_train_loop.py",
    "mel.py",
    "model.py",
    "nsynth_tune_fingerprint.py",
    "tune_nsynth.py",
)


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def code_digest_for_instrument_sources(salt: str = FINGERPRINT_SALT) -> str:
    """SHA-256 over sorted file paths and contents under instruments/."""
    h = hashlib.sha256()
    h.update(salt.encode())
    h.update(b"\0")
    for name in sorted(FINGERPRINT_SOURCE_NAMES):
        path = _INSTRUMENTS_DIR / name
        h.update(name.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def fingerprint_payload(
    *,
    dataset_name: str,
    dataset_version: str,
    code_digest: str,
) -> dict[str, str]:
    return {
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "code_digest": code_digest,
        "tensorflow": _package_version("tensorflow") or "unknown",
        "tensorflow_datasets": _package_version("tensorflow-datasets") or "unknown",
    }


def study_suffix_from_payload(payload: dict[str, str]) -> str:
    """Stable short id for Optuna study name (first 12 hex chars of canonical JSON hash)."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    full = hashlib.sha256(canonical.encode()).hexdigest()
    return full[:12]


def nsynth_study_name(suffix: str) -> str:
    return f"nsynth_family_{suffix}"


def default_tune_cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "song_analyzer" / "tune"


def code_digest_for_test_files(file_contents: dict[str, bytes]) -> str:
    """Deterministic digest for tests (same layout as code_digest but from a mapping)."""
    h = hashlib.sha256()
    h.update(FINGERPRINT_SALT.encode())
    h.update(b"\0")
    for name in sorted(file_contents.keys()):
        h.update(name.encode())
        h.update(b"\0")
        h.update(file_contents[name])
        h.update(b"\0")
    return h.hexdigest()
