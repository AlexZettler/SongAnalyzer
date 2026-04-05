"""Map NSynth instrument families to htdemucs four-stem buckets for synthetic full mixes."""

from __future__ import annotations

from song_analyzer.instruments.constants import NSYNTH_FAMILIES

# Heuristic mapping for 4-stem Demucs (htdemucs). NSynth has no dedicated "drums" family.
FAMILY_TO_DEMUCS_BUCKET: dict[str, str] = {
    "bass": "bass",
    "brass": "other",
    "flute": "other",
    "guitar": "other",
    "keyboard": "other",
    "mallet": "other",
    "organ": "other",
    "reed": "other",
    "string": "other",
    "synth_lead": "other",
    "vocal": "vocals",
}

DEMUCS_FOUR_STEMS: tuple[str, ...] = ("drums", "bass", "other", "vocals")


def demucs_bucket_for_family(family: str) -> str:
    if family not in NSYNTH_FAMILIES:
        raise ValueError(f"unknown NSynth family {family!r}; expected one of {NSYNTH_FAMILIES}")
    return FAMILY_TO_DEMUCS_BUCKET[family]
