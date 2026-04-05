from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class LyricsResult:
    text: str
    attribution: str


@runtime_checkable
class LyricsConnector(Protocol):
    """Fetches plain lyrics text for a track (title/artist or external id)."""

    name: str

    def fetch_lyrics(
        self,
        *,
        title: str | None,
        artist: str | None,
        source_id: str | None = None,
    ) -> LyricsResult:
        """May perform network I/O. Caller is responsible for ToS and copyright."""
        ...
