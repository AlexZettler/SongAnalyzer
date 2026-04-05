from __future__ import annotations

"""
Template scraper connector. Implement HTML/API parsing in a subclass; respect site ToS.

Rate limiting: use ``time.sleep`` between requests; many lyrics sites forbid automated access.
"""

import time

from song_analyzer.corpus.connectors.protocol import LyricsConnector, LyricsResult


class GeniusLyricsStubConnector:
    """Placeholder for a Genius (or similar) integration — not implemented in-tree."""

    def __init__(self, *, min_interval_seconds: float = 1.0) -> None:
        self.name = "genius"
        self._min_interval = min_interval_seconds
        self._last_request_ts: float = 0.0

    def _throttle(self) -> None:
        now = time.monotonic()
        wait = self._last_request_ts + self._min_interval - now
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.monotonic()

    def fetch_lyrics(
        self,
        *,
        title: str | None,
        artist: str | None,
        source_id: str | None = None,
    ) -> LyricsResult:
        self._throttle()
        raise NotImplementedError(
            "Genius / HTML lyrics scraping is not implemented. "
            "Subclass GeniusLyricsStubConnector or add a connector that returns LyricsResult. "
            "Ensure you comply with the site's terms of service and copyright."
        )


_CONNECTORS: dict[str, type[LyricsConnector]] = {
    "genius": GeniusLyricsStubConnector,
}


def get_lyrics_connector(name: str, **kwargs: object) -> LyricsConnector:
    key = name.strip().lower()
    cls = _CONNECTORS.get(key)
    if cls is None:
        raise ValueError(f"Unknown lyrics connector: {name!r}. Known: {sorted(_CONNECTORS)}")
    return cls(**kwargs)  # type: ignore[misc,arg-type]


def register_lyrics_connector(name: str, cls: type[LyricsConnector]) -> None:
    _CONNECTORS[name.strip().lower()] = cls
