"""Pluggable metadata and lyrics connectors (optional ``[corpus]`` extra for HTTP)."""

from song_analyzer.corpus.connectors.protocol import LyricsConnector, LyricsResult
from song_analyzer.corpus.connectors.stub import GeniusLyricsStubConnector, get_lyrics_connector

__all__ = [
    "GeniusLyricsStubConnector",
    "LyricsConnector",
    "LyricsResult",
    "get_lyrics_connector",
]
