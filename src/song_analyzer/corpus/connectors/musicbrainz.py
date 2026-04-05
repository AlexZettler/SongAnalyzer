from __future__ import annotations

"""
MusicBrainz API client (JSON). Requires ``httpx`` (``pip install -e ".[corpus]"``).

https://musicbrainz.org/doc/MusicBrainz_API — use a unique User-Agent with contact info.
"""

import time
from dataclasses import dataclass
from typing import Any

DEFAULT_USER_AGENT = "SongAnalyzerCorpus/0.1 (local corpus tool)"


@dataclass(frozen=True)
class RecordingInfo:
    mbid: str
    title: str
    artist: str
    raw: dict[str, Any]


class MusicBrainzClient:
    """Small read-only client with simple rate limiting (1 req / second default)."""

    BASE = "https://musicbrainz.org/ws/2"

    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        min_interval_seconds: float = 1.0,
    ) -> None:
        self._user_agent = user_agent
        self._min_interval = min_interval_seconds
        self._last_request_ts: float = 0.0

    def _throttle(self) -> None:
        now = time.monotonic()
        wait = self._last_request_ts + self._min_interval - now
        if wait > 0:
            time.sleep(wait)

    def _get_json(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        import httpx

        self._throttle()
        url = f"{self.BASE}/{path}"
        with httpx.Client(headers={"User-Agent": self._user_agent}, timeout=30.0) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
        self._last_request_ts = time.monotonic()
        return r.json()

    def fetch_recording(self, mbid: str) -> RecordingInfo:
        data = self._get_json(
            f"recording/{mbid}",
            {"fmt": "json", "inc": "artist-credits"},
        )
        title = str(data.get("title") or "")
        artist = _format_artist_credit(data.get("artist-credit") or [])
        return RecordingInfo(mbid=mbid, title=title, artist=artist, raw=data)


def _format_artist_credit(credit_list: list[Any]) -> str:
    """Best-effort line from MusicBrainz ``artist-credit`` (mix of str and dict)."""
    parts: list[str] = []
    for item in credit_list:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            parts.append(str(item.get("joinphrase", "") or ""))
            if "artist" in item and isinstance(item["artist"], dict):
                parts.append(str(item["artist"].get("name", "") or ""))
            elif "name" in item:
                parts.append(str(item["name"] or ""))
    return "".join(parts).strip()
