"""Disk-persisted registry of active Gemini caches (safety lock).

Prevents zombie caches from running up storage bills when the process crashes.
On startup, any caches in the registry that weren't properly cleaned up can be
detected and deleted.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger("cache_guard")

_DEFAULT_REGISTRY_PATH = os.path.expanduser("~/.cache/cache_guard/gemini_registry.json")


class CacheRegistry:
    """Persistent registry of active Gemini CachedContent objects."""

    def __init__(self, path: str | None = None) -> None:
        self._path = Path(path or _DEFAULT_REGISTRY_PATH)
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def add(
        self,
        cache_name: str,
        model: str,
        session_id: str,
        ttl_seconds: int,
    ) -> None:
        """Register a newly created cache."""
        self._data[cache_name] = {
            "cache_name": cache_name,
            "model": model,
            "session_id": session_id,
            "ttl_seconds": ttl_seconds,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
        }
        self._save()

    def touch(self, cache_name: str) -> None:
        """Update last-accessed time for a cache."""
        if cache_name in self._data:
            self._data[cache_name]["last_accessed"] = datetime.now().isoformat()
            self._save()

    def remove(self, cache_name: str) -> None:
        """Remove a cache from the registry."""
        self._data.pop(cache_name, None)
        self._save()

    def get_stale_entries(self, max_age_hours: float = 2.0) -> list[dict[str, Any]]:
        """Get entries that haven't been accessed within max_age_hours.

        These are likely zombie caches from crashed processes.
        """
        stale = []
        cutoff = datetime.now() - timedelta(hours=max_age_hours)

        for entry in self._data.values():
            last_accessed = datetime.fromisoformat(entry["last_accessed"])
            if last_accessed < cutoff:
                stale.append(entry)

        return stale

    def get_all(self) -> list[dict[str, Any]]:
        return list(self._data.values())

    def get_by_session(self, session_id: str) -> list[dict[str, Any]]:
        return [e for e in self._data.values() if e["session_id"] == session_id]

    @property
    def count(self) -> int:
        return len(self._data)

    def _load(self) -> None:
        """Load registry from disk."""
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load cache registry: %s", e)
                self._data = {}

    def _save(self) -> None:
        """Persist registry to disk."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data, indent=2))
        except OSError as e:
            logger.warning("Failed to save cache registry: %s", e)
