"""Sliding-window rate limiter for Graphiti MCP search endpoints.

Isolated in its own module so it can be imported and unit-tested without
pulling in the full MCP server dependency tree (fastmcp, neo4j, etc.).
"""

from __future__ import annotations

import asyncio
import time as _time


class SlidingWindowRateLimiter:
    """Asyncio-safe in-process sliding-window rate limiter.

    Each key tracks its own independent request window; keys that are never
    used consume no memory.  The default key ``'__global__'`` is used when
    no per-caller key is available.

    Thread-safety: safe within a single asyncio event loop (uses
    ``asyncio.Lock``).  Do *not* share instances across threads/processes.

    Args:
        max_requests: Maximum number of requests allowed per window.
        window_seconds: Duration of the sliding window in seconds.

    Raises:
        ValueError: If ``max_requests`` or ``window_seconds`` is not positive.
    """

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        if max_requests <= 0:
            raise ValueError(f'max_requests must be positive, got {max_requests}')
        if window_seconds <= 0:
            raise ValueError(f'window_seconds must be positive, got {window_seconds}')
        self._max = max_requests
        self._window = window_seconds
        # Lazy-initialised so the class can be instantiated at module scope
        # (before the event loop exists).
        self._lock: asyncio.Lock | None = None
        self._timestamps: dict[str, list[float]] = {}

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def is_allowed(self, key: str = '__global__') -> bool:
        """Return ``True`` if the request is within quota; ``False`` if rate-limited.

        Args:
            key: Caller/session identifier.  Requests with different keys are
                tracked independently.  Defaults to ``'__global__'``.
        """
        now = _time.monotonic()
        cutoff = now - self._window

        async with self._get_lock():
            ts = self._timestamps.setdefault(key, [])
            # Evict timestamps outside the window (list stays O(max_requests) in length).
            while ts and ts[0] < cutoff:
                ts.pop(0)
            # Cleanup: delete empty keys to prevent unbounded dict growth when
            # unique caller keys cycle through and their windows expire.
            if not ts:
                del self._timestamps[key]
                ts = []
            if len(ts) >= self._max:
                return False
            ts.append(now)
            self._timestamps[key] = ts
            return True
