"""Sliding-window rate limiter for Graphiti MCP search endpoints.

Isolated in its own module so it can be imported and unit-tested without
pulling in the full MCP server dependency tree (fastmcp, neo4j, etc.).
"""

from __future__ import annotations

import asyncio
import time as _time
from collections import OrderedDict


class SlidingWindowRateLimiter:
    """Asyncio-safe in-process sliding-window rate limiter.

    Each key tracks its own independent request window; keys that are never
    used consume no memory.  The default key ``'__global__'`` is used when
    no per-caller key is available.

    Key cardinality is bounded by ``max_keys`` (default 1024).  When a new
    key would exceed the limit the least-recently-used key is evicted first,
    so an adversary cannot exhaust server memory by cycling through unique
    keys.

    Thread-safety: safe within a single asyncio event loop (uses
    ``asyncio.Lock``).  Do *not* share instances across threads/processes.

    Args:
        max_requests: Maximum number of requests allowed per window.
        window_seconds: Duration of the sliding window in seconds.
        max_keys: Hard upper bound on the number of tracked keys.  LRU
            eviction is used when the limit is reached.  Must be positive.

    Raises:
        ValueError: If any of the constructor arguments is not positive.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        max_keys: int = 1024,
    ) -> None:
        if max_requests <= 0:
            raise ValueError(f'max_requests must be positive, got {max_requests}')
        if window_seconds <= 0:
            raise ValueError(f'window_seconds must be positive, got {window_seconds}')
        if max_keys <= 0:
            raise ValueError(f'max_keys must be positive, got {max_keys}')
        self._max = max_requests
        self._window = window_seconds
        self._max_keys = max_keys
        # Lazy-initialised so the class can be instantiated at module scope
        # (before the event loop exists).
        self._lock: asyncio.Lock | None = None
        # OrderedDict maintains LRU order: least-recently-used at the front,
        # most-recently-used at the back.
        self._timestamps: OrderedDict[str, list[float]] = OrderedDict()

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
            # LRU: move to end (most-recently-used position) on each access.
            if key in self._timestamps:
                self._timestamps.move_to_end(key)

            ts = self._timestamps.get(key, [])

            # Evict timestamps outside the window (list stays O(max_requests) in length).
            while ts and ts[0] < cutoff:
                ts.pop(0)

            # Cleanup: delete empty keys to prevent unbounded dict growth when
            # unique caller keys cycle through and their windows expire.
            if not ts and key in self._timestamps:
                del self._timestamps[key]

            if len(ts) >= self._max:
                return False

            # Enforce cardinality limit before inserting a new key.
            # Evict the least-recently-used entry (front of the OrderedDict).
            if key not in self._timestamps and len(self._timestamps) >= self._max_keys:
                self._timestamps.popitem(last=False)

            ts.append(now)
            self._timestamps[key] = ts
            return True
