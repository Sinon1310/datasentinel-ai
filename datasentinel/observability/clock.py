"""Time sources for pipeline observability.

Two clocks are deliberately kept apart:

* ``utc_now_ms`` — UTC wall-clock time as integer epoch milliseconds, used for every persisted
  timestamp (it can jump when the system clock is adjusted).
* ``monotonic_ns`` — a monotonic counter, used only for in-process elapsed-time measurement
  (never persisted as a timestamp; it has no meaning across processes).

:class:`FixedClock` makes tests and deterministic seeding reproducible.
"""

from __future__ import annotations

import threading
import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Source of UTC wall-clock time and monotonic time."""

    def utc_now_ms(self) -> int:
        """Current UTC time in integer milliseconds since the Unix epoch."""
        ...

    def monotonic_ns(self) -> int:
        """Monotonic nanoseconds; meaningful only as a difference within one process."""
        ...


class SystemClock:
    """The real clock: ``time.time_ns`` for UTC and ``time.monotonic_ns`` for elapsed time."""

    def utc_now_ms(self) -> int:
        return time.time_ns() // 1_000_000

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()


class FixedClock:
    """A deterministic, manually advanced clock (thread-safe).

    Wall-clock and monotonic time advance together by default. :meth:`set_wall_ms` moves only the
    wall clock, which lets tests simulate system-clock adjustments (for example, an end time
    earlier than the start time).
    """

    def __init__(self, start_ms: int = 1_700_000_000_000, monotonic_start_ns: int = 0) -> None:
        if start_ms < 0 or monotonic_start_ns < 0:
            raise ValueError("FixedClock start values must be non-negative")
        self._wall_ms = int(start_ms)
        self._mono_ns = int(monotonic_start_ns)
        self._lock = threading.Lock()

    def utc_now_ms(self) -> int:
        with self._lock:
            return self._wall_ms

    def monotonic_ns(self) -> int:
        with self._lock:
            return self._mono_ns

    def advance(self, *, ms: int = 0, seconds: float = 0.0) -> None:
        """Advance both clocks by a non-negative amount."""
        delta_ms = int(ms + seconds * 1000)
        if delta_ms < 0:
            raise ValueError("FixedClock cannot move backwards with advance(); use set_wall_ms()")
        with self._lock:
            self._wall_ms += delta_ms
            self._mono_ns += delta_ms * 1_000_000

    def set_wall_ms(self, value: int) -> None:
        """Set only the wall clock (simulates a system-clock adjustment; monotonic time is unaffected)."""
        with self._lock:
            self._wall_ms = int(value)
