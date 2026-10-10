"""
Usage caps for the public demo.

Counters live in process memory: they reset on restart/scale-to-zero,
which is acceptable for a cost guard (not a security boundary).
"""

from __future__ import annotations

import threading
from collections import defaultdict

from config.settings import (
    DEMO_MAX_QUESTIONS_PER_SESSION,
    DEMO_MAX_QUESTIONS_TOTAL,
)


class QuestionLimitReached(Exception):
    """Raised when a session or the whole demo hit its question cap."""


class QuestionLimiter:
    def __init__(self, per_session: int, total: int) -> None:
        self.per_session = per_session
        self.total = total
        self._lock = threading.Lock()
        self._by_session: dict[str, int] = defaultdict(int)
        self._total_count = 0

    def remaining(self, session_id: str) -> int:
        with self._lock:
            return max(0, self.per_session - self._by_session[session_id])

    def acquire(self, session_id: str) -> None:
        """Count one question or raise QuestionLimitReached."""

        with self._lock:
            if self._total_count >= self.total:
                raise QuestionLimitReached(
                    "The demo has reached its daily question budget. "
                    "Please try again later."
                )
            if self._by_session[session_id] >= self.per_session:
                raise QuestionLimitReached(
                    f"Session limit reached ({self.per_session} questions). "
                    "Run the project locally for unlimited use."
                )
            self._by_session[session_id] += 1
            self._total_count += 1


limiter = QuestionLimiter(
    per_session=DEMO_MAX_QUESTIONS_PER_SESSION,
    total=DEMO_MAX_QUESTIONS_TOTAL,
)
