"""Level 2 web dashboard (stretch goal). Track C, owner: Swechha03. See docs/PHASE1_PLAN.md, Section 8.1 step 5.

Must bind to 127.0.0.1 only, never 0.0.0.0, so the video can't be reached from the venue Wi-Fi.
"""
from __future__ import annotations

from sentra.events import StatusUpdate


class WebDashboard:
    """Implements sentra.events.StatusSink as a local web page. Build only after Level 1 works."""

    def __init__(self, privacy: bool = False, host: str = "127.0.0.1", port: int = 8000) -> None:
        raise NotImplementedError("WebDashboard is a Level 2 stretch goal (docs/PHASE1_PLAN.md, Section 8.1)")

    def publish(self, update: StatusUpdate) -> None:
        raise NotImplementedError

    def should_stop(self) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
