"""Camera input. Track B, owner: babarchechi. See docs/PHASE1_PLAN.md, Section 7.1 step 2."""
from __future__ import annotations

import numpy as np


def parse_source(raw: str) -> int | str:
    """Turn a webcam index like "0" into an int; leave anything else (a stream URL) as a string."""
    return int(raw) if raw.isdecimal() else raw


class FrameSource:
    """Reads frames on a background thread and keeps only the latest one. Reconnects URL sources."""

    def __init__(self, source: int | str) -> None:
        self.source = source
        # TODO(babarchechi): open cv2.VideoCapture and start the reader thread.
        raise NotImplementedError("FrameSource is not built yet (docs/PHASE1_PLAN.md, Section 7.1)")

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        """Return (ok, latest BGR frame, time.monotonic() timestamp when that frame was read)."""
        raise NotImplementedError

    def close(self) -> None:
        """Stop the reader thread and release the camera."""
        raise NotImplementedError
