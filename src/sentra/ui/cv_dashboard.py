"""Level 1 dashboard in an OpenCV window. Track C, owner: Swechha03. See docs/PHASE1_PLAN.md, Section 8.1 step 3.

STUB: shows the annotated frame (or a black screen in privacy mode) and quits on q.
"""
from __future__ import annotations

import cv2
import numpy as np

from sentra.events import StatusUpdate

WINDOW_TITLE = "Sentra - prototype, not a medical device (q to quit)"


class CvDashboard:
    """Implements sentra.events.StatusSink with an OpenCV window."""

    def __init__(self, privacy: bool = False) -> None:
        self.privacy = privacy
        self._stop = False

    def publish(self, update: StatusUpdate) -> None:
        # TODO(Swechha03): side panel, alert banner + sound, c/f/p keys, skeleton-only privacy view,
        # alert CSV log (plan Section 8.1 step 3).
        if self.privacy:
            view = np.zeros_like(update.frame)  # never show video pixels in privacy mode
            cv2.putText(view, "PRIVACY MODE (skeleton view not built yet)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            view = update.annotated
        cv2.imshow(WINDOW_TITLE, view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self._stop = True

    def should_stop(self) -> bool:
        return self._stop

    def close(self) -> None:
        cv2.destroyAllWindows()
