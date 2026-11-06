"""Posture analysis. Track A, owner: iesa242. See docs/PHASE1_PLAN.md, Section 6.

STUB: every person is reported as UNKNOWN so the rest of the app can run before this is built.
Replace the internals freely, but keep the public interface (PostureConfig, PostureAnalyzer.update)
unless the whole team agrees to change it.
"""
from __future__ import annotations

from dataclasses import dataclass

from sentra.events import PersonStatus, PoseFrame, PostureState


@dataclass
class PostureConfig:
    """State-machine thresholds. These are starting guesses to tune (plan Section 6.3)."""

    min_keypoint_conf: float = 0.5        # ignore keypoints below this confidence
    upright_max_angle_deg: float = 35.0   # torso within this of vertical counts as upright
    lying_min_angle_deg: float = 60.0     # torso beyond this counts as lying
    lying_min_aspect: float = 1.2         # box width/height above this supports "lying"
    descent_speed: float = 1.5            # torso-lengths per second of hip drop that counts as "fast"
    descent_window_s: float = 2.0         # lying must start within this long after the fast drop
    on_ground_alert_s: float = 5.0        # seconds on the ground before POSSIBLE_FALL (short for the demo)
    recovery_s: float = 2.0               # upright this long clears POSSIBLE_FALL
    state_hold_frames: int = 3            # frames a new state must persist before switching
    track_timeout_s: float = 3.0          # forget a track not seen for this long


class PostureAnalyzer:
    """Keeps a short history per track_id and returns one PersonStatus per person each frame."""

    def __init__(self, config: PostureConfig | None = None) -> None:
        self.config = config or PostureConfig()

    def update(self, pose: PoseFrame) -> tuple[PersonStatus, ...]:
        """Call once per frame. STUB: reports everyone as UNKNOWN."""
        # TODO(iesa242): compute features (6.1), run the state machine (6.2), forget stale tracks.
        return tuple(
            PersonStatus(
                track_id=person.track_id,
                state=PostureState.UNKNOWN,
                confidence=0.0,
                reason="posture analysis not implemented yet",
                since=pose.timestamp,
            )
            for person in pose.people
        )
