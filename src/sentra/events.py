"""Shared data contract between pipeline (B), posture analysis (A) and UI (C).

Changing anything in this file needs agreement from all three track owners.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

# COCO-17 keypoint order produced by YOLO pose models.
KEYPOINT_NAMES: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
KP: dict[str, int] = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# Pairs of keypoint indices to connect when drawing a skeleton (0-indexed).
SKELETON: tuple[tuple[int, int], ...] = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
)


@dataclass(frozen=True)
class PersonPose:
    track_id: int                              # temporary ByteTrack ID — NOT an identity
    keypoints: np.ndarray                      # shape (17, 3) float32: x px, y px, confidence 0..1
    box: tuple[float, float, float, float]     # x1, y1, x2, y2 in pixels
    box_confidence: float                      # 0..1


@dataclass(frozen=True)
class PoseFrame:
    timestamp: float                           # seconds from time.monotonic() — use this, never frame counts
    frame_width: int
    frame_height: int
    people: tuple[PersonPose, ...]             # empty tuple when nobody is tracked


class PostureState(str, Enum):
    UNKNOWN = "unknown"              # not enough visible keypoints to judge
    UPRIGHT = "upright"              # standing / walking / sitting upright
    DESCENDING = "descending"        # hips dropping fast right now
    ON_GROUND = "on_ground"          # lying / horizontal, no fast drop seen before it
    POSSIBLE_FALL = "possible_fall"  # fast drop, then on the ground for N seconds → alert


@dataclass(frozen=True)
class PersonStatus:
    track_id: int
    state: PostureState
    confidence: float                # 0..1, how sure the analyzer is about `state`
    reason: str                      # short text for the UI, e.g. "torso 78° for 6.1 s after fast drop"
    since: float                     # timestamp when this person entered `state`


@dataclass(frozen=True)
class StatusUpdate:
    pose: PoseFrame
    statuses: tuple[PersonStatus, ...]   # one per person in pose.people — match on track_id, not order
    frame: np.ndarray                    # raw BGR camera frame (never written to disk)
    annotated: np.ndarray                # BGR frame with the Ultralytics skeleton overlay
    fps: float                           # pipeline frames per second


class StatusSink(Protocol):
    """Anything that displays StatusUpdates (the dashboard). Implemented by Track C."""

    def publish(self, update: StatusUpdate) -> None: ...   # called once per processed frame
    def should_stop(self) -> bool: ...                     # True when the user quit (e.g. pressed q)
    def close(self) -> None: ...                           # release windows / servers
