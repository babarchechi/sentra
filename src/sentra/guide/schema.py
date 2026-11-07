"""Sentra Guide data contract, schema "sentra.guide/1". Shared by every guide module.

Changing this file needs approval from both other teammates (docs/TEAM_PLAN.md, "Review rules").
Bump SCHEMA_VERSION when a field changes meaning or is removed; adding an optional field is fine.

    camera -> ObjectDetector -> DetectionFrame
           -> DepthProvider (optional) + SpatialEstimator -> SpatialObject
           -> AlertPolicy -> GuideAlert -> Speaker / GuideEventLog / GuideSink (debug view, later a web page)

Wording rule for everything built on this contract: say only what was measured. Objects are "detected",
ranges are "approximate" and only spoken when the depth source passed tape-measure tests
(DepthEstimate.validated). Nothing may ever say a path is clear or safe.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Protocol

import numpy as np

SCHEMA_VERSION = "sentra.guide/1"

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels of the (rotated) frame


class Region(str, Enum):
    """Where an object's box centre is across the image. A screen region, not a verified walking route."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class Severity(str, Enum):
    """How much an object matters for the person holding the camera. Used to order announcements."""

    LOW = "low"        # off to one side
    MEDIUM = "medium"  # in, or overlapping, the central band
    HIGH = "high"      # in the central band and large in view (big or near; box size is not a distance)


SEVERITY_RANK = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2}


@dataclass(frozen=True)
class Detection:
    """One object the detector found in one frame."""

    label: str                    # model class name, e.g. "chair", "person", "dining table"
    class_id: int
    confidence: float             # detector score, 0..1
    box: Box
    track_id: int | None = None   # ByteTrack ID, stable across frames while tracked. Never an identity.


@dataclass(frozen=True)
class DetectionFrame:
    """Everything the detector found in one camera frame."""

    frame_id: int                 # counts processed frames from 1
    timestamp: float              # time.perf_counter() when the camera delivered the frame
    frame_width: int
    frame_height: int
    detections: tuple[Detection, ...]
    source: str = ""              # camera description with credentials removed (capture.describe_source)
    inference_ms: float = 0.0


@dataclass(frozen=True)
class DepthEstimate:
    """Distance to one detection. meters=None means unknown, which is the normal camera-only case."""

    meters: float | None = None
    confidence: float = 0.0       # share of valid depth pixels behind the estimate, 0..1
    source: str = "none"          # "none", "depth_anything_v2", "record3d"
    validated: bool = False       # True only for a provider that passed tape-measure tests (docs/VALIDATION.md)

    @property
    def speakable(self) -> bool:
        """Only validated ranges may be spoken. Unvalidated ones may be shown on screen, marked "est."."""
        return self.meters is not None and self.validated


NO_DEPTH = DepthEstimate()


@dataclass(frozen=True)
class SpatialObject:
    """A detection placed in the image: which side it is on and how much it matters."""

    detection: Detection
    region: Region
    center_x: float               # box centre across the frame: 0 = left edge, 1 = right edge
    in_corridor: bool             # the box overlaps the central band
    view_height: float            # box height / frame height. A size cue only, NOT a distance.
    severity: Severity
    depth: DepthEstimate = NO_DEPTH


@dataclass(frozen=True)
class GuideAlert:
    """One spoken (or, when muted, displayed) announcement and why it fired."""

    timestamp: float              # perf_counter of the frame that triggered it
    frame_id: int
    label: str                    # spoken name: "chair", "person", or "obstacle" for an uncertain detection
    count: int                    # how many of them in that region
    region: Region
    severity: Severity
    confidence: float             # highest detector score in the group
    text: str                     # exactly what is spoken, e.g. "Chair ahead."
    reason: str                   # why it fired, for the event log and validation
    track_id: int | None = None   # set when count == 1 and the tracker gave an ID
    meters: float | None = None   # only when the range is speakable (validated depth)
    depth_source: str = "none"
    depth_confidence: float = 0.0
    latency_ms: float = 0.0       # frame capture -> alert decided (speech starts after this)
    source: str = ""
    caution: bool = False         # a danger warning ("Caution, chair ahead."): spoken first; the phone vibrates
    schema: str = SCHEMA_VERSION
    type: str = "alert"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (enums as strings), with schema and type first."""
        data = asdict(self)
        data["region"] = self.region.value
        data["severity"] = self.severity.value
        return {"schema": data.pop("schema"), "type": data.pop("type"), **data}


@dataclass
class GuideUpdate:
    """What a GuideSink receives once per processed frame."""

    frame: DetectionFrame
    objects: tuple[SpatialObject, ...]
    alerts: tuple[GuideAlert, ...]  # decided on this frame (usually empty)
    image: np.ndarray | None        # BGR frame, already rotated. Sinks must not save it.
    fps: float
    latency_ms: float               # frame capture -> this update
    depth_status: str               # e.g. "Distance: off (camera only)"
    stale: bool = False             # True when the frame was too old to announce anything


class GuideSink(Protocol):
    """A user interface for the guide: the OpenCV debug view today, a local web page later."""

    def publish(self, update: GuideUpdate) -> None: ...

    def should_stop(self) -> bool: ...

    def close(self) -> None: ...
