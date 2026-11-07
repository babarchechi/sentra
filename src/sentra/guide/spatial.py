"""Place detections in the image: left / center / right, central-band overlap and severity.

The central band is a stand-in for "in front of the camera". It is not a measured walking corridor:
the phone may be tilted or pointed away from where the person is going.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sentra.guide.schema import NO_DEPTH, DepthEstimate, DetectionFrame, Region, Severity, SpatialObject


@dataclass(frozen=True)
class SpatialConfig:
    left_edge: float = 0.35         # box centres left of this fraction of the width are LEFT
    right_edge: float = 0.65        # right of this are RIGHT; in between is CENTER ("ahead")
    hysteresis: float = 0.04        # a tracked object near an edge keeps its previous region (no flip-flopping)
    high_view_height: float = 0.5   # central and at least this tall (fraction of frame height) -> HIGH
    forget_after_s: float = 5.0     # drop remembered regions of tracks not seen this long


class SpatialEstimator:
    def __init__(self, config: SpatialConfig | None = None) -> None:
        self.config = config or SpatialConfig()
        if not 0.0 < self.config.left_edge < self.config.right_edge < 1.0:
            raise ValueError("Need 0 < left_edge < right_edge < 1")
        self._last_region: dict[int, tuple[Region, float]] = {}  # track_id -> (region, last seen)

    def region(self, center_x: float, previous: Region | None = None) -> Region:
        """Region for a box centre (0..1). With `previous`, stay put unless clearly past the edge."""
        cfg = self.config
        left, right = cfg.left_edge, cfg.right_edge
        if previous is Region.LEFT:
            left += cfg.hysteresis
        elif previous is Region.RIGHT:
            right -= cfg.hysteresis
        elif previous is Region.CENTER:
            left -= cfg.hysteresis
            right += cfg.hysteresis
        if center_x < left:
            return Region.LEFT
        if center_x > right:
            return Region.RIGHT
        return Region.CENTER

    def locate(
        self, frame: DetectionFrame, depths: Sequence[DepthEstimate] | None = None
    ) -> tuple[SpatialObject, ...]:
        """One SpatialObject per detection, in the same order. `depths` lines up with frame.detections."""
        cfg = self.config
        width, height = max(frame.frame_width, 1), max(frame.frame_height, 1)
        objects = []
        for i, det in enumerate(frame.detections):
            x1, y1, x2, y2 = det.box
            center_x = (x1 + x2) / 2.0 / width
            previous = None
            if det.track_id is not None and det.track_id in self._last_region:
                previous = self._last_region[det.track_id][0]
            region = self.region(center_x, previous)
            if det.track_id is not None:
                self._last_region[det.track_id] = (region, frame.timestamp)

            in_corridor = x1 / width < cfg.right_edge and x2 / width > cfg.left_edge
            view_height = max(0.0, (y2 - y1) / height)
            if region is Region.CENTER:
                severity = Severity.HIGH if view_height >= cfg.high_view_height else Severity.MEDIUM
            else:
                severity = Severity.MEDIUM if in_corridor else Severity.LOW
            depth = depths[i] if depths is not None and i < len(depths) else NO_DEPTH
            objects.append(SpatialObject(det, region, center_x, in_corridor, view_height, severity, depth))

        self._forget(frame.timestamp)
        return tuple(objects)

    def reset(self) -> None:
        """Forget remembered regions, e.g. after switching cameras."""
        self._last_region.clear()

    def _forget(self, now: float) -> None:
        old = [tid for tid, (_, seen) in self._last_region.items() if now - seen > self.config.forget_after_s]
        for tid in old:
            del self._last_region[tid]
