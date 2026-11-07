"""Optional distance providers. Camera-only (NoDepth) is the default and the demo must work with it.

Planned adapters, each in its own module so a slow or broken one can't break camera-only mode
(see docs/DEPTH_OPTIONS.md):
- depth_anything.py: Depth Anything V2 Small, metric indoor checkpoint, estimated from the RGB frame (Iesa).
- record3d.py: real LiDAR depth from an iPhone over USB with the Record3D app and its Python client (Babar).

A provider returns one DepthEstimate per detection. It sets validated=True only after its ranges were
checked against a tape measure (docs/VALIDATION.md, section 4); until then ranges are shown, never spoken.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from sentra.guide.schema import NO_DEPTH, Box, DepthEstimate, DetectionFrame


class DepthProvider(Protocol):
    name: str

    def estimate(self, image: np.ndarray, frame: DetectionFrame) -> list[DepthEstimate]:
        """One DepthEstimate per frame.detections, in the same order."""
        ...

    def status(self) -> str:
        """Short text for the screen, e.g. "Distance: off (camera only)"."""
        ...

    def close(self) -> None: ...


class NoDepth:
    """Camera-only mode: every distance is unknown and no numbers are ever shown or spoken."""

    name = "none"

    def estimate(self, image: np.ndarray, frame: DetectionFrame) -> list[DepthEstimate]:
        return [NO_DEPTH] * len(frame.detections)

    def status(self) -> str:
        return "Distance: off (camera only)"

    def close(self) -> None:
        pass


def box_depth(
    depth_m: np.ndarray,
    box: Box,
    frame_size: tuple[int, int],
    *,
    confidence: np.ndarray | None = None,
    min_confidence: float = 0.0,
    shrink: float = 0.5,
    min_valid_fraction: float = 0.3,
    percentile: float = 50.0,
) -> tuple[float | None, float]:
    """Robust distance to one detected object from a depth map. Returns (meters or None, valid fraction).

    For adapters to share, so every depth source follows the same rules (docs/DEPTH_OPTIONS.md):
    - `box` is in RGB-frame pixels and `frame_size` is (width, height) of that frame. The box is scaled to the
      depth map's resolution, which is often much lower (Record3D LiDAR depth is 256 x 192). Check alignment
      with a known object first: a crop or rotation difference makes this scaling wrong.
    - Only the central `shrink` share of the box is used, so background around the object counts less.
    - Pixels that are non-finite, <= 0, or below `min_confidence` in `confidence` are ignored.
    - The median of the remaining pixels is returned (or another `percentile`: a low one such as 20 measures the
      nearest surfaces, which suits see-through objects like chairs), or None if fewer than `min_valid_fraction`
      are valid.
    """
    frame_w, frame_h = frame_size
    depth_h, depth_w = depth_m.shape[:2]
    sx, sy = depth_w / max(frame_w, 1), depth_h / max(frame_h, 1)
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w, half_h = (x2 - x1) * shrink / 2.0, (y2 - y1) * shrink / 2.0
    left = int(np.clip(np.floor((cx - half_w) * sx), 0, depth_w - 1))
    right = int(np.clip(np.ceil((cx + half_w) * sx), left + 1, depth_w))
    top = int(np.clip(np.floor((cy - half_h) * sy), 0, depth_h - 1))
    bottom = int(np.clip(np.ceil((cy + half_h) * sy), top + 1, depth_h))

    patch = depth_m[top:bottom, left:right].astype(np.float64)
    valid = np.isfinite(patch) & (patch > 0)
    if confidence is not None:
        valid &= confidence[top:bottom, left:right] >= min_confidence
    fraction = float(valid.mean()) if valid.size else 0.0
    if fraction < min_valid_fraction:
        return None, fraction
    return float(np.percentile(patch[valid], percentile)), fraction
