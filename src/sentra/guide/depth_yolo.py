"""Estimated distance from the normal camera picture, with YOLO26n-depth (yolo26n-depth.pt). Owner: babarchechi.

This is NOT LiDAR. One camera can only estimate depth, the way a person guesses distance from a photo. Every
estimate is marked validated=False, so it is shown on screen as "~1.8 m est." and is never spoken, until the
tape-measure test (docs/VALIDATION.md, Section 4) shows it is accurate enough and a reviewed PR changes that.

Speed: about 62 ms per frame on the demo laptop at imgsz 512. To keep the video smooth the model runs on every
`every`-th frame. In between, a tracked object keeps its last estimate for up to `max_age_s`; anything else is
unknown. Frames wider than `max_width` are shrunk first (the estimate doesn't need full resolution).

Each object's distance is the `percentile` (default 20) of the depth inside its box: the nearest surfaces, not the
middle, because the middle of a chair's box is mostly the wall behind it. `scale` corrects the model's overall
scale for one camera (1-point calibration, src/depth_check.py --calibrate); on 2026-11-07 the uncorrected model
read a chair 1.0 m away as 3-6 m.
"""
from __future__ import annotations

import time
from collections import deque
from statistics import median
from typing import Any

import cv2
import numpy as np

from sentra.guide.depth import box_depth
from sentra.guide.schema import NO_DEPTH, DepthEstimate, DetectionFrame

DEFAULT_DEPTH_MODEL = "yolo26n-depth.pt"
SOURCE_NAME = "yolo26_depth"


class YoloDepth:
    """DepthProvider backed by an Ultralytics depth model. `model` lets tests pass a fake."""

    name = SOURCE_NAME

    def __init__(
        self,
        model_path: str = DEFAULT_DEPTH_MODEL,
        *,
        model: Any | None = None,
        imgsz: int = 512,
        every: int = 3,
        max_width: int = 640,
        max_age_s: float = 0.5,
        shrink: float = 0.5,
        percentile: float = 20.0,
        scale: float = 1.0,
    ) -> None:
        if every < 1:
            raise ValueError("every must be at least 1")
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.model_path = model_path
        self.model = model
        self.imgsz = imgsz
        self.every = every
        self.max_width = max_width
        self.max_age_s = max_age_s
        self.shrink = shrink
        self.percentile = percentile
        self.scale = scale
        self._frames = 0
        self._cache: dict[int, tuple[DepthEstimate, float]] = {}  # track_id -> (estimate, frame timestamp)
        self.run_ms: deque[float] = deque(maxlen=100)

    def load(self) -> Any:
        if self.model is None:
            from ultralytics import YOLO  # imported here so importing this module stays fast

            model = YOLO(self.model_path)
            if model.task != "depth":
                raise ValueError(f"{self.model_path} is a '{model.task}' model; need a depth model such as "
                                 f"{DEFAULT_DEPTH_MODEL}")
            self.model = model
        return self.model

    def warmup(self) -> None:
        self.load().predict(np.zeros((480, 640, 3), np.uint8), imgsz=self.imgsz, verbose=False)

    def depth_map(self, image: np.ndarray) -> np.ndarray:
        """Estimated meters per pixel, for `image` shrunk to at most max_width wide."""
        h, w = image.shape[:2]
        if w > self.max_width:
            image = cv2.resize(image, (self.max_width, round(h * self.max_width / w)), interpolation=cv2.INTER_AREA)
        start = time.perf_counter()
        result = self.load().predict(image, imgsz=self.imgsz, verbose=False)[0]
        self.run_ms.append((time.perf_counter() - start) * 1000.0)
        return result.depth.data.cpu().numpy()

    def estimate(self, image: np.ndarray, frame: DetectionFrame) -> list[DepthEstimate]:
        self._frames += 1
        if (self._frames - 1) % self.every == 0:
            return self._measure(image, frame)
        out = []
        for det in frame.detections:
            cached = self._cache.get(det.track_id) if det.track_id is not None else None
            if cached is not None and frame.timestamp - cached[1] <= self.max_age_s:
                out.append(cached[0])
            else:
                out.append(NO_DEPTH)
        return out

    def _measure(self, image: np.ndarray, frame: DetectionFrame) -> list[DepthEstimate]:
        if not frame.detections:
            return []
        depth = self.depth_map(image)
        size = (frame.frame_width, frame.frame_height)
        out = []
        for det in frame.detections:
            meters, valid = self.box_meters(depth, det.box, size)
            est = NO_DEPTH if meters is None else DepthEstimate(round(meters, 2), round(valid, 2), SOURCE_NAME,
                                                                validated=False)
            out.append(est)
            if det.track_id is not None:
                self._cache[det.track_id] = (est, frame.timestamp)
        old = [tid for tid, (_, t) in self._cache.items() if frame.timestamp - t > 5.0]
        for tid in old:
            del self._cache[tid]
        return out

    def box_meters(self, depth: np.ndarray, box, frame_size: tuple[int, int]) -> tuple[float | None, float]:
        """Corrected distance to one box (meters or None) and the share of valid depth pixels."""
        meters, valid = box_depth(depth, box, frame_size, shrink=self.shrink, percentile=self.percentile)
        return (None if meters is None else meters * self.scale), valid

    def status(self) -> str:
        ms = f", {median(self.run_ms):.0f} ms" if self.run_ms else ""
        calibrated = f", x{self.scale:.2f} calibrated" if self.scale != 1.0 else ""
        return f"Distance: estimated, not validated (yolo26n-depth{calibrated}{ms})"

    def close(self) -> None:
        pass
