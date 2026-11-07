"""Object detection with YOLO26n (yolo26n.pt) and optional ByteTrack.

The model knows the 80 COCO classes (person, chair, couch, dining table, bicycle, ...). It does not know
glass doors, curbs, steps, drop-offs, poles or low-hanging objects, so a missing alert never means
nothing is there.
"""
from __future__ import annotations

import time
from collections import deque
from statistics import median
from typing import Any

import numpy as np

from sentra.guide.schema import Detection, DetectionFrame

DEFAULT_DETECT_MODEL = "yolo26n.pt"
DEFAULT_TRACKER = "bytetrack.yaml"


def detection_frame_from_result(
    result: Any, frame_id: int, timestamp: float, source: str = "", inference_ms: float = 0.0
) -> DetectionFrame:
    """Convert one Ultralytics Results object into a DetectionFrame.

    track_id is None for boxes the tracker hasn't confirmed yet, and for every box when tracking is off.
    """
    height, width = result.orig_shape[:2]
    boxes = result.boxes
    detections: tuple[Detection, ...] = ()
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
        names = result.names
        detections = tuple(
            Detection(
                label=str(names[int(cls[i])]),
                class_id=int(cls[i]),
                confidence=float(conf[i]),
                box=(float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])),
                track_id=None if ids[i] is None else int(ids[i]),
            )
            for i in range(len(xyxy))
        )
    return DetectionFrame(
        frame_id=frame_id,
        timestamp=timestamp,
        frame_width=int(width),
        frame_height=int(height),
        detections=detections,
        source=source,
        inference_ms=inference_ms,
    )


class ObjectDetector:
    """Wraps a YOLO detection model. `model` lets tests pass a fake instead of loading weights."""

    def __init__(
        self,
        model_path: str = DEFAULT_DETECT_MODEL,
        *,
        model: Any | None = None,
        conf: float = 0.25,
        imgsz: int = 640,
        track: bool = True,
        tracker: str = DEFAULT_TRACKER,
    ) -> None:
        self.model_path = model_path
        self.model = model
        self.conf = conf
        self.imgsz = imgsz
        self.track = track
        self.tracker = tracker
        self._inference_ms: deque[float] = deque(maxlen=200)

    def load(self) -> Any:
        """Load the weights (Ultralytics downloads yolo26n.pt on first use) and check it's a detector."""
        if self.model is None:
            from ultralytics import YOLO  # imported here so importing this module stays fast

            model = YOLO(self.model_path)
            if model.task != "detect":
                raise ValueError(
                    f"{self.model_path} is a '{model.task}' model; Sentra Guide needs a detection model "
                    f"such as {DEFAULT_DETECT_MODEL}"
                )
            self.model = model
        return self.model

    def warmup(self) -> None:
        """Run once on a blank image so the first camera frame isn't slowed by model start-up (~2 s on CPU)."""
        self.load().predict(np.zeros((self.imgsz, self.imgsz, 3), np.uint8), imgsz=self.imgsz, verbose=False)

    def detect(self, image: np.ndarray, frame_id: int, timestamp: float, source: str = "") -> DetectionFrame:
        model = self.load()
        start = time.perf_counter()
        options = {"conf": self.conf, "imgsz": self.imgsz, "verbose": False}
        if self.track:
            results = model.track(image, persist=True, tracker=self.tracker, **options)
        else:
            results = model.predict(image, **options)
        inference_ms = (time.perf_counter() - start) * 1000.0
        self._inference_ms.append(inference_ms)
        return detection_frame_from_result(results[0], frame_id, timestamp, source, inference_ms)

    def reset_tracking(self) -> None:
        """Forget every track and restart IDs from 1, e.g. after switching cameras (old IDs mean nothing there)."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()

    @property
    def median_inference_ms(self) -> float:
        """Median over the last 200 frames (0.0 before the first frame)."""
        return median(self._inference_ms) if self._inference_ms else 0.0
