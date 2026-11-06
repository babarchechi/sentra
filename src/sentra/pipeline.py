"""Processing pipeline. Track B, owner: babarchechi. See docs/PHASE1_PLAN.md, Section 7.1 step 3.

camera -> YOLO pose + ByteTrack -> PoseFrame -> PostureAnalyzer -> StatusUpdate -> StatusSink
"""
from __future__ import annotations

import csv
import logging
import time
from collections import deque
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from sentra.capture import FrameSource
from sentra.events import PersonPose, PersonStatus, PoseFrame, PostureState, StatusSink, StatusUpdate
from sentra.posture import PostureAnalyzer

DEFAULT_MODEL = "yolo26n-pose.pt"
DEFAULT_TRACKER = "bytetrack.yaml"
DEFAULT_OUTPUT_DIR = Path("outputs")

log = logging.getLogger(__name__)


def pose_frame_from_result(result: Any, timestamp: float) -> PoseFrame:
    """Convert one Ultralytics Results object into a PoseFrame.

    Only people that ByteTrack has given an ID are included. Until it does (boxes.id is None),
    the frame has no people.
    """
    height, width = result.orig_shape[:2]
    boxes, keypoints = result.boxes, result.keypoints
    if boxes is None or keypoints is None or boxes.id is None or len(boxes) == 0:
        return PoseFrame(timestamp=timestamp, frame_width=int(width), frame_height=int(height), people=())

    track_ids = boxes.id.int().cpu().tolist()
    xyxy = boxes.xyxy.cpu().numpy()
    box_conf = boxes.conf.cpu().numpy()
    kpts = keypoints.data.cpu().numpy().astype(np.float32)  # (N, 17, 3): x, y, confidence
    if kpts.shape[-1] == 2:  # model without keypoint confidences: treat every keypoint as visible
        kpts = np.concatenate([kpts, np.ones((*kpts.shape[:-1], 1), np.float32)], axis=-1)

    people = tuple(
        PersonPose(
            track_id=int(track_id),
            keypoints=kpts[i],
            box=(float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])),
            box_confidence=float(box_conf[i]),
        )
        for i, track_id in enumerate(track_ids)
    )
    return PoseFrame(timestamp=timestamp, frame_width=int(width), frame_height=int(height), people=people)


class TransitionLog:
    """Writes state changes, never images, to <output_dir>/transitions_<YYYYmmdd_HHMMSS>.csv."""

    COLUMNS = ("timestamp", "wall_time", "track_id", "from_state", "to_state", "confidence", "reason")

    def __init__(self, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.path = output_dir / f"transitions_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.COLUMNS)
        self._file.flush()
        self._last_state: dict[int, PostureState] = {}

    def record(self, statuses: Iterable[PersonStatus], timestamp: float) -> int:
        """Log every person whose state differs from last time. Returns how many rows were written."""
        written = 0
        for status in statuses:
            previous = self._last_state.get(status.track_id)
            if previous is status.state:
                continue
            self._writer.writerow([
                f"{timestamp:.3f}",
                datetime.now().isoformat(timespec="milliseconds"),
                status.track_id,
                previous.value if previous else "",
                status.state.value,
                f"{status.confidence:.2f}",
                status.reason,
            ])
            self._last_state[status.track_id] = status.state
            written += 1
        if written:
            self._file.flush()
        return written

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


class FpsMeter:
    """Frames per second averaged over the last `window` frames."""

    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self, now: float | None = None) -> float:
        self._times.append(time.perf_counter() if now is None else now)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


class Pipeline:
    """Camera -> YOLO pose + ByteTrack -> PoseFrame -> PostureAnalyzer -> StatusUpdate -> StatusSink."""

    def __init__(
        self,
        source: FrameSource,
        analyzer: PostureAnalyzer,
        sink: StatusSink,
        model_path: str = DEFAULT_MODEL,
        *,
        model: Any | None = None,
        tracker: str = DEFAULT_TRACKER,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ) -> None:
        """`model` is an already-loaded YOLO model; leave it None to load `model_path` (tests pass a fake)."""
        self.source = source
        self.analyzer = analyzer
        self.sink = sink
        self.model_path = model_path
        self.model = model
        self.tracker = tracker
        self.output_dir = output_dir

    def run(self) -> None:
        """Process frames until sink.should_stop(). Ctrl+C (KeyboardInterrupt) also stops it cleanly."""
        model = self.model if self.model is not None else self._load_model()
        transitions = TransitionLog(self.output_dir)
        fps = FpsMeter()
        missed_reads = 0
        log.info("Logging state changes (no images) to %s", transitions.path)
        try:
            while not self.sink.should_stop():
                ok, frame, timestamp = self.source.read(timeout=1.0)
                if not ok:
                    missed_reads += 1
                    if missed_reads == 3:
                        log.warning("Waiting for frames from the camera... (Ctrl+C to quit)")
                    continue
                missed_reads = 0

                result = model.track(frame, persist=True, tracker=self.tracker, verbose=False)[0]
                pose = pose_frame_from_result(result, timestamp)
                statuses = self.analyzer.update(pose)
                transitions.record(statuses, pose.timestamp)
                self.sink.publish(
                    StatusUpdate(
                        pose=pose,
                        statuses=statuses,
                        frame=frame,
                        annotated=result.plot(),
                        fps=fps.tick(),
                    )
                )
        finally:
            transitions.close()
            log.info("State changes saved to %s", transitions.path)

    def _load_model(self) -> Any:
        from ultralytics import YOLO  # imported here so importing this module stays fast

        model = YOLO(self.model_path)
        if model.task != "pose":
            raise ValueError(
                f"{self.model_path} is a '{model.task}' model; Sentra needs a pose model such as {DEFAULT_MODEL}"
            )
        return model
