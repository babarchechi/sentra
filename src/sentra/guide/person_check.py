"""Person check: only announce a "person" when the pose model sees a body, not just a hand or an arm.

The object model learned "person" from photos where any visible part of someone counts, so the camera holder's own
hands and arms come up as "person" with scores as high as a real person's (0.5-0.87 in our runs). A score threshold
can't separate them. The pose model (yolo26n-pose.pt, the one fall detection uses) finds body keypoints, so:

    a "person" box is CONFIRMED if a pose detection overlapping it has at least `min_body_keypoints` confident
    keypoints on the head, shoulders, hips or legs. Elbows and wrists don't count: a hand or forearm alone isn't
    someone standing there.

An unconfirmed "person" box is relabelled UNCONFIRMED_PERSON ("unconfirmed person"). The alert policy never names
it (sentra.guide.alerts.OBSTACLE_ONLY_LABELS): off to the side it says nothing, which silences the holder's hands,
and when it is large and in the middle it becomes the generic "Possible obstacle ahead.". That matters because a
real person right in front of the camera can fill the frame with only their chest and waist visible, and the pose
model may find no head, shoulder, hip or leg keypoints then (babar's crops of bus.jpg/zidane.jpg in #12).
The debug view draws these boxes grey.

Cost: the pose model is about as slow as the object model (~30 ms on CPU), so it only runs when a person needs a
verdict. Verdicts are reused for a while (confirmed for 2 s, rejected for 0.5 s, so a person who walks in
hand-first is confirmed once the body appears): per track ID, or without tracking, for a box overlapping a recently
checked one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from statistics import median
from typing import Any

import numpy as np

from sentra.guide.schema import Box, Detection, DetectionFrame

DEFAULT_POSE_MODEL = "yolo26n-pose.pt"
PERSON = "person"
UNCONFIRMED_PERSON = "unconfirmed person"  # the label an unconfirmed "person" box gets

# COCO keypoint order (sentra.events.KEYPOINT_NAMES). Elbows (7, 8) and wrists (9, 10) are left out on purpose.
BODY_KEYPOINTS = (0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16)  # head, shoulders, hips, knees, ankles


@dataclass(frozen=True)
class PersonCheckConfig:
    min_keypoint_conf: float = 0.5   # a keypoint counts as seen at this confidence
    min_body_keypoints: int = 2      # head/shoulder/hip/leg keypoints needed to confirm a person
    match_iou: float = 0.3           # a pose detection belongs to a "person" box if they overlap this much
    pose_conf: float = 0.25          # the pose model's own detection threshold
    confirmed_ttl_s: float = 2.0     # re-check a confirmed track after this long (IDs can jump between people)
    rejected_ttl_s: float = 0.5      # re-check a rejected track this soon (someone may walk in hand-first)
    forget_after_s: float = 5.0      # drop verdicts for tracks not seen this long
    reuse_iou: float = 0.5           # without tracking: reuse the verdict of a recently checked box this similar


@dataclass
class _Verdict:
    confirmed: bool
    checked_at: float
    seen_at: float


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pose_bodies(result: Any) -> list[tuple[Box, np.ndarray]]:
    """(box, keypoints (17, 3): x, y, confidence) for every person the pose model found in one Results object."""
    boxes, keypoints = result.boxes, result.keypoints
    if boxes is None or keypoints is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    kpts = keypoints.data.cpu().numpy().astype(np.float32)
    if kpts.shape[-1] == 2:  # no keypoint confidences: treat every keypoint as seen
        kpts = np.concatenate([kpts, np.ones((*kpts.shape[:-1], 1), np.float32)], axis=-1)
    return [((float(b[0]), float(b[1]), float(b[2]), float(b[3])), kpts[i]) for i, b in enumerate(xyxy)]


class PersonCheck:
    """Relabels "person" detections the pose model can't confirm as a body. `model` lets tests pass a fake."""

    def __init__(
        self,
        model_path: str = DEFAULT_POSE_MODEL,
        *,
        model: Any | None = None,
        imgsz: int = 640,
        config: PersonCheckConfig | None = None,
        clock=time.perf_counter,
    ) -> None:
        self.model_path = model_path
        self.model = model
        self.imgsz = imgsz
        self.config = config or PersonCheckConfig()
        self._clock = clock
        self._verdicts: dict[int, _Verdict] = {}                 # by track ID
        self._untracked: list[tuple[Box, _Verdict]] = []         # by box, for detections without a track ID
        self.checks = 0           # pose model runs
        self.rejected = 0         # "person" detections relabelled as unconfirmed
        self.check_ms: list[float] = []

    # Model ------------------------------------------------------------------------------------------------

    def load(self) -> Any:
        if self.model is None:
            from ultralytics import YOLO  # imported here so importing this module stays fast

            model = YOLO(self.model_path)
            if model.task != "pose":
                raise ValueError(f"{self.model_path} is a '{model.task}' model; the person check needs a pose "
                                 f"model such as {DEFAULT_POSE_MODEL}")
            self.model = model
        return self.model

    def warmup(self) -> None:
        self.load().predict(np.zeros((self.imgsz, self.imgsz, 3), np.uint8), imgsz=self.imgsz, verbose=False)

    @property
    def median_check_ms(self) -> float:
        return median(self.check_ms[-200:]) if self.check_ms else 0.0

    # Checking ---------------------------------------------------------------------------------------------

    def is_body(self, keypoints: np.ndarray) -> bool:
        cfg = self.config
        seen = keypoints[list(BODY_KEYPOINTS), 2] >= cfg.min_keypoint_conf
        return int(seen.sum()) >= cfg.min_body_keypoints

    def filter(self, image: np.ndarray, frame: DetectionFrame) -> DetectionFrame:
        """The same frame, with "person" detections that aren't a confirmed body relabelled UNCONFIRMED_PERSON."""
        people = [d for d in frame.detections if d.label == PERSON]
        if not people:
            self._forget(self._clock())
            return frame
        now = self._clock()
        verdicts: dict[int, bool] = {}  # index in frame.detections -> confirmed
        to_check: list[int] = []
        for i, det in enumerate(frame.detections):
            if det.label != PERSON:
                continue
            cached = self._cached(det, now)
            if cached is None:
                to_check.append(i)
            else:
                verdicts[i] = cached

        if to_check:
            bodies = self._run_pose(image)
            for i in to_check:
                det = frame.detections[i]
                confirmed = self._matches_body(det, bodies)
                verdicts[i] = confirmed
                if det.track_id is not None:
                    self._verdicts[det.track_id] = _Verdict(confirmed, now, now)
                else:
                    self._untracked.append((det.box, _Verdict(confirmed, now, now)))

        out, changed = [], False
        for i, det in enumerate(frame.detections):
            if det.label == PERSON and not verdicts[i]:
                self.rejected += 1
                det, changed = replace(det, label=UNCONFIRMED_PERSON), True
            out.append(det)
        self._forget(now)
        return replace(frame, detections=tuple(out)) if changed else frame

    def reset(self) -> None:
        """Forget all verdicts, e.g. after switching cameras (track IDs start again)."""
        self._verdicts.clear()
        self._untracked.clear()

    # Internals --------------------------------------------------------------------------------------------

    def _ttl(self, verdict: _Verdict) -> float:
        return self.config.confirmed_ttl_s if verdict.confirmed else self.config.rejected_ttl_s

    def _cached(self, det: Detection, now: float) -> bool | None:
        if det.track_id is None:
            # No track ID: reuse the newest still-valid verdict for a box in nearly the same place.
            best: _Verdict | None = None
            for box, verdict in self._untracked:
                if (now - verdict.checked_at < self._ttl(verdict) and _iou(det.box, box) >= self.config.reuse_iou
                        and (best is None or verdict.checked_at > best.checked_at)):
                    best = verdict
            return None if best is None else best.confirmed
        verdict = self._verdicts.get(det.track_id)
        if verdict is None:
            return None
        verdict.seen_at = now
        return verdict.confirmed if now - verdict.checked_at < self._ttl(verdict) else None

    def _run_pose(self, image: np.ndarray) -> list[tuple[Box, np.ndarray]]:
        model = self.load()
        start = time.perf_counter()
        result = model.predict(image, imgsz=self.imgsz, conf=self.config.pose_conf, verbose=False)[0]
        self.check_ms.append((time.perf_counter() - start) * 1000.0)
        self.checks += 1
        return pose_bodies(result)

    def _matches_body(self, det: Detection, bodies: list[tuple[Box, np.ndarray]]) -> bool:
        best_iou, best_kpts = 0.0, None
        for box, kpts in bodies:
            iou = _iou(det.box, box)
            if iou > best_iou:
                best_iou, best_kpts = iou, kpts
        return best_kpts is not None and best_iou >= self.config.match_iou and self.is_body(best_kpts)

    def _forget(self, now: float) -> None:
        old = [tid for tid, v in self._verdicts.items() if now - v.seen_at > self.config.forget_after_s]
        for tid in old:
            del self._verdicts[tid]
        self._untracked = [(box, v) for box, v in self._untracked if now - v.checked_at < self._ttl(v)]
