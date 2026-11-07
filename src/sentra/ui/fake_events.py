"""Fake StatusUpdate generator. Track C, owner: Swechha03. See docs/PHASE1_PLAN.md, Section 8.1 step 1.

Lets the dashboard be built and tested with no camera, model, or analyzer. Two simulated people on a 640x480 canvas:

- **Person #1** stands for 5 s, drops to the floor in 0.5 s, lies there for 5 s, then raises ``POSSIBLE_FALL`` and
  stays down until the alert is answered, then gets back up. Each loop plays the next of three variants, so every
  dashboard path gets exercised:

  0. *Clear fall:* ``ON_GROUND`` while lying, then a confident ``POSSIBLE_FALL``.
  1. *Uncertain fall:* hips and shoulders are hidden after the drop (``UNKNOWN``), then a low-confidence
     ``POSSIBLE_FALL`` (plan Section 6.2, "When the picture is uncertain").
  2. *Gets up alone:* stands back up 4 s into the alert even if nobody answered (the dashboard logs "recovered").

- **Person #2** always stays ``UPRIGHT``, to test multi-person display.

Nothing here touches a camera or the disk.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator

import cv2
import numpy as np

from sentra.events import KP, SKELETON, PersonPose, PersonStatus, PoseFrame, PostureState, StatusUpdate

WIDTH, HEIGHT = 640, 480
FLOOR_Y = 430          # pixel row of the floor; figures rest on it
TORSO_PX = 80.0        # torso length in pixels; a standing figure is ~2.7 torso lengths tall

# Script timings in seconds (plan Section 8.1 step 1).
UPRIGHT_S = 5.0
DROP_S = 0.5
DOWN_S = 5.0
RISE_S = 1.0
GET_UP_ALONE_S = 4.0   # variant 2: gets up this long into the alert, answered or not
MIN_ALERT_S = 1.0      # never get up sooner than this after the alert starts
VARIANTS = 3

LYING_ANGLE_DEG = 88.0
HIDDEN_CONF = 0.15     # keypoint confidence for "can't see it" (below every threshold we use)
_HIDDEN_WHEN_UNCERTAIN = (KP["left_shoulder"], KP["right_shoulder"], KP["left_hip"], KP["right_hip"])

# Standing pose in torso lengths: hip centre at (0, 0), image y pointing down, COCO-17 order.
_STANDING = np.array([
    (0.00, -1.55),                  # nose
    (-0.07, -1.62), (0.07, -1.62),  # eyes
    (-0.14, -1.57), (0.14, -1.57),  # ears
    (-0.25, -1.00), (0.25, -1.00),  # shoulders
    (-0.32, -0.55), (0.32, -0.55),  # elbows
    (-0.35, -0.12), (0.35, -0.12),  # wrists
    (-0.15, 0.00), (0.15, 0.00),    # hips
    (-0.16, 0.55), (0.16, 0.55),    # knees
    (-0.17, 1.10), (0.17, 1.10),    # ankles
], dtype=np.float32)
_FEET = _STANDING[[KP["left_ankle"], KP["right_ankle"]]].mean(axis=0)

# Limbs drawn as the simulated person's "body" in the fake camera frame (no face lines).
_BODY_LIMBS = tuple((a, b) for a, b in SKELETON if a >= KP["left_shoulder"] and b >= KP["left_shoulder"])


def stick_figure(
    feet_x: float,
    angle_deg: float,
    *,
    hidden: tuple[int, ...] = (),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return (17, 3) float32 keypoints for a person tipped `angle_deg` from vertical, resting on the floor.

    0 degrees is standing; ~90 is lying on the floor (they tip over to the left, pivoting on their feet).
    Keypoints listed in `hidden` get a low confidence, as if the model couldn't see them.
    """
    a = math.radians(-angle_deg)  # negative: tip towards the left of the image
    rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=np.float32)
    xy = (_STANDING - _FEET) @ rotation.T * TORSO_PX
    xy[:, 0] += feet_x
    xy[:, 1] += FLOOR_Y - xy[:, 1].max()  # lowest point sits on the floor
    conf = np.full((17, 1), 0.9, dtype=np.float32)
    if rng is not None:
        xy += rng.normal(0.0, 0.6, xy.shape).astype(np.float32)  # small jitter, like a real detector
        conf += rng.uniform(-0.05, 0.05, conf.shape).astype(np.float32)
    conf[list(hidden)] = HIDDEN_CONF
    return np.concatenate([xy, conf], axis=1).astype(np.float32)


def box_around(keypoints: np.ndarray) -> tuple[float, float, float, float]:
    """A person box (x1, y1, x2, y2) around all keypoints, with a margin, clipped to the canvas."""
    xs, ys = keypoints[:, 0], keypoints[:, 1]
    return (
        float(max(xs.min() - 15, 0)),
        float(max(ys.min() - 20, 0)),
        float(min(xs.max() + 15, WIDTH - 1)),
        float(min(ys.max() + 8, HEIGHT - 1)),
    )


def _room() -> np.ndarray:
    """Background of the simulated camera: a wall, a floor and a window, so privacy mode visibly hides it."""
    img = np.full((HEIGHT, WIDTH, 3), (38, 34, 30), np.uint8)
    img[FLOOR_Y:] = (60, 72, 84)
    cv2.rectangle(img, (60, 60), (200, 180), (120, 100, 80), -1)
    cv2.rectangle(img, (60, 60), (200, 180), (170, 160, 150), 3)
    cv2.line(img, (130, 60), (130, 180), (170, 160, 150), 2)
    cv2.putText(img, "SIMULATED CAMERA (ui_demo) - no real video", (10, HEIGHT - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    return img


def _draw_body(img: np.ndarray, kpts: np.ndarray) -> None:
    """The simulated person as the 'camera' would see them: thick limbs and a head."""
    pts = kpts[:, :2].round().astype(int)
    for a, b in _BODY_LIMBS:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (95, 120, 170), 14, cv2.LINE_AA)
    head = tuple(pts[[KP["left_ear"], KP["right_ear"]]].mean(axis=0).round().astype(int))
    cv2.circle(img, head, int(TORSO_PX * 0.22), (120, 150, 200), -1, cv2.LINE_AA)


def _draw_detection(img: np.ndarray, person: PersonPose) -> None:
    """Roughly what Ultralytics' results.plot() adds: skeleton, keypoints, box and an id label."""
    k = person.keypoints
    visible = k[:, 2] >= 0.5
    pts = k[:, :2].round().astype(int)
    for a, b in SKELETON:
        if visible[a] and visible[b]:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), (255, 180, 0), 2, cv2.LINE_AA)
    for i in np.flatnonzero(visible):
        cv2.circle(img, tuple(pts[i]), 4, (0, 255, 255), -1, cv2.LINE_AA)
    x1, y1, x2, y2 = (int(round(v)) for v in person.box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 120, 0), 2)
    cv2.putText(img, f"person id:{person.track_id} {person.box_confidence:.2f}", (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1, cv2.LINE_AA)


class _Faller:
    """Person #1's script: upright -> dropping -> down -> alert -> rising -> upright, then the next variant."""

    def __init__(self, track_id: int, feet_x: float, start: float, auto_get_up_s: float) -> None:
        self.track_id = track_id
        self.feet_x = feet_x
        self.auto_get_up_s = auto_get_up_s
        self.variant = 0
        self.phase = "upright"
        self.phase_start = start
        self.state = PostureState.UPRIGHT
        self.state_since = start

    def step(
        self, now: float, alert_answered: Callable[[int], bool] | None, rng: np.random.Generator
    ) -> tuple[np.ndarray, PersonStatus]:
        self._advance(now, alert_answered)
        elapsed = now - self.phase_start
        uncertain = self.variant == 1
        hidden: tuple[int, ...] = ()

        if self.phase == "upright":
            angle = 3.0 * math.sin(now)
            state, conf, reason = PostureState.UPRIGHT, 0.92, f"torso {abs(angle):.0f}° from vertical"
        elif self.phase == "dropping":
            p = min(elapsed / DROP_S, 1.0)
            angle = LYING_ANGLE_DEG * p * p  # speeds up, like a fall
            state, conf = PostureState.DESCENDING, 0.70
            reason = f"hips dropping {2.0 + 2.0 * p:.1f} torso-lengths/s"
        elif self.phase == "down":
            angle = LYING_ANGLE_DEG
            if uncertain:
                hidden = _HIDDEN_WHEN_UNCERTAIN
                state, conf, reason = PostureState.UNKNOWN, 0.0, "hips and shoulders not visible"
            else:
                state, conf = PostureState.ON_GROUND, 0.75
                reason = f"torso {angle:.0f}° for {elapsed:.1f} s after fast drop"
        elif self.phase == "alert":
            angle = LYING_ANGLE_DEG
            down_for = DOWN_S + elapsed
            state = PostureState.POSSIBLE_FALL
            if uncertain:
                hidden = _HIDDEN_WHEN_UNCERTAIN
                conf, reason = 0.42, f"fast drop, then hips and shoulders hidden for {down_for:.1f} s"
            else:
                conf, reason = 0.86, f"torso {angle:.0f}° for {down_for:.1f} s after fast drop, still"
        else:  # rising: still POSSIBLE_FALL until fully upright, like the analyzer's recovery_s
            p = min(elapsed / RISE_S, 1.0)
            angle = LYING_ANGLE_DEG * (1.0 - p * p * (3 - 2 * p))
            state = PostureState.POSSIBLE_FALL
            conf, reason = (0.42 if uncertain else 0.86), f"torso {angle:.0f}°, getting up"

        if state is not self.state:
            self.state, self.state_since = state, now
        keypoints = stick_figure(self.feet_x, angle, hidden=hidden, rng=rng)
        return keypoints, PersonStatus(self.track_id, state, conf, reason, self.state_since)

    def _advance(self, now: float, alert_answered: Callable[[int], bool] | None) -> None:
        elapsed = now - self.phase_start
        if self.phase == "upright" and elapsed >= UPRIGHT_S:
            self._go("dropping", now)
        elif self.phase == "dropping" and elapsed >= DROP_S:
            self._go("down", now)
        elif self.phase == "down" and elapsed >= DOWN_S:
            self._go("alert", now)
        elif self.phase == "alert" and elapsed >= MIN_ALERT_S:
            if alert_answered is None:
                get_up = elapsed >= self.auto_get_up_s
            else:
                get_up = alert_answered(self.track_id)
            if get_up or (self.variant == 2 and elapsed >= GET_UP_ALONE_S):
                self._go("rising", now)
        elif self.phase == "rising" and elapsed >= RISE_S:
            self._go("upright", now)
            self.variant = (self.variant + 1) % VARIANTS

    def _go(self, phase: str, now: float) -> None:
        self.phase, self.phase_start = phase, now


def fake_updates(
    fps: float = 10.0,
    *,
    alert_answered: Callable[[int], bool] | None = None,
    realtime: bool = True,
    auto_get_up_s: float = 8.0,
) -> Iterator[StatusUpdate]:
    """Yield scripted StatusUpdates forever at roughly `fps`.

    One stick figure stands, falls and lies down on a loop; a second figure always stays upright.

    `alert_answered(track_id)` tells the script whether a human has answered that person's alert; the fallen
    figure stays down until it returns True (pass ``lambda tid: not dashboard.has_open_alert(tid)``). Without it,
    the figure gets up by itself after `auto_get_up_s` seconds of alert.

    `realtime=False` skips sleeping and steps timestamps by exactly 1/fps, for tests.
    """
    rng = np.random.default_rng(7)
    dt = 1.0 / fps
    start = time.perf_counter() if realtime else 1000.0
    faller = _Faller(track_id=1, feet_x=300.0, start=start, auto_get_up_s=auto_get_up_s)
    room = _room()
    previous = start - dt
    measured_fps = fps
    i = 0
    while True:
        if realtime:
            wait = start + i * dt - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            now = time.perf_counter()
        else:
            now = start + i * dt
        if now > previous:
            measured_fps = 0.9 * measured_fps + 0.1 / (now - previous)
        previous = now

        kpts_1, status_1 = faller.step(now, alert_answered, rng)
        sway = now - start
        angle_2 = 2.0 * math.sin(sway * 1.3)
        kpts_2 = stick_figure(520.0 + 18.0 * math.sin(sway * 0.6), angle_2, rng=rng)
        status_2 = PersonStatus(2, PostureState.UPRIGHT, 0.9, f"torso {abs(angle_2):.0f}° from vertical", start)

        people = tuple(
            PersonPose(track_id=tid, keypoints=k, box=box_around(k), box_confidence=0.91)
            for tid, k in ((1, kpts_1), (2, kpts_2))
        )
        frame = room.copy()
        for person in people:
            _draw_body(frame, person.keypoints)
        annotated = frame.copy()
        for person in people:
            _draw_detection(annotated, person)

        yield StatusUpdate(
            pose=PoseFrame(timestamp=now, frame_width=WIDTH, frame_height=HEIGHT, people=people),
            statuses=(status_2, status_1),  # deliberately not in pose.people order: match on track_id
            frame=frame,
            annotated=annotated,
            fps=measured_fps,
        )
        i += 1
