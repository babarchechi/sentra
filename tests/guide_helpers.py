"""Shared builders and fakes for the Sentra Guide tests (not a test module itself).

Shared by all three owners: tell the team before changing anything here.
"""
import time

import numpy as np
import torch

from sentra.guide.schema import Detection, DetectionFrame

WIDTH, HEIGHT = 1000, 1000


def det(label="chair", cx=0.5, height=0.4, conf=0.8, track_id=None, width=0.1):
    """A detection centred at cx (0..1 across the frame) and `height` tall (share of frame height)."""
    x1, x2 = (cx - width / 2) * WIDTH, (cx + width / 2) * WIDTH
    y2 = 0.9 * HEIGHT
    return Detection(label, 0, conf, (x1, y2 - height * HEIGHT, x2, y2), track_id)


def frame(*detections, frame_id=1, t=100.0):
    return DetectionFrame(frame_id, t, WIDTH, HEIGHT, tuple(detections), source="test")


class FakeClock:
    """A perf_counter stand-in. `lag` is how long after each frame's timestamp it is "processed"."""

    def __init__(self, lag=0.05):
        self.lag = lag
        self.now = 0.0

    def __call__(self):
        return self.now


# Stand-ins for the parts of Ultralytics' Results object the detector reads -------------------------

NAMES = {0: "person", 56: "chair", 41: "cup"}


class FakeBoxes:
    def __init__(self, rows, with_ids):
        # rows: (class_id, conf, x1, y1, x2, y2)
        self.xyxy = torch.tensor([r[2:] for r in rows], dtype=torch.float32).reshape(-1, 4)
        self.conf = torch.tensor([r[1] for r in rows], dtype=torch.float32)
        self.cls = torch.tensor([r[0] for r in rows], dtype=torch.float32)
        self.id = torch.arange(1, len(rows) + 1, dtype=torch.float32) if with_ids and rows else None

    def __len__(self):
        return len(self.xyxy)


class FakeResult:
    def __init__(self, rows, shape=(480, 640), with_ids=True):
        self.orig_shape = shape
        self.boxes = FakeBoxes(rows, with_ids)
        self.names = NAMES


class FakeModel:
    task = "detect"

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def track(self, image, **kwargs):
        self.calls.append(("track", kwargs))
        return [FakeResult(self.rows, image.shape[:2])]

    def predict(self, image, **kwargs):
        self.calls.append(("predict", kwargs))
        return [FakeResult(self.rows, image.shape[:2], with_ids=False)]


class FakeSource:
    """`n` blank frames, each stamped "captured just now" so it is fresh when processed; then ends."""

    description = "fake camera"

    def __init__(self, n, shape=(480, 640, 3)):
        self.remaining = n
        self.shape = shape
        self.ended = False

    def read(self, timeout=1.0):
        if self.remaining <= 0:
            self.ended = True
            return False, None, 0.0
        self.remaining -= 1
        return True, np.zeros(self.shape, np.uint8), time.perf_counter()


class RecordingSpeaker:
    def __init__(self):
        self.said = []

    def say(self, text, created=None):
        self.said.append(text)


CHAIR_AHEAD = [(56, 0.85, 280, 100, 360, 400)]  # centre x = 320 of 640 -> ahead
