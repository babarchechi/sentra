"""Tests for the processing pipeline. Track B, owner: babarchechi."""
import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from sentra.events import PersonStatus, PostureState
from sentra.pipeline import FpsMeter, Pipeline, TransitionLog, pose_frame_from_result
from sentra.posture import PostureAnalyzer

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "yolo26n-pose.pt"


# Minimal stand-ins for the parts of Ultralytics' Results object that the pipeline reads -------------

class FakeBoxes:
    def __init__(self, track_ids, n):
        self.id = None if track_ids is None else torch.tensor(track_ids, dtype=torch.float32)
        self.xyxy = torch.tensor([[10.0 * i, 20.0, 10.0 * i + 50, 140.0] for i in range(n)])
        self.conf = torch.full((n,), 0.9)

    def __len__(self):
        return len(self.xyxy)


class FakeKeypoints:
    def __init__(self, n):
        self.data = torch.rand(n, 17, 3)


class FakeResult:
    def __init__(self, track_ids, n=None, shape=(480, 640)):
        n = len(track_ids) if n is None else n
        self.orig_shape = shape
        self.boxes = FakeBoxes(track_ids, n)
        self.keypoints = FakeKeypoints(n)

    def plot(self):
        return np.zeros((*self.orig_shape, 3), np.uint8)


class FakeModel:
    def __init__(self, track_ids):
        self.track_ids = track_ids
        self.calls = []

    def track(self, frame, **kwargs):
        self.calls.append(kwargs)
        return [FakeResult(self.track_ids, shape=frame.shape[:2])]


class FakeSource:
    def __init__(self, frames, start=100.0, step=0.1):
        self.items = [(frame, start + i * step) for i, frame in enumerate(frames)]

    def read(self, timeout=1.0):
        if not self.items:
            return False, None, 0.0
        frame, timestamp = self.items.pop(0)
        return True, frame, timestamp


class RecordingSink:
    def __init__(self, stop_after):
        self.stop_after = stop_after
        self.updates = []

    def publish(self, update):
        self.updates.append(update)

    def should_stop(self):
        return len(self.updates) >= self.stop_after

    def close(self):
        pass


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


# Tests -----------------------------------------------------------------------------------------------

def test_no_track_ids_means_no_people():
    pose = pose_frame_from_result(FakeResult(None, n=2), timestamp=5.0)
    assert pose.people == ()
    assert (pose.frame_width, pose.frame_height, pose.timestamp) == (640, 480, 5.0)


def test_result_maps_to_people():
    result = FakeResult([4, 9])
    pose = pose_frame_from_result(result, timestamp=1.5)
    assert [p.track_id for p in pose.people] == [4, 9]
    first = pose.people[0]
    assert first.keypoints.shape == (17, 3) and first.keypoints.dtype == np.float32
    assert first.box == (0.0, 20.0, 50.0, 140.0)
    assert first.box_confidence == pytest.approx(0.9)


def test_transition_log_writes_only_changes(tmp_path):
    log = TransitionLog(tmp_path)

    def status(track_id, state):
        return PersonStatus(track_id=track_id, state=state, confidence=0.5, reason="r", since=0.0)

    assert log.record([status(1, PostureState.UNKNOWN)], timestamp=1.0) == 1
    assert log.record([status(1, PostureState.UNKNOWN)], timestamp=1.1) == 0
    assert log.record([status(1, PostureState.UPRIGHT), status(2, PostureState.UNKNOWN)], timestamp=1.2) == 2
    log.close()

    rows = read_csv(log.path)
    assert rows[0] == list(TransitionLog.COLUMNS)
    assert [(r[2], r[3], r[4]) for r in rows[1:]] == [
        ("1", "", "unknown"),
        ("1", "unknown", "upright"),
        ("2", "", "unknown"),
    ]
    assert log.path.name.startswith("transitions_") and log.path.suffix == ".csv"


def test_fps_meter():
    meter = FpsMeter()
    assert meter.tick(0.0) == 0.0
    meter.tick(0.1)
    assert meter.tick(0.2) == pytest.approx(10.0)


def test_pipeline_end_to_end_with_fakes(tmp_path):
    frames = [np.zeros((480, 640, 3), np.uint8) for _ in range(3)]
    model = FakeModel([1, 2])
    sink = RecordingSink(stop_after=3)
    Pipeline(FakeSource(frames), PostureAnalyzer(), sink, model=model, output_dir=tmp_path).run()

    assert len(sink.updates) == 3
    assert all(call["persist"] is True and call["tracker"] == "bytetrack.yaml" for call in model.calls)
    last = sink.updates[-1]
    assert last.pose.timestamp == pytest.approx(100.2)  # capture time from the source, not processing time
    assert [p.track_id for p in last.pose.people] == [1, 2]
    assert {s.track_id for s in last.statuses} == {1, 2}
    assert last.annotated.shape == (480, 640, 3)

    (csv_path,) = tmp_path.glob("transitions_*.csv")
    assert len(read_csv(csv_path)) == 3  # header + the first state of each of the two people


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="yolo26n-pose.pt not downloaded yet (run the camera test once)")
def test_pipeline_with_real_model_on_sample_image(tmp_path):
    import cv2
    from ultralytics.utils import ASSETS

    image = cv2.imread(str(ASSETS / "bus.jpg"))
    sink = RecordingSink(stop_after=3)
    Pipeline(FakeSource([image] * 3), PostureAnalyzer(), sink, model_path=str(MODEL_PATH), output_dir=tmp_path).run()

    last = sink.updates[-1]
    assert len(last.pose.people) >= 1, "ByteTrack should have assigned IDs by the third frame"
    assert all(p.keypoints.shape == (17, 3) for p in last.pose.people)
    assert last.annotated.shape == image.shape
