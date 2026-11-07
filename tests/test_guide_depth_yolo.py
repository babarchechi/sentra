"""Estimated distance (YOLO26n-depth) as a DepthProvider. Owner: babarchechi. No real model: a fake depth map."""
from pathlib import Path

import numpy as np
import pytest
import torch

from guide_helpers import FakeClock
from sentra.guide.alerts import AlertPolicy
from sentra.guide.depth_yolo import YoloDepth
from sentra.guide.schema import Detection, DetectionFrame
from sentra.guide.spatial import SpatialEstimator

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPTH_MODEL = REPO_ROOT / "yolo26n-depth.pt"


class FakeDepthModel:
    """Returns a depth map the same size as the image: 4 m everywhere, 1.5 m in the left half, 2.5 m bottom right."""

    task = "depth"

    def __init__(self):
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append(image.shape)
        h, w = image.shape[:2]
        depth = np.full((h, w), 4.0, np.float32)
        depth[:, : w // 2] = 1.5
        depth[h // 2:, 3 * w // 4:] = 2.5

        class R:
            pass

        r = R()
        r.depth = R()
        r.depth.data = torch.from_numpy(depth)
        return [r]


def frame(*dets, t=0.0, w=1280, h=720):
    return DetectionFrame(1, t, w, h, tuple(dets), source="test")


LEFT_CHAIR = Detection("chair", 56, 0.9, (100, 200, 400, 700), track_id=1)      # left half -> 1.5 m
RIGHT_BOTTOM = Detection("person", 0, 0.9, (1000, 400, 1260, 710), track_id=2)  # bottom right -> 2.5 m


def test_estimates_follow_the_depth_map_and_are_never_validated():
    model = FakeDepthModel()
    provider = YoloDepth(model=model, every=1, max_width=640)
    chair, person = provider.estimate(np.zeros((720, 1280, 3), np.uint8), frame(LEFT_CHAIR, RIGHT_BOTTOM))
    assert chair.meters == pytest.approx(1.5) and person.meters == pytest.approx(2.5)
    assert chair.source == "yolo26_depth" and not chair.validated and not chair.speakable
    assert model.calls == [(360, 640, 3)]  # shrunk to max_width before the model; boxes scaled to match
    assert "not validated" in provider.status()


def test_runs_every_nth_frame_and_reuses_tracked_estimates_briefly():
    model = FakeDepthModel()
    provider = YoloDepth(model=model, every=3, max_age_s=0.5)
    img = np.zeros((720, 1280, 3), np.uint8)
    untracked = Detection("chair", 56, 0.9, (100, 200, 400, 700), track_id=None)
    results = [provider.estimate(img, frame(LEFT_CHAIR, untracked, t=0.1 * i)) for i in range(4)]
    assert len(model.calls) == 2  # frames 1 and 4
    assert results[1][0].meters == pytest.approx(1.5)  # tracked: reused
    assert results[1][1].meters is None               # untracked: unknown between model runs

    provider = YoloDepth(model=FakeDepthModel(), every=10, max_age_s=0.5)
    provider.estimate(img, frame(LEFT_CHAIR, t=0.0))
    assert provider.estimate(img, frame(LEFT_CHAIR, t=0.4))[0].meters == pytest.approx(1.5)
    assert provider.estimate(img, frame(LEFT_CHAIR, t=0.9))[0].meters is None  # too old


def test_no_model_run_without_detections_and_bad_settings_rejected():
    model = FakeDepthModel()
    assert YoloDepth(model=model, every=1).estimate(np.zeros((480, 640, 3), np.uint8), frame()) == []
    assert model.calls == []
    with pytest.raises(ValueError):
        YoloDepth(every=0)


def test_unvalidated_distance_is_shown_but_never_spoken():
    provider = YoloDepth(model=FakeDepthModel(), every=1)
    spatial = SpatialEstimator()
    clock = FakeClock()
    policy = AlertPolicy(clock=clock)
    img = np.zeros((720, 1280, 3), np.uint8)
    chair = Detection("chair", 56, 0.9, (500, 100, 780, 700), track_id=1)  # central
    said = []
    for i in range(5):
        f = frame(chair, t=float(i) * 0.125)
        clock.now = f.timestamp + 0.05
        objs = spatial.locate(f, provider.estimate(img, f))
        assert objs[0].depth.meters is not None
        said += policy.update(objs, f)
    assert said and all("meter" not in a.text for a in said)
    assert said[0].meters is None and "unvalidated distance not spoken" in said[0].reason


def test_rejects_a_model_that_is_not_a_depth_model(monkeypatch):
    class Detect:
        task = "detect"

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", lambda path: Detect())
    with pytest.raises(ValueError, match="depth model"):
        YoloDepth("yolo26n.pt").load()


@pytest.mark.skipif(not DEPTH_MODEL.exists(), reason="yolo26n-depth.pt not downloaded yet")
def test_real_depth_model_gives_plausible_meters_on_sample_image():
    import cv2
    from ultralytics.utils import ASSETS

    image = cv2.imread(str(ASSETS / "bus.jpg"))
    provider = YoloDepth(str(DEPTH_MODEL), every=1)
    h, w = image.shape[:2]
    bus = Detection("bus", 5, 0.9, (0.05 * w, 0.22 * h, 0.95 * w, 0.68 * h))
    (est,) = provider.estimate(image, DetectionFrame(1, 0.0, w, h, (bus,)))
    assert est.meters is not None and 1.0 < est.meters < 30.0  # a street scene, a few meters away


def test_scale_corrects_the_estimate_and_low_percentile_measures_nearest_surfaces():
    # A see-through chair: inside its box, half the pixels are the chair (1 m), half the wall behind it (4 m).
    class SeeThrough(FakeDepthModel):
        def predict(self, image, **kwargs):
            [r] = super().predict(image, **kwargs)
            h, w = image.shape[:2]
            depth = np.full((h, w), 4.0, np.float32)
            depth[:, ::2] = 1.0  # every other column is chair
            r.depth.data = torch.from_numpy(depth)
            return [r]

    img = np.zeros((720, 1280, 3), np.uint8)
    nearest = YoloDepth(model=SeeThrough(), every=1, percentile=20)
    middle = YoloDepth(model=SeeThrough(), every=1, percentile=50)
    assert nearest.estimate(img, frame(LEFT_CHAIR))[0].meters == pytest.approx(1.0)
    assert middle.estimate(img, frame(LEFT_CHAIR))[0].meters > 1.5  # the median is pulled toward the wall
    scaled = YoloDepth(model=FakeDepthModel(), every=1, scale=0.5)
    est = scaled.estimate(img, frame(LEFT_CHAIR))[0]
    assert est.meters == pytest.approx(0.75) and not est.validated and "calibrated" in scaled.status()
    with pytest.raises(ValueError):
        YoloDepth(scale=0)
