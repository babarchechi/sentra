"""Wall coverage must keep cautioning even when no floor is visible."""
import numpy as np
import pytest

from guide_helpers import FakeClock
from sentra.guide.alerts import AlertConfig, AlertPolicy
from sentra.guide.obstacles import WALL_OBSTACLE, VIEW_UNCLEAR
from sentra.guide.spatial import SpatialEstimator
from test_guide_obstacles import (H, W, WALL, CEILING, IMAGE, frame, room, scanner, scan_twice, run_pipeline)


def wall():
    return np.full((H, W), WALL, np.uint8)


def test_full_wall_is_confirmed_without_floor():
    s = scanner(wall())
    s.submit(IMAGE, 0.0)
    for _ in range(20):
        assert not s.augment(frame()).detections
    s.submit(IMAGE, 0.0)
    assert s.augment(frame()).detections[0].label == WALL_OBSTACLE
    assert "distance unknown" in s.last.status


@pytest.mark.parametrize("all_labels", [False, True])
def test_wall_cautions_repeat_then_stop_when_view_changes(all_labels):
    clock = FakeClock()
    s = scanner(wall(), clock=clock)
    policy = AlertPolicy(AlertConfig(**({"labels": None} if all_labels else {})), clock=clock)
    spatial = SpatialEstimator()
    alerts = []
    for i in range(32):  # eight seconds of persistent wall; refresh scan every quarter second
        clock.now = i * 0.25
        s.submit(IMAGE, clock.now)
        f = s.augment(frame(t=clock.now), policy=policy)
        alerts.extend(policy.update(spatial.locate(f), f))
    assert len(alerts) == 4
    assert all(a.caution and a.text == "Caution, possible obstacle ahead." for a in alerts)
    assert all("wall fills central view" in a.reason and a.meters is None for a in alerts)
    assert [b.timestamp - a.timestamp for a, b in zip(alerts, alerts[1:])] == [2.0] * 3
    s.model.classes = room()
    for i in range(32, 48):
        clock.now = i * 0.25
        s.submit(IMAGE, clock.now)
        f = s.augment(frame(t=clock.now), policy=policy)
        assert not policy.update(spatial.locate(f), f)


def test_wall_uses_pipeline_urgent_caution(tmp_path):
    said, stats = run_pipeline(wall(), [], tmp_path)
    assert said == ["Caution, possible obstacle ahead."]
    assert stats.obstacle_frames > 0


@pytest.mark.parametrize("scene", ["floor", "ceiling", "side", "upper", "lower"])
def test_wall_requires_broad_coverage_of_both_halves(scene):
    classes = np.full((H, W), CEILING, np.uint8)
    if scene == "floor":
        classes = room()
    elif scene == "side":
        classes[:, :20] = WALL
    elif scene == "upper":
        classes[:H // 2] = WALL
    elif scene == "lower":
        classes[H // 2:] = WALL
    assert not scan_twice(scanner(classes)).detections


def test_obstacle_scan_does_not_count_as_wall_confirmation():
    s = scanner(room(box=(26, 36, 30)))
    scan_twice(s)
    s.model.classes = wall()
    s.submit(IMAGE, 0.0)
    assert not s.augment(frame()).detections
    s.submit(IMAGE, 0.0)
    assert s.augment(frame()).detections[0].label == WALL_OBSTACLE


def test_wall_does_not_confirm_an_unrelated_floor_obstacle():
    s = scanner(wall())
    scan_twice(s)
    s.model.classes = room(box=(26, 36, 30))
    s.submit(IMAGE, 0.0)
    assert not s.augment(frame()).detections


def test_stale_wall_cannot_keep_cautioning_or_confirm_after_gap():
    clock = FakeClock()
    s = scanner(wall(), clock=clock)
    scan_twice(s)
    clock.now = 2.0
    assert not s.augment(frame(t=2.0)).detections
    s.submit(IMAGE, 2.0)
    assert not s.augment(frame(t=2.0)).detections
    s.submit(IMAGE, 2.25)
    clock.now = 2.25
    assert s.augment(frame(t=2.25)).detections


def test_reset_requires_new_wall_confirmation():
    s = scanner(wall())
    scan_twice(s)
    s.reset()
    s.submit(IMAGE, 0.0)
    assert not s.augment(frame()).detections


def test_no_caution_option_is_respected():
    clock = FakeClock()
    s = scanner(wall(), clock=clock)
    policy = AlertPolicy(AlertConfig(caution=False), clock=clock)
    spatial = SpatialEstimator()
    alerts = []
    for _ in range(8):
        s.submit(IMAGE, 0.0)
        f = s.augment(frame(), policy=policy)
        alerts.extend(policy.update(spatial.locate(f), f))
    assert alerts and all(not a.caution for a in alerts)


@pytest.mark.parametrize("colour", [(0, 0, 0), (128, 128, 128), (255, 255, 255), (50, 100, 200)])
def test_uniform_unrecognised_view_keeps_warning_then_recovers(colour):
    clock = FakeClock()
    s = scanner(np.full((H, W), CEILING, np.uint8), clock=clock)
    policy, spatial = AlertPolicy(clock=clock), SpatialEstimator()
    blank = np.full_like(IMAGE, colour)
    alerts = []
    for i in range(24):
        clock.now = i * 0.25
        s.submit(blank, clock.now)
        f = s.augment(frame(t=clock.now))
        if i == 0:
            assert not f.detections
        alerts.extend(policy.update(spatial.locate(f), f))
    assert len(alerts) == 3
    assert all(a.text == "Caution, camera can't see." and a.caution and a.meters is None for a in alerts)
    assert all(a.label == VIEW_UNCLEAR for a in alerts)
    # A textured ceiling is unjudgeable, but no longer a featureless-camera warning.
    clock.now = 6.0
    s.submit(IMAGE, clock.now)
    assert not s.augment(frame(t=clock.now)).detections


def test_single_featureless_scan_cannot_confirm_over_reused_frames():
    s = scanner(np.full((H, W), CEILING, np.uint8))
    s.submit(np.zeros_like(IMAGE), 0.0)
    for _ in range(20):
        assert not s.augment(frame()).detections


@pytest.mark.parametrize("shape", [(480, 640), (640, 480)])
@pytest.mark.parametrize("lighting", ["horizontal", "vertical", "curved"])
def test_smooth_lighting_gradient_repeats_visibility_warning(shape, lighting):
    h, w = shape
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    values = {"horizontal": 100 + 50 * x, "vertical": 100 + 50 * y,
              "curved": 80 + 100 * (x * x + y * y) / 2}[lighting]
    image = np.repeat(values.astype(np.uint8)[:, :, None], 3, axis=2)
    clock = FakeClock()
    s = scanner(np.full((H, W), CEILING, np.uint8), clock=clock)
    policy, spatial, alerts = AlertPolicy(clock=clock), SpatialEstimator(), []
    for i in range(24):
        clock.now = i * 0.25
        s.submit(image, clock.now)
        f = s.augment(frame(t=clock.now))
        if i == 0:
            assert not f.detections
        alerts.extend(policy.update(spatial.locate(f), f))
    assert len(alerts) == 3
    assert all(a.text == "Caution, camera can't see." and a.caution for a in alerts)
    assert all(b.timestamp - a.timestamp == 2.0 for a, b in zip(alerts, alerts[1:]))


@pytest.mark.parametrize("kind", ["noise", "stripes", "colour"])
def test_textured_view_does_not_trigger_visibility_warning(kind):
    if kind == "noise":
        image = np.random.default_rng(0).integers(0, 256, IMAGE.shape, dtype=np.uint8)
    else:
        bands = (np.arange(640) // 8 % 2).astype(bool)
        if kind == "colour":
            row = np.where(bands[:, None], [255, 0, 0], [0, 0, 255]).astype(np.uint8)
        else:
            row = np.repeat((bands * 255).astype(np.uint8)[:, None], 3, axis=1)
        image = np.broadcast_to(row, IMAGE.shape).copy()
    s = scanner(np.full((H, W), CEILING, np.uint8))
    for _ in range(3):
        s.submit(image, 0.0)
        assert not s.latest().featureless
        assert not s.augment(frame()).detections


def test_unclear_view_respects_freshness_and_reset():
    clock = FakeClock()
    s = scanner(np.full((H, W), CEILING, np.uint8), clock=clock)
    for _ in range(2):
        s.submit(np.zeros_like(IMAGE), 0.0)
        s.augment(frame())
    assert s.last.obstacle.label == VIEW_UNCLEAR
    clock.now = 2.0
    assert not s.augment(frame(t=2.0)).detections
    s.reset()
    s.submit(np.zeros_like(IMAGE), 2.0)
    assert not s.augment(frame(t=2.0)).detections
