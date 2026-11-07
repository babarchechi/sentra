"""Unknown-obstacle layer (floor scan): warns about things YOLO can't name, never claims the way is clear.
Owner: iesa242."""
import threading
import json
import time
from pathlib import Path

import numpy as np
import pytest
from guide_helpers import FakeClock, FakeModel, FakeSource, RecordingSpeaker

from sentra.guide.alerts import AlertConfig, AlertPolicy
from sentra.guide.detector import ObjectDetector
from sentra.guide.obstacles import (CANT_JUDGE, NONE_FOUND, OBSTRUCTION, UNKNOWN_OBSTACLE, ObstacleScanner,
                                    free_space)
from sentra.guide.pipeline import GuideControls, GuidePipeline
from sentra.guide.schema import Detection, DetectionFrame
from sentra.guide.spatial import SpatialEstimator
from sentra.guide.view import GuideView, HeadlessSink

WALL, FLOOR, BOX, POLE, CHAIR = 0, 1, 2, 3, 4
NAMES = {WALL: "wall", FLOOR: "floor", BOX: "box", POLE: "pole", CHAIR: "chair"}
H, W = 48, 64  # class map size; the corridor is columns 22..40


def room(**things):
    """Floor in the lower 60%, wall above; things given as name=(col1, col2, row1) blocks reaching the bottom."""
    classes = np.full((H, W), FLOOR, np.uint8)
    classes[: int(H * 0.4)] = WALL
    for name, (c1, c2, r1) in things.items():
        classes[r1:, c1:c2] = {"box": BOX, "pole": POLE, "chair": CHAIR}[name.rstrip("0123456789")]
    return classes


class FakeSegModel:
    task = "semantic"
    names = NAMES

    def __init__(self, classes, delay=0.0):
        self.classes, self.delay, self.calls = classes, delay, 0

    def predict(self, image, **kwargs):
        self.calls += 1
        time.sleep(self.delay)
        mask = type("Mask", (), {"data": self.classes})()
        return [type("Result", (), {"semantic_mask": mask})()]


def scanner(classes, clock=None, **kwargs):
    return ObstacleScanner("indoor", model=FakeSegModel(classes), background=False, clock=clock or FakeClock(),
                           **kwargs)


def frame(*detections, t=0.0):
    return DetectionFrame(1, t, 640, 480, tuple(detections))


IMAGE = np.zeros((480, 640, 3), np.uint8)


def scan_twice(s, f=None):
    for _ in range(2):  # an obstruction needs 2 of the last 3 scans
        s.submit(IMAGE, 0.0)
        s.augment(f or frame())
    return s.augment(f or frame())


# Free space ---------------------------------------------------------------------------------------------------

def test_free_space_is_how_far_the_floor_reaches_from_the_bottom():
    walkable = np.ones((10, 3), bool)
    walkable[:4, 0] = False        # floor up to row 4 (60% of the height)
    walkable[7:, 1] = False        # blocked right at the bottom
    walkable[5, 2] = False         # a 1-pixel speck doesn't block (min run 2)
    assert free_space(walkable, 2).tolist() == pytest.approx([0.6, 0.0, 1.0])


# Decisions ----------------------------------------------------------------------------------------------------

def test_open_floor_says_nothing_and_never_claims_the_way_is_clear():
    s = scanner(room())
    out = scan_twice(s)
    assert out.detections == () and s.last.status == NONE_FOUND
    assert "not proof the way is clear" in s.status() and "clear" not in s.status().replace("not proof the way is clear", "")


def test_a_box_ahead_that_yolo_cant_name_becomes_an_unknown_obstacle():
    s = scanner(room(box=(26, 36, 30)))
    s.submit(IMAGE, 0.0)
    assert s.augment(frame()).detections == ()  # one scan isn't enough
    out = scan_twice(s)
    (obstacle,) = out.detections
    assert obstacle.label == UNKNOWN_OBSTACLE and obstacle.track_id is None
    x1, y1, x2, y2 = obstacle.box
    assert 250 <= x1 < x2 <= 370 and y2 == 480  # columns 26-36 of 64, scaled to 640 px
    assert s.last.status.startswith(OBSTRUCTION) and "box" in s.last.status


def test_a_thin_pole_is_enough():
    out = scan_twice(scanner(room(pole=(30, 33, 20))))
    assert [d.label for d in out.detections] == [UNKNOWN_OBSTACLE]


def test_no_floor_in_view_means_cant_judge_not_an_obstacle():
    s = scanner(np.full((H, W), WALL, np.uint8))  # e.g. the phone points at a wall or the ceiling
    out = scan_twice(s)
    assert out.detections == () and s.last.status == CANT_JUDGE


def test_something_off_to_the_side_is_not_an_obstruction():
    out = scan_twice(scanner(room(box=(2, 14, 25))))
    assert out.detections == ()


def test_a_thing_yolo_already_names_is_left_to_yolo():
    chair = Detection("chair", 56, 0.85, (260.0, 250.0, 360.0, 478.0))  # stands where the floor stops
    out = scan_twice(scanner(room(chair=(26, 36, 30))), frame(chair))
    assert out.detections == (chair,)  # no second, generic obstacle for the same thing
    unsure = Detection("chair", 56, 0.3, (260.0, 250.0, 360.0, 478.0))
    out = scan_twice(scanner(room(chair=(26, 36, 30))), frame(unsure))
    assert [d.label for d in out.detections] == ["chair", UNKNOWN_OBSTACLE]  # too unsure to explain it


def test_closer_means_bigger_in_view():
    near = scan_twice(scanner(room(box=(26, 36, 40)))).detections[0]   # floor blocked right at the bottom
    s = scanner(room())
    classes = room()
    classes[30:40, 26:36] = BOX  # floor visible below the box: further away
    far = scan_twice(scanner(classes)).detections[0]
    view = lambda d: (d.box[3] - d.box[1]) / 480  # noqa: E731
    assert view(near) == pytest.approx(1.0) and 0.4 <= view(far) < view(near)
    assert s.latest() is None  # never scanned


def test_old_scans_are_not_used():
    clock = FakeClock()
    s = scanner(room(box=(26, 36, 30)), clock=clock)
    scan_twice(s)
    clock.now = 5.0
    assert s.augment(frame()).detections == () and s.last.status == "scan too old"


def test_reset_forgets_scans():
    s = scanner(room(box=(26, 36, 30)))
    scan_twice(s)
    s.reset()
    s.submit(IMAGE, 0.0)
    assert s.augment(frame()).detections == ()  # needs two fresh scans again


def test_background_scanning_never_blocks_the_caller():
    s = ObstacleScanner("indoor", model=FakeSegModel(room(box=(26, 36, 30)), delay=0.2), clock=FakeClock())
    try:
        start = time.perf_counter()
        s.submit(IMAGE, 0.0)
        assert time.perf_counter() - start < 0.05
        deadline = time.perf_counter() + 2.0
        while s.latest() is None and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert s.latest() is not None
    finally:
        s.close()


def test_environments():
    assert ObstacleScanner("outdoor", model=FakeSegModel(room())).model_path == "yolo26n-sem.pt"
    with pytest.raises(ValueError):
        ObstacleScanner("underwater")


# Cues -----------------------------------------------------------------------------------------------------------

def run_pipeline(classes, rows, tmp_path, n=8):
    speaker = RecordingSpeaker()
    obstacles = scanner(classes)
    pipeline = GuidePipeline(FakeSource(n), ObjectDetector(model=FakeModel(rows)), HeadlessSink(), speaker=speaker,
                             output_dir=tmp_path, obstacles=obstacles)
    stats = pipeline.run()
    return speaker.said, stats


def test_a_close_unknown_obstacle_is_a_caution_and_never_named(tmp_path):
    said, stats = run_pipeline(room(box=(26, 36, 30)), [], tmp_path)
    assert said == ["Caution, possible obstacle ahead."]
    summary = stats.as_dict()
    assert summary["obstacle_scans"] == 8 and summary["obstacle_frames"] >= 1
    assert "floor scan ran 8 times" in stats.summary()
    saved = json.loads(next(tmp_path.glob("*.jsonl")).read_text().splitlines()[-1])
    for key in ("obstacle_scans", "median_obstacle_scan_ms", "obstacle_frames"):
        assert saved[key] == summary[key]


def test_a_further_unknown_obstacle_is_a_plain_possible_obstacle(tmp_path):
    classes = room()
    classes[25:35, 26:36] = BOX  # floor reaches 13 of 48 rows (27%) under it: close, but not very close
    said, _ = run_pipeline(classes, [], tmp_path)
    assert said == ["Possible obstacle ahead."]


def test_open_floor_stays_silent(tmp_path):
    said, _ = run_pipeline(room(), [], tmp_path)
    assert said == []


def test_reason_says_it_came_from_the_floor_scan():
    s = scanner(room(box=(26, 36, 30)))
    policy, spatial, clock = AlertPolicy(clock=FakeClock()), SpatialEstimator(), None
    alerts = []
    for i in range(4):
        s.submit(IMAGE, 0.0)
        f = s.augment(DetectionFrame(i + 1, 0.0, 640, 480, ()))
        alerts += policy.update(spatial.locate(f), f)
    assert alerts and "floor blocked close ahead" in alerts[0].reason


def test_developer_view_draws_the_floor_scan(tmp_path):
    updates = []

    class Keep(HeadlessSink):
        def publish(self, update):
            updates.append(update)

    obstacles = scanner(room(box=(26, 36, 30)))
    GuidePipeline(FakeSource(3), ObjectDetector(model=FakeModel([])), Keep(), output_dir=tmp_path,
                  obstacles=obstacles).run()
    view = GuideView(GuideControls(), AlertPolicy(), show_window=False, obstacles=obstacles)
    plain = GuideView(GuideControls(), AlertPolicy(), show_window=False)
    with_scan, without = view.render(updates[-1]), plain.render(updates[-1])
    assert with_scan.shape == without.shape and not np.array_equal(with_scan, without)


# Real model (skipped until the indoor segmentation weights have been downloaded once) ---------------------------

@pytest.mark.parametrize("first", ["blip", "explained", "no_floor"])
def test_confirmation_requires_two_qualifying_unexplained_scans(first):
    classes = room(pole=(30, 31, 20)) if first == "blip" else room(box=(26, 36, 30))
    if first == "no_floor":
        classes[:] = WALL
    s = scanner(classes)
    chair = Detection("chair", 56, 0.85, (260., 250., 360., 478.))
    s.submit(IMAGE, 0.0)
    s.augment(frame(chair) if first == "explained" else frame())
    s.model.classes = room(box=(26, 36, 30))
    s.submit(IMAGE, 0.0)
    for _ in range(10):
        assert not s.augment(frame()).detections  # reusing a scan is not confirmation
    s.submit(IMAGE, 0.0)
    assert s.augment(frame()).detections


@pytest.mark.parametrize("label", ["handbag", "skateboard", "bottle", "laptop", "unconfirmed person"])
def test_unspoken_labels_do_not_suppress_unknown_obstacles(label):
    det = Detection(label, 0, 0.7, (260., 250., 360., 478.))
    out = scan_twice(scanner(room(box=(26, 36, 30))), frame(det))
    assert out.detections[-1].label == UNKNOWN_OBSTACLE


def test_active_policy_controls_which_boxes_explain_obstacles():
    det = Detection("handbag", 0, 0.7, (260., 250., 360., 478.))
    s = scanner(room(box=(26, 36, 30)))
    policy = AlertPolicy(AlertConfig(labels=None))
    for _ in range(3):
        s.submit(IMAGE, 0.0)
        assert s.augment(frame(det), policy=policy).detections == (det,)


def test_reset_discards_a_background_scan_already_running():
    started, release = threading.Event(), threading.Event()

    class BlockingModel(FakeSegModel):
        def predict(self, image, **kwargs):
            started.set()
            assert release.wait(2)
            return super().predict(image, **kwargs)

    s = ObstacleScanner(model=BlockingModel(room(box=(26, 36, 30))), clock=FakeClock())
    try:
        s.submit(IMAGE, 0.0)
        assert started.wait(2)
        s.reset()
        release.set()
        s.close()  # join ensures the in-flight result has attempted to store
        assert s.latest() is None
        assert not s.augment(frame()).detections
    finally:
        release.set()
        s.close()

WEIGHTS = Path(__file__).resolve().parents[1] / "yolo26n-sem-ade20k.pt"


@pytest.mark.skipif(not WEIGHTS.exists(), reason="yolo26n-sem-ade20k.pt not downloaded yet")
def test_real_segmentation_finds_the_street_in_the_sample_photo():
    import cv2
    from ultralytics.utils import ASSETS

    s = ObstacleScanner("indoor", background=False)
    result = s.scan(cv2.imread(str(ASSETS / "bus.jpg")), 0.0)
    assert result.width == 360 and result.walkable_share > 0.15  # road and pavement count as walkable
    assert result.free.shape == (360,)
