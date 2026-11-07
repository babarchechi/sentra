"""Person check: "person" boxes are announced only when the pose model sees a body. Owner: iesa242."""
import numpy as np
import pytest
import torch
from guide_helpers import FakeClock, FakeModel, FakeSource, RecordingSpeaker

from sentra.events import KEYPOINT_NAMES
from sentra.guide.detector import ObjectDetector
from sentra.guide.person_check import PersonCheck, PersonCheckConfig, pose_bodies
from sentra.guide.pipeline import GuidePipeline
from sentra.guide.schema import Detection, DetectionFrame
from sentra.guide.view import HeadlessSink

IMAGE = np.zeros((480, 640, 3), np.uint8)
PERSON_BOX = (200.0, 40.0, 440.0, 470.0)
HAND_BOX = (500.0, 300.0, 630.0, 470.0)


def keypoints(*names, conf=0.9):
    """(17, 3) keypoints where only `names` are confidently seen."""
    kpts = np.zeros((17, 3), np.float32)
    for name in names:
        kpts[KEYPOINT_NAMES.index(name)] = (1.0, 1.0, conf)
    return kpts


FACE_AND_SHOULDERS = keypoints("nose", "left_eye", "right_eye", "left_shoulder", "right_shoulder")
LEGS_ONLY = keypoints("left_hip", "left_knee", "left_ankle")
HAND_ONLY = keypoints("right_wrist", "right_elbow")


class FakePoseBoxes:
    def __init__(self, boxes):
        self.xyxy = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)

    def __len__(self):
        return len(self.xyxy)


class FakePoseResult:
    def __init__(self, bodies):
        self.boxes = FakePoseBoxes([b for b, _ in bodies])
        data = torch.tensor(np.array([k for _, k in bodies]), dtype=torch.float32) if bodies else torch.zeros(0, 17, 3)
        self.keypoints = type("Keypoints", (), {"data": data})()


class FakePoseModel:
    task = "pose"

    def __init__(self, bodies):
        self.bodies = bodies  # [(box, keypoints)] found in every frame
        self.calls = 0

    def predict(self, image, **kwargs):
        self.calls += 1
        return [FakePoseResult(self.bodies)]


def person(box=PERSON_BOX, track_id=None, conf=0.85):
    return Detection("person", 0, conf, box, track_id)


def chair():
    return Detection("chair", 56, 0.8, (10.0, 200.0, 120.0, 460.0))


def frame(*detections, frame_id=1):
    return DetectionFrame(frame_id, 0.0, 640, 480, tuple(detections))


def labels(f):
    return [(d.label, d.box) for d in f.detections]


# Verdicts ----------------------------------------------------------------------------------------------

def test_hands_and_arms_are_dropped_and_real_bodies_kept():
    pose = FakePoseModel([(PERSON_BOX, FACE_AND_SHOULDERS), (HAND_BOX, HAND_ONLY)])
    check = PersonCheck(model=pose)
    out = check.filter(IMAGE, frame(person(), person(HAND_BOX), chair()))
    assert labels(out) == [("person", PERSON_BOX), ("chair", (10.0, 200.0, 120.0, 460.0))]
    assert check.rejected == 1 and check.checks == 1


def test_legs_alone_are_a_body_too():
    # A camera pointed low sees someone's legs: that person is still in the way.
    check = PersonCheck(model=FakePoseModel([(PERSON_BOX, LEGS_ONLY)]))
    assert labels(check.filter(IMAGE, frame(person()))) == [("person", PERSON_BOX)]


def test_a_person_box_with_no_matching_pose_is_dropped():
    far_away = (0.0, 0.0, 50.0, 50.0)
    check = PersonCheck(model=FakePoseModel([(far_away, FACE_AND_SHOULDERS)]))
    assert check.filter(IMAGE, frame(person())).detections == ()
    assert PersonCheck(model=FakePoseModel([])).filter(IMAGE, frame(person())).detections == ()


def test_one_faint_keypoint_is_not_enough():
    faint = keypoints("nose", "left_shoulder", conf=0.3)
    single = keypoints("nose")
    for kpts in (faint, single):
        check = PersonCheck(model=FakePoseModel([(PERSON_BOX, kpts)]))
        assert check.filter(IMAGE, frame(person())).detections == ()
    lenient = PersonCheck(model=FakePoseModel([(PERSON_BOX, single)]), config=PersonCheckConfig(min_body_keypoints=1))
    assert len(lenient.filter(IMAGE, frame(person())).detections) == 1


def test_frames_without_people_never_run_the_pose_model():
    pose = FakePoseModel([])
    check = PersonCheck(model=pose)
    f = frame(chair())
    assert check.filter(IMAGE, f) is f and pose.calls == 0


# Caching per track --------------------------------------------------------------------------------------

def test_a_confirmed_track_is_rechecked_only_every_two_seconds():
    clock = FakeClock()
    pose = FakePoseModel([(PERSON_BOX, FACE_AND_SHOULDERS)])
    check = PersonCheck(model=pose, clock=clock)
    for i in range(20):  # 1.9 s at 10 fps
        clock.now = i * 0.1
        assert len(check.filter(IMAGE, frame(person(track_id=7))).detections) == 1
    assert pose.calls == 1
    clock.now = 2.05
    check.filter(IMAGE, frame(person(track_id=7)))
    assert pose.calls == 2


def test_a_rejected_track_is_rechecked_soon_so_someone_walking_in_hand_first_is_found():
    clock = FakeClock()
    pose = FakePoseModel([(PERSON_BOX, HAND_ONLY)])
    check = PersonCheck(model=pose, clock=clock)
    assert check.filter(IMAGE, frame(person(track_id=3))).detections == ()
    clock.now = 0.3
    assert check.filter(IMAGE, frame(person(track_id=3))).detections == () and pose.calls == 1  # cached
    pose.bodies = [(PERSON_BOX, FACE_AND_SHOULDERS)]  # the rest of them comes into view
    clock.now = 0.6
    assert len(check.filter(IMAGE, frame(person(track_id=3))).detections) == 1 and pose.calls == 2


def test_without_tracking_every_frame_with_a_person_is_checked():
    pose = FakePoseModel([(PERSON_BOX, FACE_AND_SHOULDERS)])
    check = PersonCheck(model=pose)
    for _ in range(3):
        check.filter(IMAGE, frame(person()))
    assert pose.calls == 3


def test_reset_and_forgetting_old_tracks():
    clock = FakeClock()
    pose = FakePoseModel([(PERSON_BOX, FACE_AND_SHOULDERS)])
    check = PersonCheck(model=pose, clock=clock)
    check.filter(IMAGE, frame(person(track_id=1)))
    check.reset()  # e.g. after switching cameras
    check.filter(IMAGE, frame(person(track_id=1)))
    assert pose.calls == 2
    clock.now = 10.0
    check.filter(IMAGE, frame(chair()))  # track 1 not seen for 10 s
    assert check._verdicts == {}


# Model handling ----------------------------------------------------------------------------------------

def test_pose_bodies_reads_results_and_handles_missing_confidences():
    assert pose_bodies(FakePoseResult([])) == []
    result = FakePoseResult([(PERSON_BOX, FACE_AND_SHOULDERS)])
    result.keypoints.data = result.keypoints.data[..., :2]  # a model without keypoint confidences
    ((box, kpts),) = pose_bodies(result)
    assert box == PERSON_BOX and kpts.shape == (17, 3) and (kpts[:, 2] == 1.0).all()


def test_load_rejects_a_detection_model(monkeypatch):
    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", lambda path: type("M", (), {"task": "detect"})())
    with pytest.raises(ValueError, match="pose model"):
        PersonCheck("yolo26n.pt").load()


# Real models (skipped until yolo26n.pt and yolo26n-pose.pt have been downloaded once) ---------------------

from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = [REPO_ROOT / "yolo26n.pt", REPO_ROOT / "yolo26n-pose.pt"]


@pytest.mark.skipif(not all(w.exists() for w in WEIGHTS), reason="model weights not downloaded yet")
def test_real_people_in_sample_photos_are_kept():
    import cv2
    from ultralytics.utils import ASSETS

    image = cv2.imread(str(ASSETS / "bus.jpg"))
    detected = ObjectDetector(str(WEIGHTS[0]), track=False).detect(image, 1, 0.0)
    kept = PersonCheck(str(WEIGHTS[1])).filter(image, detected)
    full_bodies = [d for d in detected.detections if d.label == "person" and d.box[0] > 30]
    assert len(full_bodies) >= 3 and all(d in kept.detections for d in full_bodies)
    assert any(d.label == "bus" for d in kept.detections)


# In the pipeline -----------------------------------------------------------------------------------------

PERSON_AHEAD = [(0, 0.85, 200, 40, 440, 470)]  # in the middle of a 640x480 frame


@pytest.mark.parametrize("body, said", [(HAND_ONLY, []), (FACE_AND_SHOULDERS, ["Person ahead."])])
def test_pipeline_announces_only_confirmed_people(tmp_path, body, said):
    speaker = RecordingSpeaker()
    check = PersonCheck(model=FakePoseModel([(PERSON_BOX, body)]))
    pipeline = GuidePipeline(FakeSource(8), ObjectDetector(model=FakeModel(PERSON_AHEAD)), HeadlessSink(),
                             speaker=speaker, output_dir=tmp_path, person_check=check)
    stats = pipeline.run()
    assert speaker.said == said
    summary = stats.as_dict()
    assert summary["person_checks"] >= 1 and summary["median_person_check_ms"] is not None
    if not said:
        assert summary["unconfirmed_people"] == 8 and "ignored 8 hand/arm/partial" in stats.summary()
    else:
        assert summary["unconfirmed_people"] == 0


def test_pipeline_without_a_person_check_is_unchanged(tmp_path):
    stats = GuidePipeline(FakeSource(3), ObjectDetector(model=FakeModel(PERSON_AHEAD)), HeadlessSink(),
                          output_dir=tmp_path).run()
    assert "person_checks" not in stats.as_dict() and "person check" not in stats.summary()
