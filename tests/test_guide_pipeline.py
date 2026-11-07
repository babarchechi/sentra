"""Sentra Guide perception and pipeline: detector conversion, pipeline with fakes, event log, real model.

Owner: iesa242. Speech and the debug view are tested in test_guide_speech_view.py (Swechha03)."""
import json
from pathlib import Path

import numpy as np
import pytest

from sentra.guide.detector import ObjectDetector, detection_frame_from_result
from sentra.guide.pipeline import GuideControls, GuidePipeline
from sentra.guide.schema import Region
from sentra.guide.view import HeadlessSink

from guide_helpers import CHAIR_AHEAD, FakeModel, FakeResult, FakeSource, RecordingSpeaker

REPO_ROOT = Path(__file__).resolve().parents[1]
DETECT_MODEL = REPO_ROOT / "yolo26n.pt"

def test_detection_frame_conversion_with_and_without_track_ids():
    rows = [(56, 0.9, 10, 20, 110, 220), (0, 0.6, 500, 0, 600, 480)]
    f = detection_frame_from_result(FakeResult(rows), frame_id=3, timestamp=1.5, source="cam", inference_ms=12.0)
    assert (f.frame_id, f.timestamp, f.frame_width, f.frame_height, f.source) == (3, 1.5, 640, 480, "cam")
    chair, person = f.detections
    assert (chair.label, chair.class_id, chair.track_id) == ("chair", 56, 1)
    assert chair.box == (10.0, 20.0, 110.0, 220.0) and chair.confidence == pytest.approx(0.9)
    assert person.label == "person" and person.track_id == 2
    untracked = detection_frame_from_result(FakeResult(rows, with_ids=False), 1, 0.0)
    assert [d.track_id for d in untracked.detections] == [None, None]
    assert detection_frame_from_result(FakeResult([]), 1, 0.0).detections == ()


def test_detector_uses_tracking_by_default_and_predict_without_it():
    model = FakeModel(CHAIR_AHEAD)
    ObjectDetector(model=model, conf=0.3, imgsz=480).detect(np.zeros((480, 640, 3), np.uint8), 1, 0.0)
    ObjectDetector(model=model, track=False).detect(np.zeros((480, 640, 3), np.uint8), 2, 0.0)
    (kind1, kw1), (kind2, _) = model.calls
    assert kind1 == "track" and kw1["persist"] and kw1["conf"] == 0.3 and kw1["imgsz"] == 480
    assert kind2 == "predict"


def test_pipeline_end_to_end_with_fakes_speaks_logs_and_never_saves_images(tmp_path):
    speaker = RecordingSpeaker()
    sink = HeadlessSink()
    pipeline = GuidePipeline(FakeSource(12), ObjectDetector(model=FakeModel(CHAIR_AHEAD)), sink,
                             speaker=speaker, output_dir=tmp_path, label="unit")
    stats = pipeline.run()
    assert speaker.said == ["Chair ahead."]
    assert stats.frames == 12 and stats.alerts == 1 and stats.stale_frames == 0

    assert pipeline.log_path.name.endswith("_unit.jsonl")
    lines = [json.loads(line) for line in pipeline.log_path.read_text(encoding="utf-8").splitlines()]
    alert, summary = lines
    assert alert["type"] == "alert" and alert["text"] == "Chair ahead." and alert["spoken"] is True
    assert alert["region"] == "center" and alert["meters"] is None and alert["source"] == "fake camera"
    assert summary["type"] == "session_summary" and summary["frames"] == 12 and summary["alerts"] == 1
    assert [p.suffix for p in tmp_path.iterdir()] == [".jsonl"]  # nothing but the log


def test_muted_alerts_are_logged_but_not_spoken(tmp_path):
    speaker = RecordingSpeaker()
    pipeline = GuidePipeline(FakeSource(6), ObjectDetector(model=FakeModel(CHAIR_AHEAD)), HeadlessSink(),
                             speaker=speaker, controls=GuideControls(muted=True), output_dir=tmp_path)
    assert pipeline.run().alerts == 1
    assert speaker.said == []
    assert json.loads(pipeline.log_path.read_text(encoding="utf-8").splitlines()[0])["spoken"] is False


def test_detector_rejects_a_pose_model(monkeypatch):
    class PoseModel:
        task = "pose"

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", lambda path: PoseModel())
    with pytest.raises(ValueError, match="detection model"):
        ObjectDetector("yolo26n-pose.pt").load()


# Real model (skipped until yolo26n.pt has been downloaded once) --------------------------------------

@pytest.mark.skipif(not DETECT_MODEL.exists(), reason="yolo26n.pt not downloaded yet (run python src/guide.py once)")
def test_real_model_finds_people_and_bus_in_sample_image(tmp_path):
    import cv2
    from ultralytics.utils import ASSETS

    from sentra.guide.spatial import SpatialEstimator

    image = cv2.imread(str(ASSETS / "bus.jpg"))
    detector = ObjectDetector(str(DETECT_MODEL), track=False)
    frame = detector.detect(image, 1, 0.0, "bus.jpg")
    labels = [d.label for d in frame.detections]
    assert labels.count("person") >= 3 and "bus" in labels
    regions = {d.label: o.region for d, o in zip(frame.detections, SpatialEstimator().locate(frame))}
    assert regions["bus"] is Region.CENTER
    lefts = [o for o in SpatialEstimator().locate(frame) if o.detection.label == "person" and o.region is Region.LEFT]
    rights = [o for o in SpatialEstimator().locate(frame) if o.detection.label == "person" and o.region is Region.RIGHT]
    assert lefts and rights
