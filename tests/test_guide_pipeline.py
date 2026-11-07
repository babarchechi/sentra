"""Sentra Guide perception and pipeline: detector conversion, pipeline with fakes, event log, real model.

Owner: iesa242. Speech and the debug view are tested in test_guide_speech_view.py (Swechha03)."""
import json
import time
from pathlib import Path

import numpy as np
import pytest

from sentra.guide.alerts import AlertConfig, AlertPolicy
from sentra.guide.cameras import CameraChoice
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


# Stale-frame rule: a frame that is too old by the time it's analysed never triggers a cue ---------------

class KeepUpdates(HeadlessSink):
    def __init__(self):
        super().__init__()
        self.updates = []

    def publish(self, update):
        self.updates.append(update)


class AgedSource(FakeSource):
    """Like FakeSource, but frame i (from 1) was captured `ages[i - 1]` seconds before it's read."""

    def __init__(self, ages):
        super().__init__(len(ages))
        self.ages = list(ages)

    def read(self, timeout=1.0):
        ok, image, timestamp = super().read(timeout)
        if ok:
            timestamp -= self.ages.pop(0)
        return ok, image, timestamp


class SlowOnce(FakeModel):
    """FakeModel that takes `seconds` extra on call number `slow_call` (from 1)."""

    def __init__(self, rows, slow_call, seconds):
        super().__init__(rows)
        self.slow_call, self.seconds = slow_call, seconds

    def track(self, image, **kwargs):
        if len(self.calls) + 1 == self.slow_call:
            time.sleep(self.seconds)
        return super().track(image, **kwargs)


def _alert_frames(sink):
    return [(u.frame.frame_id, a.text) for u in sink.updates for a in u.alerts]


def test_a_frame_that_arrives_late_is_never_announced(tmp_path):
    # Frame 3 is the first with the chair in 3 of the last frames, but it was captured 1.5 s ago (e.g. a stalled
    # stream delivering an old frame). It must not speak; the next fresh frame does.
    speaker, sink = RecordingSpeaker(), KeepUpdates()
    stats = GuidePipeline(AgedSource([0, 0, 1.5, 0, 0, 0]), ObjectDetector(model=FakeModel(CHAIR_AHEAD)), sink,
                          speaker=speaker, output_dir=tmp_path).run()
    assert _alert_frames(sink) == [(4, "Chair ahead.")]
    assert [u.stale for u in sink.updates] == [False, False, True, False, False, False]
    assert speaker.said == ["Chair ahead."] and stats.stale_frames == 1


def test_a_frame_the_model_was_slow_on_is_never_announced(tmp_path):
    # Same, but the frame was fresh when read and the model took too long on it. The age limit is lowered so
    # the test only has to wait 0.15 s.
    policy = AlertPolicy(AlertConfig(max_frame_age_s=0.1))
    sink = KeepUpdates()
    detector = ObjectDetector(model=SlowOnce(CHAIR_AHEAD, slow_call=3, seconds=0.15))
    stats = GuidePipeline(FakeSource(5), detector, sink, policy=policy, output_dir=tmp_path).run()
    assert _alert_frames(sink) == [(4, "Chair ahead.")]
    assert sink.updates[2].stale and stats.stale_frames == 1


def test_frames_that_are_all_late_never_speak(tmp_path):
    speaker = RecordingSpeaker()
    stats = GuidePipeline(AgedSource([1.2] * 8), ObjectDetector(model=FakeModel(CHAIR_AHEAD)), HeadlessSink(),
                          speaker=speaker, output_dir=tmp_path).run()
    assert speaker.said == [] and stats.alerts == 0 and stats.stale_frames == 8


# Camera switching ---------------------------------------------------------------------------------------

class LiveSource:
    """A camera that keeps delivering fresh frames until closed. `source` is what was opened."""

    def __init__(self, source):
        self.source = source
        self.description = f"cam {source}"
        self.closed = False

    def read(self, timeout=1.0):
        if self.closed:
            return False, None, 0.0
        return True, np.zeros((480, 640, 3), np.uint8), time.perf_counter()

    def close(self):
        self.closed = True


class SwitchingSink(HeadlessSink):
    """Asks for `choice` after `switch_after` frames; stops after `frames` frames or `max_idles` idle updates."""

    def __init__(self, controls, choice, switch_after=4, frames=10, max_idles=500):
        super().__init__()
        self.controls, self.choice = controls, choice
        self.switch_after, self.frames, self.max_idles = switch_after, frames, max_idles
        self.sources, self.idles = [], []

    def publish(self, update):
        self.sources.append(update.frame.source)
        if len(self.sources) == self.switch_after:
            self.controls.requested_camera = self.choice

    def idle(self, message):
        self.idles.append(message)

    def should_stop(self):
        return len(self.sources) >= self.frames or len(self.idles) >= self.max_idles


def run_switch(choice, open_source, tmp_path, **sink_options):
    controls = GuideControls(camera_label="Webcam 0", camera_source=0)
    speaker = RecordingSpeaker()
    sink = SwitchingSink(controls, choice, **sink_options)
    first = LiveSource(0)
    pipeline = GuidePipeline(first, ObjectDetector(model=FakeModel(CHAIR_AHEAD)), sink, speaker=speaker,
                             controls=controls, output_dir=tmp_path, open_source=open_source)
    resets = []
    pipeline.detector.reset_tracking = lambda: resets.append(True)
    pipeline.run()
    return pipeline, controls, sink, speaker, first, resets


def test_switching_cameras_closes_the_old_one_and_starts_fresh_on_the_new_one(tmp_path):
    choice = CameraChoice("win:droid", "DroidCam Video", "windows:DroidCam Video")
    pipeline, controls, sink, speaker, first, resets = run_switch(choice, LiveSource, tmp_path)
    assert first.closed and pipeline.source.source == "windows:DroidCam Video"
    assert (controls.camera_label, controls.camera_source) == ("DroidCam Video", "windows:DroidCam Video")
    assert controls.requested_camera is None and not controls.switching
    assert controls.notice == "Now using DroidCam Video" and pipeline.switches == 1
    assert sink.sources[-1] == "cam windows:DroidCam Video"  # frames (and the event log) name the new camera
    assert sink.idles and sink.idles[0] == "Switching to DroidCam Video..."  # the window kept updating
    # Cues start fresh: the chair is announced again on the new camera, inside the old 10 s cooldown.
    assert speaker.said == ["Chair ahead.", "Chair ahead."] and resets == [True]


def test_a_camera_that_wont_open_goes_back_to_the_old_one(tmp_path):
    def open_source(spec):
        if spec == "windows:Broken":
            raise RuntimeError("Cannot open camera source")
        return LiveSource(spec)

    choice = CameraChoice("win:broken", "Broken", "windows:Broken")
    pipeline, controls, sink, _, first, _ = run_switch(choice, open_source, tmp_path)
    assert len(sink.sources) == 10  # frames came back after the failed switch
    assert first.closed and isinstance(pipeline.source, LiveSource) and pipeline.source is not first
    assert pipeline.source.source == 0 and not pipeline.source.closed
    assert (controls.camera_label, controls.camera_source) == ("Webcam 0", 0)
    assert controls.notice == "Couldn't open Broken; back to Webcam 0" and pipeline.switches == 0


def test_if_nothing_reopens_it_waits_for_another_pick_and_keeps_the_window_alive(tmp_path):
    def open_source(spec):
        raise RuntimeError("Cannot open camera source")

    choice = CameraChoice("win:broken", "Broken", "windows:Broken")
    pipeline, controls, sink, _, first, _ = run_switch(choice, open_source, tmp_path, max_idles=20)
    assert pipeline.source is first and first.closed
    assert controls.notice == "Couldn't open Broken. Pick another camera."
    assert len(sink.idles) == 20  # no frames, but the sink kept getting idle updates (no busy spin, no freeze)


def test_picking_the_camera_already_in_use_does_nothing(tmp_path):
    choice = CameraChoice("cv:0", "Webcam 0", 0)
    opened = []
    pipeline, controls, sink, _, first, _ = run_switch(
        choice, lambda spec: opened.append(spec) or LiveSource(spec), tmp_path)
    assert opened == [] and pipeline.source is first and controls.requested_camera is None
    assert sink.idles == [] and set(sink.sources) == {"cam 0"}


def test_without_an_opener_a_switch_is_refused_with_a_message(tmp_path):
    choice = CameraChoice("win:droid", "DroidCam Video", "windows:DroidCam Video")
    pipeline, controls, sink, *_ = run_switch(choice, None, tmp_path)
    assert controls.notice == "Camera switching isn't available in this mode"
    assert set(sink.sources) == {"cam 0"} and controls.camera_source == 0


def test_a_stalled_camera_gives_idle_updates_and_then_says_it_is_waiting(tmp_path, monkeypatch):
    import sentra.guide.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "READ_TIMEOUT_S", 0.02)
    monkeypatch.setattr(pipeline_module, "WAITING_AFTER_S", 0.2)

    class Stalled(LiveSource):
        def read(self, timeout=1.0):
            time.sleep(timeout)
            return False, None, 0.0

    class Sink(HeadlessSink):
        idles = []

        def idle(self, message):
            self.idles.append(message)

        def should_stop(self):
            return "Waiting for the camera..." in self.idles

    sink = Sink()
    GuidePipeline(Stalled(0), ObjectDetector(model=FakeModel([])), sink, output_dir=tmp_path).run()
    assert sink.idles[0] == "" and sink.idles[-1] == "Waiting for the camera..."


def test_detector_reset_tracking_resets_every_tracker():
    class Tracker:
        resets = 0

        def reset(self):
            self.resets += 1

    model = FakeModel([])
    model.predictor = type("Predictor", (), {"trackers": [Tracker(), Tracker()]})()
    ObjectDetector(model=model).reset_tracking()
    assert [t.resets for t in model.predictor.trackers] == [1, 1]
    ObjectDetector(model=FakeModel([])).reset_tracking()  # no tracker yet: nothing to do


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
