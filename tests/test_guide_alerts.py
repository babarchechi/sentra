"""Alert policy: persistence, cooldown, priority, wording and honesty rules for Sentra Guide."""
from dataclasses import replace

import pytest

from guide_helpers import FakeClock, det, frame
from sentra.guide.alerts import AlertConfig, AlertPolicy, alert_text, plural
from sentra.guide.schema import DepthEstimate, Region
from sentra.guide.spatial import SpatialEstimator

STEP = 0.125


class Scene:
    """Feeds frames 0.125 s apart (exact in binary floating point) through spatial + policy; collects what is said."""

    def __init__(self, config=None, lag=0.05):
        self.clock = FakeClock()
        # These tests check timing, wording and grouping, including side cues, so they announce side objects too.
        # The default path-only rules (quiet sides and far objects) are tested in test_guide_path.py.
        config = replace(config or AlertConfig(), announce_sides=True)
        self.policy = AlertPolicy(config, clock=self.clock)
        self.spatial = SpatialEstimator()
        self.lag = lag
        self.t = 100.0
        self.frame_id = 0
        self.said = []  # (time, alert)

    def step(self, *detections, depths=None, lag=None):
        self.frame_id += 1
        f = frame(*detections, frame_id=self.frame_id, t=self.t)
        self.clock.now = self.t + (self.lag if lag is None else lag)
        alerts = self.policy.update(self.spatial.locate(f, depths), f)
        self.said += [(self.t, a) for a in alerts]
        self.t += STEP
        return alerts

    def run(self, seconds, *detections, **kwargs):
        for _ in range(round(seconds / STEP)):
            self.step(*detections, **kwargs)

    def texts(self):
        return [a.text for _, a in self.said]


def test_needs_three_frames_before_speaking():
    scene = Scene()
    assert scene.step(det()) == ()
    assert scene.step(det()) == ()
    (alert,) = scene.step(det())
    assert alert.text == "Chair ahead."
    assert alert.region is Region.CENTER and alert.label == "chair"
    assert "3/3 recent frames" in alert.reason


def test_one_frame_flicker_is_never_spoken():
    scene = Scene()
    for _ in range(10):
        scene.step(det())
        scene.step()
        scene.step()
    assert scene.said == []


def test_something_that_stays_in_view_is_not_repeated_until_the_central_reminder():
    scene = Scene(AlertConfig(repeat_after_s=10.0, remind_after_s=30.0))
    scene.run(25, det(cx=0.5), det("person", cx=0.9))   # chair ahead, person to the right, both stay
    assert sorted(scene.texts()) == ["Chair ahead.", "Person to your right."]  # once each in 25 s
    scene.run(15, det(cx=0.5), det("person", cx=0.9))
    reminders = [(t - 100.0, a) for t, a in scene.said[2:]]
    # only the central chair gets a reminder, 30 s after it was first said; the side person never repeats
    assert [a.text for _, a in reminders] == ["Chair ahead."]
    assert reminders[0][0] == 30.25 and "reminder" in reminders[0][1].reason


def test_no_reminders_when_turned_off():
    scene = Scene(AlertConfig(remind_after_s=None))
    scene.run(70, det())
    assert scene.texts() == ["Chair ahead."]


def test_said_again_when_it_comes_back_after_being_out_of_view():
    scene = Scene(AlertConfig(repeat_after_s=10.0, forget_after_s=3.0, remind_after_s=None))
    scene.run(5, det())
    scene.run(1, )                 # gone for 1 s: a flicker, not "gone"
    scene.run(5, det())
    assert scene.texts() == ["Chair ahead."]
    scene.run(4, )                 # gone for 4 s
    scene.run(2, det())
    assert scene.texts() == ["Chair ahead.", "Chair ahead."]
    assert "back in view" in scene.said[-1][1].reason


def test_said_again_when_more_of_them_appear_but_not_when_fewer():
    scene = Scene(AlertConfig(repeat_after_s=10.0, remind_after_s=None))
    scene.run(11, det("chair", cx=0.1))
    scene.run(2, det("chair", cx=0.1), det("chair", cx=0.2))
    assert scene.texts() == ["Chair to your left.", "2 chairs to your left."]
    assert "more of them (1 -> 2)" in scene.said[-1][1].reason
    scene.run(12, det("chair", cx=0.1))
    assert len(scene.said) == 2


def test_never_repeats_sooner_than_the_floor_even_if_it_flickers_away():
    scene = Scene(AlertConfig(repeat_after_s=10.0, forget_after_s=3.0))
    scene.run(1, det())
    scene.run(4, )
    scene.run(2, det())            # back after 4 s, but only 7 s since it was said
    assert scene.texts() == ["Chair ahead."]
    scene.run(4, det())            # 11 s since: now the "back in view" rule allows it
    assert scene.texts() == ["Chair ahead.", "Chair ahead."]


def test_moving_chair_from_centre_to_side_is_announced_again_with_the_new_direction():
    scene = Scene()
    scene.run(3, det(cx=0.5))
    scene.run(3, det(cx=0.15))
    assert scene.texts() == ["Chair ahead.", "Chair to your left."]


def test_minimum_gap_between_any_two_announcements_and_priority_order():
    scene = Scene(AlertConfig(min_gap_s=2.5, caution=False))  # cautions are tested in test_guide_caution.py
    objects = (det("person", cx=0.85), det("chair", cx=0.5, height=0.6), det("dog", cx=0.1))
    scene.run(8, *objects)
    times = [t - 100.0 for t, _ in scene.said]
    # most important first: the large central chair, then the others, never closer than 2.5 s
    assert scene.texts()[0] == "Chair ahead."
    assert sorted(scene.texts()[1:]) == ["Dog to your left.", "Person to your right."]
    assert all(b - a >= 2.5 for a, b in zip(times, times[1:]))


def test_counts_and_plurals():
    scene = Scene()
    scene.run(1, det("person", cx=0.8), det("person", cx=0.9))
    assert scene.texts() == ["2 people to your right."]
    assert plural("bench") == "benches" and plural("chair") == "chairs" and plural("bus") == "buses"
    assert alert_text("dining table", 1, Region.LEFT).startswith("Dining table")


def test_spoken_names_are_friendly():
    scene = Scene()
    scene.run(1, det("dining table"))
    assert scene.texts() == ["Table ahead."]


def test_objects_not_on_the_list_tiny_boxes_and_low_scores_are_not_announced():
    scene = Scene()
    scene.run(3, det("cup", cx=0.5), det("chair", cx=0.1, height=0.05), det("person", cx=0.9, conf=0.2))
    assert scene.said == []


def test_all_labels_mode_announces_any_class():
    scene = Scene(AlertConfig(labels=None))
    scene.run(1, det("cup"))
    assert scene.texts() == ["Cup ahead."]


def test_uncertain_large_central_detection_is_a_possible_obstacle_without_its_label():
    scene = Scene(AlertConfig(caution=False))  # this big couch is in the way: with cautions it's "Caution, ..."
    scene.run(1, det("couch", cx=0.5, height=0.5, conf=0.4))
    (alert,) = [a for _, a in scene.said]
    assert alert.text == "Possible obstacle ahead."
    assert alert.label == "obstacle" and "couch" not in alert.text
    # ...but not when it is small, off to the side, or the same box as a confident detection
    for objects in [(det("couch", height=0.2, conf=0.4),),
                    (det("couch", cx=0.1, height=0.5, conf=0.4),),
                    (det("chair", height=0.5, conf=0.8), det("couch", height=0.5, conf=0.4))]:
        scene = Scene()
        scene.run(3, *objects)
        assert "Possible obstacle ahead." not in scene.texts()


def test_frames_that_are_too_old_never_trigger_speech():
    scene = Scene(AlertConfig(max_frame_age_s=1.0))
    scene.run(2, det(), lag=1.5)
    assert scene.said == []
    (alert,) = scene.step(det(), lag=0.2)  # fresh again: it has been visible all along, so it can speak
    assert alert.latency_ms == pytest.approx(200.0)


def test_meters_are_spoken_only_for_validated_depth():
    unvalidated = DepthEstimate(1.43, 0.9, "depth_anything_v2", validated=False)
    scene = Scene()
    scene.run(1, det(), depths=[unvalidated])
    (alert,) = [a for _, a in scene.said]
    assert alert.text == "Chair ahead." and alert.meters is None
    assert "unvalidated distance not spoken" in alert.reason

    validated = DepthEstimate(1.43, 0.9, "record3d", validated=True)
    scene = Scene()
    scene.run(1, det(), depths=[validated])
    (alert,) = [a for _, a in scene.said]
    assert alert.text == "Chair ahead, approximately 1.4 meters."
    assert (alert.meters, alert.depth_source) == (1.4, "record3d")


def test_never_claims_a_path_is_clear_or_safe():
    scene = Scene(AlertConfig(labels=None, repeat_after_s=0.5, min_gap_s=0.1))
    labels = ["person", "chair", "dining table", "bench", "dog", "cup", "bus"]
    for i in range(300):
        visible = [det(labels[(i // 7 + k) % len(labels)], cx=(0.15 + 0.35 * k) % 1.0, conf=0.3 + 0.1 * k)
                   for k in range(i % 4)]
        scene.step(*visible)
    assert scene.said, "the scene should have produced some alerts"
    for text in scene.texts():
        assert "clear" not in text.lower() and "safe" not in text.lower()


def test_alert_dict_is_json_ready_and_versioned():
    import json

    scene = Scene()
    scene.run(1, det(track_id=4))
    (alert,) = [a for _, a in scene.said]
    data = alert.to_dict()
    assert list(data)[:2] == ["schema", "type"]
    assert data["schema"] == "sentra.guide/1" and data["region"] == "center" and data["track_id"] == 4
    json.dumps(data)


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        AlertConfig(persist_frames=6, persist_window=5)
    with pytest.raises(ValueError):
        AlertConfig(uncertain_confidence=0.6, announce_confidence=0.5)
