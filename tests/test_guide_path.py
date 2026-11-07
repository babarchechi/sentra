"""Path-relevant cues: things in the way are named; things fully to the side or far away stay quiet. Owner: iesa242."""
import pytest
from guide_helpers import FakeClock, det, frame

from sentra.guide.alerts import AlertConfig, AlertPolicy
from sentra.guide.spatial import SpatialEstimator
from sentra.guide.view import not_announced_reason


def play(detections, config=None, seconds=0.5, fps=10):
    clock = FakeClock()
    policy, spatial = AlertPolicy(config, clock=clock), SpatialEstimator()
    said, t = [], 100.0
    for i in range(round(seconds * fps)):
        clock.now = t + 0.03
        f = frame(*detections, frame_id=i + 1, t=t)
        said += [a.text for a in policy.update(spatial.locate(f), f)]
        t += 1.0 / fps
    return said


def located(d):
    return SpatialEstimator().locate(frame(d))[0]


# The brief's table: what Sentra should say

def test_chair_against_the_left_wall_stays_quiet():
    assert play([det("chair", cx=0.1, height=0.4)]) == []


def test_chair_directly_ahead_is_named():
    assert play([det("chair", cx=0.5, height=0.4)]) == ["Chair ahead."]


def test_someone_standing_far_away_stays_quiet():
    assert play([det("person", cx=0.5, height=0.15)]) == []  # small: far away in the path


def test_a_thing_partly_in_the_path_is_ahead_slightly_left_or_right():
    # centre at 0.3 (left region) but the box reaches into the central band (0.25-0.35 wide box)
    assert play([det("chair", cx=0.3, height=0.4, width=0.12)]) == ["Chair ahead, slightly left."]
    assert play([det("chair", cx=0.7, height=0.4, width=0.12)]) == ["Chair ahead, slightly right."]


def test_people_to_the_side_stay_quiet_while_something_ahead_is_named():
    said = play([det("person", cx=0.9, height=0.6), det("dog", cx=0.08, height=0.3), det("chair", cx=0.5)],
                seconds=6.0)
    assert said == ["Chair ahead."]


def test_announce_sides_brings_back_every_direction():
    sides = AlertConfig(announce_sides=True)
    assert play([det("chair", cx=0.1, height=0.4)], sides) == ["Chair to your left."]
    assert play([det("person", cx=0.5, height=0.15)], sides) == ["Person ahead."]


def test_the_path_limit_is_tunable():
    assert play([det("person", cx=0.5, height=0.15)], AlertConfig(path_min_view_height=0.1)) == ["Person ahead."]


# The dashboard explains why something isn't said, with the same rules

@pytest.mark.parametrize("d, reason", [
    (det("chair", cx=0.1, height=0.4), "to the side, not in the path"),
    (det("person", cx=0.5, height=0.15), "far away (small) in the path"),
    (det("chair", cx=0.5, height=0.4), ""),
])
def test_dashboard_reason_matches_the_policy(d, reason):
    policy = AlertPolicy()
    obj = located(d)
    assert not_announced_reason(policy, obj) == reason
    assert policy.is_confident(obj) == (reason == "")
