"""Tests for posture analysis. Track A, owner: iesa242. See docs/PHASE1_PLAN.md, Section 6.4.

Everything here uses synthetic keypoints: `body()` draws a stick figure with a given torso angle and hip position,
and `run()` feeds a timeline of them to the analyzer at ~10 fps (or with random 5-15 fps jitter).
"""
import math
import random

import numpy as np
import pytest

from sentra.events import KP, PersonPose, PoseFrame, PostureState
from sentra.posture import PostureAnalyzer

FALL = PostureState.POSSIBLE_FALL
STAND_Y, FLOOR_Y = 300.0, 420.0


def make_frame(*people: PersonPose, timestamp: float = 0.0) -> PoseFrame:
    return PoseFrame(timestamp=timestamp, frame_width=640, frame_height=480, people=people)


# --- Synthetic people --------------------------------------------------------------------------------


def body(track_id=1, angle=5.0, hip_x=320.0, hip_y=STAND_Y, torso=100.0, conf=0.9, torso_conf=None, shift=(0.0, 0.0)):
    """A stick figure whose torso is `angle` degrees from vertical, with mid-hip at (hip_x, hip_y)."""
    a = math.radians(angle)
    up = np.array([math.sin(a), -math.cos(a)])        # hip -> shoulder direction (image y grows downward)
    side = np.array([-up[1], up[0]])                   # perpendicular, for left/right
    hip = np.array([hip_x, hip_y]) + np.array(shift)
    shoulder = hip + torso * up
    pts = {
        "nose": shoulder + 0.35 * torso * up,
        "left_eye": shoulder + 0.4 * torso * up + 0.05 * torso * side,
        "right_eye": shoulder + 0.4 * torso * up - 0.05 * torso * side,
        "left_ear": shoulder + 0.35 * torso * up + 0.1 * torso * side,
        "right_ear": shoulder + 0.35 * torso * up - 0.1 * torso * side,
        "left_shoulder": shoulder + 0.2 * torso * side,
        "right_shoulder": shoulder - 0.2 * torso * side,
        "left_elbow": shoulder + 0.25 * torso * side - 0.45 * torso * up,
        "right_elbow": shoulder - 0.25 * torso * side - 0.45 * torso * up,
        "left_wrist": shoulder + 0.25 * torso * side - 0.9 * torso * up,
        "right_wrist": shoulder - 0.25 * torso * side - 0.9 * torso * up,
        "left_hip": hip + 0.12 * torso * side,
        "right_hip": hip - 0.12 * torso * side,
        "left_knee": hip + 0.12 * torso * side - 0.55 * torso * up,
        "right_knee": hip - 0.12 * torso * side - 0.55 * torso * up,
        "left_ankle": hip + 0.12 * torso * side - 1.1 * torso * up,
        "right_ankle": hip - 0.12 * torso * side - 1.1 * torso * up,
    }
    keypoints = np.zeros((17, 3), np.float32)
    for name, (x, y) in pts.items():
        keypoints[KP[name]] = (x, y, conf)
    if torso_conf is not None:
        for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            keypoints[KP[name], 2] = torso_conf
    xs, ys = keypoints[:, 0], keypoints[:, 1]
    box = (float(xs.min() - 10), float(ys.min() - 10), float(xs.max() + 10), float(ys.max() + 10))
    return PersonPose(track_id=track_id, keypoints=keypoints, box=box, box_confidence=0.9)


def interp(t, keyframes):
    """Piecewise-linear value at time t from [(time, value), ...]; constant before the first and after the last."""
    if t <= keyframes[0][0]:
        return keyframes[0][1]
    for (t0, v0), (t1, v1) in zip(keyframes, keyframes[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return keyframes[-1][1]


def moving_body(angle_keys, hip_keys, **kwargs):
    return lambda t: body(angle=interp(t, angle_keys), hip_y=interp(t, hip_keys), **kwargs)


# Stand 2 s, then drop to lying in 0.5 s (hips 120 px = 1.2 torso-lengths in 0.5 s), then stay down.
FALL_ANGLE = [(0, 5), (2.0, 5), (2.5, 85)]
FALL_HIPS = [(0, STAND_Y), (2.0, STAND_Y), (2.5, FLOOR_Y)]


def run(scene, duration, fps=10.0, jitter=False, seed=0, analyzer=None):
    """Feed `scene(t) -> list[PersonPose]` to an analyzer. Returns [(t, {track_id: PersonStatus})]."""
    analyzer = analyzer or PostureAnalyzer()
    rng = random.Random(seed)
    t, out = 0.0, []
    while t <= duration + 1e-9:
        statuses = analyzer.update(make_frame(*scene(t), timestamp=t))
        out.append((t, {s.track_id: s for s in statuses}))
        t += 1.0 / (rng.uniform(5, 15) if jitter else fps)
    return out


def states(timeline, track_id=1):
    return [(t, s[track_id].state) for t, s in timeline if track_id in s]


def first_time(timeline, state, track_id=1):
    return next((t for t, st in states(timeline, track_id) if st is state), None)


def ever(timeline, state, track_id=1):
    return first_time(timeline, state, track_id) is not None


# --- Starter tests ---------------------------------------------------------------------------------


def test_one_status_per_person_and_invisible_keypoints_are_unknown():
    invisible = PersonPose(
        track_id=7,
        keypoints=np.zeros((17, 3), np.float32),  # every keypoint has confidence 0
        box=(0.0, 0.0, 50.0, 120.0),
        box_confidence=0.9,
    )
    statuses = PostureAnalyzer().update(make_frame(invisible))
    assert [s.track_id for s in statuses] == [7]
    assert statuses[0].state is PostureState.UNKNOWN


def test_empty_frame_returns_no_statuses():
    assert PostureAnalyzer().update(make_frame()) == ()


# --- Core scenarios (6.4) --------------------------------------------------------------------------


def test_standing_still_is_upright_throughout():
    timeline = run(lambda t: [body()], 5.0)
    assert all(st is PostureState.UPRIGHT for _, st in states(timeline))


def test_fast_drop_then_lying_raises_possible_fall_after_about_five_seconds():
    timeline = run(lambda t: [moving_body(FALL_ANGLE, FALL_HIPS)(t)], 9.0)
    alert_at = first_time(timeline, FALL)
    assert alert_at is not None
    assert 7.0 <= alert_at <= 8.3  # lying starts ~2.35 s, plus on_ground_alert_s = 5 s
    assert ever(timeline, PostureState.DESCENDING)
    assert not ever(timeline, PostureState.ON_GROUND)  # a fall is never reported as a slow lie-down
    status = timeline[-1][1][1]
    assert "after fast drop" in status.reason
    assert 0.0 < status.confidence <= 1.0


def test_getting_up_after_two_seconds_on_the_ground_never_alerts():
    angle = FALL_ANGLE + [(4.5, 85), (5.0, 5)]
    hips = FALL_HIPS + [(4.5, FLOOR_Y), (5.0, STAND_Y)]
    timeline = run(lambda t: [moving_body(angle, hips)(t)], 12.0)
    assert not ever(timeline, FALL)
    assert timeline[-1][1][1].state is PostureState.UPRIGHT


def test_lying_down_slowly_is_on_ground_without_alert():
    timeline = run(lambda t: [moving_body([(0, 5), (1, 5), (4, 85)], [(0, STAND_Y), (1, STAND_Y), (4, FLOOR_Y)])(t)],
                   12.0)
    assert not ever(timeline, FALL)
    assert not ever(timeline, PostureState.DESCENDING)
    assert timeline[-1][1][1].state is PostureState.ON_GROUND


def test_sitting_down_never_alerts():
    # Hips drop 0.6 torso-lengths in 1 s and the torso stays upright.
    timeline = run(lambda t: [moving_body([(0, 10)], [(0, STAND_Y), (2, STAND_Y), (3, STAND_Y + 60)])(t)], 10.0)
    assert not ever(timeline, FALL)
    assert not ever(timeline, PostureState.ON_GROUND)
    assert timeline[-1][1][1].state is PostureState.UPRIGHT


def test_possible_fall_clears_after_being_upright_for_recovery_time():
    angle = FALL_ANGLE + [(9.0, 85), (9.5, 5)]
    hips = FALL_HIPS + [(9.0, FLOOR_Y), (9.5, STAND_Y)]
    timeline = run(lambda t: [moving_body(angle, hips)(t)], 13.0)
    assert ever(timeline, FALL)
    still_alerting = [st for t, st in states(timeline) if 9.5 <= t <= 10.5]
    assert all(st is FALL for st in still_alerting)  # one upright second is not enough
    assert timeline[-1][1][1].state is PostureState.UPRIGHT  # upright ~3 s


def test_low_confidence_hips_and_shoulders_are_unknown():
    timeline = run(lambda t: [body(torso_conf=0.2)], 3.0)
    assert all(st is PostureState.UNKNOWN for _, st in states(timeline))


def test_tiny_or_jittery_skeletons_are_unknown_not_falls():
    # A skeleton a few pixels tall (far away or a bad detection) jumping around randomly must not look like a drop.
    rng = np.random.default_rng(0)
    analyzer = PostureAnalyzer()
    for i in range(30):
        keypoints = rng.random((17, 3)).astype(np.float32)
        person = PersonPose(track_id=1, keypoints=keypoints, box=(0.0, 0.0, 5.0, 5.0), box_confidence=0.9)
        (status,) = analyzer.update(make_frame(person, timestamp=i * 0.1))
        assert status.state is PostureState.UNKNOWN


def test_two_people_only_the_one_who_falls_alerts():
    faller = moving_body(FALL_ANGLE, FALL_HIPS, track_id=2, hip_x=480.0)
    timeline = run(lambda t: [body(track_id=1, hip_x=150.0), faller(t)], 9.0)
    assert ever(timeline, FALL, track_id=2)
    assert not ever(timeline, FALL, track_id=1)
    assert all(st is PostureState.UPRIGHT for _, st in states(timeline, track_id=1))


def test_person_disappearing_is_forgotten_without_crashing():
    analyzer = PostureAnalyzer()
    timeline = run(lambda t: [body()] if t < 2.0 else [], 7.0, analyzer=analyzer)
    assert analyzer.tracked_ids == ()
    assert timeline[-1][1] == {}
    # Coming back later starts fresh.
    assert analyzer.update(make_frame(body(), timestamp=7.5))[0].state is PostureState.UPRIGHT


@pytest.mark.parametrize("seed", range(5))
def test_uneven_frame_timing_gives_the_same_outcomes(seed):
    fall = run(lambda t: [moving_body(FALL_ANGLE, FALL_HIPS)(t)], 9.0, jitter=True, seed=seed)
    alert_at = first_time(fall, FALL)
    assert alert_at is not None and 7.0 <= alert_at <= 8.5

    standing = run(lambda t: [body()], 5.0, jitter=True, seed=seed)
    assert all(st is PostureState.UPRIGHT for _, st in states(standing))

    slow = run(lambda t: [moving_body([(0, 5), (1, 5), (4, 85)], [(0, STAND_Y), (1, STAND_Y), (4, FLOOR_Y)])(t)],
               12.0, jitter=True, seed=seed)
    assert not ever(slow, FALL)


# --- When the picture is uncertain (6.2) ----------------------------------------------------------


def test_track_id_change_while_lying_keeps_the_fall_timer():
    pose_at = moving_body(FALL_ANGLE, FALL_HIPS)

    def scene(t):
        person = pose_at(t)
        if t >= 3.5:  # ByteTrack hands out a new ID one second after landing
            person = PersonPose(track_id=2, keypoints=person.keypoints, box=person.box, box_confidence=0.9)
        return [person]

    timeline = run(scene, 9.0)
    alert_at = first_time(timeline, FALL, track_id=2)
    assert alert_at is not None and 7.0 <= alert_at <= 8.3  # on time, as if the ID never changed


def test_fast_drop_then_hidden_raises_low_confidence_alert():
    pose_at = moving_body(FALL_ANGLE, FALL_HIPS)

    def scene(t):
        person = pose_at(t)
        if t >= 2.5:  # hips and shoulders go low-confidence right after the drop (e.g. behind a bed)
            person = body(angle=85, hip_y=FLOOR_Y, torso_conf=0.2)
        return [person]

    timeline = run(scene, 9.0)
    alerts = [s[1] for t, s in timeline if s[1].state is FALL]
    assert alerts, "expected a POSSIBLE_FALL"
    assert all(a.confidence <= 0.5 for a in alerts)
    assert "hidden" in alerts[0].reason


def test_fast_drop_disappear_and_reappear_upright_does_not_alert():
    pose_at = moving_body(FALL_ANGLE, FALL_HIPS)

    def scene(t):
        if t < 2.4:
            return [pose_at(t)]
        if t < 5.0:
            return []  # out of shot
        return [body()]  # back on their feet

    timeline = run(scene, 9.0)
    assert not ever(timeline, FALL)
    assert timeline[-1][1][1].state is PostureState.UPRIGHT


def test_still_after_fall_is_more_confident_than_moving():
    still = run(lambda t: [moving_body(FALL_ANGLE, FALL_HIPS)(t)], 9.0)
    crawl = run(lambda t: [body(angle=interp(t, FALL_ANGLE), hip_y=interp(t, FALL_HIPS),
                                shift=(max(0.0, t - 2.5) * 15.0, 0.0))], 9.0)  # moving 0.15 torso-lengths/s
    still_alert = next(s[1] for _, s in still if s[1].state is FALL)
    crawl_alert = next((s[1] for _, s in crawl if s[1].state is FALL), None)
    assert crawl_alert is not None, "moving on the ground must not cancel the alert"
    assert still_alert.confidence > crawl_alert.confidence
    assert "still" in still_alert.reason


def test_lying_down_slowly_and_moving_for_25_s_never_alerts():
    def scene(t):
        drift = 15.0 * (abs(((t / 4.0) % 2) - 1) * 4.0)  # back and forth, 15 px/s, turning every 4 s
        return [body(angle=interp(t, [(0, 5), (1, 5), (4, 85)]),
                     hip_y=interp(t, [(0, STAND_Y), (1, STAND_Y), (4, FLOOR_Y)]), shift=(drift, 0.0))]

    timeline = run(scene, 30.0)
    assert not ever(timeline, FALL)
    assert timeline[-1][1][1].state is PostureState.ON_GROUND


def test_stretch_lying_still_for_a_long_time_without_a_drop_raises_low_confidence_alert():
    timeline = run(lambda t: [moving_body([(0, 5), (1, 5), (4, 85)], [(0, STAND_Y), (1, STAND_Y), (4, FLOOR_Y)])(t)],
                   28.0)
    alerts = [s[1] for _, s in timeline if s[1].state is FALL]
    assert alerts, "expected a low-confidence POSSIBLE_FALL after ~20 s still"
    assert all(a.confidence <= 0.5 for a in alerts)
    assert first_time(timeline, FALL) >= 4.0 + 20.0


def test_since_is_when_the_state_started():
    timeline = run(lambda t: [moving_body(FALL_ANGLE, FALL_HIPS)(t)], 9.0)
    alert_at = first_time(timeline, FALL)
    assert timeline[-1][1][1].since == pytest.approx(alert_at)
