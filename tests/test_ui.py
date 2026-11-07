"""Tests for the dashboard and fake events. Track C, owner: Swechha03.

No window is opened: these call CvDashboard.process(), render() and handle_key() directly.
"""
import csv
import itertools

import numpy as np
import pytest

from sentra.events import PersonPose, PersonStatus, PoseFrame, PostureState, StatusUpdate
from sentra.posture import PostureAnalyzer
from sentra.ui.cv_dashboard import (
    CONFIRMED,
    FALSE_ALARM,
    NO_RESPONSE,
    RECOVERED,
    TOP_BAR_H,
    CvDashboard,
    ascii_text,
    fit_text,
    format_duration,
)
from sentra.ui.fake_events import HEIGHT, WIDTH, fake_updates, stick_figure
from test_posture import FALL_ANGLE, FALL_HIPS, make_frame, moving_body  # Track A's synthetic-body helpers

S = PostureState
VIDEO_COLOR = (13, 37, 211)  # a colour the dashboard never draws itself


# ----- helpers -----------------------------------------------------------------------------------------------

def make_update(t, states, frame_shape=(480, 640)):
    """One StatusUpdate at time t. `states` maps track_id -> state or (state, confidence)."""
    people, statuses = [], []
    for i, (track_id, spec) in enumerate(sorted(states.items())):
        state, conf = spec if isinstance(spec, tuple) else (spec, 0.9)
        kpts = stick_figure(150.0 + 200 * i, 88.0 if state == S.POSSIBLE_FALL else 0.0)
        people.append(PersonPose(track_id, kpts, (10.0, 10.0, 100.0, 200.0), 0.9))
        statuses.append(PersonStatus(track_id, state, conf, f"reason for #{track_id}", t))
    frame = np.full((*frame_shape, 3), VIDEO_COLOR, np.uint8)
    return StatusUpdate(
        pose=PoseFrame(t, frame_shape[1], frame_shape[0], tuple(people)),
        statuses=tuple(statuses),
        frame=frame,
        annotated=frame.copy(),
        fps=9.5,
    )


class SoundCounter:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1


@pytest.fixture
def sound():
    return SoundCounter()


@pytest.fixture
def dash(tmp_path, sound):
    d = CvDashboard(output_dir=tmp_path, alert_sound=sound)
    yield d
    d.close()


def read_log(dash):
    dash.close()
    with dash.log_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ----- alerts --------------------------------------------------------------------------------------------------

def test_alert_opens_once_and_stays_until_answered(dash, sound):
    dash.process(make_update(0.0, {1: S.UPRIGHT}))
    assert dash.open_alert_ids == ()
    for t in range(1, 30):
        dash.process(make_update(float(t), {1: S.POSSIBLE_FALL}))
    assert dash.open_alert_ids == (1,)
    assert sound.calls == 1
    assert dash.alert_count == 1


def test_confirm_and_false_alarm_are_logged(dash):
    dash.process(make_update(0.0, {1: S.UPRIGHT, 2: S.UPRIGHT}))
    dash.process(make_update(1.0, {1: S.POSSIBLE_FALL, 2: S.UPRIGHT}))
    dash.process(make_update(2.0, {1: S.POSSIBLE_FALL, 2: S.POSSIBLE_FALL}))
    dash.process(make_update(4.5, {1: S.POSSIBLE_FALL, 2: S.POSSIBLE_FALL}))
    dash.handle_key(ord("c"))  # answers the oldest alert first: person 1
    assert dash.open_alert_ids == (2,)
    dash.handle_key(ord("F"))
    assert dash.open_alert_ids == ()

    rows = read_log(dash)
    assert [(r["track_id"], r["response"]) for r in rows] == [("1", CONFIRMED), ("2", FALSE_ALARM)]
    assert rows[0]["seconds_to_response"] == "3.5"
    assert rows[0]["reason"] == "reason for #1"
    assert list(rows[0]) == ["alert_time", "track_id", "reason", "response", "response_time",
                             "seconds_to_response", "confidence", "earlier_track_ids"]


def test_answered_alert_does_not_fire_again_while_still_down(dash, sound):
    dash.process(make_update(0.0, {1: S.POSSIBLE_FALL}))
    dash.handle_key(ord("c"))
    for t in range(1, 10):
        dash.process(make_update(float(t), {1: S.POSSIBLE_FALL}))
    assert dash.open_alert_ids == ()
    assert sound.calls == 1


def test_new_fall_after_getting_up_is_a_new_alert(dash, sound):
    dash.process(make_update(0.0, {1: S.POSSIBLE_FALL}))
    dash.handle_key(ord("f"))
    dash.process(make_update(1.0, {1: S.UPRIGHT}))
    dash.process(make_update(2.0, {1: S.POSSIBLE_FALL}))
    assert dash.open_alert_ids == (1,)
    assert sound.calls == 2


def test_getting_up_before_an_answer_is_logged_as_recovered(dash):
    dash.process(make_update(0.0, {1: S.POSSIBLE_FALL}))
    getting_up = make_update(2.0, {1: S.POSSIBLE_FALL})
    getting_up = StatusUpdate(getting_up.pose, (PersonStatus(1, S.POSSIBLE_FALL, 0.9, "getting up", 0.0),),
                              getting_up.frame, getting_up.annotated, getting_up.fps)
    dash.process(getting_up)
    assert dash.open_alert(1).reason == "getting up"  # the screen shows the latest reason
    dash.process(make_update(3.0, {1: S.UPRIGHT}))
    assert dash.open_alert_ids == ()
    rows = read_log(dash)
    assert [r["response"] for r in rows] == [RECOVERED]
    assert rows[0]["reason"] == "reason for #1"  # the log keeps why the alert fired


@pytest.mark.parametrize("state", [S.UNKNOWN, S.ON_GROUND, S.DESCENDING])
def test_alert_stays_open_when_not_upright(dash, state):
    dash.process(make_update(0.0, {1: S.POSSIBLE_FALL}))
    dash.process(make_update(1.0, {1: state}))
    assert dash.open_alert_ids == (1,)


def test_alert_stays_open_when_person_leaves_view(dash):
    dash.process(make_update(0.0, {1: S.POSSIBLE_FALL}))
    for t in range(1, 30):
        dash.process(make_update(float(t), {}))
    assert dash.open_alert_ids == (1,)
    dash.render(make_update(30.0, {}))  # draws a "not in view" row without crashing


def test_alert_for_track_with_no_skeleton_is_shown(dash):
    """Plan 8.1: an alert can arrive for a track_id that isn't in pose.people (out-of-frame status)."""
    update = make_update(0.0, {})
    update = StatusUpdate(update.pose, (PersonStatus(7, S.POSSIBLE_FALL, 0.4, "lost from view", 0.0),),
                          update.frame, update.annotated, update.fps)
    dash.process(update)
    assert dash.open_alert_ids == (7,)
    dash.render(update)


def test_low_confidence_alert_is_uncertain(dash):
    dash.process(make_update(0.0, {1: (S.POSSIBLE_FALL, 0.42), 2: (S.POSSIBLE_FALL, 0.86)}))
    assert dash.open_alert(1).uncertain
    assert not dash.open_alert(2).uncertain


def test_close_logs_unanswered_alerts(dash):
    dash.process(make_update(0.0, {1: S.POSSIBLE_FALL}))
    assert [r["response"] for r in read_log(dash)] == [NO_RESPONSE]


def test_answer_key_with_no_alert_does_nothing(dash):
    dash.handle_key(ord("c"))
    dash.handle_key(-1)
    assert read_log(dash) == []


# ----- track ID changes mid-fall (review on #7) -----------------------------------------------------------------

def drive_real_analyzer(dash, scene, duration, keys=(), fps=10.0):
    """Feed `scene(t) -> [PersonPose]` through the real PostureAnalyzer into the dashboard.

    `keys` is [(time, key)]: each key is pressed right after the first frame at or after that time.
    """
    analyzer = PostureAnalyzer()
    pending = sorted(keys)
    frame = np.zeros((480, 640, 3), np.uint8)
    for i in range(int(duration * fps) + 1):
        t = i / fps
        pose = make_frame(*scene(t), timestamp=t)
        dash.process(StatusUpdate(pose, analyzer.update(pose), frame, frame, fps))
        while pending and pending[0][0] <= t + 1e-9:
            dash.handle_key(ord(pending.pop(0)[1]))


def falling_person(id_changes):
    """One person who falls at 2 s and stays down. `id_changes` is [(time, new_track_id)], like ByteTrack
    re-assigning the ID while their detection flickers on the floor."""
    fall = moving_body(FALL_ANGLE, FALL_HIPS)

    def scene(t):
        track_id = 1
        for at, new_id in id_changes:
            if t >= at:
                track_id = new_id
        p = fall(t)
        return [PersonPose(track_id, p.keypoints, p.box, p.box_confidence)]
    return scene


def test_id_change_moves_an_open_alert_instead_of_raising_a_new_one(dash, sound):
    drive_real_analyzer(dash, falling_person([(9.0, 5)]), duration=15)
    assert dash.alert_count == 1
    assert sound.calls == 1
    assert dash.open_alert_ids == (5,)  # no second "NOT IN VIEW" alert left behind for #1
    dash.handle_key(ord("c"))
    rows = read_log(dash)
    assert [(r["track_id"], r["response"], r["earlier_track_ids"]) for r in rows] == [("5", CONFIRMED, "1")]


def test_id_change_after_confirming_does_not_alarm_again(dash, sound):
    drive_real_analyzer(dash, falling_person([(9.0, 5)]), duration=15, keys=[(8.5, "c")])
    assert dash.alert_count == 1
    assert sound.calls == 1
    assert dash.open_alert_ids == ()
    assert [(r["track_id"], r["response"]) for r in read_log(dash)] == [("1", CONFIRMED)]


def test_repeated_id_changes_keep_one_alert(dash, sound):
    drive_real_analyzer(dash, falling_person([(9.0, 5), (11.0, 9), (13.0, 12)]), duration=16)
    assert (dash.alert_count, sound.calls, dash.open_alert_ids) == (1, 1, (12,))
    assert read_log(dash)[0]["earlier_track_ids"] == "1;5;9"


def test_id_change_after_a_false_alarm_stays_answered(dash, sound):
    drive_real_analyzer(dash, falling_person([(9.0, 5), (12.0, 8)]), duration=15, keys=[(8.0, "f")])
    assert (dash.alert_count, sound.calls, dash.open_alert_ids) == (1, 1, ())


def test_new_id_with_matching_since_but_no_box_still_continues(dash, sound):
    """Matching on `since` alone works too, e.g. for an alert sent with no skeleton on screen."""
    def update(t, track_id, people=True):
        u = make_update(t, {track_id: S.POSSIBLE_FALL} if people else {})
        return StatusUpdate(u.pose, (PersonStatus(track_id, S.POSSIBLE_FALL, 0.9, "down", 3.25),),
                            u.frame, u.annotated, u.fps)
    dash.process(update(4.0, 1))
    dash.process(update(5.0, 6, people=False))
    assert (dash.alert_count, sound.calls, dash.open_alert_ids) == (1, 1, (6,))


def test_two_people_falling_together_get_two_alerts(dash, sound):
    """Both IDs are on screen at once, so neither can be a renamed copy of the other."""
    left = moving_body(FALL_ANGLE, FALL_HIPS, track_id=1, hip_x=200.0)
    right = moving_body(FALL_ANGLE, FALL_HIPS, track_id=2, hip_x=450.0)
    drive_real_analyzer(dash, lambda t: [left(t), right(t)], duration=10)
    assert dash.open_alert_ids == (1, 2)
    assert sound.calls == 2


def test_new_fall_elsewhere_is_a_new_alert(dash, sound):
    """A different person (different since, no box overlap) is not merged into an open alert."""
    dash.process(make_update(0.0, {1: S.POSSIBLE_FALL}))
    far_away = make_update(1.0, {})
    far_away = StatusUpdate(
        PoseFrame(1.0, 640, 480, (PersonPose(3, stick_figure(500, 88.0), (450.0, 380.0, 630.0, 440.0), 0.9),)),
        (PersonStatus(3, S.POSSIBLE_FALL, 0.9, "down", 0.5),), far_away.frame, far_away.annotated, 10.0)
    dash.process(far_away)
    assert dash.open_alert_ids == (1, 3)
    assert sound.calls == 2


def test_old_lost_track_is_not_matched_after_continuation_window(dash, sound):
    dash.process(make_update(0.0, {1: S.POSSIBLE_FALL}))
    dash.process(make_update(30.0, {}))  # #1 gone for 30 s, alert still open as NOT IN VIEW
    dash.process(make_update(31.0, {7: S.POSSIBLE_FALL}))  # same box, but far too late to be an ID change
    assert dash.open_alert_ids == (1, 7)


# ----- keys and drawing -----------------------------------------------------------------------------------------

def test_keys_toggle_privacy_and_quit(dash):
    assert not dash.privacy and not dash.should_stop()
    dash.handle_key(ord("p"))
    assert dash.privacy
    dash.handle_key(ord("p"))
    assert not dash.privacy
    dash.handle_key(ord("q"))
    assert dash.should_stop()


def contains_video_color(img):
    return bool(np.any(np.all(img == VIDEO_COLOR, axis=-1)))


def test_live_mode_shows_video_and_privacy_mode_never_does(dash):
    update = make_update(0.0, {1: S.UPRIGHT, 2: S.POSSIBLE_FALL})
    dash.process(update)
    assert contains_video_color(dash.render(update))
    dash.handle_key(ord("p"))
    view = dash.render(update)
    assert not contains_video_color(view)
    assert view[TOP_BAR_H:, :WIDTH].any()  # but the skeletons are drawn


def test_large_frames_are_scaled_to_fit(dash):
    update = make_update(0.0, {1: S.UPRIGHT}, frame_shape=(1080, 1920))
    view = dash.render(update)
    assert view.shape[0] <= TOP_BAR_H + 540
    assert view.shape[1] <= 960 + 400


def test_render_handles_empty_and_crowded_frames(dash):
    dash.render(make_update(0.0, {}))
    crowd = {i: S.UPRIGHT for i in range(1, 12)}
    dash.render(make_update(0.0, crowd))


def test_text_helpers():
    assert ascii_text("torso 78° — ok") == "torso 78 deg - ok"
    assert fit_text("x" * 500, 100, 0.5).endswith("...")
    assert format_duration(3.25) == "3.2 s"
    assert format_duration(42) == "42 s"
    assert format_duration(125) == "2m 05s"


# ----- fake events ----------------------------------------------------------------------------------------------

def run_fake(seconds, fps=10.0, **kwargs):
    return list(itertools.islice(fake_updates(fps, realtime=False, **kwargs), int(seconds * fps)))


def states_of(updates, track_id):
    return [next(s.state for s in u.statuses if s.track_id == track_id) for u in updates]


def test_fake_updates_follow_the_contract():
    for u in run_fake(3):
        assert u.frame.shape == u.annotated.shape == (HEIGHT, WIDTH, 3)
        assert u.frame.dtype == np.uint8
        assert {p.track_id for p in u.pose.people} == {s.track_id for s in u.statuses} == {1, 2}
        for p in u.pose.people:
            assert p.keypoints.shape == (17, 3) and p.keypoints.dtype == np.float32


def test_fake_script_order_and_timing():
    updates = run_fake(14)
    states = states_of(updates, 1)
    collapsed = [s for s, _ in itertools.groupby(states)]
    assert collapsed == [S.UPRIGHT, S.DESCENDING, S.ON_GROUND, S.POSSIBLE_FALL]
    first_alert = states.index(S.POSSIBLE_FALL) / 10.0
    assert first_alert == pytest.approx(10.5, abs=0.2)
    assert set(states_of(updates, 2)) == {S.UPRIGHT}
    since_values = {s.since for u in updates for s in u.statuses if s.track_id == 1 and s.state == S.ON_GROUND}
    assert len(since_values) == 1  # `since` only changes when the state changes


def test_fake_figure_waits_for_an_answer():
    states = states_of(run_fake(60, alert_answered=lambda tid: False), 1)
    assert states[-1] == S.POSSIBLE_FALL  # nobody answered, so still down a minute later

    # Answered as soon as allowed: alert at 10.5 s, up again by 12.5 s, standing until the next loop at 17.5 s.
    states = states_of(run_fake(14, alert_answered=lambda tid: True), 1)
    assert states[-1] == S.UPRIGHT


def test_fake_variants_cover_uncertain_and_recovered():
    # Answer instantly, so each loop is ~13 s: variant 0, then 1 (uncertain), then 2.
    updates = run_fake(45, alert_answered=lambda tid: True)
    statuses = [s for u in updates for s in u.statuses if s.track_id == 1]
    assert any(s.state == S.UNKNOWN for s in statuses)
    alert_confidences = {s.confidence for s in statuses if s.state == S.POSSIBLE_FALL}
    assert min(alert_confidences) <= 0.5 < max(alert_confidences)


def test_fake_events_drive_the_dashboard(tmp_path):
    """The whole demo loop, without a window: confirm the first alert, call the uncertain one a false alarm,
    and leave the third unanswered so the figure gets up by itself ('recovered')."""
    dash = CvDashboard(output_dir=tmp_path, alert_sound=None)

    def alert_answered(track_id):
        alert = dash.open_alert(track_id)
        if alert is not None and dash.alert_count == 1:
            dash.handle_key(ord("c"))
        elif alert is not None and alert.uncertain:
            dash.handle_key(ord("f"))
        return not dash.has_open_alert(track_id)

    for update in itertools.islice(fake_updates(10.0, realtime=False, alert_answered=alert_answered), 450):
        dash.process(update)
        dash.render(update)
    assert [r["response"] for r in read_log(dash)][:3] == [CONFIRMED, FALSE_ALARM, RECOVERED]
