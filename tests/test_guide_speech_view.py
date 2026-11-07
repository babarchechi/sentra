"""Sentra Guide speech queue and debug view. Owner: Swechha03."""
import sys
import time

import pytest

from sentra.guide.alerts import AlertPolicy
from sentra.guide.detector import ObjectDetector
from sentra.guide.pipeline import GuideControls, GuidePipeline
from sentra.guide.speech import PrintSpeech, Speaker, WindowsSpeech
from sentra.guide.view import GuideView, HeadlessSink
from guide_helpers import CHAIR_AHEAD, FakeModel, FakeSource


def test_view_close_after_window_closed_with_x_does_not_crash():
    """Phone test 2026-11-07: closing the window with its X, then quitting, raised cv2.error in close()."""
    from sentra.guide.pipeline import GuideControls

    view = GuideView(GuideControls(), AlertPolicy())
    view._window_open = True  # as if shown, then closed by the user: no window exists any more
    view.close()
    view.close()  # and closing twice is fine


# Speech ------------------------------------------------------------------------------------------------

class SlowBackend:
    name = "slow"

    def __init__(self, delay=0.3, fail_start=False, fail_speak=False):
        self.delay, self.fail_start, self.fail_speak = delay, fail_start, fail_speak
        self.said = []

    def start(self):
        if self.fail_start:
            raise RuntimeError("no voice here")

    def speak(self, text):
        if self.fail_speak:
            raise RuntimeError("voice crashed")
        time.sleep(self.delay)
        self.said.append(text)

    def close(self):
        pass


def test_speaker_drops_backlog_and_stale_messages():
    backend = SlowBackend(delay=0.3)
    speaker = Speaker(backend, max_age_s=2.0, max_pending=1)
    speaker.say("first")
    time.sleep(0.05)  # "first" is being spoken
    speaker.say("second")
    speaker.say("third")  # queue holds one: "second" is dropped, newest wins
    speaker.say("old news", created=time.perf_counter() - 5.0)  # replaces "third", then is too old to say
    time.sleep(0.8)
    speaker.close()
    assert backend.said == ["first"]
    assert speaker.dropped == 3


def test_speaker_falls_back_to_printing_when_the_voice_fails(capsys):
    speaker = Speaker(SlowBackend(fail_start=True))
    speaker.say("Chair ahead.")
    time.sleep(0.2)
    speaker.close()
    assert speaker.failed and speaker.spoken == ["Chair ahead."]
    assert "[voice] Chair ahead." in capsys.readouterr().out

    speaker = Speaker(SlowBackend(fail_speak=True))
    speaker.say("Person to your left.")
    time.sleep(0.2)
    speaker.close()
    assert speaker.failed and speaker.spoken == ["Person to your left."]


def test_print_speech_and_windows_command_is_self_contained():
    PrintSpeech().speak("hello")
    cmd = WindowsSpeech(rate=2).command()
    assert "-EncodedCommand" in cmd and "-NoProfile" in cmd
    with pytest.raises(ValueError):
        WindowsSpeech(rate=11)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows voice only exists on Windows")
def test_windows_voice_starts_and_speaks():
    voice = WindowsSpeech(rate=10)
    try:
        voice.start()      # loads System.Speech in PowerShell
        voice.speak("   ")  # blank: checks the round trip without making a sound during tests
    finally:
        voice.close()



# Debug view --------------------------------------------------------------------------------------------

def test_view_renders_regions_boxes_and_privacy_and_handles_keys(tmp_path):
    class Keep(HeadlessSink):
        updates = []

        def publish(self, update):
            self.updates.append(update)

    GuidePipeline(FakeSource(1, shape=(1080, 1920, 3)), ObjectDetector(model=FakeModel(CHAIR_AHEAD)), Keep(),
                  output_dir=tmp_path).run()
    (update,) = Keep.updates
    update.image[:] = 200  # a bright "camera image"
    controls = GuideControls()
    view = GuideView(controls, AlertPolicy(), show_window=False)
    canvas = view.render(update)
    assert canvas.shape[1] <= 1280  # big phone frames are scaled down to fit
    assert canvas[200, 100].tolist() == [200, 200, 200]  # camera image visible

    view.handle_key(ord("m"))
    view.handle_key(ord("p"))
    assert controls.muted and controls.privacy
    assert canvas.shape == view.render(update).shape
    assert view.render(update)[200, 100].tolist() == [0, 0, 0]  # privacy: image hidden
    assert not view.should_stop()
    view.handle_key(ord("q"))
    assert view.should_stop()


# Dashboard panel ---------------------------------------------------------------------------------------

def _dashboard_update(*objects_spec, alerts=()):
    """A 480x640 GuideUpdate with SpatialObjects built through the real SpatialEstimator."""
    import numpy as np

    from sentra.guide.schema import Detection, DetectionFrame, GuideUpdate
    from sentra.guide.spatial import SpatialEstimator

    dets = tuple(Detection(label, 0, conf, (cx * 640 - 40, 480 - h * 480, cx * 640 + 40, 480.0))
                 for label, cx, h, conf in objects_spec)
    frame = DetectionFrame(1, 0.0, 640, 480, dets, source="test")
    image = np.full((480, 640, 3), 90, np.uint8)
    return GuideUpdate(frame, SpatialEstimator().locate(frame), tuple(alerts), image, 25.0, 30.0, "Distance: off")


def test_not_announced_reasons_match_the_alert_rules():
    from sentra.guide.view import not_announced_reason, path_position

    policy = AlertPolicy()
    chair, cup, tiny, unsure, low_side = _dashboard_update(
        ("chair", 0.5, 0.6, 0.9), ("cup", 0.1, 0.4, 0.9), ("person", 0.9, 0.05, 0.9),
        ("couch", 0.5, 0.5, 0.4), ("person", 0.85, 0.4, 0.3)).objects
    assert not_announced_reason(policy, chair) == ""
    assert not_announced_reason(policy, cup) == "not on alert list"
    assert not_announced_reason(policy, tiny) == "small / far away"
    assert "possible obstacle" in not_announced_reason(policy, unsure)
    assert not_announced_reason(policy, low_side) == "score under 50%"
    assert path_position(chair) == "ahead, large in view"


def test_dashboard_puts_video_left_and_panel_right_and_lists_cues():
    from sentra.guide.schema import GuideAlert, Region, Severity
    from sentra.guide.view import BAR, PANEL_BG, PANEL_W

    controls = GuideControls()
    view = GuideView(controls, AlertPolicy(), show_window=False)
    alert = GuideAlert(0.0, 1, "chair", 1, Region.CENTER, Severity.HIGH, 0.9, "Chair ahead.", "test")
    update = _dashboard_update(("chair", 0.5, 0.6, 0.9), ("person", 0.9, 0.4, 0.8), alerts=(alert,))
    view.publish(update)  # show_window=False: records the cue without opening a window
    canvas = view.render(update)
    assert canvas.shape == (480 + 2 * BAR, 640 + PANEL_W, 3)
    assert canvas[BAR + 300, 20].tolist() != PANEL_BG and canvas[BAR + 300, 640 + PANEL_W - 5].tolist() == list(PANEL_BG)
    assert len(view._recent) == 1 and view._recent[0][1] == "Chair ahead."

    # privacy hides only the camera image; the panel (text, no pixels from the camera) stays
    controls.privacy = True
    hidden = view.render(update)
    assert hidden[BAR + 300, 20].tolist() == [0, 0, 0]
    assert (hidden[BAR:, 640:] == canvas[BAR:, 640:]).mean() > 0.95


def test_dashboard_never_says_clear_or_safe(monkeypatch):
    import sentra.guide.view as view_module

    written = []
    real_put = view_module._put
    monkeypatch.setattr(view_module, "_put", lambda img, text, *a, **k: (written.append(text), real_put(img, text, *a, **k)))
    view = GuideView(GuideControls(), AlertPolicy(), show_window=False)
    view.render(_dashboard_update())                         # nothing detected
    view.render(_dashboard_update(("cup", 0.1, 0.4, 0.9)))   # something irrelevant to the side
    view.render_idle()
    text = " ".join(written).lower()
    assert "nothing detected" in text and "not proof the way is clear" in text
    assert "is clear" not in text.replace("not proof the way is clear", "") and "safe" not in text

