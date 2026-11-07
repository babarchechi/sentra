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
