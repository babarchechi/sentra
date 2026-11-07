"""Caution cues (danger warnings): "Caution, chair ahead." for something big in the middle of the view, said first,
repeated while it stays, and vibrated on the phone page. There is never a "safe" signal. Owner: iesa242."""
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from guide_helpers import CHAIR_AHEAD, CHAIR_IN_THE_WAY, FakeClock, FakeModel, FakeSource, RecordingSpeaker, det, frame

from sentra.guide.alerts import AlertConfig, AlertPolicy, alert_text
from sentra.guide.detector import ObjectDetector
from sentra.guide.pipeline import GuideControls, GuidePipeline
from sentra.guide.schema import SCHEMA_VERSION, Region, Severity
from sentra.guide.spatial import SpatialEstimator
from sentra.guide.speech import Speaker
from sentra.guide.view import CAUTION_RED, BAR, GuideView, HeadlessSink

STATIC_DIR = Path(__file__).resolve().parents[1] / "src" / "sentra" / "guide" / "static"


def play(scenes, config=None, fps=10):
    """Run the policy over [(seconds, [detections])] at `fps`. Returns [(time, alert)] for every alert."""
    clock = FakeClock()
    policy, spatial = AlertPolicy(config, clock=clock), SpatialEstimator()
    t, frame_id, said = 100.0, 0, []
    for seconds, detections in scenes:
        for _ in range(round(seconds * fps)):
            frame_id += 1
            clock.now = t + 0.03
            f = frame(*detections, frame_id=frame_id, t=t)
            said += [(t - 100.0, a) for a in policy.update(spatial.locate(f), f)]
            t += 1.0 / fps
    return said


def texts(said):
    return [a.text for _, a in said]


BIG_CHAIR = det("chair", cx=0.5, height=0.6)      # in the middle and 60% of the frame tall: in the way
CHAIR = det("chair", cx=0.5, height=0.4)          # ahead, but not big enough to be in the way
BIG_CHAIR_LEFT = det("chair", cx=0.15, height=0.6)  # big, but off to the side


# The policy ------------------------------------------------------------------------------------------------

def test_something_big_in_the_middle_gets_a_caution():
    ((t, alert),) = play([(0.5, [BIG_CHAIR])])
    assert alert.text == "Caution, chair ahead." and alert.caution and alert.severity is Severity.HIGH
    assert alert.reason.startswith("in the way: chair") and t == pytest.approx(0.2)  # 3rd frame


def test_ahead_but_not_big_and_big_but_to_the_side_are_ordinary_cues():
    assert [(a.text, a.caution) for _, a in play([(0.5, [CHAIR])])] == [("Chair ahead.", False)]
    assert [(a.text, a.caution) for _, a in play([(0.5, [BIG_CHAIR_LEFT])])] == [("Chair to your left.", False)]


def test_one_big_frame_is_not_enough():
    said = play([(0.1, [BIG_CHAIR]), (0.5, [CHAIR])])
    assert texts(said) == ["Chair ahead."]


def test_a_caution_keeps_repeating_every_two_seconds_while_it_stays_and_nothing_plain_in_between():
    said = play([(10.0, [BIG_CHAIR])])
    assert texts(said) == ["Caution, chair ahead."] * 5
    times = [t for t, _ in said]
    assert all(b - a == pytest.approx(2.0, abs=0.11) for a, b in zip(times, times[1:]))


def test_cautioning_stops_once_it_is_no_longer_in_the_way_and_says_nothing_reassuring():
    # In the way for 5 s, then walked past (gone), then off to the side.
    said = play([(5.0, [BIG_CHAIR]), (4.0, []), (4.0, [BIG_CHAIR_LEFT])])
    cautions = [t for t, a in said if a.caution]
    assert len(cautions) == 3 and cautions[-1] < 5.0  # at ~0.2, 2.2, 4.2 s, none after it left
    after = [a.text for t, a in said if t >= 5.0]
    assert after == ["Chair to your left."]  # no caution for something to the side, and no "clear"


def test_something_announced_that_comes_close_gets_its_caution_straight_away():
    said = play([(1.0, [CHAIR]), (2.0, [BIG_CHAIR])])
    assert texts(said) == ["Chair ahead.", "Caution, chair ahead."]
    gap = said[1][0] - said[0][0]
    assert 1.0 <= gap < 2.5  # inside the plain 10 s repeat and the 2.5 s gap: a caution can't wait


def test_a_caution_goes_before_other_cues():
    said = play([(0.5, [det("person", cx=0.85), BIG_CHAIR])])
    assert texts(said)[0] == "Caution, chair ahead."


def test_an_unsure_big_thing_in_the_way_is_a_possible_obstacle_caution():
    said = play([(0.5, [det("couch", cx=0.5, height=0.6, conf=0.4)])])
    assert texts(said) == ["Caution, possible obstacle ahead."] and said[0][1].label == "obstacle"


def test_cautions_can_be_turned_off():
    said = play([(0.5, [BIG_CHAIR])], AlertConfig(caution=False))
    assert [(a.text, a.caution) for _, a in said] == [("Chair ahead.", False)]


def test_there_is_never_a_safe_or_clear_signal():
    scenes = [[(12.0, [BIG_CHAIR])], [(3.0, [CHAIR]), (3.0, [])], [(5.0, [])],
              [(3.0, [det("couch", cx=0.5, height=0.6, conf=0.4)])]]
    words = " ".join(" ".join(texts(play(scene))) for scene in scenes).lower()
    assert "caution" in words and "safe" not in words and "clear" not in words and "go ahead" not in words
    assert play([(5.0, [])]) == []  # nothing detected: silence, not reassurance


def test_wording_and_the_json_contract():
    assert alert_text("chair", 1, Region.CENTER, caution=True) == "Caution, chair ahead."
    assert alert_text("person", 2, Region.CENTER, caution=True) == "Caution, 2 people ahead."
    assert alert_text("obstacle", 1, Region.CENTER) == "Possible obstacle ahead."
    ((_, alert),) = play([(0.5, [BIG_CHAIR])])
    data = json.loads(json.dumps(alert.to_dict()))
    assert data["caution"] is True and data["schema"] == SCHEMA_VERSION == "sentra.guide/1"
    ((_, plain),) = play([(0.5, [CHAIR])])
    assert plain.to_dict()["caution"] is False


def test_reset_forgets_cautions():
    clock = FakeClock()
    policy, spatial = AlertPolicy(clock=clock), SpatialEstimator()
    for i in range(3):
        f = frame(BIG_CHAIR, frame_id=i + 1, t=100.0 + i * 0.1)
        clock.now = f.timestamp
        out = policy.update(spatial.locate(f), f)
    assert out and out[0].caution
    policy.reset()  # e.g. after switching cameras
    for i in range(3):
        f = frame(BIG_CHAIR, frame_id=10 + i, t=101.0 + i * 0.1)
        clock.now = f.timestamp
        out = policy.update(spatial.locate(f), f)
    assert out and out[0].text == "Caution, chair ahead."  # not held back by the old repeat timer


# The laptop voice ------------------------------------------------------------------------------------------

class BlockingBackend:
    name = "blocking"

    def __init__(self):
        self.said, self.speaking, self.release = [], threading.Event(), threading.Event()

    def start(self):
        pass

    def speak(self, text):
        self.speaking.set()
        self.release.wait(2.0)
        self.said.append(text)

    def close(self):
        pass


def test_a_caution_jumps_ahead_of_cues_still_waiting():
    backend = BlockingBackend()
    speaker = Speaker(backend, max_age_s=5.0)
    speaker.say("Person to your left.")
    assert backend.speaking.wait(2.0)  # busy saying the first cue
    speaker.say("Dog to your right.")
    speaker.say("Caution, chair ahead.", urgent=True)
    backend.release.set()
    deadline = time.perf_counter() + 2.0
    while len(backend.said) < 2 and time.perf_counter() < deadline:  # close() would drop what's still waiting
        time.sleep(0.01)
    speaker.close()
    assert backend.said == ["Person to your left.", "Caution, chair ahead."] and speaker.dropped == 1


# Pipeline and debug view -----------------------------------------------------------------------------------

def test_pipeline_speaks_a_caution_urgently_and_logs_it(tmp_path):
    speaker = RecordingSpeaker()
    pipeline = GuidePipeline(FakeSource(6), ObjectDetector(model=FakeModel(CHAIR_IN_THE_WAY)), HeadlessSink(),
                             speaker=speaker, output_dir=tmp_path)
    pipeline.run()
    assert speaker.said == ["Caution, chair ahead."] and speaker.urgent == ["Caution, chair ahead."]
    alert = json.loads(pipeline.log_path.read_text(encoding="utf-8").splitlines()[0])
    assert alert["caution"] is True and alert["text"] == "Caution, chair ahead."


def test_plain_cues_are_not_urgent(tmp_path):
    speaker = RecordingSpeaker()
    GuidePipeline(FakeSource(6), ObjectDetector(model=FakeModel(CHAIR_AHEAD)), HeadlessSink(), speaker=speaker,
                  output_dir=tmp_path).run()
    assert speaker.said == ["Chair ahead."] and speaker.urgent == []


def test_debug_view_frames_the_video_in_red_after_a_caution(tmp_path):
    updates = []

    class Keep(HeadlessSink):
        def publish(self, update):
            updates.append(update)

    GuidePipeline(FakeSource(4), ObjectDetector(model=FakeModel(CHAIR_IN_THE_WAY)), Keep(), output_dir=tmp_path).run()
    view = GuideView(GuideControls(), AlertPolicy(), show_window=False)
    plain = view.render(updates[0])
    assert plain[BAR + 2, 2].tolist() != list(CAUTION_RED)
    caution_update = next(u for u in updates if u.alerts)
    view.publish(caution_update)  # the window learns about alerts when they're published
    assert view.caution_showing()
    assert view.render(caution_update)[BAR + 2, 2].tolist() == list(CAUTION_RED)


# The phone page --------------------------------------------------------------------------------------------

PAGE_HARNESS = r"""
const vm = require("vm"), fs = require("fs");
const log = { vibrate: [], utterances: [], cancels: 0 };
function el(id) {
  return { id, textContent: "", hidden: false, attrs: {}, listeners: {}, children: [], lastElementChild: null,
    classList: { set: new Set(), add(...c) { c.forEach((x) => this.set.add(x)); },
                 remove(...c) { c.forEach((x) => this.set.delete(x)); },
                 toggle(c, on) { if (on) this.set.add(c); else this.set.delete(c); }, contains(c) { return this.set.has(c); } },
    setAttribute(k, v) { this.attrs[k] = v; }, addEventListener(t, f) { this.listeners[t] = f; },
    replaceChildren() {}, prepend() {}, append() {} };
}
const elements = {};
const get = (id) => elements[id] || (elements[id] = el(id));
class Utterance { constructor(text) { this.text = text; } }
const speechSynthesis = { speak(u) { log.utterances.push(u); }, cancel() { log.cancels += 1; } };
const navigator = { vibrate(pattern) { log.vibrate.push(pattern); return true; } };
const ctx = { document: { getElementById: get, querySelector: get, createElement: () => el("x"),
                          createTextNode: () => ({}), addEventListener() {}, visibilityState: "visible" },
              window: { speechSynthesis, SpeechSynthesisUtterance: Utterance }, navigator,
              SpeechSynthesisUtterance: Utterance, EventSource: class { addEventListener() {} },
              setInterval: () => 0, setTimeout: () => 0, Date, JSON, console };
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx);
const spoken = () => log.utterances.map((u) => u.text);
const out = {};

ctx.onAlert({ text: "Caution, chair ahead.", caution: true, severity: "high" });
out.beforeTap = { vibrate: log.vibrate.length, spoken: spoken() };          // no tap yet: no buzz, no voice

get("enable").listeners.click();                                           // "Tap to turn on sound"
out.afterTap = { vibrate: log.vibrate.slice(), vibrateButtonShown: !get("vibrate").hidden };

ctx.onAlert({ text: "Person to your left.", severity: "medium", caution: false });
const plain = log.utterances[log.utterances.length - 1];
const cancelsBefore = log.cancels;
ctx.onAlert({ text: "Caution, chair ahead.", severity: "high", caution: true });
out.interrupt = { cancelled: log.cancels - cancelsBefore, spoken: spoken(), vibrate: log.vibrate.slice(),
                  red: get(".cue").classList.contains("cue--caution") };

plain.onend && plain.onend();                       // the cut-off cue's late "end" must not end the caution
ctx.onAlert({ text: "Dog to your right.", severity: "low", caution: false });
out.whileCaution = { spoken: spoken(), cancels: log.cancels };
log.utterances[log.utterances.length - 1].onend(); // the caution finishes: the waiting cue is spoken
out.afterCaution = spoken();

get("vibrate").listeners.click();                   // Vibration: off
const buzzes = log.vibrate.length;
ctx.onAlert({ text: "Caution, chair ahead.", caution: true, severity: "high" });
out.vibrationOff = log.vibrate.length - buzzes;
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js to run the page script")
def test_phone_page_vibrates_on_a_caution_and_lets_it_cut_in(tmp_path):
    harness = tmp_path / "harness.js"
    harness.write_text(PAGE_HARNESS, encoding="utf-8")
    result = subprocess.run(["node", str(harness), str(STATIC_DIR / "app.js")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["beforeTap"] == {"vibrate": 0, "spoken": []}
    assert out["afterTap"] == {"vibrate": [80], "vibrateButtonShown": True}
    assert out["interrupt"]["cancelled"] == 1 and out["interrupt"]["red"]
    assert out["interrupt"]["spoken"][-2:] == ["Person to your left.", "Caution, chair ahead."]
    assert out["interrupt"]["vibrate"] == [80, [300, 120, 300]]
    # an ordinary cue arriving during a caution waits; the caution is not cut off
    assert out["whileCaution"]["spoken"][-1] == "Caution, chair ahead."
    assert out["whileCaution"]["cancels"] == out["interrupt"]["cancelled"] + 1  # no extra cancel (1 was the tap)
    assert out["afterCaution"][-1] == "Dog to your right."
    assert out["vibrationOff"] == 0


def test_phone_page_has_the_vibration_toggle_and_iphone_note():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert 'id="vibrate"' in html and 'id="no-vibrate"' in html and "safe" not in html.lower()
    css = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
    assert ".cue--caution" in css


def test_cli_caution_options():
    from guide import build_parser, build_policy

    args = build_parser().parse_args([])
    assert build_policy(args).config.caution and build_policy(args).config.caution_repeat_after_s == 2.0
    args = build_parser().parse_args(["--no-caution", "--caution-every", "3"])
    config = build_policy(args).config
    assert not config.caution and config.caution_repeat_after_s == 3.0
