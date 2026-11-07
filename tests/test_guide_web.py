"""Phone speech page (sentra.guide.web). Owner: Swechha03.

Starts the real server on 127.0.0.1 with port 0 (any free port) and reads the event stream like a phone would.
"""
import http.client
import json
import subprocess
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from sentra.guide.detector import ObjectDetector
from sentra.guide.pipeline import GuideControls, GuidePipeline
from sentra.guide.schema import SCHEMA_VERSION, GuideAlert, GuideUpdate, Region, Severity
from sentra.guide.view import HeadlessSink
from sentra.guide.web import STATIC_DIR, Broadcaster, TeeSink, WebSpeechPage, check_host
from guide_helpers import CHAIR_AHEAD, FakeModel, FakeSource, frame

REPO = Path(__file__).resolve().parents[1]


# ----- helpers ------------------------------------------------------------------------------------------------

def alert(text="Chair ahead.", t=100.0, severity=Severity.MEDIUM):
    return GuideAlert(timestamp=t, frame_id=1, label="chair", count=1, region=Region.CENTER, severity=severity,
                      confidence=0.85, text=text, reason="test", source="test")


def update(*alerts, stale=False):
    return GuideUpdate(frame=frame(), objects=(), alerts=tuple(alerts),
                       image=np.full((480, 640, 3), 7, np.uint8), fps=24.6, latency_ms=41.2,
                       depth_status="Distance: off (camera only)", stale=stale)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def page():
    p = WebSpeechPage(GuideControls(), host="127.0.0.1", port=0, clock=Clock())
    yield p
    p.close()


def get(page, path):
    conn = http.client.HTTPConnection(page.host, page.port, timeout=5)
    conn.request("GET", path)
    response = conn.getresponse()
    return response, response.read(), conn


class Stream:
    """Reads Server-Sent Events from /events, like the phone's EventSource."""

    def __init__(self, page):
        self.conn = http.client.HTTPConnection(page.host, page.port, timeout=5)
        self.conn.request("GET", "/events")
        self.response = self.conn.getresponse()
        assert self.response.status == 200
        assert self.response.getheader("Content-Type").startswith("text/event-stream")

    def next_event(self):
        """Next (event, data) pair, skipping the retry line and keepalive comments."""
        event, data = None, None
        while True:
            line = self.response.fp.readline()
            if not line:
                return None  # stream ended
            line = line.decode("utf-8").rstrip("\n")
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
            elif line == "" and event is not None:
                return event, data

    def close(self):
        self.conn.close()


# ----- network rule -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("host", ["0.0.0.0", "", "::", "8.8.8.8", "localhost", "phone"])
def test_refuses_every_network_public_and_non_ip_hosts(host):
    with pytest.raises(ValueError):
        check_host(host)


@pytest.mark.parametrize("host", ["127.0.0.1", "172.20.10.2", "192.168.1.20", "10.0.0.5"])
def test_accepts_loopback_and_private_hotspot_addresses(host):
    assert check_host(host) == host


def test_page_refuses_to_start_on_every_network():
    with pytest.raises(ValueError, match="every network"):
        WebSpeechPage(host="0.0.0.0", port=0)


# ----- routes ---------------------------------------------------------------------------------------------------

def test_serves_the_page_with_strict_headers(page):
    response, body, conn = get(page, "/")
    conn.close()
    assert response.status == 200
    assert b"Tap to turn on sound" in body
    assert b"Not a navigation aid" in body
    csp = response.getheader("Content-Security-Policy")
    assert "default-src 'self'" in csp and "img-src 'none'" in csp and "media-src 'none'" in csp
    for path in ("/app.js", "/style.css"):
        response, body, conn = get(page, path)
        conn.close()
        assert response.status == 200 and body


@pytest.mark.parametrize("path", ["/video", "/frame.jpg", "/../schema.py", "/static/app.js", "/stream"])
def test_nothing_else_is_served(page, path):
    response, _, conn = get(page, path)
    conn.close()
    assert response.status == 404


def test_page_has_no_inline_script_or_style():
    """The Content-Security-Policy blocks inline code, so the page must not rely on any."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "<script>" not in html and "style=" not in html and "<style" not in html
    assert 'src="/app.js"' in html and 'href="/style.css"' in html


def test_health(page):
    response, body, conn = get(page, "/health")
    conn.close()
    assert response.status == 200
    assert json.loads(body) == {"ok": True, "schema": SCHEMA_VERSION, "viewers": 0}


# ----- the stream ---------------------------------------------------------------------------------------------

def test_alert_reaches_the_phone_as_the_schema_json(page):
    stream = Stream(page)
    try:
        event, hello = stream.next_event()
        assert event == "hello" and hello["recent"] == [] and hello["status"] is None
        page.publish(update(alert()))
        assert stream.next_event()[0] == "status"  # status first, so the page knows whether it's muted
        event, data = stream.next_event()
        assert event == "alert"
        assert data == alert().to_dict()  # exactly the contract: GuideAlert.to_dict()
    finally:
        stream.close()


def test_status_heartbeat_is_throttled_and_carries_mute(page):
    clock = page._clock
    stream = Stream(page)
    try:
        stream.next_event()  # hello
        page.publish(update())             # t=0: status
        clock.now = 0.5
        page.publish(update())             # too soon: nothing
        page.controls.muted = True
        clock.now = 1.1
        page.publish(update(stale=True))   # t=1.1: status again
        first, second = stream.next_event(), stream.next_event()
        assert first[0] == second[0] == "status"
        assert first[1]["muted"] is False and second[1]["muted"] is True
        assert second[1]["stale"] is True
        assert second[1]["fps"] == 24.6 and second[1]["depth_status"] == "Distance: off (camera only)"
    finally:
        stream.close()


def test_no_image_data_is_ever_sent(page):
    stream = Stream(page)
    try:
        stream.next_event()
        page.publish(update(alert()))
        for _ in range(2):
            _, data = stream.next_event()
            assert set(data) <= set(alert().to_dict()) | {"fps", "latency_ms", "depth_status", "muted",
                                                         "stale", "objects", "source"}
            assert "image" not in data and "box" not in data
    finally:
        stream.close()


def test_new_page_gets_recent_cues_to_show_not_speak(page):
    for i in range(7):
        page.publish(update(alert(f"Cue {i}.")))
    stream = Stream(page)
    try:
        event, hello = stream.next_event()
        assert event == "hello"
        assert [a["text"] for a in hello["recent"]] == [f"Cue {i}." for i in range(2, 7)]  # last 5
        assert hello["status"]["type"] == "status"
    finally:
        stream.close()


def test_close_ends_open_streams(page):
    stream = Stream(page)
    stream.next_event()
    page.close()
    assert stream.next_event() is None
    stream.close()


def test_a_stuck_viewer_loses_old_messages_but_never_blocks():
    b = Broadcaster(max_pending=3)
    q = b.subscribe()  # holds "hello"
    for i in range(10):
        b.send("alert", {"text": str(i)})  # never blocks, even though nobody reads
    items = [q.get_nowait() for _ in range(3)]
    assert [data["text"] for _, data in items] == ["7", "8", "9"]  # newest news wins


def test_busy_port_fails_at_start(page):
    with pytest.raises(OSError):
        WebSpeechPage(host="127.0.0.1", port=page.port)


# ----- with the real pipeline -----------------------------------------------------------------------------------

def test_pipeline_cue_reaches_the_phone(page, tmp_path):
    stream = Stream(page)
    try:
        stream.next_event()  # hello
        sink = TeeSink(HeadlessSink(), page)
        GuidePipeline(FakeSource(8), ObjectDetector(model=FakeModel(CHAIR_AHEAD)), sink,
                      output_dir=tmp_path).run()
        alerts = []
        while not alerts:
            event, data = stream.next_event()
            if event == "alert":
                alerts.append(data)
        assert alerts[0]["text"] == "Chair ahead."
        assert alerts[0]["schema"] == SCHEMA_VERSION
    finally:
        stream.close()


def test_tee_sink_fans_out_and_closes_everything():
    class Sink:
        def __init__(self, stop=False, fail_close=False):
            self.updates, self.stop, self.fail_close, self.closed = [], stop, fail_close, False

        def publish(self, u):
            self.updates.append(u)

        def should_stop(self):
            return self.stop

        def close(self):
            self.closed = True
            if self.fail_close:
                raise RuntimeError("boom")

    a, b = Sink(fail_close=True), Sink()
    tee = TeeSink(a, b)
    tee.publish(update())
    assert len(a.updates) == len(b.updates) == 1
    assert not tee.should_stop()
    b.stop = True
    assert tee.should_stop()
    tee.close()  # a's failure doesn't stop b from closing
    assert a.closed and b.closed


# ----- CLI and page script ----------------------------------------------------------------------------------------

def test_cli_has_web_options():
    sys.path.insert(0, str(REPO / "src"))
    try:
        import guide
    finally:
        sys.path.pop(0)
    args = guide.build_parser().parse_args(["--web", "--web-host", "172.20.10.2", "--web-port", "9000"])
    assert (args.web, args.web_host, args.web_port) == (True, "172.20.10.2", 9000)
    defaults = guide.build_parser().parse_args([])
    assert (defaults.web, defaults.web_host) == (False, "127.0.0.1")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_page_script_parses():
    result = subprocess.run(["node", "--check", str(STATIC_DIR / "app.js")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
