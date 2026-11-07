"""Tests for camera input. Track B, owner: babarchechi.

No real camera is used: sentra.capture._open_capture is replaced with fakes.
"""
import time

import numpy as np
import pytest

import sentra.capture as capture
from sentra.capture import FrameSource, describe_source, parse_source


class FakeCapture:
    """Delivers `frames` frames whose pixels equal a running counter, then fails every read."""

    def __init__(self, frames: int, start: int = 0) -> None:
        self.remaining = frames
        self.value = start
        self.released = False

    def read(self):
        time.sleep(0.002)
        if self.remaining <= 0:
            return False, None
        self.remaining -= 1
        self.value += 1
        return True, np.full((4, 4, 3), self.value % 256, np.uint8)

    def release(self):
        self.released = True


class MarkerCapture(FakeCapture):
    """Frames 2 rows x 3 columns with one bright pixel in the top-left corner, to check rotation."""

    def read(self):
        ok, _ = super().read()
        if not ok:
            return False, None
        frame = np.zeros((2, 3, 3), np.uint8)
        frame[0, 0] = 255
        return True, frame


@pytest.fixture
def make_source(monkeypatch):
    """Build a FrameSource whose successive opens return the given fakes (None = open fails)."""
    created = []

    def factory(captures, source=0, **kwargs):
        queue = list(captures)
        opens = []

        def fake_open(_source):
            cap = queue.pop(0) if queue else None
            opens.append(cap)
            return cap

        monkeypatch.setattr(capture, "_open_capture", fake_open)
        src = FrameSource(
            source, max_failed_reads=2, backoff_start_s=0.01, backoff_max_s=0.02, open_timeout_s=2, **kwargs
        )
        created.append(src)
        return src, opens

    yield factory
    for src in created:
        src.close()


@pytest.mark.parametrize(
    "rotate, shape, marker",
    [
        (0, (2, 3, 3), (0, 0)),
        (90, (3, 2, 3), (0, 1)),   # clockwise: the top-left pixel ends up top-right
        (180, (2, 3, 3), (1, 2)),  # bottom-right
        (270, (3, 2, 3), (2, 0)),  # counter-clockwise: bottom-left
    ],
)
def test_rotate_turns_frames(make_source, rotate, shape, marker):
    src, _ = make_source([MarkerCapture(frames=10_000)], rotate=rotate)
    ok, frame, _ = src.read(timeout=1)
    assert ok
    assert frame.shape == shape
    assert frame[marker][0] == 255


def test_invalid_rotate_rejected_before_opening_camera(monkeypatch):
    opened = []
    monkeypatch.setattr(capture, "_open_capture", lambda source: opened.append(source))
    with pytest.raises(ValueError):
        FrameSource(0, rotate=45)
    assert opened == []


def test_parse_source():
    assert parse_source("0") == 0
    assert parse_source("12") == 12
    assert parse_source("http://192.168.1.5:8080/video") == "http://192.168.1.5:8080/video"


def test_describe_source_hides_credentials_and_query():
    url = "rtsp://admin:hunter2@192.168.1.20:554/live?token=abc123"
    assert describe_source(url) == "rtsp://192.168.1.20:554/live"
    assert describe_source(0) == "webcam 0"


def test_open_failure_raises_without_leaking_credentials(monkeypatch):
    monkeypatch.setattr(capture, "_open_capture", lambda _source: None)
    with pytest.raises(RuntimeError) as excinfo:
        FrameSource("rtsp://admin:hunter2@10.0.0.5/live", open_timeout_s=2)
    assert "hunter2" not in str(excinfo.value)
    assert "10.0.0.5" in str(excinfo.value)


def test_read_returns_each_frame_once_with_increasing_timestamps(make_source):
    src, _ = make_source([FakeCapture(frames=10_000)])
    ok1, frame1, t1 = src.read(timeout=1)
    ok2, frame2, t2 = src.read(timeout=1)
    assert ok1 and ok2
    assert frame1.shape == (4, 4, 3)
    assert t2 > t1


def test_read_times_out_when_no_new_frame(make_source):
    src, _ = make_source([FakeCapture(frames=1)])  # later reopen attempts fail (None)
    ok, frame, _ = src.read(timeout=1)
    assert ok and frame is not None
    start = time.monotonic()
    assert src.read(timeout=0.1) == (False, None, 0.0)
    assert time.monotonic() - start < 1.0


def test_reconnects_after_source_drops(make_source):
    first = FakeCapture(frames=3)
    second = FakeCapture(frames=10_000, start=100)
    src, opens = make_source([first, None, second])  # drop, one failed reopen, then success

    deadline = time.monotonic() + 3
    reconnected = False
    while time.monotonic() < deadline:
        ok, frame, _ = src.read(timeout=0.5)
        if ok and frame[0, 0, 0] > 100:
            reconnected = True
            break
    assert reconnected, "never received a frame from the reopened source"
    assert first.released
    assert len(opens) >= 3


def test_close_stops_reader_thread(make_source):
    src, _ = make_source([FakeCapture(frames=10_000)])
    assert src.read(timeout=1)[0]
    src.close()
    assert not src._thread.is_alive()
    assert src.read(timeout=0.5) == (False, None, 0.0)
    src.close()  # second close is harmless


# Video files (consented recordings) --------------------------------------------------------------------

class FileCapture:
    def __init__(self, frames, fps):
        self.remaining, self.fps = frames, fps

    def get(self, prop):
        return self.fps

    def read(self):
        if self.remaining <= 0:
            return False, None
        self.remaining -= 1
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self):
        pass


def test_video_file_plays_at_its_frame_rate_then_ends_without_reconnecting(tmp_path, monkeypatch):
    clip = tmp_path / "private" / "clip.mp4"
    clip.parent.mkdir()
    clip.write_bytes(b"not really a video")
    opens = []
    monkeypatch.setattr(capture, "_open_capture", lambda s: opens.append(s) or FileCapture(5, fps=20.0))
    source = FrameSource(str(clip))
    try:
        assert source.is_file and source.description == "clip.mp4"  # no folder names in logs
        stamps = []
        while True:
            ok, _, ts = source.read(timeout=1.0)
            if not ok:
                break
            stamps.append(ts)
        assert len(stamps) == 5 and source.ended
        assert stamps[-1] - stamps[0] == pytest.approx(4 / 20.0, abs=0.04)  # paced, not decoded flat out
        assert len(opens) == 1
    finally:
        source.close()
    assert describe_source("phone") == "phone"
