"""Tests for the opt-in Windows camera source (Phone Link / DroidCam). No real camera is used."""
from types import SimpleNamespace

import pytest

import sentra.capture as capture
from sentra.capture import PHONE_HINT, describe_source, windows_camera_name

# --- Routing: runs on every platform --------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (0, None),
        (1, None),
        ("http://192.168.1.5:8080/video", None),
        ("rtsp://cam.local/stream", None),
        ("phone", PHONE_HINT),
        ("PHONE", PHONE_HINT),
        ("windows:DroidCam", "DroidCam"),
        ("Windows: S25 Ultra ", "S25 Ultra"),
        ("windows:", None),
    ],
)
def test_only_opt_in_sources_go_to_windows(source, expected):
    assert windows_camera_name(source) == expected


def test_numbered_webcams_and_urls_stay_on_opencv(monkeypatch):
    opened_with_cv, opened_with_windows = [], []

    class FakeVideoCapture:
        def __init__(self, source):
            opened_with_cv.append(source)

        def isOpened(self):
            return True

        def set(self, *_):
            pass

    monkeypatch.setattr(capture.cv2, "VideoCapture", FakeVideoCapture)
    monkeypatch.setattr(capture, "_open_windows_camera", lambda name: opened_with_windows.append(name))

    for source in (0, 1, "http://192.168.1.5:8080/video"):
        capture._open_capture(source)
    capture._open_capture("phone")
    capture._open_capture("windows:DroidCam")

    assert opened_with_cv == [0, 1, "http://192.168.1.5:8080/video"]
    assert opened_with_windows == [PHONE_HINT, "DroidCam"]


def test_windows_source_names_are_printable():
    assert describe_source("phone") == "phone"
    assert describe_source("windows:DroidCam") == "windows:DroidCam"


# --- Windows reader helpers: need the winrt packages -----------------------------------------------

windows_camera = pytest.importorskip("sentra.windows_camera", reason="winrt packages are Windows-only")


def fmt(width, height, fps=30):
    return SimpleNamespace(
        video_format=SimpleNamespace(width=width, height=height),
        frame_rate=SimpleNamespace(numerator=fps, denominator=1),
    )


def test_pick_format_prefers_largest_non_square_up_to_max_width():
    formats = [fmt(800, 800), fmt(640, 480), fmt(1280, 720), fmt(1920, 1080), fmt(2560, 1440)]
    best = windows_camera._pick_format(formats)
    assert (best.video_format.width, best.video_format.height) == (1280, 720)


def test_pick_format_prefers_30_fps_at_equal_size():
    best = windows_camera._pick_format([fmt(1280, 720, 60), fmt(1280, 720, 30), fmt(1280, 720, 25)])
    assert best.frame_rate.numerator == 30


def test_pick_format_returns_none_when_nothing_fits():
    assert windows_camera._pick_format([fmt(800, 800), fmt(1920, 1080), fmt(0, 0)]) is None


def test_missing_timestamps_are_not_treated_as_repeats():
    # A camera that never reports SystemRelativeTime must still deliver its first and later frames.
    assert not windows_camera._is_repeat(None, None)
    assert not windows_camera._is_repeat(None, 123)
    assert windows_camera._is_repeat(123, 123)
    assert not windows_camera._is_repeat(124, 123)
