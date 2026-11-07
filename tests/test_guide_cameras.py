"""Camera dropdown: the live camera list (sentra.guide.cameras) and the menu in the debug view. Owner: iesa242."""
import threading
import time

import cv2
import numpy as np
import pytest
from guide_helpers import FakeClock

import sentra.guide.view as view_module
from sentra.guide.alerts import AlertPolicy
from sentra.guide.cameras import CameraCatalog, CameraChoice, nice_name, stream_label
from sentra.guide.pipeline import GuideControls
from sentra.guide.view import BAR, MENU_X, ROW_H, YELLOW, GuideView

BUILT_IN = ("id-hp", "HP 5MP Camera")
DROIDCAM = ("id-droid", "DroidCam Video")
PHONE_LINK = ("id-phone", "Pixel 9 (Windows Virtual Camera)")
IPHONE = "http://10.0.0.7:4747/video"
ANDROID = "http://10.0.0.8:8080/video"


class Devices:
    """A lister whose answer can change between scans, like plugging a camera in."""

    def __init__(self, *devices):
        self.devices = list(devices)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return list(self.devices)


def scanned(catalog):
    catalog.refresh(force=True)
    catalog.wait()
    return catalog


# The camera list -------------------------------------------------------------------------------------------

def test_lists_streams_then_windows_cameras_with_friendly_names():
    catalog = scanned(CameraCatalog([IPHONE, ANDROID, IPHONE], lister=Devices(BUILT_IN, PHONE_LINK)))
    assert [c.label for c in catalog.choices] == [
        "Stream 1 (port 4747)", "Stream 2 (port 8080)", "HP 5MP Camera", "Pixel 9 (Phone Link)"]
    assert [c.source for c in catalog.choices] == [
        IPHONE, ANDROID, "windows:HP 5MP Camera", "windows:Pixel 9 (Windows Virtual Camera)"]
    assert all("10.0.0" not in c.label for c in catalog.choices)  # stream addresses are never shown


def test_a_camera_connected_later_appears_after_the_next_scan():
    devices = Devices(BUILT_IN)
    catalog = scanned(CameraCatalog(lister=devices))
    assert [c.label for c in catalog.choices] == ["HP 5MP Camera"]
    devices.devices.append(DROIDCAM)  # e.g. the DroidCam client was started
    scanned(catalog)
    assert [c.label for c in catalog.choices] == ["HP 5MP Camera", "DroidCam Video"]
    devices.devices.remove(BUILT_IN)  # and one unplugged
    assert [c.label for c in scanned(catalog).choices] == ["DroidCam Video"]


def test_refresh_is_throttled_unless_forced():
    clock = FakeClock()
    devices = Devices(BUILT_IN)
    catalog = CameraCatalog(lister=devices, min_refresh_s=2.0, clock=clock)
    assert catalog.refresh()
    catalog.wait()
    clock.now = 1.0
    assert not catalog.refresh()       # too soon
    assert catalog.refresh(force=True)  # opening the dropdown always re-reads
    catalog.wait()
    clock.now = 4.0
    assert catalog.refresh()
    catalog.wait()
    assert devices.calls == 3


def test_scanning_never_blocks_the_caller():
    release = threading.Event()

    def slow():
        release.wait(2.0)
        return [BUILT_IN]

    catalog = CameraCatalog(lister=slow)
    start = time.perf_counter()
    assert catalog.refresh(force=True)
    assert time.perf_counter() - start < 0.1 and catalog.scanning
    assert not catalog.refresh(force=True)  # one scan at a time
    release.set()
    catalog.wait()
    assert not catalog.scanning and [c.label for c in catalog.choices] == ["HP 5MP Camera"]


def test_without_a_device_list_it_offers_webcam_numbers_and_keeps_the_last_good_list():
    assert [c.source for c in scanned(CameraCatalog(lister=lambda: None)).choices] == [0, 1]

    def broken():
        raise OSError("camera service unavailable")

    assert [c.label for c in scanned(CameraCatalog(lister=broken)).choices] == ["Webcam 0", "Webcam 1"]

    answers = [[BUILT_IN], None]
    catalog = scanned(CameraCatalog(lister=lambda: answers.pop(0)))
    assert [c.label for c in scanned(catalog).choices] == ["HP 5MP Camera"]  # a failed re-scan changes nothing


def test_labels_for_the_start_up_source():
    catalog = scanned(CameraCatalog([IPHONE], lister=Devices(BUILT_IN)))
    assert catalog.label_for(IPHONE) == "Stream 1 (port 4747)"
    assert catalog.label_for(0) == "Webcam 0"
    assert catalog.label_for("windows:DroidCam") == "DroidCam"
    assert catalog.label_for("phone") == "phone"
    assert catalog.label_for("http://10.0.0.9:1234/x") == "Stream"
    assert nice_name("Pixel 9 (Windows Virtual Camera)") == "Pixel 9 (Phone Link)"
    assert stream_label(3, "http://host/video") == "Stream 3"


# The dropdown in the debug view ----------------------------------------------------------------------------

def make_view(*devices, streams=()):
    controls = GuideControls(camera_label="HP 5MP Camera", camera_source="windows:HP 5MP Camera")
    catalog = scanned(CameraCatalog(streams, lister=Devices(*devices)))
    view = GuideView(controls, AlertPolicy(), show_window=False, cameras=catalog)
    view.render_idle()  # draws the button once, which sets its size
    return view, controls


def click(view, x, y):
    view.menu.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y)
    view.render_idle()


def row_y(slot):
    """Middle of the button (slot 0) or of list row `slot` (1 = first camera)."""
    return view_module.BAR + (BAR - ROW_H) // 2 + slot * ROW_H + ROW_H // 2


def test_clicking_the_button_opens_the_list_and_clicking_a_camera_requests_it():
    view, controls = make_view(BUILT_IN, DROIDCAM)
    menu = view.menu
    assert not menu.open
    click(view, MENU_X + 5, row_y(0))
    assert menu.open and [c.label for c in menu.rows] == ["HP 5MP Camera", "DroidCam Video"]
    click(view, MENU_X + 5, row_y(2))
    assert controls.requested_camera == CameraChoice("win:id-droid", "DroidCam Video", "windows:DroidCam Video")
    assert not menu.open
    click(view, MENU_X + 5, row_y(0))
    click(view, 600, 400)  # anywhere else closes it
    assert not menu.open


def test_opening_the_list_rescans_so_new_cameras_show_up():
    view, _ = make_view(BUILT_IN)
    view.menu.catalog._lister.devices.append(DROIDCAM)
    view.handle_key(ord("c"))
    view.menu.catalog.wait()
    view.render_idle()
    assert [c.label for c in view.menu.rows] == ["HP 5MP Camera", "DroidCam Video"]


def test_keys_open_pick_and_close_without_quitting():
    view, controls = make_view(BUILT_IN, DROIDCAM)
    view.handle_key(ord("c"))
    assert view.menu.open
    view.handle_key(27)  # Esc closes the list, it doesn't quit
    assert not view.menu.open and not view.should_stop()
    view.handle_key(ord("c"))
    view.render_idle()
    view.handle_key(ord("2"))
    assert controls.requested_camera.label == "DroidCam Video"
    view.handle_key(ord("1"))  # list closed again, so number keys do nothing
    assert controls.requested_camera.label == "DroidCam Video"
    view.handle_key(27)
    assert view.should_stop()


def test_the_camera_in_use_is_highlighted():
    view, _ = make_view(BUILT_IN, DROIDCAM)
    view.handle_key(ord("c"))
    canvas = view.render_idle()
    current = canvas[row_y(1), MENU_X + 3].tolist()
    other = canvas[row_y(2), MENU_X + 3].tolist()
    assert current == list(view_module.ACCENT) and other != current


def test_idle_screen_keeps_the_last_frame_and_shows_the_message():
    view, controls = make_view(BUILT_IN)
    controls.switching = True
    controls.tell("Switching to DroidCam Video...")
    canvas = view.render_idle("Switching to DroidCam Video...")
    assert canvas.shape == (480 + 3 * BAR, 640, 3)  # blank screen before the first frame: bars + toolbar
    middle = canvas[BAR * 2:-BAR]
    assert (middle == YELLOW).all(axis=2).any()  # the message is drawn over the video area


def _frame_update(color=(0, 255, 0)):
    """A GuideUpdate with a solid-colour 480x640 camera image and nothing detected."""
    from sentra.guide.schema import DetectionFrame, GuideUpdate

    image = np.zeros((480, 640, 3), np.uint8)
    image[:] = color
    frame = DetectionFrame(1, 0.0, 640, 480, (), source="test")
    return GuideUpdate(frame, (), (), image, fps=25.0, latency_ms=30.0, depth_status="Distance: off")


def _green_pixels(canvas):
    return int((canvas == [0, 255, 0]).all(axis=2).sum())


@pytest.mark.parametrize("with_menu", [True, False])
def test_privacy_pressed_while_waiting_hides_the_frozen_frame(with_menu):
    # babar's repro on #11: privacy turned on after the last frame, while switching or with a stalled stream.
    if with_menu:
        view, controls = make_view(BUILT_IN)
    else:
        controls = GuideControls()
        view = GuideView(controls, AlertPolicy(), show_window=False)
    assert _green_pixels(view.render(_frame_update())) > 250_000  # privacy off: the camera image is shown
    controls.privacy = True
    assert _green_pixels(view.render_idle("Waiting for the camera...")) == 0
    assert _green_pixels(view.render_idle()) == 0
    controls.privacy = False
    assert _green_pixels(view.render_idle()) > 250_000  # back on: the last frame may be shown again


def test_close_after_the_window_was_closed_with_x_does_not_crash(monkeypatch):
    view = GuideView(GuideControls(), AlertPolicy(), show_window=False)
    view._window_open = True

    def gone(_title):
        raise cv2.error("NULL window")

    monkeypatch.setattr(view_module.cv2, "destroyWindow", gone)
    view.close()  # used to raise cv2.error and skip the rest of shutdown
    view.close()


def test_view_without_a_camera_list_has_no_toolbar():
    view = GuideView(GuideControls(), AlertPolicy(), show_window=False)
    assert view.menu is None and view.render_idle().shape == (480 + 2 * BAR, 640, 3)
    view.handle_key(ord("c"))  # no list to open; nothing happens
    assert not view.should_stop()


def test_r_rotates_the_current_camera_and_remembers_it():
    view, controls = make_view(BUILT_IN)
    view.handle_key(ord("r"))
    view.handle_key(ord("r"))
    assert controls.rotation == 180 and controls.rotations == {"windows:HP 5MP Camera": 180}
    assert controls.notice == "Rotation 180 degrees"
    for _ in range(2):
        view.handle_key(ord("r"))
    assert controls.rotation == 0 and not view.should_stop()


def test_tee_sink_passes_idle_updates_to_the_window():
    from sentra.guide.view import HeadlessSink
    from sentra.guide.web import TeeSink

    class Window(HeadlessSink):
        def __init__(self):
            super().__init__()
            self.idles = []

        def idle(self, message):
            self.idles.append(message)

    window = Window()
    TeeSink(window, HeadlessSink()).idle("Switching to HP 5MP Camera...")  # the page sink has no idle(): fine
    assert window.idles == ["Switching to HP 5MP Camera..."]


def test_extra_sources_take_an_optional_rotation():
    from guide import build_parser, parse_extra_sources

    args = build_parser().parse_args(
        ["--extra-source", "http://10.0.0.7:4747/video", "90", "--extra-source", "http://10.0.0.8:8080/video"])
    assert parse_extra_sources(args.extra_source) == [
        ("http://10.0.0.7:4747/video", 90), ("http://10.0.0.8:8080/video", 0)]
    for bad in (["http://x/video", "45"], ["0"], ["http://x/video", "90", "extra"]):
        with pytest.raises(ValueError):
            parse_extra_sources([bad])


@pytest.mark.parametrize("key", [ord("m"), ord("p")])
def test_other_keys_still_work_with_the_menu(key):
    view, controls = make_view(BUILT_IN)
    view.handle_key(key)
    assert controls.muted or controls.privacy
    assert isinstance(view.render_idle(), np.ndarray)
