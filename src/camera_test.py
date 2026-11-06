import argparse
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

WINDOW = "Sentra - Pose Test (click the camera menu to switch, Q: quit)"
WINDOWS_PREFIX = "windows:"
PHONE_HINT = "Windows Virtual Camera"
REFRESH_SECONDS = 2.0  # how often the camera list is re-scanned while the dropdown is open
STATUS_SECONDS = 4.0  # how long a status message stays on screen

# Camera dropdown drawn on top of the video (OpenCV windows have no native widgets).
MENU_X, MENU_Y = 10, 10
ROW_H = 32
PAD = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
WHITE, DARK, HOVER, ACCENT = (255, 255, 255), (40, 40, 40), (80, 80, 80), (60, 160, 60)


class Camera:
    """One entry in the dropdown: a stable key, a label, and what to open."""

    def __init__(self, key: str, label: str, source):
        self.key, self.label, self.source = key, label, source


def windows_devices() -> list[tuple[str, str]] | None:
    """(id, name) of every Windows camera, or None when the Windows camera API is unavailable."""
    if sys.platform != "win32":
        return None
    try:
        from windows_camera import list_devices

        return list_devices()
    except Exception:  # winrt packages missing or camera service unavailable
        return None


def nice_name(name: str) -> str:
    return name.replace(f"({PHONE_HINT})", "(phone)").strip()


def discover_cameras(extra_sources: list) -> list[Camera]:
    """Current camera list. On Windows: every camera Windows knows about, re-scanned on each call."""
    cameras = []
    for source in extra_sources:  # camera URLs given with --source
        cameras.append(Camera(f"url:{source}", "Stream", source))  # never show the URL on screen
    devices = windows_devices()
    if devices is None:  # other platforms: OpenCV webcam indices only
        for index in sorted({0, *[s for s in extra_sources if isinstance(s, int)]}):
            cameras.append(Camera(f"cv:{index}", f"Webcam {index}", index))
    else:
        for device_id, name in devices:
            cameras.append(Camera(f"win:{device_id}", nice_name(name), (WINDOWS_PREFIX, device_id)))
    return cameras


def pick_initial(raw: str, cameras: list[Camera]) -> Camera | None:
    """Match --source: an index into the list, 'phone', 'windows:<part of name>', or a URL."""
    windows = [c for c in cameras if c.key.startswith("win:")]
    lowered = raw.lower()
    if lowered == "phone":
        return next((c for c in windows if "(phone)" in c.label), None)
    if lowered.startswith(WINDOWS_PREFIX):
        wanted = lowered[len(WINDOWS_PREFIX):]
        return next((c for c in windows if wanted in c.label.lower()), None)
    if raw.isdecimal():
        if windows:  # on Windows, 0 = first camera Windows lists (the IR face-unlock camera is not listed)
            index = int(raw)
            return windows[index] if index < len(windows) else None
        return next((c for c in cameras if c.key == f"cv:{raw}"), None)
    return next((c for c in cameras if c.key == f"url:{raw}"), None)


def open_capture(camera: Camera):
    source = camera.source
    if isinstance(source, tuple) and source[0] == WINDOWS_PREFIX:
        from windows_camera import WindowsCamera  # Windows-only dependency, imported on demand

        return WindowsCamera(device_id=source[1])
    return cv2.VideoCapture(source)


def try_open(camera: Camera):
    """Open a camera and wait for its first frame. Returns (capture, frame) or (None, None)."""
    try:
        capture = open_capture(camera)
    except Exception as error:  # e.g. the device vanished or is in use by another app
        print(f"Cannot open {camera.label}: {error}")
        return None, None
    if capture.isOpened():
        ok, frame = capture.read()
        if ok:
            return capture, frame
    capture.release()
    return None, None


def reset_tracking(model: YOLO) -> None:
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", None) or []:
        tracker.reset()


class CameraMenu:
    """A click-to-open camera dropdown drawn onto each frame."""

    def __init__(self):
        self.labels: list[str] = []
        self.width = 0
        self.open = False
        self.hover = None  # row index under the mouse (-1 = header)
        self.chosen = None  # row index clicked, waiting for the main loop to switch
        self.just_opened = False

    def set_labels(self, labels: list[str]) -> None:
        self.labels = labels
        texts = [f"Camera: {label}" for label in labels] or ["Camera: none"]
        widest = max(cv2.getTextSize(text, FONT, FONT_SCALE, 1)[0][0] for text in texts)
        self.width = widest + 2 * PAD + 24  # room for the arrow
        if self.hover is not None and self.hover >= len(labels):
            self.hover = None

    def _row_at(self, x: int, y: int):
        """-1 for the header button, a row index for an open list item, None for anywhere else."""
        if not MENU_X <= x <= MENU_X + self.width:
            return None
        if MENU_Y <= y < MENU_Y + ROW_H:
            return -1
        if self.open:
            row = (y - MENU_Y - ROW_H) // ROW_H
            if 0 <= row < len(self.labels):
                return row
        return None

    def on_mouse(self, event, x, y, _flags, _param):
        row = self._row_at(x, y)
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover = row
        elif event == cv2.EVENT_LBUTTONDOWN:
            if row == -1:
                self.open = not self.open
                self.just_opened = self.open
            elif row is not None:
                self.chosen = row
                self.open = False
            else:
                self.open = False

    def draw(self, frame, current_label: str, current_row) -> None:
        self._box(frame, 0, f"Camera: {current_label}", HOVER if self.hover == -1 else DARK)
        cx, cy = MENU_X + self.width - PAD - 6, MENU_Y + ROW_H // 2
        tip, base = (cy + 4, cy - 3) if not self.open else (cy - 4, cy + 3)
        arrow = np.array([[cx - 6, base], [cx + 6, base], [cx, tip]], np.int32)
        cv2.fillPoly(frame, [arrow], WHITE, cv2.LINE_AA)
        if not self.open:
            return
        for row, label in enumerate(self.labels):
            fill = ACCENT if row == current_row else HOVER if row == self.hover else DARK
            self._box(frame, row + 1, label, fill)

    def _box(self, frame, slot: int, text: str, fill) -> None:
        y1 = MENU_Y + slot * ROW_H
        y2 = y1 + ROW_H
        cv2.rectangle(frame, (MENU_X, y1), (MENU_X + self.width, y2), fill, -1)
        cv2.rectangle(frame, (MENU_X, y1), (MENU_X + self.width, y2), WHITE, 1)
        cv2.putText(frame, text, (MENU_X + PAD, y1 + 21), FONT, FONT_SCALE, WHITE, 1, cv2.LINE_AA)


def draw_status(frame, text: str) -> None:
    y = frame.shape[0] - 15
    cv2.putText(frame, text, (MENU_X, y), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (MENU_X, y), FONT, 0.6, (0, 220, 255), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentra basic YOLO pose camera test")
    parser.add_argument(
        "--source",
        default="0",
        help="Starting camera: index (0 = first camera), approved camera URL, 'phone' (Phone Link camera), "
        "or 'windows:<part of camera name>'",
    )
    parser.add_argument("--track", action="store_true", help="Enable ByteTrack person tracking")
    parser.add_argument("--list-cameras", action="store_true", help="List camera names and exit")
    args = parser.parse_args()

    raw = args.source
    url_sources = [raw] if "://" in raw else []
    cameras = discover_cameras(url_sources)
    if args.list_cameras:
        for index, camera in enumerate(cameras):
            print(f"{index}: {camera.label}")
        return

    print("Cameras:", ", ".join(c.label for c in cameras) or "none found")
    model = YOLO("yolo26n-pose.pt")

    initial = pick_initial(raw, cameras)
    if initial is None:
        raise SystemExit(f"No camera matches --source {raw!r}. Run with --list-cameras to see them.")
    current = initial
    capture, frame = try_open(current)
    if capture is None:
        others = [c for c in cameras if c.key != initial.key]
        for camera in others:
            capture, frame = try_open(camera)
            if capture is not None:
                print(f"{initial.label} sent no video; using {camera.label} instead.")
                current = camera
                break
    if capture is None:
        raise SystemExit("Cannot open camera source. Check permissions, source, and network.")
    print(f"Using {current.label}")

    menu = CameraMenu()
    menu.set_labels([c.label for c in cameras])
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, menu.on_mouse)
    status, status_until = "", 0.0
    last_refresh = time.monotonic()
    last_annotated = None

    def show_status(text: str) -> None:
        nonlocal status, status_until
        status, status_until = text, time.monotonic() + STATUS_SECONDS
        print(text)

    def refresh() -> None:
        nonlocal cameras, last_refresh
        cameras = discover_cameras(url_sources)
        menu.set_labels([c.label for c in cameras])
        last_refresh = time.monotonic()

    def switch_to(camera: Camera) -> bool:
        nonlocal capture, current, frame
        if last_annotated is not None:  # tell the user we're working while the camera starts up
            waiting = last_annotated.copy()
            draw_status(waiting, f"Connecting to {camera.label}...")
            cv2.imshow(WINDOW, waiting)
            cv2.waitKey(1)
        new_capture, new_frame = try_open(camera)
        if new_capture is None:
            return False
        capture.release()
        capture, current, frame = new_capture, camera, new_frame
        reset_tracking(model)  # tracking IDs from the old camera mean nothing on the new one
        return True

    try:
        while True:
            if frame is None:
                ok, frame = capture.read()
                if not ok:
                    lost = current.label
                    refresh()
                    fallback = [c for c in cameras if c.key != current.key]
                    if not any(switch_to(c) for c in fallback):
                        print(f"Lost {lost} and no other camera works; ending camera test.")
                        break
                    show_status(f"Lost {lost}; switched to {current.label}")
            if args.track:
                results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
            else:
                results = model.predict(frame, verbose=False)
            frame = None
            annotated = results[0].plot()
            last_annotated = annotated.copy()

            keys = [c.key for c in cameras]
            current_row = keys.index(current.key) if current.key in keys else None
            menu.draw(annotated, current.label, current_row)
            if status and time.monotonic() < status_until:
                draw_status(annotated, status)
            cv2.imshow(WINDOW, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if menu.just_opened or (menu.open and time.monotonic() - last_refresh > REFRESH_SECONDS):
                menu.just_opened = False
                refresh()

            chosen, menu.chosen = menu.chosen, None
            if chosen is not None and chosen < len(cameras) and cameras[chosen].key != current.key:
                target = cameras[chosen]
                if switch_to(target):
                    show_status(f"Switched to {current.label}")
                else:
                    show_status(f"No video from {target.label}; staying on {current.label}")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
