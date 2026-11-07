"""OpenCV debug view for Sentra Guide: what the camera sees, what was detected, and what was said.

Keys (click the window first): m mute or unmute the voice, p privacy (hide the camera image), q or Esc quit,
c open the camera list (then 1-9 to pick, Esc to close it). The list can also be opened and used with the mouse.
r turns the current camera's picture another 90 degrees clockwise (remembered per camera).

Top bar: camera-on indicator, source, fps, latency, distance mode, voice state.
Toolbar (when a camera list is given): the camera dropdown and status messages such as "Switching to ...".
The dropdown re-reads the connected cameras while it's open, so newly connected cameras appear.
Video: dashed lines mark LEFT | AHEAD | RIGHT. Boxes are red (central and large), orange (in or touching the
central band), green (to the side) or grey (not announced: low score, small, or not on the label list).
Bottom bar: the last announcement and how long ago it was made, plus the "not a navigation aid" reminder.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from sentra.guide.alerts import AlertPolicy
from sentra.guide.cameras import CameraCatalog, CameraChoice
from sentra.guide.pipeline import GuideControls
from sentra.guide.schema import GuideAlert, GuideUpdate, Severity, SpatialObject
from sentra.guide.spatial import SpatialConfig

WINDOW_TITLE = "Sentra Guide - experimental, not a navigation aid (q to quit)"
FONT = cv2.FONT_HERSHEY_SIMPLEX
MAX_WIDTH = 1280
MAX_HEIGHT = 720
BAR = 30
BLANK_SIZE = (480, 640)  # video area shown before the first frame arrives
NOTICE_S = 6.0           # how long a status message stays in the toolbar

RED, ORANGE, GREEN, GREY = (40, 40, 230), (0, 150, 255), (60, 190, 60), (150, 150, 150)
WHITE, BLACK, YELLOW = (255, 255, 255), (0, 0, 0), (0, 220, 255)
DARK, HOVER, ACCENT, TOOLBAR_BG = (45, 45, 45), (85, 85, 85), (60, 140, 60), (25, 25, 25)
SEVERITY_COLORS = {Severity.HIGH: RED, Severity.MEDIUM: ORANGE, Severity.LOW: GREEN}

MENU_X, MENU_PAD, ROW_H, MENU_FONT = 6, 8, 24, 0.5


def _ascii(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")  # OpenCV fonts are ASCII only


def _put(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.5,
         color: tuple[int, int, int] = WHITE, thickness: int = 1) -> None:
    cv2.putText(img, _ascii(text), org, FONT, scale, color, thickness, cv2.LINE_AA)


def _text_width(text: str, scale: float = MENU_FONT) -> int:
    return cv2.getTextSize(_ascii(text), FONT, scale, 1)[0][0]


class CameraMenu:
    """The camera dropdown: a button in the toolbar that opens a list of cameras over the video.

    Rows come from a CameraCatalog, which re-reads the connected cameras in the background while the list is
    open. Picking a row puts it in `controls.requested_camera`; the pipeline does the switch.
    """

    def __init__(self, catalog: CameraCatalog, controls: GuideControls, top: int = BAR) -> None:
        self.catalog = catalog
        self.controls = controls
        self.top = top + (BAR - ROW_H) // 2   # y of the button
        self.open = False
        self.hover: int | None = None          # -1 = the button, 0.. = a row
        self.rows: list[CameraChoice] = []     # as last drawn, so clicks match what's on screen
        self.width = 0

    def header(self) -> str:
        return f"Camera: {self.controls.camera_label or 'none'}"

    def toggle(self) -> None:
        self.open = not self.open
        if self.open:
            self.catalog.refresh(force=True)  # new cameras show up as soon as the scan finishes
            self.rows = self.catalog.choices

    def pick(self, index: int) -> None:
        if 0 <= index < len(self.rows):
            self.controls.requested_camera = self.rows[index]
            self.open = False

    def _row_at(self, x: int, y: int) -> int | None:
        if not MENU_X <= x <= MENU_X + self.width:
            return None
        if self.top <= y < self.top + ROW_H:
            return -1
        if self.open:
            row = (y - self.top - ROW_H) // ROW_H
            if 0 <= row < len(self.rows):
                return row
        return None

    def on_mouse(self, event: int, x: int, y: int, _flags: int = 0, _param: object = None) -> None:
        row = self._row_at(x, y)
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover = row
        elif event == cv2.EVENT_LBUTTONDOWN:
            if row == -1:
                self.toggle()
            elif row is not None:
                self.pick(row)
            else:
                self.open = False

    def handle_key(self, key: int) -> bool:
        """Keys for the menu. Returns True if the key was used."""
        if key == ord("c"):
            self.toggle()
            return True
        if not self.open:
            return False
        if key == 27:  # Esc closes the list instead of quitting
            self.open = False
            return True
        if ord("1") <= key <= ord("9"):
            self.pick(key - ord("1"))
            return True
        return False

    def draw(self, canvas: np.ndarray) -> None:
        if self.open:
            self.catalog.refresh()  # throttled; re-reads the cameras every couple of seconds while open
            self.rows = self.catalog.choices
        texts = [f"{i}. {c.label}" if i <= 9 else c.label for i, c in enumerate(self.rows, 1)]
        extra = []
        if self.open and self.catalog.scanning:
            extra.append("Looking for cameras...")
        elif self.open and not self.rows:
            extra.append("No cameras found")
        header = self.header()
        self.width = max(_text_width(t) for t in [header + "    ", *texts, *extra]) + 2 * MENU_PAD + 16

        self._box(canvas, 0, header, HOVER if self.hover == -1 else DARK)
        cx, cy = MENU_X + self.width - MENU_PAD - 6, self.top + ROW_H // 2
        tip, base = (cy + 4, cy - 3) if not self.open else (cy - 4, cy + 3)
        cv2.fillPoly(canvas, [np.array([[cx - 6, base], [cx + 6, base], [cx, tip]], np.int32)], WHITE, cv2.LINE_AA)
        if not self.open:
            return
        for i, (choice, text) in enumerate(zip(self.rows, texts)):
            current = choice.source == self.controls.camera_source
            self._box(canvas, i + 1, text, ACCENT if current else HOVER if self.hover == i else DARK)
        for j, text in enumerate(extra):
            self._box(canvas, len(self.rows) + 1 + j, text, DARK, GREY)

    def _box(self, canvas: np.ndarray, slot: int, text: str, fill, color=WHITE) -> None:
        y1 = self.top + slot * ROW_H
        y2 = min(y1 + ROW_H, canvas.shape[0] - 1)
        if y1 >= y2:
            return
        x2 = min(MENU_X + self.width, canvas.shape[1] - 1)
        cv2.rectangle(canvas, (MENU_X, y1), (x2, y2), fill, -1)
        cv2.rectangle(canvas, (MENU_X, y1), (x2, y2), (120, 120, 120), 1)
        _put(canvas, text, (MENU_X + MENU_PAD, y1 + ROW_H - 8), MENU_FONT, color, 1)


class GuideView:
    """Implements GuideSink in an OpenCV window."""

    def __init__(
        self,
        controls: GuideControls,
        policy: AlertPolicy,
        spatial: SpatialConfig | None = None,
        *,
        voice_name: str = "windows",
        show_window: bool = True,
        cameras: CameraCatalog | None = None,
    ) -> None:
        self.controls = controls
        self.policy = policy
        self.spatial = spatial or SpatialConfig()
        self.voice_name = voice_name
        self.show_window = show_window
        self.menu = CameraMenu(cameras, controls) if cameras is not None else None
        self.toolbar = BAR if self.menu is not None else 0
        self._stop = False
        self._last_alert: GuideAlert | None = None
        self._last_alert_at = 0.0
        self._window_open = False
        self._last_base: np.ndarray | None = None  # last frame's canvas without the menu, for idle()

    # GuideSink -------------------------------------------------------------------------------------------

    def publish(self, update: GuideUpdate) -> None:
        if update.alerts:
            self._last_alert = update.alerts[-1]
            self._last_alert_at = time.perf_counter()
        self._show(self.render(update))

    def idle(self, message: str = "") -> None:
        """No new frame (switching cameras, or the stream stalled): redraw so keys and the dropdown keep working."""
        self._show(self.render_idle(message))

    def should_stop(self) -> bool:
        return self._stop

    def close(self) -> None:
        if self._window_open:
            self._window_open = False
            try:
                cv2.destroyWindow(WINDOW_TITLE)
                cv2.waitKey(1)
            except cv2.error:
                pass  # already closed with the window's X button: nothing left to destroy

    def _show(self, canvas: np.ndarray) -> None:
        if not self.show_window:
            return
        cv2.imshow(WINDOW_TITLE, canvas)
        if not self._window_open:
            self._window_open = True
            if self.menu is not None:
                cv2.setMouseCallback(WINDOW_TITLE, self.menu.on_mouse)
        self.handle_key(cv2.waitKey(1) & 0xFF)
        if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
            self._stop = True  # closed with the window's X button

    # Keys --------------------------------------------------------------------------------------------------

    def handle_key(self, key: int) -> None:
        if self.menu is not None and self.menu.handle_key(key):
            return
        if key in (ord("q"), 27):
            self._stop = True
        elif key == ord("m"):
            self.controls.muted = not self.controls.muted
        elif key == ord("p"):
            self.controls.privacy = not self.controls.privacy
        elif key == ord("r"):
            self.controls.rotate_clockwise()

    # Drawing -----------------------------------------------------------------------------------------------

    def render_idle(self, message: str = "") -> np.ndarray:
        """The last frame's screen (or a blank one) with `message` over the video, and a fresh toolbar."""
        if self._last_base is not None:
            canvas = self._last_base.copy()
            if self.controls.privacy:
                # The last frame may have been drawn before p was pressed: never show it while privacy is on.
                canvas[BAR + self.toolbar:canvas.shape[0] - BAR] = 0
        else:
            h, w = BLANK_SIZE
            canvas = np.zeros((h + 2 * BAR + self.toolbar, w, 3), np.uint8)
            cv2.rectangle(canvas, (0, 0), (w, BAR), (40, 40, 40), -1)
            _put(canvas, "Starting camera...", (10, BAR - 10), 0.45, WHITE, 1)
            self._draw_bottom_bar(canvas, canvas.shape[0] - BAR)
        if message:
            top = BAR + self.toolbar
            bottom = canvas.shape[0] - BAR
            width = _text_width(message, 0.7)
            x = max((canvas.shape[1] - width) // 2, 10)
            y = (top + bottom) // 2
            cv2.rectangle(canvas, (x - 14, y - 26), (x + width + 14, y + 14), (30, 30, 30), -1)
            _put(canvas, message, (x, y), 0.7, YELLOW, 2)
        self._draw_toolbar(canvas)
        if self.menu is not None:
            self.menu.draw(canvas)
        return canvas

    def render(self, update: GuideUpdate) -> np.ndarray:
        frame = update.frame
        h, w = frame.frame_height, frame.frame_width
        if update.image is not None and not self.controls.privacy:
            video = update.image.copy()
        else:
            video = np.zeros((h, w, 3), np.uint8)
        scale = min(MAX_WIDTH / w, MAX_HEIGHT / h, 1.0)
        if scale < 1.0:
            video = cv2.resize(video, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        vh, vw = video.shape[:2]

        self._draw_regions(video)
        for obj in update.objects:
            self._draw_object(video, obj, scale)
        if update.stale:
            _put(video, "LAGGING: frame too old, not announcing", (10, vh - 12), 0.55, YELLOW, 2)

        canvas_w = max(vw, 640)
        top = BAR + self.toolbar
        canvas = np.zeros((vh + 2 * BAR + self.toolbar, canvas_w, 3), np.uint8)
        canvas[top:top + vh, :vw] = video
        self._draw_top_bar(canvas, update)
        self._draw_bottom_bar(canvas, top + vh)
        self._last_base = canvas.copy()
        self._draw_toolbar(canvas)
        if self.menu is not None:
            self.menu.draw(canvas)
        return canvas

    def _draw_toolbar(self, canvas: np.ndarray) -> None:
        """Camera dropdown button (drawn by the menu) and a status message to its right."""
        if not self.toolbar:
            return
        cv2.rectangle(canvas, (0, BAR), (canvas.shape[1], BAR + self.toolbar), TOOLBAR_BG, -1)
        controls = self.controls
        recent = time.perf_counter() - controls.notice_at < NOTICE_S
        text, color = ((controls.notice, YELLOW) if controls.notice and (controls.switching or recent)
                       else (f"c: choose camera   r: rotate ({controls.rotation} deg)", GREY))
        x = MENU_X + max(self.menu.width, _text_width(self.menu.header() + "    ") + 2 * MENU_PAD + 16) + 14
        _put(canvas, text, (x, BAR + self.toolbar - 10), 0.5, color, 1)

    def _draw_regions(self, video: np.ndarray) -> None:
        vh, vw = video.shape[:2]
        for edge in (self.spatial.left_edge, self.spatial.right_edge):
            x = int(edge * vw)
            for y in range(0, vh, 16):
                cv2.line(video, (x, y), (x, min(y + 8, vh)), WHITE, 1)
        for text, center in (("LEFT", self.spatial.left_edge / 2),
                             ("AHEAD", (self.spatial.left_edge + self.spatial.right_edge) / 2),
                             ("RIGHT", (1 + self.spatial.right_edge) / 2)):
            size = cv2.getTextSize(text, FONT, 0.5, 1)[0]
            _put(video, text, (int(center * vw - size[0] / 2), 18), 0.5, WHITE, 1)

    def _draw_object(self, video: np.ndarray, obj: SpatialObject, scale: float) -> None:
        det = obj.detection
        x1, y1, x2, y2 = (int(v * scale) for v in det.box)
        color = SEVERITY_COLORS[obj.severity] if self.policy.is_confident(obj) else GREY
        cv2.rectangle(video, (x1, y1), (x2, y2), color, 2)
        parts = [det.label, f"{det.confidence:.2f}"]
        if det.track_id is not None:
            parts.append(f"#{det.track_id}")
        if obj.depth.meters is not None:
            parts.append(f"{obj.depth.meters:.1f} m" + ("" if obj.depth.validated else " est."))
        text = " ".join(parts)
        size = cv2.getTextSize(text, FONT, 0.5, 1)[0]
        top = max(y1 - size[1] - 6, 0)
        cv2.rectangle(video, (x1, top), (x1 + size[0] + 6, top + size[1] + 6), color, -1)
        _put(video, text, (x1 + 3, top + size[1] + 2), 0.5, BLACK, 1)

    def _draw_top_bar(self, canvas: np.ndarray, update: GuideUpdate) -> None:
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], BAR), (40, 40, 40), -1)
        cv2.circle(canvas, (14, BAR // 2), 6, RED, -1)
        camera = "CAMERA ON" + (" (image hidden)" if self.controls.privacy else "")
        voice = "muted" if self.controls.muted else self.voice_name
        text = (f"{camera} | {update.frame.source} | {update.fps:.1f} fps | {update.latency_ms:.0f} ms | "
                f"{update.depth_status} | voice: {voice} (m) | p privacy | q quit")
        _put(canvas, text, (26, BAR - 10), 0.45, WHITE, 1)

    def _draw_bottom_bar(self, canvas: np.ndarray, top: int) -> None:
        cv2.rectangle(canvas, (0, top), (canvas.shape[1], top + BAR), (40, 40, 40), -1)
        if self._last_alert is not None:
            ago = time.perf_counter() - self._last_alert_at
            muted = " (muted)" if self.controls.muted else ""
            text = f'Last: "{self._last_alert.text}"{muted} {ago:.0f} s ago'
        else:
            text = "Nothing announced yet"
        _put(canvas, text, (10, top + BAR - 10), 0.5, YELLOW, 1)
        note = "Experimental. Not a navigation aid."
        size = cv2.getTextSize(note, FONT, 0.45, 1)[0]
        _put(canvas, note, (canvas.shape[1] - size[0] - 10, top + BAR - 10), 0.45, GREY, 1)


class HeadlessSink:
    """No window (e.g. --no-view): alerts still go to the voice, the console and the log."""

    def __init__(self, max_seconds: float | None = None) -> None:
        self.max_seconds = max_seconds
        self._started = time.perf_counter()

    def publish(self, update: GuideUpdate) -> None:
        for alert in update.alerts:
            print(f"[alert] {alert.text}  ({alert.reason})", flush=True)

    def should_stop(self) -> bool:
        return self.max_seconds is not None and time.perf_counter() - self._started >= self.max_seconds

    def close(self) -> None:
        pass
