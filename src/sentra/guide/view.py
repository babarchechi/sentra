"""Sentra Guide dashboard (OpenCV window): the camera on the left, what the system knows on the right.

Keys (click the window first): m mute or unmute the voice, p privacy (hide the camera image), q or Esc quit,
c open the camera list (then 1-9 to pick, Esc to close it). The list can also be opened and used with the mouse.
r turns the current camera's picture another 90 degrees clockwise (remembered per camera).

Layout:
- Top bar: camera-on indicator, source, fps, latency, distance mode, voice state.
- Left: toolbar with the camera dropdown (when a camera list is given), then the video. Dashed lines mark
  LEFT | AHEAD | RIGHT; the zone of the latest cue lights up briefly. Boxes are red (central and large), orange
  (in or touching the central band), green (to the side) or grey (not announced).
- Right panel:
  - LAST CUE: what was just said, how long ago, and a LEFT / AHEAD / RIGHT strip showing where.
  - DETECTED AHEAD: detected objects in or touching the central band of the picture, most important first, each
    with its detector confidence. Called "detected ahead", not "in your path": without validated depth the
    system knows where something is in the picture, not whether it blocks the route. Each object has its detector
    confidence. When nothing is detected there it says so, and that this is not proof the way is clear.
  - ALSO IN VIEW: the other detections, with why each one isn't announced (score, size, not on the alert list).
  - RECENT CUES: the last few announcements with the time.
- Bottom bar: keys, and the reminder that this is experimental and the model knows only 80 kinds of objects.

The panel shows only what the detector reported. It never ranks danger, and it never says the way is clear.
"""
from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from sentra.guide.alerts import AlertPolicy, spoken_name
from sentra.guide.cameras import CameraCatalog, CameraChoice
from sentra.guide.obstacles import ObstacleScanner
from sentra.guide.pipeline import GuideControls
from sentra.guide.schema import SEVERITY_RANK, GuideAlert, GuideUpdate, Region, Severity, SpatialObject
from sentra.guide.spatial import SpatialConfig

WINDOW_TITLE = "Sentra Guide - experimental, not a navigation aid (q to quit)"
FONT = cv2.FONT_HERSHEY_SIMPLEX
MAX_WIDTH = 880          # largest video area; with the panel the window is at most 1280 wide
MAX_HEIGHT = 720
BAR = 30
PANEL_W = 400            # right-hand panel
BLANK_SIZE = (480, 640)  # video area shown before the first frame arrives
NOTICE_S = 6.0           # how long a status message stays in the toolbar
FLASH_S = 1.5            # how long the zone of a new cue stays lit
PATH_ROWS, OTHER_ROWS, RECENT_ROWS = 5, 4, 4

RED, ORANGE, GREEN, GREY = (40, 40, 230), (0, 150, 255), (60, 190, 60), (150, 150, 150)
WHITE, BLACK, YELLOW = (255, 255, 255), (0, 0, 0), (0, 220, 255)
DARK, HOVER, ACCENT, TOOLBAR_BG = (45, 45, 45), (85, 85, 85), (60, 140, 60), (25, 25, 25)
PANEL_BG, CARD_BG, RULE, DIM = (22, 22, 22), (38, 38, 38), (70, 70, 70), (110, 110, 110)
CAUTION_RED = (40, 40, 255)  # BGR
CAUTION_BG = (20, 20, 110)   # the last-cue card behind a caution
CAUTION_FLASH_S = 1.5        # how long the video keeps a red frame after "Caution, ..."
SEVERITY_COLORS = {Severity.HIGH: RED, Severity.MEDIUM: ORANGE, Severity.LOW: GREEN}
PLACE = {Region.LEFT: "left", Region.CENTER: "ahead", Region.RIGHT: "right"}

MENU_X, MENU_PAD, ROW_H, MENU_FONT = 6, 8, 24, 0.5


def _ascii(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")  # OpenCV fonts are ASCII only


def _put(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.5,
         color: tuple[int, int, int] = WHITE, thickness: int = 1) -> None:
    cv2.putText(img, _ascii(text), org, FONT, scale, color, thickness, cv2.LINE_AA)


def _text_width(text: str, scale: float = MENU_FONT, thickness: int = 1) -> int:
    return cv2.getTextSize(_ascii(text), FONT, scale, thickness)[0][0]


def _wrap(text: str, max_width: int, scale: float, thickness: int = 1) -> list[str]:
    """Split text into lines that fit max_width pixels."""
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and _text_width(candidate, scale, thickness) > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines + ([line] if line else [])


def not_announced_reason(policy: AlertPolicy, obj: SpatialObject) -> str:
    """Why an object won't be named in a cue ("" if it can be). Mirrors AlertPolicy's rules."""
    cfg = policy.config
    det = obj.detection
    if policy.is_confident(obj):
        return ""
    if (obj.region is Region.CENTER and cfg.uncertain_confidence <= det.confidence < cfg.announce_confidence
            and obj.view_height >= cfg.uncertain_min_view_height):
        return "unsure: 'possible obstacle'"
    if cfg.labels is not None and det.label not in cfg.labels:
        return "not on alert list"
    if obj.view_height < cfg.min_view_height:
        return "small / far away"
    if det.confidence < cfg.announce_confidence:
        return f"score under {cfg.announce_confidence:.0%}"
    return policy.quiet_reason(obj) or "not announced"


def path_position(obj: SpatialObject) -> str:
    """Where an object in or touching the central band is, in plain words (box size is not a distance)."""
    if obj.region is Region.CENTER:
        return "ahead, large in view" if obj.severity is Severity.HIGH else "ahead"
    return f"edge of centre ({PLACE[obj.region]})"


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
    """Implements GuideSink as the dashboard window."""

    def __init__(
        self,
        controls: GuideControls,
        policy: AlertPolicy,
        spatial: SpatialConfig | None = None,
        *,
        voice_name: str = "windows",
        show_window: bool = True,
        cameras: CameraCatalog | None = None,
        obstacles: ObstacleScanner | None = None,
    ) -> None:
        self.controls = controls
        self.obstacles = obstacles  # the experimental floor scan, drawn over the video when given
        self.policy = policy
        self.spatial = spatial or SpatialConfig()
        self.voice_name = voice_name
        self.show_window = show_window
        self.menu = CameraMenu(cameras, controls) if cameras is not None else None
        self.toolbar = BAR if self.menu is not None else 0
        self._stop = False
        self._last_alert: GuideAlert | None = None
        self._last_alert_at = 0.0
        self._recent: deque[tuple[str, str, bool]] = deque(maxlen=RECENT_ROWS)  # (clock time, text, muted)
        self._window_open = False
        self._last_base: np.ndarray | None = None  # last frame's canvas without the menu, for idle()
        self._video_w = BLANK_SIZE[1]               # width of the video area in _last_base

    # GuideSink -------------------------------------------------------------------------------------------

    def publish(self, update: GuideUpdate) -> None:
        for alert in update.alerts:
            self._last_alert = alert
            self._last_alert_at = time.perf_counter()
            self._recent.appendleft((time.strftime("%H:%M:%S"), alert.text, self.controls.muted))
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

    # Screens -----------------------------------------------------------------------------------------------

    def render_idle(self, message: str = "") -> np.ndarray:
        """The last frame's screen (or a blank one) with `message` over the video, and a fresh toolbar."""
        top = BAR + self.toolbar
        if self._last_base is not None:
            canvas = self._last_base.copy()
            if self.controls.privacy:
                # The last frame may have been drawn before p was pressed: never show it while privacy is on.
                canvas[top:canvas.shape[0] - BAR, :self._video_w] = 0
        else:
            h, w = BLANK_SIZE
            canvas = np.zeros((h + 2 * BAR + self.toolbar, w + PANEL_W, 3), np.uint8)
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], BAR), (40, 40, 40), -1)
            _put(canvas, "Starting camera...", (10, BAR - 10), 0.45, WHITE, 1)
            self._draw_panel(canvas, w, top, canvas.shape[0] - BAR, None)
            self._draw_bottom_bar(canvas, canvas.shape[0] - BAR)
        if message:
            bottom = canvas.shape[0] - BAR
            width = _text_width(message, 0.7, 2)
            x = max((self._video_w - width) // 2, 10)
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
        self._draw_floor_scan(video)
        for obj in update.objects:
            self._draw_object(video, obj, scale)
        if update.stale:
            _put(video, "LAGGING: frame too old, not announcing", (10, vh - 12), 0.55, YELLOW, 2)

        top = BAR + self.toolbar
        body_h = max(vh, BLANK_SIZE[0])
        canvas = np.zeros((body_h + top + BAR, vw + PANEL_W, 3), np.uint8)
        canvas[top:top + vh, :vw] = video
        self._video_w = vw
        self._draw_top_bar(canvas, update)
        self._draw_panel(canvas, vw, top, top + body_h, update)
        self._draw_bottom_bar(canvas, top + body_h)
        if self.caution_showing():
            cv2.rectangle(canvas, (0, top), (vw - 1, top + vh - 1), CAUTION_RED, 6)  # red frame around the video
        self._last_base = canvas.copy()
        self._draw_toolbar(canvas)
        if self.menu is not None:
            self.menu.draw(canvas)
        return canvas

    def caution_showing(self) -> bool:
        """True for CAUTION_FLASH_S after a caution: the video gets a red frame."""
        return (self._last_alert is not None and self._last_alert.caution
                and time.perf_counter() - self._last_alert_at < CAUTION_FLASH_S)

    # Video area --------------------------------------------------------------------------------------------

    def _draw_toolbar(self, canvas: np.ndarray) -> None:
        """Camera dropdown button (drawn by the menu) and a status message to its right."""
        if not self.toolbar:
            return
        cv2.rectangle(canvas, (0, BAR), (self._video_w, BAR + self.toolbar), TOOLBAR_BG, -1)
        controls = self.controls
        recent = time.perf_counter() - controls.notice_at < NOTICE_S
        text, color = ((controls.notice, YELLOW) if controls.notice and (controls.switching or recent)
                       else (f"c: choose camera   r: rotate ({controls.rotation} deg)", GREY))
        x = MENU_X + max(self.menu.width, _text_width(self.menu.header() + "    ") + 2 * MENU_PAD + 16) + 14
        _put(canvas, text, (x, BAR + self.toolbar - 10), 0.5, color, 1)

    def _draw_floor_scan(self, video: np.ndarray) -> None:
        """Experimental open-space view: how far the floor seems to reach in each column, and the scan's verdict.

        Green = open-looking floor, red = the floor stops close in the path with nothing named there. Marked
        unverified: open-looking pixels are not proof that the way is walkable, and nothing here is spoken as a
        direction to take.
        """
        if self.obstacles is None:
            return
        vh, vw = video.shape[:2]
        scan = self.obstacles.latest()
        if scan is not None:
            unexplained = self.obstacles.last.unexplained
            xs = (np.arange(scan.width) + 0.5) * vw / scan.width
            ys = vh * (1.0 - scan.free)
            tint = video.copy()
            for i in range(scan.width):
                bad = unexplained.size == scan.width and unexplained[i]
                x1, x2 = int(i * vw / scan.width), int((i + 1) * vw / scan.width)
                cv2.rectangle(tint, (x1, int(ys[i])), (x2, vh), CAUTION_RED if bad else GREEN, -1)
            video[:] = cv2.addWeighted(tint, 0.22, video, 0.78, 0)
            pts = np.stack([xs, ys], axis=1).astype(np.int32)
            cv2.polylines(video, [pts], False, WHITE, 1, cv2.LINE_AA)
            thirds = scan.thirds(self.obstacles.config)
            share = " ".join(f"{name} {thirds[r]:.0%}" for r, name in
                             ((Region.LEFT, "L"), (Region.CENTER, "A"), (Region.RIGHT, "R")))
            _put(video, f"open-looking floor (unverified): {share}", (8, vh - 32), 0.45, WHITE, 1)
        _put(video, self.obstacles.status(), (8, vh - 12), 0.45, YELLOW, 1)

    def _flash_region(self) -> Region | None:
        if self._last_alert is not None and time.perf_counter() - self._last_alert_at < FLASH_S:
            return self._last_alert.region
        return None

    def _draw_regions(self, video: np.ndarray) -> None:
        vh, vw = video.shape[:2]
        edges = {Region.LEFT: (0.0, self.spatial.left_edge),
                 Region.CENTER: (self.spatial.left_edge, self.spatial.right_edge),
                 Region.RIGHT: (self.spatial.right_edge, 1.0)}
        lit = self._flash_region()
        if lit is not None:  # the zone the latest cue was about lights up briefly
            x1, x2 = (int(e * vw) for e in edges[lit])
            tint = video.copy()
            cv2.rectangle(tint, (x1, 0), (x2, vh), YELLOW, -1)
            video[:] = cv2.addWeighted(tint, 0.18, video, 0.82, 0)
        for edge in (self.spatial.left_edge, self.spatial.right_edge):
            x = int(edge * vw)
            for y in range(0, vh, 16):
                cv2.line(video, (x, y), (x, min(y + 8, vh)), WHITE, 1)
        for region, text in ((Region.LEFT, "LEFT"), (Region.CENTER, "AHEAD"), (Region.RIGHT, "RIGHT")):
            a, b = edges[region]
            size = cv2.getTextSize(text, FONT, 0.5, 1)[0]
            _put(video, text, (int((a + b) / 2 * vw - size[0] / 2), 18), 0.5, YELLOW if region is lit else WHITE, 1)

    def _draw_object(self, video: np.ndarray, obj: SpatialObject, scale: float) -> None:
        det = obj.detection
        x1, y1, x2, y2 = (int(v * scale) for v in det.box)
        color = SEVERITY_COLORS[obj.severity] if self.policy.is_confident(obj) else GREY
        cv2.rectangle(video, (x1, y1), (x2, y2), color, 2)
        parts = [det.label, f"{det.confidence:.0%}"]
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
                f"{update.depth_status} | voice: {voice}")
        _put(canvas, text, (26, BAR - 10), 0.45, WHITE, 1)

    def _draw_bottom_bar(self, canvas: np.ndarray, top: int) -> None:
        cv2.rectangle(canvas, (0, top), (canvas.shape[1], top + BAR), (40, 40, 40), -1)
        keys = "c camera   r rotate   m mute   p privacy   q quit"
        _put(canvas, keys, (10, top + BAR - 10), 0.45, WHITE, 1)
        note = "Experimental, not a navigation aid. Knows 80 kinds of objects; misses steps, glass, poles."
        size = cv2.getTextSize(note, FONT, 0.42, 1)[0]
        _put(canvas, note, (max(canvas.shape[1] - size[0] - 10, _text_width(keys, 0.45) + 30), top + BAR - 10),
             0.42, GREY, 1)

    # Right panel -------------------------------------------------------------------------------------------

    def _draw_panel(self, canvas: np.ndarray, x0: int, y0: int, y1: int, update: GuideUpdate | None) -> None:
        cv2.rectangle(canvas, (x0, y0), (canvas.shape[1], y1), PANEL_BG, -1)
        cv2.line(canvas, (x0, y0), (x0, y1), RULE, 1)
        x, w = x0 + 14, PANEL_W - 28
        y = self._panel_last_cue(canvas, x, y0 + 10, w)
        objects = update.objects if update is not None else ()
        in_path = sorted((o for o in objects if o.in_corridor), key=self._path_order)
        others = sorted((o for o in objects if not o.in_corridor), key=lambda o: -o.detection.confidence)

        y = self._section(canvas, x, y + 8, w, f"DETECTED AHEAD ({len(in_path)})")
        if update is None:
            y = self._note(canvas, x, y, w, "Waiting for the camera...")
        elif not in_path:
            y = self._note(canvas, x, y, w, "Nothing detected in the central band.", WHITE)
            y = self._note(canvas, x, y, w, "Not proof the way is clear: unknown objects are missed.")
        for obj in in_path[:PATH_ROWS]:
            y = self._object_row(canvas, x, y, w, obj, path_position(obj))
        if len(in_path) > PATH_ROWS:
            y = self._note(canvas, x, y, w, f"+ {len(in_path) - PATH_ROWS} more")

        y = self._section(canvas, x, y + 6, w, f"ALSO IN VIEW ({len(others)})")
        if update is not None and not others:
            y = self._note(canvas, x, y, w, "Nothing else detected.")
        for obj in others[:OTHER_ROWS]:
            y = self._object_row(canvas, x, y, w, obj, PLACE[obj.region])
        if len(others) > OTHER_ROWS:
            y = self._note(canvas, x, y, w, f"+ {len(others) - OTHER_ROWS} more")

        if y + 40 < y1:
            y = self._section(canvas, x, y + 6, w, "RECENT CUES")
            if not self._recent:
                y = self._note(canvas, x, y, w, "None yet.")
            for clock, text, muted in self._recent:
                if y + 20 > y1:
                    break
                _put(canvas, clock, (x, y + 14), 0.42, DIM, 1)
                _put(canvas, text + (" (muted)" if muted else ""), (x + 74, y + 14), 0.45, WHITE, 1)
                y += 20

    def _panel_last_cue(self, canvas: np.ndarray, x: int, y: int, w: int) -> int:
        caution = self._last_alert is not None and self._last_alert.caution
        y = self._section(canvas, x, y, w, "LAST CUE: CAUTION" if caution else "LAST CUE")
        cv2.rectangle(canvas, (x, y), (x + w, y + 96), CAUTION_BG if caution else CARD_BG, -1)
        if caution:
            cv2.rectangle(canvas, (x, y), (x + w, y + 96), CAUTION_RED, 2)
        if self._last_alert is None:
            _put(canvas, "Nothing announced yet", (x + 12, y + 34), 0.6, DIM, 1)
            region = None
        else:
            alert = self._last_alert
            lines = _wrap(f'"{alert.text}"', w - 24, 0.75, 2)[:2]
            for i, line in enumerate(lines):
                _put(canvas, line, (x + 12, y + 30 + 26 * i), 0.75, WHITE if caution else YELLOW, 2)
            ago = time.perf_counter() - self._last_alert_at
            muted = "  muted, not spoken" if self.controls.muted else ""
            _put(canvas, f"{ago:.0f} s ago  |  score {alert.confidence:.0%}{muted}", (x + 12, y + 84), 0.42, GREY, 1)
            region = alert.region if ago < 10 else None
        # Where: LEFT / AHEAD / RIGHT strip under the card
        y += 102
        third = (w - 8) // 3
        for i, (r, label) in enumerate(((Region.LEFT, "LEFT"), (Region.CENTER, "AHEAD"), (Region.RIGHT, "RIGHT"))):
            bx = x + i * (third + 4)
            on = r is region
            cv2.rectangle(canvas, (bx, y), (bx + third, y + 22), YELLOW if on else CARD_BG, -1)
            tw = _text_width(label, 0.45)
            _put(canvas, label, (bx + (third - tw) // 2, y + 16), 0.45, BLACK if on else DIM, 1)
        return y + 26

    def _path_order(self, obj: SpatialObject):
        return (-SEVERITY_RANK[obj.severity], not self.policy.is_confident(obj), -obj.view_height,
                -obj.detection.confidence)

    def _section(self, canvas: np.ndarray, x: int, y: int, w: int, title: str) -> int:
        _put(canvas, title, (x, y + 14), 0.45, GREY, 1)
        cv2.line(canvas, (x, y + 20), (x + w, y + 20), RULE, 1)
        return y + 26

    def _note(self, canvas: np.ndarray, x: int, y: int, w: int, text: str, color=DIM) -> int:
        for line in _wrap(text, w, 0.45):
            _put(canvas, line, (x, y + 14), 0.45, color, 1)
            y += 18
        return y

    def _object_row(self, canvas: np.ndarray, x: int, y: int, w: int, obj: SpatialObject, where: str) -> int:
        """Colour chip, name, confidence bar and %, then a second line: where it is and whether it's announced."""
        det = obj.detection
        reason = not_announced_reason(self.policy, obj)
        chip = SEVERITY_COLORS[obj.severity] if not reason else GREY
        cv2.rectangle(canvas, (x, y + 4), (x + 10, y + 16), chip, -1)
        name = spoken_name(det.label)
        _put(canvas, name, (x + 18, y + 15), 0.52, WHITE if not reason else (200, 200, 200), 1)
        bar_x, bar_w = x + w - 150, 100
        fill = int(bar_w * max(0.0, min(det.confidence, 1.0)))
        bar_color = GREEN if det.confidence >= 0.7 else YELLOW if det.confidence >= 0.5 else GREY
        cv2.rectangle(canvas, (bar_x, y + 6), (bar_x + bar_w, y + 15), CARD_BG, -1)
        cv2.rectangle(canvas, (bar_x, y + 6), (bar_x + fill, y + 15), bar_color, -1)
        _put(canvas, f"{det.confidence:.0%}", (bar_x + bar_w + 8, y + 15), 0.45, WHITE, 1)
        detail = where
        if obj.depth.meters is not None:
            detail += f", {obj.depth.meters:.1f} m" + ("" if obj.depth.validated else " est.")
        detail += f"  |  {reason}" if reason else "  |  named in cues"
        _put(canvas, detail, (x + 18, y + 33), 0.4, DIM if reason else GREY, 1)
        return y + 42


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
