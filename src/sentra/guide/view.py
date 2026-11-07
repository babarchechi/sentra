"""OpenCV debug view for Sentra Guide: what the camera sees, what was detected, and what was said.

Keys (click the window first): m mute or unmute the voice, p privacy (hide the camera image), q or Esc quit.

Top bar: camera-on indicator, source, fps, latency, distance mode, voice state.
Video: dashed lines mark LEFT | AHEAD | RIGHT. Boxes are red (central and large), orange (in or touching the
central band), green (to the side) or grey (not announced: low score, small, or not on the label list).
Bottom bar: the last announcement and how long ago it was made, plus the "not a navigation aid" reminder.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from sentra.guide.alerts import AlertPolicy
from sentra.guide.pipeline import GuideControls
from sentra.guide.schema import GuideAlert, GuideUpdate, Severity, SpatialObject
from sentra.guide.spatial import SpatialConfig

WINDOW_TITLE = "Sentra Guide - experimental, not a navigation aid (q to quit)"
FONT = cv2.FONT_HERSHEY_SIMPLEX
MAX_WIDTH = 1280
MAX_HEIGHT = 720
BAR = 30

RED, ORANGE, GREEN, GREY = (40, 40, 230), (0, 150, 255), (60, 190, 60), (150, 150, 150)
WHITE, BLACK, YELLOW = (255, 255, 255), (0, 0, 0), (0, 220, 255)
SEVERITY_COLORS = {Severity.HIGH: RED, Severity.MEDIUM: ORANGE, Severity.LOW: GREEN}


def _ascii(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")  # OpenCV fonts are ASCII only


def _put(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.5,
         color: tuple[int, int, int] = WHITE, thickness: int = 1) -> None:
    cv2.putText(img, _ascii(text), org, FONT, scale, color, thickness, cv2.LINE_AA)


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
    ) -> None:
        self.controls = controls
        self.policy = policy
        self.spatial = spatial or SpatialConfig()
        self.voice_name = voice_name
        self.show_window = show_window
        self._stop = False
        self._last_alert: GuideAlert | None = None
        self._last_alert_at = 0.0
        self._window_open = False

    # GuideSink -------------------------------------------------------------------------------------------

    def publish(self, update: GuideUpdate) -> None:
        if update.alerts:
            self._last_alert = update.alerts[-1]
            self._last_alert_at = time.perf_counter()
        canvas = self.render(update)
        if not self.show_window:
            return
        cv2.imshow(WINDOW_TITLE, canvas)
        self._window_open = True
        self.handle_key(cv2.waitKey(1) & 0xFF)
        if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
            self._stop = True  # closed with the window's X button

    def should_stop(self) -> bool:
        return self._stop

    def close(self) -> None:
        if self._window_open:
            cv2.destroyWindow(WINDOW_TITLE)
            cv2.waitKey(1)
            self._window_open = False

    # Keys --------------------------------------------------------------------------------------------------

    def handle_key(self, key: int) -> None:
        if key in (ord("q"), 27):
            self._stop = True
        elif key == ord("m"):
            self.controls.muted = not self.controls.muted
        elif key == ord("p"):
            self.controls.privacy = not self.controls.privacy

    # Drawing -----------------------------------------------------------------------------------------------

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
        canvas = np.zeros((vh + 2 * BAR, canvas_w, 3), np.uint8)
        canvas[BAR:BAR + vh, :vw] = video
        self._draw_top_bar(canvas, update)
        self._draw_bottom_bar(canvas, vh + BAR)
        return canvas

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
