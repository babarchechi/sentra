"""Level 1 dashboard in an OpenCV window. Track C, owner: Swechha03. See docs/PHASE1_PLAN.md, Section 8.1 step 3.

Layout: a top bar (LIVE or PRIVACY MODE, fps, disclaimer), the video on the left, one row per person on the right.

When someone first enters ``POSSIBLE_FALL``, a red banner and a sound ask a human to check. The alert stays until
they press ``c`` (confirmed: they needed help) or ``f`` (false alarm). If the person gets back up first, it is logged
as "recovered". Alerts with confidence <= 0.5 (uncertain evidence, plan Section 6.2) get an "UNCERTAIN" tag but still
need an answer. Every alert is written to ``outputs/alerts_<YYYYmmdd_HHMMSS>.csv``. No images are ever saved.

Keys (click the window first): c confirm, f false alarm, p privacy mode on/off, q quit.

OpenCV's built-in fonts are ASCII only, so on screen "—" becomes "-" and "°" becomes " deg".
"""
from __future__ import annotations

import csv
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from sentra.events import SKELETON, PersonPose, PersonStatus, PostureState, StatusUpdate

log = logging.getLogger(__name__)

WINDOW_TITLE = "Sentra - prototype, not a medical device (q to quit)"
DEFAULT_OUTPUT_DIR = Path("outputs")

UNCERTAIN_MAX_CONFIDENCE = 0.5  # plan Section 6.3, uncertain_confidence_max
MIN_KEYPOINT_CONF = 0.5         # skeleton lines are drawn only between keypoints at least this confident
FORGET_AFTER_S = 10.0           # drop per-person memory for tracks not seen this long (unless an alert is open)
NOTICE_S = 4.0                  # how long a short notice ("logged as false alarm") stays on screen

# Layout, in pixels
MAX_VIDEO_W, MAX_VIDEO_H = 960, 540
MIN_BODY_H = 480
PANEL_W = 380
TOP_BAR_H = 40
ROW_H = 78
FOOTER_H = 58
BANNER_H = 96

# Colours (BGR)
WHITE = (255, 255, 255)
LIGHT = (205, 205, 205)
DIM = (140, 140, 140)
BLACK = (0, 0, 0)
GREEN = (80, 190, 80)
GREY = (150, 150, 150)
AMBER = (0, 170, 255)
RED = (40, 40, 230)
YELLOW = (0, 215, 255)
BLUE = (200, 120, 40)
BAR_BG = (25, 25, 25)
PANEL_BG = (34, 34, 34)

STATE_COLORS = {
    PostureState.UPRIGHT: GREEN,
    PostureState.UNKNOWN: GREY,
    PostureState.DESCENDING: AMBER,
    PostureState.ON_GROUND: AMBER,
    PostureState.POSSIBLE_FALL: RED,
}
STATE_LABELS = {
    PostureState.UPRIGHT: "Upright",
    PostureState.UNKNOWN: "Unknown",
    PostureState.DESCENDING: "Descending",
    PostureState.ON_GROUND: "On ground",
    PostureState.POSSIBLE_FALL: "POSSIBLE FALL",
}

# Alert responses written to the CSV
CONFIRMED = "confirmed"
FALSE_ALARM = "false_alarm"
RECOVERED = "recovered"
NO_RESPONSE = "no_response"  # the app was closed while the alert was still open

FONT = cv2.FONT_HERSHEY_SIMPLEX
_REPLACEMENTS = {"°": " deg", "—": "-", "–": "-", "→": "->", "…": "...", "×": "x", "≤": "<=", "≥": ">="}


def ascii_text(text: str) -> str:
    """Make text safe for OpenCV's ASCII-only fonts (anything else would show up as '?')."""
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    return text.encode("ascii", "replace").decode("ascii")


def format_duration(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    if seconds < 10:
        return f"{seconds:.1f} s"
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s"


def play_alert_sound() -> None:
    """Play the Windows exclamation sound without blocking the video loop; do nothing elsewhere."""
    try:
        import winsound

        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
    except Exception:  # not Windows, or no sound device: the banner is still there
        log.debug("Alert sound unavailable", exc_info=True)


@dataclass
class Alert:
    """An open "possible fall" alert for one track, waiting for a human to answer it."""

    track_id: int
    reason: str             # latest reason from the analyzer, shown on screen
    confidence: float       # latest confidence
    opened_at: float        # pose timestamp (time.perf_counter() clock) when the alert opened
    opened_wall: datetime   # wall-clock time, for the CSV
    opened_reason: str = "" # why it fired, for the CSV (the latest reason may be "getting up")

    def __post_init__(self) -> None:
        self.opened_reason = self.opened_reason or self.reason

    @property
    def uncertain(self) -> bool:
        return self.confidence <= UNCERTAIN_MAX_CONFIDENCE


class AlertLog:
    """Writes one row per alert, never images, to <output_dir>/alerts_<YYYYmmdd_HHMMSS>.csv."""

    COLUMNS = ("alert_time", "track_id", "reason", "response", "response_time", "seconds_to_response", "confidence")

    def __init__(self, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"alerts_{datetime.now():%Y%m%d_%H%M%S}"
        self.path = output_dir / f"{stem}.csv"
        n = 2
        while self.path.exists():  # two runs started in the same second must not overwrite each other
            self.path = output_dir / f"{stem}_{n}.csv"
            n += 1
        self._file = self.path.open("x", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.COLUMNS)
        self._file.flush()

    def write(self, alert: Alert, response: str, response_wall: datetime, seconds_to_response: float) -> None:
        self._writer.writerow([
            alert.opened_wall.isoformat(timespec="seconds"),
            alert.track_id,
            alert.opened_reason,
            response,
            response_wall.isoformat(timespec="seconds"),
            f"{seconds_to_response:.1f}",
            f"{alert.confidence:.2f}",
        ])
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


class CvDashboard:
    """Implements sentra.events.StatusSink with an OpenCV window.

    `publish()` = `process()` (alert logic) + `render()` (drawing) + showing the window and reading keys.
    Tests call `process()`, `render()` and `handle_key()` directly, without opening a window.
    """

    def __init__(
        self,
        privacy: bool = False,
        *,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        alert_sound: Callable[[], None] | None = play_alert_sound,
    ) -> None:
        self.privacy = privacy
        self.alert_count = 0
        self._alert_sound = alert_sound
        self._log = AlertLog(output_dir)
        self._open: dict[int, Alert] = {}            # insertion order = oldest first
        self._last_status: dict[int, PersonStatus] = {}
        self._last_seen: dict[int, float] = {}
        self._now = 0.0
        self._notice: tuple[str, tuple[int, int, int], float] | None = None
        self._stop = False
        self._window_open = False
        self._closed = False
        log.info("Logging alerts (no images) to %s", self._log.path)

    # ----- StatusSink -------------------------------------------------------------------------------------

    def publish(self, update: StatusUpdate) -> None:
        self.process(update)
        view = self.render(update)
        cv2.imshow(WINDOW_TITLE, view)
        self._window_open = True
        self.handle_key(cv2.waitKey(1))
        if self._window_was_closed():
            self._stop = True

    def should_stop(self) -> bool:
        return self._stop

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for track_id in list(self._open):
            self._resolve(track_id, NO_RESPONSE)
        self._log.close()
        log.info("Alerts saved to %s (%d this session)", self._log.path, self.alert_count)
        if self._window_open:
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except cv2.error:
                pass

    # ----- Alert logic ------------------------------------------------------------------------------------

    @property
    def log_path(self) -> Path:
        return self._log.path

    @property
    def open_alert_ids(self) -> tuple[int, ...]:
        """Track IDs with an unanswered alert, oldest first."""
        return tuple(self._open)

    def has_open_alert(self, track_id: int) -> bool:
        return track_id in self._open

    def open_alert(self, track_id: int) -> Alert | None:
        return self._open.get(track_id)

    def process(self, update: StatusUpdate) -> None:
        """Open, update and auto-resolve alerts from this frame's statuses."""
        self._now = update.pose.timestamp
        for status in update.statuses:
            track_id = status.track_id
            previous = self._last_status.get(track_id)
            self._last_status[track_id] = status
            self._last_seen[track_id] = self._now
            alert = self._open.get(track_id)
            if status.state == PostureState.POSSIBLE_FALL:
                if alert is not None:
                    alert.reason, alert.confidence = status.reason, status.confidence
                elif previous is None or previous.state != PostureState.POSSIBLE_FALL:
                    self._open_alert(status)
            elif alert is not None and status.state == PostureState.UPRIGHT:
                # Back on their feet before anyone answered. UNKNOWN or out of view keeps the alert open.
                self._resolve(track_id, RECOVERED)
        self._forget_stale_tracks()

    def handle_key(self, key: int) -> None:
        """React to a key code from cv2.waitKey (-1 means no key)."""
        if key is None or key < 0:
            return
        code = key & 0xFF
        char = chr(code).lower()
        if char == "q" or code == 27:  # q or Esc
            self._stop = True
        elif char == "p":
            self.privacy = not self.privacy
            self._show_notice("Privacy mode ON: skeleton only, no video" if self.privacy else "Live video ON", BLUE)
        elif char in ("c", "f"):
            if not self._open:
                self._show_notice("No open alert to answer", DIM)
                return
            oldest = next(iter(self._open))
            self._resolve(oldest, CONFIRMED if char == "c" else FALSE_ALARM)

    def _open_alert(self, status: PersonStatus) -> None:
        alert = Alert(status.track_id, status.reason, status.confidence, self._now, datetime.now())
        self._open[status.track_id] = alert
        self.alert_count += 1
        log.warning(
            "Possible fall - person #%d - please check (%s%s)",
            alert.track_id, ascii_text(alert.reason), ", uncertain" if alert.uncertain else "",
        )
        if self._alert_sound is not None:
            self._alert_sound()

    def _resolve(self, track_id: int, response: str) -> None:
        alert = self._open.pop(track_id)
        self._log.write(alert, response, datetime.now(), self._now - alert.opened_at)
        log.info("Alert for person #%d logged as %s", track_id, response)
        if response == CONFIRMED:
            self._show_notice(f"Person #{track_id}: logged as CONFIRMED (needed help)", WHITE)
        elif response == FALSE_ALARM:
            self._show_notice(f"Person #{track_id}: logged as false alarm", WHITE)
        elif response == RECOVERED:
            self._show_notice(f"Person #{track_id} got up before anyone answered: logged as recovered", AMBER)

    def _forget_stale_tracks(self) -> None:
        for track_id, seen in list(self._last_seen.items()):
            if self._now - seen > FORGET_AFTER_S and track_id not in self._open:
                del self._last_seen[track_id]
                self._last_status.pop(track_id, None)

    def _show_notice(self, text: str, color: tuple[int, int, int]) -> None:
        self._notice = (text, color, self._now + NOTICE_S)

    def _window_was_closed(self) -> bool:
        try:
            return cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return False

    # ----- Drawing ----------------------------------------------------------------------------------------

    def render(self, update: StatusUpdate) -> np.ndarray:
        """Build the whole dashboard image for this frame. Pure drawing: no window, no keys, no alert changes."""
        video = self._video_panel(update)
        vh, vw = video.shape[:2]
        body_h = max(vh, MIN_BODY_H)
        canvas = np.zeros((TOP_BAR_H + body_h, vw + PANEL_W, 3), np.uint8)
        canvas[TOP_BAR_H:TOP_BAR_H + vh, :vw] = video
        self._draw_top_bar(canvas, update.fps)
        self._draw_side_panel(canvas[TOP_BAR_H:, vw:], update)
        return canvas

    def _video_panel(self, update: StatusUpdate) -> np.ndarray:
        h, w = update.frame.shape[:2]
        scale = min(1.0, MAX_VIDEO_W / w, MAX_VIDEO_H / h)
        size = (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1))
        statuses = {s.track_id: s for s in update.statuses}
        if self.privacy:
            # Never touch update.frame / update.annotated here: privacy mode must not show any video pixels.
            view = np.zeros((size[1], size[0], 3), np.uint8)
            draw_skeletons(view, update.pose.people, statuses, scale)
        elif scale == 1.0:
            view = update.annotated.copy()
        else:
            view = cv2.resize(update.annotated, size, interpolation=cv2.INTER_AREA)
        self._draw_person_tags(view, update.pose.people, statuses, scale)
        self._draw_alert(view)
        self._draw_notice(view)
        return view

    def _draw_person_tags(
        self, view: np.ndarray, people: Iterable[PersonPose], statuses: dict[int, PersonStatus], scale: float
    ) -> None:
        for person in people:
            status = statuses.get(person.track_id)
            state = status.state if status else PostureState.UNKNOWN
            color = STATE_COLORS[state]
            x1, y1, x2, y2 = (int(round(v * scale)) for v in person.box)
            alerting = person.track_id in self._open
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 4 if alerting else 2)
            label = f"#{person.track_id} {STATE_LABELS[state]}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
            top = max(y1 - th - 10, 0)
            cv2.rectangle(view, (x1, top), (x1 + tw + 10, top + th + 10), color, -1)
            _text(view, label, (x1 + 5, top + th + 5), 0.5, BLACK if color != RED else WHITE)

    def _draw_alert(self, view: np.ndarray) -> None:
        if not self._open:
            return
        h, w = view.shape[:2]
        alert = next(iter(self._open.values()))
        if int(self._now * 2) % 2 == 0:  # blinking red border
            cv2.rectangle(view, (0, 0), (w - 1, h - 1), RED, 8)
        banner_h = min(BANNER_H, h)
        roi = view[:banner_h]
        roi[:] = cv2.addWeighted(roi, 0.2, np.full_like(roi, (20, 20, 190)), 0.8, 0)

        tag_w = 0
        if alert.uncertain:
            (tw, th), _ = cv2.getTextSize("UNCERTAIN", FONT, 0.5, 1)
            tag_w = tw + 16
            cv2.rectangle(roi, (w - tag_w - 10, 10), (w - 10, 18 + th), YELLOW, -1)
            _text(roi, "UNCERTAIN", (w - tag_w - 2, 14 + th), 0.5, BLACK)
        headline = fit_text(f"Possible fall - person #{alert.track_id} - please check", w - tag_w - 34, 0.72, 2)
        _text(roi, headline, (14, 32), 0.72, WHITE, 2)
        ago = format_duration(self._now - alert.opened_at)
        _text(roi, fit_text(f"{alert.reason} ({ago} ago)", w - 28, 0.5), (14, 58), 0.5, WHITE)
        _text(roi, "C = confirmed (needed help)    F = false alarm", (14, 84), 0.5, YELLOW)
        if len(self._open) > 1:
            more = f"+{len(self._open) - 1} more waiting"
            (tw, _), _ = cv2.getTextSize(more, FONT, 0.5, 1)
            _text(roi, more, (w - tw - 14, 84), 0.5, WHITE)

    def _draw_notice(self, view: np.ndarray) -> None:
        if self._notice is None or self._now > self._notice[2]:
            return
        text, color, _ = self._notice
        h, w = view.shape[:2]
        text = fit_text(text, w - 40, 0.5)
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
        x, y = (w - tw) // 2, h - 20
        cv2.rectangle(view, (x - 10, y - th - 10), (x + tw + 10, y + 8), (20, 20, 20), -1)
        _text(view, text, (x, y), 0.5, color)

    def _draw_top_bar(self, canvas: np.ndarray, fps: float) -> None:
        bar = canvas[:TOP_BAR_H]
        bar[:] = BAR_BG
        _text(bar, "SENTRA", (12, 27), 0.7, WHITE, 2)
        if self.privacy:
            label, color = "PRIVACY MODE", BLUE
        else:
            label, color = "LIVE", GREEN
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(bar, (120, 9), (120 + tw + 16, 31), color, -1)
        _text(bar, label, (128, 26), 0.5, WHITE if self.privacy else BLACK)
        _text(bar, f"{fps:.1f} fps", (120 + tw + 32, 26), 0.5, LIGHT)
        disclaimer = "Prototype - not a medical device"
        (dw, _), _ = cv2.getTextSize(disclaimer, FONT, 0.5, 1)
        _text(bar, disclaimer, (canvas.shape[1] - dw - 12, 26), 0.5, DIM)

    def _draw_side_panel(self, panel: np.ndarray, update: StatusUpdate) -> None:
        panel[:] = PANEL_BG
        h, w = panel.shape[:2]
        current = sorted(update.statuses, key=lambda s: s.track_id)
        shown = {s.track_id for s in current}
        # People with an open alert who aren't in this frame (left view, or a contract change sends no status).
        missing = [self._last_status[t] for t in self._open if t not in shown and t in self._last_status]
        rows = [(s, True) for s in current] + [(s, False) for s in missing]

        _text(panel, f"People in view: {len(current)}", (16, 28), 0.6, WHITE)
        y = 44
        max_rows = max((h - y - FOOTER_H) // ROW_H, 0)
        if not rows:
            _text(panel, "Nobody in view", (16, y + 30), 0.5, DIM)
        for status, in_view in rows[:max_rows]:
            self._draw_row(panel[y:y + ROW_H], status, in_view)
            y += ROW_H
        if len(rows) > max_rows:
            _text(panel, f"+{len(rows) - max_rows} more", (16, y + 18), 0.5, DIM)

        footer_y = h - FOOTER_H
        cv2.line(panel, (0, footer_y), (w, footer_y), (60, 60, 60), 1)
        _text(panel, "C confirm   F false alarm   P privacy   Q quit", (16, footer_y + 24), 0.45, LIGHT)
        _text(panel, f"Alerts this session: {self.alert_count}   open: {len(self._open)}",
              (16, footer_y + 46), 0.45, DIM)

    def _draw_row(self, row: np.ndarray, status: PersonStatus, in_view: bool) -> None:
        w = row.shape[1]
        color = STATE_COLORS[status.state]
        cv2.rectangle(row, (0, 2), (5, ROW_H - 4), color, -1)
        _text(row, f"#{status.track_id}", (16, 22), 0.6, WHITE, 2)
        bold = 2 if status.state == PostureState.POSSIBLE_FALL else 1
        _text(row, STATE_LABELS[status.state], (66, 22), 0.6, color, bold)
        duration = "for " + format_duration(self._now - status.since)
        (dw, _), _ = cv2.getTextSize(duration, FONT, 0.45, 1)
        _text(row, duration, (w - dw - 14, 22), 0.45, LIGHT)
        _text(row, fit_text(status.reason, w - 30, 0.45), (16, 44), 0.45, LIGHT)

        x = 16
        _text(row, f"confidence {status.confidence:.2f}", (x, 66), 0.42, DIM)
        x += 130
        tags = []
        if status.track_id in self._open:
            tags.append(("ALERT OPEN", RED, WHITE))
        if status.state == PostureState.POSSIBLE_FALL and status.confidence <= UNCERTAIN_MAX_CONFIDENCE:
            tags.append(("UNCERTAIN", YELLOW, BLACK))
        if not in_view:
            tags.append(("NOT IN VIEW", GREY, BLACK))
        for label, bg, fg in tags:
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.4, 1)
            cv2.rectangle(row, (x, 66 - th - 4), (x + tw + 10, 70), bg, -1)
            _text(row, label, (x + 5, 66), 0.4, fg)
            x += tw + 16
        cv2.line(row, (12, ROW_H - 1), (w - 12, ROW_H - 1), (55, 55, 55), 1)


def draw_skeletons(
    view: np.ndarray, people: Iterable[PersonPose], statuses: dict[int, PersonStatus], scale: float = 1.0
) -> None:
    """Draw each person's skeleton from keypoints only (privacy mode), coloured by their state."""
    for person in people:
        status = statuses.get(person.track_id)
        color = STATE_COLORS[status.state if status else PostureState.UNKNOWN]
        kpts = person.keypoints
        visible = kpts[:, 2] >= MIN_KEYPOINT_CONF
        pts = (kpts[:, :2] * scale).round().astype(int)
        for a, b in SKELETON:
            if visible[a] and visible[b]:
                cv2.line(view, tuple(pts[a]), tuple(pts[b]), color, 3, cv2.LINE_AA)
        for i in np.flatnonzero(visible):
            cv2.circle(view, tuple(pts[i]), 4, WHITE, -1, cv2.LINE_AA)


def fit_text(text: str, max_width: int, scale: float, thickness: int = 1) -> str:
    """Shorten text with '...' so it fits in max_width pixels."""
    text = ascii_text(text)
    if cv2.getTextSize(text, FONT, scale, thickness)[0][0] <= max_width:
        return text
    while text and cv2.getTextSize(text + "...", FONT, scale, thickness)[0][0] > max_width:
        text = text[:-1]
    return text.rstrip() + "..."


def _text(
    img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.5,
    color: tuple[int, int, int] = WHITE, thickness: int = 1,
) -> None:
    cv2.putText(img, ascii_text(text), org, FONT, scale, color, thickness, cv2.LINE_AA)
