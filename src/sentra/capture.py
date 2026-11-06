"""Camera input. Track B, owner: babarchechi. See docs/PHASE1_PLAN.md, Section 7.1 step 2.

FrameSource reads frames on a background thread and keeps only the newest one, so slow pose
inference never works through a backlog of stale frames. Sources that stop delivering frames
(for example a phone stream on flaky Wi-Fi) are reopened automatically.
"""
from __future__ import annotations

import logging
import threading
import time
from urllib.parse import urlsplit

import cv2
import numpy as np

log = logging.getLogger(__name__)


WINDOWS_PREFIX = "windows:"
PHONE_SOURCE = "phone"
PHONE_HINT = "Windows Virtual Camera"  # how Phone Link names a phone used as a connected camera


def parse_source(raw: str) -> int | str:
    """Turn a webcam index like "0" into an int; leave anything else (a stream URL) as a string."""
    return int(raw) if raw.isdecimal() else raw


def windows_camera_name(source: int | str) -> str | None:
    """Camera-name fragment for the opt-in Windows sources, or None for everything else.

    "phone" -> the Phone Link camera; "windows:<part of name>" -> any camera Windows lists (e.g. DroidCam).
    Numbered webcams and URLs return None and stay on OpenCV.
    """
    if not isinstance(source, str):
        return None
    if source.lower() == PHONE_SOURCE:
        return PHONE_HINT
    if source.lower().startswith(WINDOWS_PREFIX):
        return source[len(WINDOWS_PREFIX):].strip() or None
    return None


def describe_source(source: int | str) -> str:
    """Printable name for a source, with any username, password and query string removed.

    Stream URLs can contain credentials, and these names end up in logs and on screen.
    """
    if isinstance(source, int):
        return f"webcam {source}"
    if "://" not in source:
        return source
    parts = urlsplit(source)
    try:
        port = f":{parts.port}" if parts.port else ""
    except ValueError:
        port = ""
    return f"{parts.scheme}://{parts.hostname or '?'}{port}{parts.path}"


def _open_windows_camera(name: str):
    """Open a Windows camera by name through MediaCapture, or return None (also on non-Windows)."""
    try:
        from sentra.windows_camera import WindowsCamera  # Windows-only dependency, imported on demand
    except ImportError:
        log.error("Source %r needs Windows and the winrt packages (pip install -r requirements.txt).", name)
        return None
    try:
        # Short frame timeout so FrameSource's failed-read counter reaches its reconnect in ~10 s, not ~50 s.
        camera = WindowsCamera(name=name, frame_timeout=1.0)
    except OSError as exc:  # e.g. the camera is in use by another app
        log.warning("Cannot open Windows camera %r: %s", name, exc)
        return None
    if not camera.isOpened():
        camera.release()
        return None
    return camera


def _open_capture(source: int | str) -> cv2.VideoCapture | None:
    """Open a camera or stream, or return None if it can't be opened. Tests replace this function."""
    windows_name = windows_camera_name(source)
    if windows_name is not None:
        return _open_windows_camera(windows_name)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        return None
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ask the backend not to queue up old frames
    return capture


# Degrees clockwise -> OpenCV rotate code. A phone held upright usually needs 90 or 270.
ROTATIONS: dict[int, int | None] = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class FrameSource:
    """Background camera reader that always hands out the newest frame.

    - read() returns each frame at most once, so the pipeline never analyzes the same frame twice.
    - If the source stops delivering frames, it is released and reopened with backoff
      (1 s, 2 s, 4 s, then every 5 s) until close() is called.
    - The first open must succeed, otherwise the constructor raises RuntimeError.
    - `rotate` (0, 90, 180 or 270 degrees clockwise) turns frames upright, e.g. for a phone held in
      portrait. This matters for fall detection: a sideways image makes a standing person look horizontal.
    """

    def __init__(
        self,
        source: int | str,
        *,
        rotate: int = 0,
        open_timeout_s: float = 15.0,
        max_failed_reads: int = 10,
        backoff_start_s: float = 1.0,
        backoff_max_s: float = 5.0,
    ) -> None:
        if rotate not in ROTATIONS:
            raise ValueError(f"rotate must be one of {sorted(ROTATIONS)}, got {rotate}")
        self.source = source
        self.rotate = rotate
        self._rotate_code = ROTATIONS[rotate]
        self.description = describe_source(source)
        self._max_failed_reads = max_failed_reads
        self._backoff_start_s = backoff_start_s
        self._backoff_max_s = backoff_max_s

        self._capture: cv2.VideoCapture | None = None
        self._new_frame = threading.Condition()
        self._frame: np.ndarray | None = None
        self._timestamp = 0.0
        self._seq = 0              # number of frames captured so far
        self._last_read_seq = 0    # value of _seq at the last successful read()
        self._stop = threading.Event()
        self._opened = threading.Event()
        self._open_failed = False

        # The capture is opened, read and released on the reader thread only.
        self._thread = threading.Thread(target=self._run, name="FrameSource", daemon=True)
        self._thread.start()
        if not self._opened.wait(open_timeout_s) or self._open_failed:
            self.close()
            raise RuntimeError(
                f"Cannot open camera source ({self.description}). Check permissions, source, and network."
            )

    def read(self, timeout: float = 1.0) -> tuple[bool, np.ndarray | None, float]:
        """Return (ok, newest BGR frame, time.perf_counter() when it was captured).

        Waits up to `timeout` seconds for a frame that hasn't been returned before.
        Returns (False, None, 0.0) if none arrives in time or the source is closed.
        The frame is already rotated by `rotate`.
        """
        deadline = time.monotonic() + timeout
        with self._new_frame:
            while True:
                if self._stop.is_set():
                    return False, None, 0.0
                if self._seq != self._last_read_seq:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None, 0.0
                self._new_frame.wait(remaining)
            self._last_read_seq = self._seq
            frame, timestamp = self._frame, self._timestamp
        if self._rotate_code is not None:
            # Outside the lock, and only for frames we hand out, so the reader thread never waits on it.
            frame = cv2.rotate(frame, self._rotate_code)
        return True, frame, timestamp

    def close(self) -> None:
        """Stop the reader thread and release the camera. Safe to call more than once."""
        self._stop.set()
        with self._new_frame:
            self._new_frame.notify_all()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=3.0)

    # Reader thread ------------------------------------------------------------------------------

    def _run(self) -> None:
        self._capture = _open_capture(self.source)
        self._open_failed = self._capture is None
        self._opened.set()
        if self._open_failed:
            return

        failed_reads = 0
        backoff = self._backoff_start_s
        try:
            while not self._stop.is_set():
                capture = self._capture
                ok, frame = capture.read() if capture is not None else (False, None)
                if ok and frame is not None:
                    self._store(frame)
                    failed_reads, backoff = 0, self._backoff_start_s
                    continue

                failed_reads += 1
                if capture is not None and failed_reads < self._max_failed_reads:
                    self._stop.wait(0.01)
                    continue

                log.warning("No frames from %s; reconnecting in %.0f s", self.description, backoff)
                self._release()
                if self._stop.wait(backoff):
                    break
                backoff = min(backoff * 2, self._backoff_max_s)
                self._capture = _open_capture(self.source)
                if self._capture is not None:
                    log.info("Reconnected to %s", self.description)
                failed_reads = 0
        finally:
            self._release()

    def _store(self, frame: np.ndarray) -> None:
        # perf_counter, not monotonic: on Python 3.12 for Windows monotonic only ticks every ~15.6 ms.
        timestamp = time.perf_counter()
        with self._new_frame:
            self._frame, self._timestamp = frame, timestamp
            self._seq += 1
            self._new_frame.notify_all()

    def _release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
