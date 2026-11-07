"""The live camera list behind the debug view's camera dropdown.

On Windows it lists every camera Windows knows about (built-in, USB, DroidCam's virtual camera, a phone
connected through Phone Link), read again in the background whenever the dropdown is opened, so a camera
plugged in or connected while the app runs shows up. Phone video streams can't be discovered, so the stream
URLs given on the command line are listed too. Elsewhere, or without the winrt packages, it falls back to
OpenCV webcam numbers.

Picking a Windows camera opens it as source "windows:<its name>" (sentra.capture), a stream by its URL.
Stream URLs are never shown: they are labelled "Stream 1 (port 4747)" and so on.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

from sentra.capture import PHONE_HINT, WINDOWS_PREFIX, describe_source

log = logging.getLogger(__name__)

Lister = Callable[[], "list[tuple[str, str]] | None"]  # () -> [(device id, name)], or None if unavailable


@dataclass(frozen=True)
class CameraChoice:
    """One row of the dropdown."""

    key: str              # stable identity across refreshes, e.g. "win:<device id>" or "url:1"
    label: str            # what the dropdown shows
    source: int | str     # what FrameSource opens


def windows_lister() -> list[tuple[str, str]] | None:
    """(device id, name) of every enabled Windows camera, or None when the Windows camera API isn't available."""
    if sys.platform != "win32":
        return None
    try:
        from sentra.windows_camera import list_devices  # Windows-only dependency, imported on demand

        return list_devices()
    except Exception as exc:  # winrt missing, or the camera service failed
        log.debug("Windows camera list unavailable: %s", exc)
        return None


def nice_name(name: str) -> str:
    """Shorter, clearer device names for the dropdown."""
    return name.replace(f"({PHONE_HINT})", "(Phone Link)").strip()


def stream_label(number: int, url: str) -> str:
    try:
        port = urlsplit(url).port
    except ValueError:
        port = None
    return f"Stream {number}" + (f" (port {port})" if port else "")


def is_stream(source: int | str) -> bool:
    return isinstance(source, str) and "://" in source


class CameraCatalog:
    """Thread-safe camera list. refresh() re-reads devices on a background thread and never blocks the caller."""

    def __init__(
        self,
        streams: Sequence[str] = (),
        *,
        lister: Lister = windows_lister,
        fallback_webcams: int = 2,
        min_refresh_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        seen: list[str] = []
        for url in streams:  # keep command-line order, drop duplicates
            if url not in seen:
                seen.append(url)
        self._streams = [CameraChoice(f"url:{i}", stream_label(i, url), url) for i, url in enumerate(seen, 1)]
        self._lister = lister
        self._fallback = [CameraChoice(f"cv:{i}", f"Webcam {i}", i) for i in range(fallback_webcams)]
        self._min_refresh_s = min_refresh_s
        self._clock = clock
        self._lock = threading.Lock()
        self._devices: list[CameraChoice] | None = None  # None until the first scan finishes
        self._thread: threading.Thread | None = None
        self._last_start = float("-inf")
        self.scans = 0

    # Reading -------------------------------------------------------------------------------------------

    @property
    def choices(self) -> list[CameraChoice]:
        """Streams first, then the devices from the latest scan (OpenCV webcam numbers if there's no device list)."""
        with self._lock:
            devices = self._devices
        return [*self._streams, *(self._fallback if devices is None else devices)]

    @property
    def scanning(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def label_for(self, source: int | str) -> str:
        """Dropdown label for a source, e.g. the one given with --source at start-up."""
        for choice in self.choices:
            if choice.source == source:
                return choice.label
        if isinstance(source, str) and source.lower().startswith(WINDOWS_PREFIX):
            return nice_name(source[len(WINDOWS_PREFIX):].strip())
        if isinstance(source, int):
            return f"Webcam {source}"
        return "Stream" if is_stream(source) else describe_source(source)

    # Refreshing ----------------------------------------------------------------------------------------

    def refresh(self, *, force: bool = False) -> bool:
        """Start a background re-scan unless one is running or the last started under `min_refresh_s` ago.

        Returns True if a scan was started.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            now = self._clock()
            if not force and now - self._last_start < self._min_refresh_s:
                return False
            self._last_start = now
            self._thread = threading.Thread(target=self._scan, name="CameraCatalog", daemon=True)
            self._thread.start()
            return True

    def wait(self, timeout: float = 5.0) -> None:
        """Block until a running scan finishes (for start-up and tests)."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _scan(self) -> None:
        try:
            found = self._lister()
        except Exception as exc:  # a broken lister must never take the app down
            log.warning("Could not list cameras: %s", exc)
            found = None
        devices = None
        if found is not None:
            devices = [
                CameraChoice(f"win:{device_id}", nice_name(name), f"{WINDOWS_PREFIX}{name}")
                for device_id, name in found
            ]
        with self._lock:
            if devices is not None or self._devices is None:
                self._devices = devices  # keep the last good list if a later scan fails
            self.scans += 1
