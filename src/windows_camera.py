"""Windows-only camera reader for cameras OpenCV cannot open.

Phone Link's "Use as a connected camera" exposes a phone as a Windows Virtual Camera. OpenCV's MSMF and
DirectShow backends do not list virtual cameras, so this module reads them through the Windows
MediaCapture API instead. It mimics the small part of cv2.VideoCapture that camera_test.py uses:
isOpened(), read() and release().

Frames are converted in memory and never saved.
"""

import asyncio
import time

import cv2
import numpy as np
from winrt.windows.devices.enumeration import DeviceClass, DeviceInformation
from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
from winrt.windows.media.capture import (
    MediaCapture,
    MediaCaptureInitializationSettings,
    MediaCaptureMemoryPreference,
    MediaCaptureSharingMode,
    StreamingCaptureMode,
)
from winrt.windows.media.capture.frames import MediaFrameSourceKind
from winrt.windows.media.mediaproperties import MediaEncodingSubtypes
from winrt.windows.storage.streams import Buffer


def _run(loop: asyncio.AbstractEventLoop, awaitable):
    return loop.run_until_complete(_await(awaitable))


async def _await(awaitable):
    return await awaitable


MAX_WIDTH = 1280  # the pose model works at 640 px, so larger frames only cost CPU


def _pick_format(formats):
    """Largest non-square format up to MAX_WIDTH, preferring frame rates closest to 30 fps."""
    best, best_key = None, None
    for fmt in formats:
        width, height = fmt.video_format.width, fmt.video_format.height
        if width == 0 or width > MAX_WIDTH or width == height:
            continue
        rate = fmt.frame_rate
        fps = rate.numerator / rate.denominator if rate.denominator else 0
        key = (width * height, -abs(fps - 30))
        if best_key is None or key > best_key:
            best, best_key = fmt, key
    return best


def list_devices() -> list[tuple[str, str]]:
    """(device id, name) for every enabled video capture device, including virtual cameras.

    Windows returns devices in no fixed order, so built-in cameras (which report a panel location) come first,
    then the rest by name. That keeps "camera 0" meaning the laptop's own webcam.
    """
    loop = asyncio.new_event_loop()
    try:
        devices = _run(loop, DeviceInformation.find_all_async(DeviceClass.VIDEO_CAPTURE))
        enabled = [d for d in devices if d.is_enabled]
        enabled.sort(key=lambda d: (d.enclosure_location is None, d.name.lower()))
        return [(d.id, d.name) for d in enabled]
    finally:
        loop.close()


def list_cameras() -> list[str]:
    """Names of all video capture devices Windows knows about, including virtual cameras."""
    return [name for _, name in list_devices()]


class WindowsCamera:
    """Opens a Windows camera by exact device id, or the first whose name contains `name` (case-insensitive)."""

    def __init__(self, name: str | None = None, device_id: str | None = None, frame_timeout: float = 5.0):
        self._loop = asyncio.new_event_loop()
        self._capture = None
        self._reader = None
        self._buffer = None
        self._last_time = None
        self._frame_timeout = frame_timeout
        self.device_name = None

        devices = _run(self._loop, DeviceInformation.find_all_async(DeviceClass.VIDEO_CAPTURE))
        if device_id is not None:
            match = next((d for d in devices if d.id == device_id), None)
        else:
            match = next((d for d in devices if name and name.lower() in d.name.lower()), None)
        if match is None:
            return
        self.device_name = match.name

        settings = MediaCaptureInitializationSettings()
        settings.video_device_id = match.id
        settings.streaming_capture_mode = StreamingCaptureMode.VIDEO
        settings.memory_preference = MediaCaptureMemoryPreference.CPU
        settings.sharing_mode = MediaCaptureSharingMode.EXCLUSIVE_CONTROL

        capture = MediaCapture()
        _run(self._loop, capture.initialize_async(settings))
        source = next(
            (s for s in capture.frame_sources.values() if s.info.source_kind == MediaFrameSourceKind.COLOR),
            None,
        )
        if source is None:
            capture.close()
            return
        best = _pick_format(source.supported_formats)
        if best is not None:
            _run(self._loop, source.set_format_async(best))
        reader = _run(self._loop, capture.create_frame_reader_async(source, MediaEncodingSubtypes.bgra8))
        _run(self._loop, reader.start_async())
        self._capture = capture
        self._reader = reader

    def isOpened(self) -> bool:
        return self._reader is not None

    def read(self):
        """Wait for the next new frame and return (ok, BGR image) like cv2.VideoCapture.read().

        Returns (False, None) when no new frame arrives in time or the stream was closed underneath us
        (camera unplugged, phone locked, Phone Link/DroidCam disconnected).
        """
        if self._reader is None:
            return False, None
        try:
            return self._read()
        except OSError:  # e.g. "The object has been closed." when the device goes away mid-stream
            return False, None

    def _read(self):
        deadline = time.monotonic() + self._frame_timeout
        while time.monotonic() < deadline:
            ref = self._reader.try_acquire_latest_frame()
            frame = ref.video_media_frame if ref is not None else None
            bitmap = frame.software_bitmap if frame is not None else None
            if bitmap is None or ref.system_relative_time == self._last_time:
                time.sleep(0.005)
                continue
            self._last_time = ref.system_relative_time
            if bitmap.bitmap_pixel_format != BitmapPixelFormat.BGRA8:
                bitmap = SoftwareBitmap.convert(bitmap, BitmapPixelFormat.BGRA8)
            width, height = bitmap.pixel_width, bitmap.pixel_height
            size = width * height * 4
            if self._buffer is None or self._buffer.capacity != size:
                self._buffer = Buffer(size)
            bitmap.copy_to_buffer(self._buffer)
            bgra = np.frombuffer(memoryview(self._buffer), dtype=np.uint8)[:size].reshape(height, width, 4)
            return True, cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        return False, None

    def release(self) -> None:
        """Stop and free the camera. Safe to call twice and on a stream that already died."""
        reader, capture = self._reader, self._capture
        self._reader = self._capture = None
        if reader is not None:
            for step in (lambda: _run(self._loop, reader.stop_async()), reader.close):
                try:
                    step()
                except Exception:  # the device may already be gone; nothing left to free
                    pass
        if capture is not None:
            try:
                capture.close()
            except Exception:
                pass
        if not self._loop.is_closed():
            self._loop.close()
