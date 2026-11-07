"""Sentra Guide pipeline: camera -> detection -> direction -> alert policy -> speech, log and screen.

    FrameSource -> ObjectDetector -> DepthProvider (NoDepth by default) -> SpatialEstimator
                -> AlertPolicy -> Speaker + GuideEventLog -> GuideSink (debug view)

Kept separate from the fall-detection pipeline (sentra.pipeline), which still runs with src/main.py.

Camera switching: the view puts a CameraChoice in `controls.requested_camera`. The pipeline closes the current
camera and opens the new one on a background thread (opening can take seconds), goes back to the old camera if
the new one won't open, and forgets tracks, regions and recent cues so nothing carries over from the old view.

While there is no frame to show (switching, or a stalled stream), a sink that has an `idle(message)` method gets
called a few times a second, so a window keeps responding to keys and can say what's happening.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from sentra.guide.alerts import AlertPolicy
from sentra.guide.cameras import CameraChoice
from sentra.guide.depth import DepthProvider, NoDepth
from sentra.guide.detector import ObjectDetector
from sentra.guide.event_log import DEFAULT_OUTPUT_DIR, GuideEventLog
from sentra.guide.schema import GuideSink, GuideUpdate
from sentra.guide.spatial import SpatialEstimator
from sentra.guide.speech import Speaker
from sentra.pipeline import FpsMeter

log = logging.getLogger(__name__)

READ_TIMEOUT_S = 0.25   # how long to wait for a frame before giving the sink an idle update
WAITING_AFTER_S = 1.0   # no frame for this long -> show "Waiting for the camera..."
WARN_AFTER_S = 3.0      # ...and log a warning after this long


@dataclass
class GuideControls:
    """Switches the user can flip while it runs (the debug view maps them to keys and the camera dropdown)."""

    muted: bool = False    # alerts still appear on screen and in the log, but aren't spoken
    privacy: bool = False  # the view hides the camera image and shows only boxes
    camera_label: str = ""                       # the camera in use, as the dropdown names it
    camera_source: int | str | None = None       # ...and what was opened for it
    requested_camera: CameraChoice | None = None  # set by the view; the pipeline switches and clears it
    switching: bool = False                      # a camera switch is in progress
    notice: str = ""                             # short status line for the view
    notice_at: float = 0.0                       # perf_counter when `notice` was set

    def tell(self, notice: str) -> None:
        self.notice, self.notice_at = notice, time.perf_counter()


class _CameraSwitch(threading.Thread):
    """Close the old camera, open the new one, and reopen the old one if that fails. Runs off the main thread."""

    def __init__(self, old: Any, choice: CameraChoice, open_source: Callable[[int | str], Any]) -> None:
        super().__init__(name="CameraSwitch", daemon=True)
        self.old, self.choice, self.open_source = old, choice, open_source
        self.new: Any = None
        self.reopened: Any = None
        self.error: Exception | None = None

    def run(self) -> None:
        _close(self.old)
        try:
            self.new = self.open_source(self.choice.source)
            return
        except Exception as exc:  # FrameSource raises RuntimeError when a camera won't open
            self.error = exc
        old_spec = getattr(self.old, "source", None)
        if old_spec is not None:
            try:
                self.reopened = self.open_source(old_spec)
            except Exception as exc:
                log.warning("Could not reopen the previous camera either: %s", exc)


def _close(source: Any) -> None:
    close = getattr(source, "close", None)
    if close is not None:
        close()


@dataclass
class RunStats:
    frames: int = 0
    stale_frames: int = 0
    alerts: int = 0
    seconds: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    inference_ms: list[float] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "seconds": round(self.seconds, 1),
            "fps": round(self.fps, 1),
            "median_latency_ms": round(median(self.latencies_ms), 1) if self.latencies_ms else None,
            "median_inference_ms": round(median(self.inference_ms), 1) if self.inference_ms else None,
            "alerts": self.alerts,
            "stale_frames": self.stale_frames,
        }

    def summary(self) -> str:
        d = self.as_dict()
        latency = f"{d['median_latency_ms']:.0f} ms" if d["median_latency_ms"] is not None else "n/a"
        inference = f"{d['median_inference_ms']:.0f} ms" if d["median_inference_ms"] is not None else "n/a"
        return (f"{d['frames']} frames in {d['seconds']} s ({d['fps']} fps); median model time {inference}, "
                f"median capture-to-decision latency {latency}; {d['alerts']} alerts; "
                f"{d['stale_frames']} frames too old to announce")


class GuidePipeline:
    def __init__(
        self,
        source: Any,                    # sentra.capture.FrameSource or anything with read()/description
        detector: ObjectDetector,
        sink: GuideSink,
        *,
        speaker: Speaker | None = None,
        spatial: SpatialEstimator | None = None,
        depth: DepthProvider | None = None,
        policy: AlertPolicy | None = None,
        controls: GuideControls | None = None,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        label: str | None = None,
        clock=time.perf_counter,
        open_source: Callable[[int | str], Any] | None = None,  # opens a camera for a switch; None = no switching
    ) -> None:
        self.source = source
        self.detector = detector
        self.sink = sink
        self.speaker = speaker
        self.spatial = spatial or SpatialEstimator()
        self.depth = depth or NoDepth()
        self.policy = policy or AlertPolicy(clock=clock)
        self.controls = controls or GuideControls()
        self.output_dir = output_dir
        self.label = label
        self.open_source = open_source
        self._clock = clock
        self._switch: _CameraSwitch | None = None
        self.switches = 0
        self.stats = RunStats()
        self.log_path: Path | None = None
        if self.controls.camera_source is None:
            self.controls.camera_source = getattr(source, "source", None)
        if not self.controls.camera_label:
            self.controls.camera_label = getattr(source, "description", "")

    def run(self) -> RunStats:
        """Process frames until the sink says stop, a video file ends, or Ctrl+C."""
        self.detector.load()
        event_log = GuideEventLog(self.output_dir, self.label)
        self.log_path = event_log.path
        log.info("Logging alerts (no images) to %s", event_log.path)
        fps = FpsMeter()
        frame_id = 0
        last_frame_at = time.perf_counter()
        warned = False
        started = self._clock()
        try:
            while not self.sink.should_stop():
                if self._poll_switch():
                    self._idle(self.controls.notice)
                    time.sleep(0.03)
                    last_frame_at = time.perf_counter()
                    continue
                ok, image, timestamp = self.source.read(timeout=READ_TIMEOUT_S)
                if not ok:
                    if getattr(self.source, "ended", False):
                        log.info("Video file finished.")
                        break
                    waited = time.perf_counter() - last_frame_at
                    if waited >= WARN_AFTER_S and not warned:
                        log.warning("Waiting for frames from the camera... (q or Ctrl+C to quit)")
                        warned = True
                    self._idle("Waiting for the camera..." if waited >= WAITING_AFTER_S else "")
                    time.sleep(0.01)  # a closed source returns at once; don't spin
                    continue
                last_frame_at, warned = time.perf_counter(), False
                frame_id += 1

                description = getattr(self.source, "description", "")
                frame = self.detector.detect(image, frame_id, timestamp, description)
                depths = self.depth.estimate(image, frame)
                objects = self.spatial.locate(frame, depths)
                alerts = self.policy.update(objects, frame)
                latency_ms = (self._clock() - timestamp) * 1000.0
                stale = latency_ms > self.policy.config.max_frame_age_s * 1000.0

                for alert in alerts:
                    spoken = self.speaker is not None and not self.controls.muted
                    if spoken:
                        self.speaker.say(alert.text, created=timestamp)
                    event_log.write_alert(alert, spoken=spoken)

                self.stats.frames += 1
                self.stats.stale_frames += stale
                self.stats.alerts += len(alerts)
                self.stats.latencies_ms.append(latency_ms)
                self.stats.inference_ms.append(frame.inference_ms)
                self.sink.publish(GuideUpdate(
                    frame=frame,
                    objects=objects,
                    alerts=alerts,
                    image=image,
                    fps=fps.tick(),
                    latency_ms=latency_ms,
                    depth_status=self.depth.status(),
                    stale=stale,
                ))
        finally:
            self.stats.seconds = self._clock() - started
            event_log.write_summary(self.stats.as_dict())
            event_log.close()
            self._abandon_switch()
        return self.stats

    # Camera switching ------------------------------------------------------------------------------------

    def _poll_switch(self) -> bool:
        """Start a requested camera switch, or finish one that's done. True while a switch is running."""
        if self._switch is not None:
            if self._switch.is_alive():
                return True
            self._finish_switch()
            return False
        choice = self.controls.requested_camera
        if choice is None:
            return False
        self.controls.requested_camera = None
        if choice.source == self.controls.camera_source:
            return False
        if self.open_source is None:
            self.controls.tell("Camera switching isn't available in this mode")
            return False
        log.info("Switching camera to %s ...", choice.label)
        self.controls.switching = True
        self.controls.tell(f"Switching to {choice.label}...")
        self._switch = _CameraSwitch(self.source, choice, self.open_source)
        self._switch.start()
        return True

    def _finish_switch(self) -> None:
        switch, self._switch = self._switch, None
        self.controls.switching = False
        if switch.new is not None:
            self.source = switch.new
            self.controls.camera_label = switch.choice.label
            self.controls.camera_source = switch.choice.source
            self.controls.tell(f"Now using {switch.choice.label}")
            self.switches += 1
            log.info("Camera: %s", switch.choice.label)
        elif switch.reopened is not None:
            self.source = switch.reopened
            self.controls.tell(f"Couldn't open {switch.choice.label}; back to {self.controls.camera_label}")
            log.warning("Could not open %s (%s); went back to %s",
                        switch.choice.label, switch.error, self.controls.camera_label)
        else:
            self.source = switch.old  # closed: read() keeps failing until another camera is picked
            self.controls.tell(f"Couldn't open {switch.choice.label}. Pick another camera.")
            log.warning("Could not open %s (%s), and the previous camera didn't reopen",
                        switch.choice.label, switch.error)
        # Whatever is on screen now is a different view: forget track IDs, remembered regions and recent cues.
        self.detector.reset_tracking()
        self.spatial.reset()
        self.policy.reset()

    def _abandon_switch(self) -> None:
        """On exit: let a running switch finish (briefly) and close whatever it opened."""
        switch, self._switch = self._switch, None
        if switch is None:
            return
        switch.join(timeout=5.0)
        for opened in (switch.new, switch.reopened):
            if opened is not None:
                self.source = opened  # so the caller's close() reaches it

    def _idle(self, message: str) -> None:
        idle = getattr(self.sink, "idle", None)
        if idle is not None:
            idle(message)
