"""Sentra Guide pipeline: camera -> detection -> direction -> alert policy -> speech, log and screen.

    FrameSource -> ObjectDetector -> DepthProvider (NoDepth by default) -> SpatialEstimator
                -> AlertPolicy -> Speaker + GuideEventLog -> GuideSink (debug view)

Kept separate from the fall-detection pipeline (sentra.pipeline), which still runs with src/main.py.

Camera switching: the view puts a CameraChoice in `controls.requested_camera`. The pipeline opens the new camera on
a background thread (opening can take seconds; see _CameraSwitch), keeps or goes back to the old camera if the new
one won't open, and forgets tracks, regions and recent cues so nothing carries over from the old view.

Rotation is per camera: `controls.rotation` is applied to each frame here (not in FrameSource), the r key changes
it, and `controls.rotations` remembers it per camera so switching back restores it.

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

import cv2

from sentra.capture import ROTATIONS
from sentra.guide.alerts import AlertPolicy
from sentra.guide.cameras import CameraChoice, is_stream
from sentra.guide.depth import DepthProvider, NoDepth
from sentra.guide.detector import ObjectDetector
from sentra.guide.event_log import DEFAULT_OUTPUT_DIR, GuideEventLog
from sentra.guide.person_check import PersonCheck
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
    rotation: int = 0                            # degrees clockwise applied to the current camera's frames
    rotations: dict[int | str, int] = field(default_factory=dict)  # remembered rotation per camera source

    def tell(self, notice: str) -> None:
        self.notice, self.notice_at = notice, time.perf_counter()

    def rotate_clockwise(self) -> None:
        """Turn the current camera another 90 degrees (the r key). Remembered for that camera."""
        self.rotation = (self.rotation + 90) % 360
        if self.camera_source is not None:
            self.rotations[self.camera_source] = self.rotation
        self.tell(f"Rotation {self.rotation} degrees")


class _CameraSwitch(threading.Thread):
    """Open the new camera off the main thread.

    - A phone stream (URL) is opened while the old camera keeps running (`keep_old`): a stream that isn't up yet
      can take ~14 s to fail, and the video shouldn't go blank meanwhile. The main thread closes the old one after.
    - A device is opened after closing the old camera (a webcam can usually be open in only one place), and the
      old camera is reopened if the new one fails.
    """

    def __init__(self, old: Any, choice: CameraChoice, open_source: Callable[[int | str], Any]) -> None:
        super().__init__(name="CameraSwitch", daemon=True)
        self.old, self.choice, self.open_source = old, choice, open_source
        self.keep_old = is_stream(choice.source)
        self.new: Any = None
        self.reopened: Any = None
        self.error: Exception | None = None

    def run(self) -> None:
        if self.keep_old:
            try:
                self.new = self.open_source(self.choice.source)
            except Exception as exc:
                self.error = exc
            return
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
    person_checks: int = 0         # pose-model runs by the person check
    person_check_ms: list[float] = field(default_factory=list)
    unconfirmed_people: int = 0    # "person" detections with no body seen (relabelled "unconfirmed person")

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        data = {
            "frames": self.frames,
            "seconds": round(self.seconds, 1),
            "fps": round(self.fps, 1),
            "median_latency_ms": round(median(self.latencies_ms), 1) if self.latencies_ms else None,
            "median_inference_ms": round(median(self.inference_ms), 1) if self.inference_ms else None,
            "alerts": self.alerts,
            "stale_frames": self.stale_frames,
        }
        if self.person_checks or self.unconfirmed_people:
            data["person_checks"] = self.person_checks
            data["median_person_check_ms"] = (round(median(self.person_check_ms), 1)
                                              if self.person_check_ms else None)
            data["unconfirmed_people"] = self.unconfirmed_people
        return data

    def summary(self) -> str:
        d = self.as_dict()
        latency = f"{d['median_latency_ms']:.0f} ms" if d["median_latency_ms"] is not None else "n/a"
        inference = f"{d['median_inference_ms']:.0f} ms" if d["median_inference_ms"] is not None else "n/a"
        text = (f"{d['frames']} frames in {d['seconds']} s ({d['fps']} fps); median model time {inference}, "
                f"median capture-to-decision latency {latency}; {d['alerts']} alerts; "
                f"{d['stale_frames']} frames too old to announce")
        if "person_checks" in d:
            check = d["median_person_check_ms"]
            text += (f"; person check ran on {d['person_checks']} frames"
                     + (f" ({check:.0f} ms median)" if check is not None else "")
                     + f" and found no body in {d['unconfirmed_people']} 'person' boxes "
                       "(hands/arms or too close: never named, at most 'Possible obstacle')")
        return text


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
        person_check: PersonCheck | None = None,  # relabels "person" boxes with no body (hands/arms); None = off
    ) -> None:
        self.source = source
        self.detector = detector
        self.person_check = person_check
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
        self._applied_rotation = self.controls.rotation
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
        if self.person_check is not None:
            self.person_check.load()
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
                if self._poll_switch() and not self._switch.keep_old:  # old camera closed: nothing to show yet
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
                image = self._rotate(image)

                description = getattr(self.source, "description", "")
                frame = self.detector.detect(image, frame_id, timestamp, description)
                if self.person_check is not None:
                    checks, rejected = self.person_check.checks, self.person_check.rejected
                    frame = self.person_check.filter(image, frame)
                    if self.person_check.checks > checks:
                        self.stats.person_checks += 1
                        self.stats.person_check_ms.append(self.person_check.check_ms[-1])
                    self.stats.unconfirmed_people += self.person_check.rejected - rejected
                depths = self.depth.estimate(image, frame)
                objects = self.spatial.locate(frame, depths)
                alerts = self.policy.update(objects, frame)
                latency_ms = (self._clock() - timestamp) * 1000.0
                stale = latency_ms > self.policy.config.max_frame_age_s * 1000.0

                for alert in alerts:
                    spoken = self.speaker is not None and not self.controls.muted
                    if spoken:
                        if alert.caution:  # a danger warning goes before anything still waiting
                            self.speaker.say(alert.text, created=timestamp, urgent=True)
                        else:
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
        if self.controls.camera_source is not None:  # remember this camera's rotation for when we come back
            self.controls.rotations[self.controls.camera_source] = self.controls.rotation
        self.controls.switching = True
        self._switch = _CameraSwitch(self.source, choice, self.open_source)
        verb = "Connecting to" if self._switch.keep_old else "Switching to"
        self.controls.tell(f"{verb} {choice.label}...")
        self._switch.start()
        return True

    def _finish_switch(self) -> None:
        switch, self._switch = self._switch, None
        self.controls.switching = False
        if switch.keep_old and switch.new is None:
            # The old camera never stopped, so nothing on screen changed: keep its tracks, regions and cues.
            self.controls.tell(f"Couldn't open {switch.choice.label}; still using {self.controls.camera_label}")
            log.warning("Could not open %s (%s); still using %s",
                        switch.choice.label, switch.error, self.controls.camera_label)
            return
        if switch.new is not None:
            if switch.keep_old:  # closing can take a moment (it waits for the reader thread): not on this thread
                threading.Thread(target=_close, args=(switch.old,), name="CloseOldCamera", daemon=True).start()
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
        # Whatever is on screen now is a different view: forget track IDs, remembered regions and recent cues,
        # and use the rotation remembered for this camera (0 for one never rotated).
        self.controls.rotation = self.controls.rotations.get(self.controls.camera_source, 0)
        self._applied_rotation = self.controls.rotation
        self._reset_scene()

    def _abandon_switch(self) -> None:
        """On exit: let a running switch finish (briefly) and close whatever it opened."""
        switch, self._switch = self._switch, None
        if switch is None:
            return
        switch.join(timeout=5.0)
        if switch.keep_old:  # the old camera is still self.source (closed by the caller); drop the new one
            _close(switch.new)
            return
        for opened in (switch.new, switch.reopened):
            if opened is not None:
                self.source = opened  # so the caller's close() reaches it

    def _rotate(self, image: Any) -> Any:
        """Turn the frame upright with the current camera's rotation. A new rotation is a new view: start fresh."""
        rotation = self.controls.rotation
        if rotation != self._applied_rotation:
            self._applied_rotation = rotation
            self._reset_scene()
        code = ROTATIONS.get(rotation)
        return image if code is None else cv2.rotate(image, code)

    def _reset_scene(self) -> None:
        """Forget track IDs, remembered regions and recent cues (after a camera switch or rotation)."""
        self.detector.reset_tracking()
        if self.person_check is not None:
            self.person_check.reset()
        self.spatial.reset()
        self.policy.reset()

    def _idle(self, message: str) -> None:
        idle = getattr(self.sink, "idle", None)
        if idle is not None:
            idle(message)
