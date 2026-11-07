"""Sentra Guide pipeline: camera -> detection -> direction -> alert policy -> speech, log and screen.

    FrameSource -> ObjectDetector -> DepthProvider (NoDepth by default) -> SpatialEstimator
                -> AlertPolicy -> Speaker + GuideEventLog -> GuideSink (debug view)

Kept separate from the fall-detection pipeline (sentra.pipeline), which still runs with src/main.py.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from sentra.guide.alerts import AlertPolicy
from sentra.guide.depth import DepthProvider, NoDepth
from sentra.guide.detector import ObjectDetector
from sentra.guide.event_log import DEFAULT_OUTPUT_DIR, GuideEventLog
from sentra.guide.schema import GuideSink, GuideUpdate
from sentra.guide.spatial import SpatialEstimator
from sentra.guide.speech import Speaker
from sentra.pipeline import FpsMeter

log = logging.getLogger(__name__)


@dataclass
class GuideControls:
    """Switches the user can flip while it runs (the debug view maps them to keys)."""

    muted: bool = False    # alerts still appear on screen and in the log, but aren't spoken
    privacy: bool = False  # the view hides the camera image and shows only boxes


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
        self._clock = clock
        self.stats = RunStats()
        self.log_path: Path | None = None

    def run(self) -> RunStats:
        """Process frames until the sink says stop, a video file ends, or Ctrl+C."""
        self.detector.load()
        event_log = GuideEventLog(self.output_dir, self.label)
        self.log_path = event_log.path
        log.info("Logging alerts (no images) to %s", event_log.path)
        fps = FpsMeter()
        description = getattr(self.source, "description", "")
        frame_id = 0
        missed_reads = 0
        started = self._clock()
        try:
            while not self.sink.should_stop():
                ok, image, timestamp = self.source.read(timeout=1.0)
                if not ok:
                    if getattr(self.source, "ended", False):
                        log.info("Video file finished.")
                        break
                    missed_reads += 1
                    if missed_reads == 3:
                        log.warning("Waiting for frames from the camera... (q or Ctrl+C to quit)")
                    continue
                missed_reads = 0
                frame_id += 1

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
        return self.stats
