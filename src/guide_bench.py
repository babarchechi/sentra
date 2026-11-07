"""Sentra Guide benchmark: how fast the real guide loop runs on this machine, per model size, tracking on/off.

Each configuration runs the same loop as src/guide.py (GuidePipeline: camera -> detector -> spatial -> alert
policy), without speech or a window, and measures every processed frame:

- model ms: detector time per frame (ObjectDetector.detect)
- capture->decision ms: from the moment the camera delivered the frame to the alert decision
- fps: frames processed per second. It can't exceed the source's own frame rate (webcams ~30, files their own fps).

The first --warmup frames of each configuration are not counted (model start-up, tracker start-up).

    python src/guide_bench.py --source 0
    python src/guide_bench.py --source "http://<phone-ip>:8080/video" --frames 300
    python src/guide_bench.py --source clip.mp4 --imgsz 640 320 --track on

It prints a Markdown table for docs/VALIDATION.md, Section 2. Stream URLs are replaced by a placeholder in the
printed command, so the table can be committed. No images are saved; alert logs go to a temporary folder.
"""
from __future__ import annotations

import argparse
import logging
import os
import platform
import shlex
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sentra.capture import FrameSource, describe_source, parse_source
from sentra.guide.detector import DEFAULT_DETECT_MODEL, ObjectDetector
from sentra.guide.pipeline import GuidePipeline
from sentra.guide.schema import GuideUpdate

log = logging.getLogger("sentra")

DEFAULT_SIZES = (640, 480, 320)


@dataclass
class Sample:
    at: float              # perf_counter when the update reached the sink
    model_ms: float
    latency_ms: float      # capture -> decision
    objects: int
    stale: bool


class BenchSink:
    """A GuideSink that records one Sample per frame and stops after `total` frames or `timeout_s` seconds."""

    def __init__(self, total: int, timeout_s: float = 120.0, clock: Callable[[], float] = time.perf_counter):
        self.total = total
        self.timeout_s = timeout_s
        self.samples: list[Sample] = []
        self._clock = clock
        self._started = clock()
        self.timed_out = False

    def publish(self, update: GuideUpdate) -> None:
        self.samples.append(Sample(
            at=self._clock(),
            model_ms=update.frame.inference_ms,
            latency_ms=update.latency_ms,
            objects=len(update.objects),
            stale=update.stale,
        ))

    def should_stop(self) -> bool:
        if len(self.samples) >= self.total:
            return True
        if self._clock() - self._started > self.timeout_s:
            self.timed_out = True
            return True
        return False

    def close(self) -> None:
        pass


@dataclass
class BenchResult:
    imgsz: int
    track: bool
    frames: int = 0                 # measured frames (after warm-up)
    seconds: float = 0.0
    fps: float = 0.0
    model_ms_median: float | None = None
    model_ms_p90: float | None = None
    latency_ms_median: float | None = None
    latency_ms_p90: float | None = None
    objects_per_frame: float | None = None
    stale_frames: int = 0
    notes: list[str] = field(default_factory=list)


def _p(values: Sequence[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 1) if len(values) else None


def summarize(samples: Sequence[Sample], imgsz: int, track: bool, warmup: int) -> BenchResult:
    """Stats over the samples after the first `warmup` ones."""
    result = BenchResult(imgsz, track)
    measured = list(samples[warmup:])
    result.frames = len(measured)
    if not measured:
        result.notes.append("no frames measured")
        return result
    if len(measured) > 1:
        span = measured[-1].at - measured[0].at  # rounded only for display: fast runs can take < 0.005 s
        result.seconds = round(span, 2)
        if span > 0:
            result.fps = round((len(measured) - 1) / span, 1)
    model = [s.model_ms for s in measured]
    latency = [s.latency_ms for s in measured]
    result.model_ms_median, result.model_ms_p90 = _p(model, 50), _p(model, 90)
    result.latency_ms_median, result.latency_ms_p90 = _p(latency, 50), _p(latency, 90)
    result.objects_per_frame = round(sum(s.objects for s in measured) / len(measured), 1)
    result.stale_frames = sum(s.stale for s in measured)
    return result


def run_config(
    open_source: Callable[[], Any],
    make_detector: Callable[[int, bool], ObjectDetector],
    imgsz: int,
    track: bool,
    *,
    frames: int,
    warmup: int,
    timeout_s: float = 120.0,
) -> BenchResult:
    """Run the guide loop for `warmup + frames` frames with one model size and tracking setting."""
    detector = make_detector(imgsz, track)
    detector.warmup()
    source = open_source()
    sink = BenchSink(warmup + frames, timeout_s)
    try:
        with tempfile.TemporaryDirectory(prefix="sentra_bench_") as tmp:
            GuidePipeline(source, detector, sink, output_dir=Path(tmp)).run()
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()
    result = summarize(sink.samples, imgsz, track, warmup)
    if sink.timed_out:
        result.notes.append(f"stopped after {timeout_s:.0f} s")
    elif len(sink.samples) < warmup + frames:
        result.notes.append("source ended early")
    return result


def markdown_table(results: Sequence[BenchResult]) -> str:
    def cell(value: float | None) -> str:
        return "n/a" if value is None else f"{value:g}"

    lines = [
        "| imgsz | Tracking | Frames / s | fps | Model ms median / p90 | Capture→decision ms median / p90 "
        "| Objects / frame | Stale | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r.imgsz} | {'on' if r.track else 'off'} | {r.frames} / {r.seconds:g} | {r.fps:g} "
            f"| {cell(r.model_ms_median)} / {cell(r.model_ms_p90)} "
            f"| {cell(r.latency_ms_median)} / {cell(r.latency_ms_p90)} "
            f"| {cell(r.objects_per_frame)} | {r.stale_frames} | {'; '.join(r.notes)} |"
        )
    return "\n".join(lines)


def printable_command(argv: Sequence[str]) -> str:
    """The command line with stream URLs replaced and file paths cut to the file name, safe to commit."""
    def safe(raw: str) -> str:
        if "://" in raw:
            return "<stream URL>"
        return os.path.basename(raw) if os.path.isfile(raw) else raw

    shown = list(argv)
    for i, arg in enumerate(shown):
        if arg.startswith("--source="):
            shown[i] = "--source=" + safe(arg.split("=", 1)[1])
        elif arg == "--source" and i + 1 < len(shown):
            shown[i + 1] = safe(shown[i + 1])
    return "python src/guide_bench.py " + " ".join(shlex.quote(a) for a in shown)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the Sentra Guide loop (fps, model time, latency)")
    parser.add_argument("--source", default="0", help="Webcam index, phone stream URL, or a local video file")
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--model", default=DEFAULT_DETECT_MODEL)
    parser.add_argument("--conf", type=float, default=0.25, help="Detector score threshold (same default as guide.py)")
    parser.add_argument("--imgsz", type=int, nargs="+", default=list(DEFAULT_SIZES), help="Model input sizes to test")
    parser.add_argument("--track", choices=("on", "off", "both"), default="both", help="ByteTrack setting(s) to test")
    parser.add_argument("--frames", type=int, default=200, help="Measured frames per configuration (default 200)")
    parser.add_argument("--warmup", type=int, default=20, help="Frames skipped at the start of each configuration")
    parser.add_argument("--timeout", type=float, default=120.0, help="Give up on a configuration after this many s")
    return parser


def main(argv: Sequence[str] | None = None) -> list[BenchResult]:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.frames < 2 or args.warmup < 0:
        raise SystemExit("Need --frames >= 2 and --warmup >= 0")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    source = parse_source(args.source)
    tracks = {"on": [True], "off": [False], "both": [True, False]}[args.track]

    def open_source() -> FrameSource:
        try:
            return FrameSource(source, rotate=args.rotate)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from None

    def make_detector(imgsz: int, track: bool) -> ObjectDetector:
        return ObjectDetector(args.model, conf=args.conf, imgsz=imgsz, track=track)

    print(f"Benchmarking {args.model} on {describe_source(source)}: {len(args.imgsz) * len(tracks)} configurations, "
          f"{args.warmup} + {args.frames} frames each. Keep the scene steady.", flush=True)
    results = []
    for imgsz in args.imgsz:
        for track in tracks:
            result = run_config(open_source, make_detector, imgsz, track,
                                frames=args.frames, warmup=args.warmup, timeout_s=args.timeout)
            print(f"  imgsz {imgsz}, tracking {'on' if track else 'off'}: {result.fps:g} fps, model "
                  f"{result.model_ms_median} ms median, capture->decision {result.latency_ms_median} ms median",
                  flush=True)
            results.append(result)

    print()
    print(f"Command: `{printable_command(argv)}`")
    print(f"Machine: {platform.platform()}, {platform.processor() or platform.machine()}, "
          f"Python {platform.python_version()}")
    print()
    print(markdown_table(results))
    return results


if __name__ == "__main__":
    main()
