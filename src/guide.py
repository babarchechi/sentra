"""Sentra Guide entry point: camera -> YOLO26n objects -> left/center/right -> short spoken cues.

Experimental supplementary awareness prototype. Not a navigation aid; never says a path is clear.

Run from the repo root (docs/SETUP_WINDOWS.md has phone instructions):
    python src/guide.py --source 0                                   laptop webcam
    python src/guide.py --source http://<phone-ip>:4747/video        iPhone (DroidCam) on the same network
    python src/guide.py --source http://<phone-ip>:8080/video        Android (IP Webcam)
    python src/guide.py --source my_clip.mp4 --no-view --voice print consented recording, no window

The fall-detection prototype is unchanged: python src/main.py
"""
from __future__ import annotations

import argparse
import logging

from sentra.capture import FrameSource, parse_source
from sentra.guide.alerts import AlertConfig, AlertPolicy
from sentra.guide.detector import DEFAULT_DETECT_MODEL, ObjectDetector
from sentra.guide.pipeline import GuideControls, GuidePipeline
from sentra.guide.spatial import SpatialEstimator
from sentra.guide.speech import PrintSpeech, Speaker, WindowsSpeech
from sentra.guide.view import GuideView, HeadlessSink

log = logging.getLogger("sentra")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sentra Guide: experimental object awareness with short spoken cues (not a navigation aid)"
    )
    parser.add_argument("--source", default="0",
                        help="Webcam index (0), phone stream URL, or path to a consented local video file")
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0,
                        help="Rotate frames clockwise so the image is upright (phones in portrait: 90 or 270)")
    parser.add_argument("--model", default=DEFAULT_DETECT_MODEL, help="Detection weights (default yolo26n.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector score threshold (default 0.25)")
    parser.add_argument("--announce-conf", type=float, default=0.5,
                        help="Score needed to say an object's name (default 0.5)")
    parser.add_argument("--imgsz", type=int, default=640, help="Model input size; 480 or 320 is faster")
    parser.add_argument("--no-track", action="store_true", help="Turn off ByteTrack (detections only)")
    parser.add_argument("--all-labels", action="store_true",
                        help="Announce every class the model knows, not just the obstacle list")
    parser.add_argument("--repeat-after", type=float, default=10.0,
                        help="Seconds before the same object and direction is said again (default 10)")
    parser.add_argument("--min-gap", type=float, default=2.5,
                        help="Minimum seconds between any two announcements (default 2.5)")
    parser.add_argument("--voice", choices=("windows", "print", "off"), default="windows",
                        help="windows: laptop's built-in voice; print: console only; off: no speech")
    parser.add_argument("--muted", action="store_true", help="Start muted (press m in the window to unmute)")
    parser.add_argument("--privacy", action="store_true", help="Start with the camera image hidden")
    parser.add_argument("--no-view", action="store_true", help="No window; alerts go to the console")
    parser.add_argument("--max-seconds", type=float, default=None, help="Stop after this many seconds")
    parser.add_argument("--label", default=None,
                        help="Tag for the event-log filename, e.g. a test-scene name from docs/VALIDATION.md")
    return parser


def build_policy(args: argparse.Namespace) -> AlertPolicy:
    config = AlertConfig(
        announce_confidence=args.announce_conf,
        uncertain_confidence=min(0.3, args.announce_conf),
        repeat_after_s=args.repeat_after,
        min_gap_s=args.min_gap,
        **({"labels": None} if args.all_labels else {}),
    )
    return AlertPolicy(config)


def build_speaker(choice: str) -> Speaker | None:
    if choice == "off":
        return None
    return Speaker(WindowsSpeech() if choice == "windows" else PrintSpeech())


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("sentra").setLevel(logging.INFO)

    detector = ObjectDetector(args.model, conf=args.conf, imgsz=args.imgsz, track=not args.no_track)
    log.info("Loading %s ...", args.model)
    try:
        detector.warmup()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from None

    try:
        source = FrameSource(parse_source(args.source), rotate=args.rotate)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    rotation = f", rotated {args.rotate} degrees" if args.rotate else ""
    log.info("Camera: %s%s. Experimental: not a navigation aid.", source.description, rotation)

    policy = build_policy(args)
    spatial = SpatialEstimator()
    controls = GuideControls(muted=args.muted, privacy=args.privacy)
    speaker = build_speaker(args.voice)
    if args.no_view:
        sink = HeadlessSink(args.max_seconds)
    else:
        sink = GuideView(controls, policy, spatial.config, voice_name=args.voice)
        if args.max_seconds is not None:
            log.warning("--max-seconds only applies with --no-view; press q to quit the window.")
        log.info("Click the video window, then: m mute, p privacy, q quit.")

    pipeline = GuidePipeline(source, detector, sink, speaker=speaker, spatial=spatial, policy=policy,
                             controls=controls, label=args.label)
    try:
        pipeline.run()
    except KeyboardInterrupt:
        log.info("Stopped with Ctrl+C.")
    finally:
        sink.close()
        source.close()
        if speaker is not None:
            speaker.close()
            if speaker.failed:
                log.warning("The Windows voice failed, so alerts were printed instead.")
        log.info("Run summary: %s", pipeline.stats.summary())
        if pipeline.log_path is not None:
            log.info("Alerts saved to %s", pipeline.log_path)


if __name__ == "__main__":
    main()
