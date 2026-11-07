"""Sentra Guide entry point: camera -> YOLO26n objects -> left/center/right -> short spoken cues.

Experimental supplementary awareness prototype. Not a navigation aid; never says a path is clear.

Run from the repo root (docs/SETUP_WINDOWS.md has phone instructions):
    python src/guide.py --source 0                                   laptop webcam
    python src/guide.py --source http://<phone-ip>:4747/video        iPhone (DroidCam) on the same network
    python src/guide.py --source http://<phone-ip>:8080/video        Android (IP Webcam)
    python src/guide.py --source my_clip.mp4 --no-view --voice print consented recording, no window
    python src/guide.py --source 0 --web --web-host 172.20.10.2      also speak cues on a phone browser
                                                                     (open http://172.20.10.2:8765/ on the phone)

In the window, switch cameras with the "Camera:" dropdown (or press c). It lists every camera Windows can see,
updated while it's open; phone streams can't be discovered, so list them with --extra-source, each with its own
rotation (--rotate is for --source only). Press r to turn the current camera another 90 degrees.
    python src/guide.py --source 0 --extra-source http://<iphone-ip>:4747/video 90 --extra-source http://<android-ip>:8080/video

The fall-detection prototype is unchanged: python src/main.py
"""
from __future__ import annotations

import argparse
import logging

from sentra.capture import FrameSource, parse_source
from sentra.guide.alerts import AlertConfig, AlertPolicy
from sentra.guide.cameras import CameraCatalog, is_stream
from sentra.guide.detector import DEFAULT_DETECT_MODEL, ObjectDetector
from sentra.guide.obstacles import ENVIRONMENTS, ObstacleScanner
from sentra.guide.person_check import DEFAULT_POSE_MODEL, PersonCheck
from sentra.guide.pipeline import GuideControls, GuidePipeline
from sentra.guide.spatial import SpatialEstimator
from sentra.guide.speech import PrintSpeech, Speaker, WindowsSpeech
from sentra.guide.view import GuideView, HeadlessSink
from sentra.guide.web import DEFAULT_HOST, DEFAULT_PORT, TeeSink, WebSpeechPage, hotspot_hint

log = logging.getLogger("sentra")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sentra Guide: experimental object awareness with short spoken cues (not a navigation aid)"
    )
    parser.add_argument("--source", default="0",
                        help="Webcam index (0), phone stream URL, or path to a consented local video file")
    parser.add_argument("--extra-source", action="append", nargs="+", default=[], metavar=("URL", "ROTATION"),
                        help="Another phone stream for the camera dropdown, optionally with its rotation "
                             "(e.g. --extra-source http://<iphone-ip>:4747/video 90). Repeatable")
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0,
                        help="Rotate --source's frames clockwise so the image is upright (phones in portrait: 90 or "
                             "270). Other cameras use their own rotation; r in the window turns the current one")
    parser.add_argument("--model", default=DEFAULT_DETECT_MODEL, help="Detection weights (default yolo26n.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector score threshold (default 0.25)")
    parser.add_argument("--announce-conf", type=float, default=0.5,
                        help="Score needed to say an object's name (default 0.5)")
    parser.add_argument("--imgsz", type=int, default=640, help="Model input size; 480 or 320 is faster")
    parser.add_argument("--no-track", action="store_true",
                        help="Turn off ByteTrack (detections only). With --person-check, the pose model then runs "
                             "more often (see docs/VALIDATION.md, Section 2.1)")
    parser.add_argument("--person-check", action="store_true",
                        help="Experimental: confirm people with the pose model, so the camera holder's hands and arms "
                             "aren't announced as 'person' (a large central one becomes 'Possible obstacle'). Needs "
                             f"{DEFAULT_POSE_MODEL}. Nearly free with tracking; with --no-track it runs more often")
    parser.add_argument("--pose-model", default=DEFAULT_POSE_MODEL,
                        help=f"Pose weights for --person-check (default {DEFAULT_POSE_MODEL})")
    parser.add_argument("--obstacles", choices=sorted(ENVIRONMENTS), default=None,
                        help="Experimental: also look for obstacles YOLO can't name (boxes, poles, bins) from where the "
                             "floor stops, with a segmentation model (indoor: ADE20K, outdoor: Cityscapes). They are "
                             "only ever called 'Possible obstacle ahead.'; the window shows the open-looking floor, "
                             "marked unverified. Off by default")
    parser.add_argument("--all-labels", action="store_true",
                        help="Announce every class the model knows, not just the obstacle list")
    parser.add_argument("--repeat-after", type=float, default=10.0,
                        help="Never repeat the same object and direction sooner than this (default 10 s). Unchanged things are not repeated anyway, unless they leave and come back or more of them appear")
    parser.add_argument("--remind-after", type=float, default=30.0,
                        help="Remind about objects still ahead this often, in seconds (default 30; 0 = never)")
    parser.add_argument("--min-gap", type=float, default=2.5,
                        help="Minimum seconds between any two announcements (default 2.5)")
    parser.add_argument("--no-caution", action="store_true",
                        help="Don't give danger warnings. By default something big in the middle of the view gets "
                             "'Caution, chair ahead.' (said first, repeated every --caution-every s while it stays, "
                             "and the phone page vibrates). There is never a 'safe' signal")
    parser.add_argument("--announce-sides", action="store_true",
                        help="Also announce things fully off to the side and far away (by default only things in "
                             "the path ahead are named, e.g. a chair against the wall stays quiet)")
    parser.add_argument("--caution-every", type=float, default=2.0,
                        help="Seconds between repeats of a caution while the thing stays in the way (default 2: it "
                             "keeps cautioning until the thing is no longer in the way)")
    parser.add_argument("--voice", choices=("windows", "print", "off"), default="windows",
                        help="windows: laptop's built-in voice; print: console only; off: no speech")
    parser.add_argument("--muted", action="store_true", help="Start muted (press m in the window to unmute)")
    parser.add_argument("--privacy", action="store_true", help="Start with the camera image hidden")
    parser.add_argument("--no-view", action="store_true", help="No window; alerts go to the console")
    parser.add_argument("--max-seconds", type=float, default=None, help="Stop after this many seconds")
    parser.add_argument("--label", default=None,
                        help="Tag for the event-log filename, e.g. a test-scene name from docs/VALIDATION.md")
    parser.add_argument("--web", action="store_true",
                        help="Serve the phone speech page: a phone browser speaks the cues (alert text only, no video)")
    parser.add_argument("--web-host", default=DEFAULT_HOST,
                        help="Address the page listens on: this laptop's IP on the phone hotspot, e.g. 172.20.10.2 "
                             "(default 127.0.0.1 = this laptop only; 0.0.0.0 is refused)")
    parser.add_argument("--web-port", type=int, default=DEFAULT_PORT, help=f"Port for --web (default {DEFAULT_PORT})")
    return parser


def build_policy(args: argparse.Namespace) -> AlertPolicy:
    config = AlertConfig(
        announce_confidence=args.announce_conf,
        uncertain_confidence=min(0.3, args.announce_conf),
        repeat_after_s=args.repeat_after,
        remind_after_s=args.remind_after or None,
        min_gap_s=args.min_gap,
        caution=not args.no_caution,
        announce_sides=args.announce_sides,
        caution_repeat_after_s=args.caution_every,
        **({"labels": None} if args.all_labels else {}),
    )
    return AlertPolicy(config)


def build_speaker(choice: str) -> Speaker | None:
    if choice == "off":
        return None
    return Speaker(WindowsSpeech() if choice == "windows" else PrintSpeech())


def start_web_page(args: argparse.Namespace, controls: GuideControls) -> WebSpeechPage:
    try:
        page = WebSpeechPage(controls, host=args.web_host, port=args.web_port)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    except OSError as exc:
        raise SystemExit(f"Can't start the phone page on {args.web_host}:{args.web_port} ({exc}). "
                         "Is another copy running, or is that address not on this laptop? Try another --web-port, "
                         "or check the address with ipconfig.") from None
    log.info("Phone speech page: open %s on the phone, then tap 'Tap to turn on sound'. Sends cue text only.",
             page.url)
    if args.web_host == DEFAULT_HOST:
        log.info("127.0.0.1 is reachable from this laptop only. %s", hotspot_hint())
    if args.voice == "windows":
        log.info("Tip: add --voice off so cues aren't spoken twice (laptop and phone).")
    return page


def parse_extra_sources(values: list[list[str]]) -> list[tuple[str, int]]:
    """--extra-source URL [ROTATION] (repeatable) -> [(url, rotation)]. Raises ValueError on a bad value."""
    extras = []
    for value in values:
        if not 1 <= len(value) <= 2:
            raise ValueError("--extra-source takes a URL and, optionally, a rotation: 0, 90, 180 or 270")
        url = value[0]
        if not is_stream(url):
            raise ValueError(f"--extra-source needs a stream URL (http://...), got {url!r}")
        rotation = value[1] if len(value) == 2 else "0"
        if rotation not in ("0", "90", "180", "270"):
            raise ValueError(f"--extra-source rotation must be 0, 90, 180 or 270, got {rotation!r}")
        extras.append((url, int(rotation)))
    return extras


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        extras = parse_extra_sources(args.extra_source)
    except ValueError as exc:
        parser.error(str(exc))
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("sentra").setLevel(logging.INFO)

    detector = ObjectDetector(args.model, conf=args.conf, imgsz=args.imgsz, track=not args.no_track)
    person_check = PersonCheck(args.pose_model, imgsz=args.imgsz) if args.person_check else None
    obstacles = ObstacleScanner(args.obstacles) if args.obstacles else None
    loading = [args.model]
    loading += [f"{args.pose_model} (person check)"] if person_check else []
    loading += [f"{obstacles.model_path} (floor scan)"] if obstacles else []
    log.info("Loading %s ...", " and ".join(loading))
    try:
        detector.warmup()
        if person_check is not None:
            person_check.warmup()
        if obstacles is not None:
            obstacles.warmup()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from None

    first = parse_source(args.source)
    controls = GuideControls(muted=args.muted, privacy=args.privacy, camera_source=first)
    page = start_web_page(args, controls) if args.web else None  # before the camera, so a busy port fails fast

    streams = [s for s in [args.source, *(url for url, _ in extras)] if is_stream(s)]
    cameras = None if args.no_view else CameraCatalog(streams)
    if cameras is not None:
        cameras.refresh(force=True)  # list the connected cameras in the background while the camera opens
    # Rotation belongs to each camera (a portrait phone needs 90, the laptop camera 0). The pipeline applies it,
    # and r in the window turns the current camera another 90 degrees.
    controls.rotations = {url: rotation for url, rotation in extras}
    controls.rotations[first] = args.rotate
    controls.rotation = args.rotate

    def open_source(spec: int | str) -> FrameSource:
        return FrameSource(spec, open_timeout_s=10.0)

    try:
        source = FrameSource(first)
    except RuntimeError as exc:
        if page is not None:
            page.close()
        raise SystemExit(str(exc)) from None
    controls.camera_label = cameras.label_for(first) if cameras else source.description
    rotation = f", rotated {args.rotate} degrees" if args.rotate else ""
    log.info("Camera: %s%s. Experimental: not a navigation aid.", source.description, rotation)

    policy = build_policy(args)
    spatial = SpatialEstimator()
    speaker = build_speaker(args.voice)
    if args.no_view:
        sink = HeadlessSink(args.max_seconds)
    else:
        sink = GuideView(controls, policy, spatial.config, voice_name=args.voice, cameras=cameras,
                         obstacles=obstacles)
        if args.max_seconds is not None:
            log.warning("--max-seconds only applies with --no-view; press q to quit the window.")
        log.info("Click the video window, then: c cameras, r rotate, m mute, p privacy, q quit.")
    if page is not None:
        sink = TeeSink(sink, page)

    pipeline = GuidePipeline(source, detector, sink, speaker=speaker, spatial=spatial, policy=policy,
                             controls=controls, label=args.label, open_source=open_source,
                             person_check=person_check, obstacles=obstacles)
    try:
        pipeline.run()
    except KeyboardInterrupt:
        log.info("Stopped with Ctrl+C.")
    finally:
        sink.close()
        source.close()
        if pipeline.source is not source:
            pipeline.source.close()  # the camera picked in the dropdown
        if obstacles is not None:
            obstacles.close()
        if speaker is not None:
            speaker.close()
            if speaker.failed:
                log.warning("The Windows voice failed, so alerts were printed instead.")
        log.info("Run summary: %s", pipeline.stats.summary())
        if pipeline.log_path is not None:
            log.info("Alerts saved to %s", pipeline.log_path)


if __name__ == "__main__":
    main()
