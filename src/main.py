"""Sentra app entry point. Track B, owner: babarchechi. See docs/PHASE1_PLAN.md, Section 7.1 step 4.

Run from the repo root:  python src/main.py --source 0
"""
from __future__ import annotations

import argparse
import logging

from sentra.capture import FrameSource, parse_source
from sentra.pipeline import DEFAULT_MODEL, Pipeline
from sentra.posture import PostureAnalyzer
from sentra.ui.cv_dashboard import CvDashboard

log = logging.getLogger("sentra")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sentra prototype (not a medical device)")
    parser.add_argument(
        "--source", default="0",
        help="Webcam index (0) or approved camera URL. Windows only, opt-in: 'phone' (Phone Link connected "
             "camera) or 'windows:<part of camera name>' (e.g. windows:DroidCam)",
    )
    parser.add_argument(
        "--rotate", type=int, choices=(0, 90, 180, 270), default=0,
        help="Rotate frames clockwise by this many degrees so people appear upright "
             "(try 90 or 270 for a phone held in portrait)",
    )
    parser.add_argument("--ui", choices=("cv", "web"), default="cv", help="Dashboard type")
    parser.add_argument("--privacy", action="store_true", help="Start in skeleton-only privacy mode")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Pose model weights file")
    return parser


def configure_logging() -> None:
    # Our own messages at INFO; other libraries only when something is wrong.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("sentra").setLevel(logging.INFO)


def main() -> None:
    args = build_parser().parse_args()
    configure_logging()
    if args.ui == "web":
        raise SystemExit("The web dashboard is a Level 2 stretch goal and isn't built yet. Use --ui cv.")

    try:
        source = FrameSource(parse_source(args.source), rotate=args.rotate)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    rotation = f", rotated {args.rotate} degrees clockwise" if args.rotate else ""
    log.info("Camera: %s%s. Click the video window and press q to quit.", source.description, rotation)

    try:
        sink = CvDashboard(privacy=args.privacy)
        try:
            Pipeline(source, PostureAnalyzer(), sink, model_path=args.model).run()
        except KeyboardInterrupt:
            log.info("Stopped with Ctrl+C.")
        finally:
            sink.close()
    finally:
        source.close()


if __name__ == "__main__":
    main()
