"""Sentra app entry point. Track B, owner: babarchechi. See docs/PHASE1_PLAN.md, Section 7.1 step 4.

Run from the repo root:  python src/main.py --source 0
"""
from __future__ import annotations

import argparse

from sentra.capture import FrameSource, parse_source
from sentra.pipeline import DEFAULT_MODEL, Pipeline
from sentra.posture import PostureAnalyzer
from sentra.ui.cv_dashboard import CvDashboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sentra prototype (not a medical device)")
    parser.add_argument("--source", default="0", help="Webcam index (0) or approved camera URL")
    parser.add_argument("--ui", choices=("cv", "web"), default="cv", help="Dashboard type")
    parser.add_argument("--privacy", action="store_true", help="Start in skeleton-only privacy mode")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Pose model weights file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.ui == "web":
        raise SystemExit("The web dashboard is a Level 2 stretch goal and isn't built yet. Use --ui cv.")

    source = FrameSource(parse_source(args.source))
    try:
        sink = CvDashboard(privacy=args.privacy)
        try:
            Pipeline(source, PostureAnalyzer(), sink, model_path=args.model).run()
        finally:
            sink.close()
    finally:
        source.close()


if __name__ == "__main__":
    main()
