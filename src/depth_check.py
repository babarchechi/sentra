"""Tape-measure test for estimated distance. Owner: babarchechi. See docs/VALIDATION.md, Section 4.

Put one object (a chair by default) in the middle of the view at a distance you measured with a tape, from the
camera lens to the front of the object, then run for a few seconds:

    python src/depth_check.py --source 0 --true 2.0
    python src/depth_check.py --source "http://<phone-ip>:4747/video" --rotate 90 --label chair --true 1.0

It prints the median estimate for the largest matching object, how much it wobbled, and the error against --true.
Nothing is saved. The estimate comes from yolo26n-depth (a guess from the picture, not LiDAR).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from sentra.capture import FrameSource, parse_source
from sentra.guide.depth_yolo import DEFAULT_DEPTH_MODEL, YoloDepth
from sentra.guide.detector import DEFAULT_DETECT_MODEL, ObjectDetector


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare estimated distance with a tape-measured one")
    ap.add_argument("--source", default="0")
    ap.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0)
    ap.add_argument("--label", default="chair", help="Object class to measure (default chair)")
    ap.add_argument("--true", type=float, default=None, help="Tape-measured distance in meters")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--min-score", type=float, default=0.4, help="Ignore weaker detections (default 0.4)")
    ap.add_argument("--scale", type=float, default=1.0, help="Correction factor from an earlier --calibrate run")
    ap.add_argument("--calibrate", action="store_true",
                    help="Print the correction factor that makes this reading match --true (needs --true)")
    ap.add_argument("--model", default=DEFAULT_DETECT_MODEL)
    ap.add_argument("--depth-model", default=DEFAULT_DEPTH_MODEL)
    args = ap.parse_args()

    detector = ObjectDetector(args.model, track=False)
    depth = YoloDepth(args.depth_model, every=1, scale=args.scale)
    detector.warmup()
    depth.warmup()
    source = FrameSource(parse_source(args.source), rotate=args.rotate)
    readings, scores, heights, others = [], [], [], []
    try:
        end = time.perf_counter() + args.seconds
        while time.perf_counter() < end:
            ok, image, ts = source.read(1.0)
            if not ok:
                continue
            frame = detector.detect(image, len(readings) + 1, ts)
            targets = [d for d in frame.detections if d.label == args.label and d.confidence >= args.min_score]
            others.append(sorted({d.label for d in frame.detections if d.label != args.label}))
            if not targets:
                continue
            # the biggest one in the picture: the test object should be the nearest one of its kind
            target = max(targets, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
            meters, _ = depth.box_meters(depth.depth_map(image), target.box, (frame.frame_width, frame.frame_height))
            if meters is not None:
                readings.append(meters)
                scores.append(target.confidence)
                heights.append((target.box[3] - target.box[1]) / frame.frame_height)
    finally:
        source.close()

    seen = sorted({label for labels in others for label in labels})
    if not readings:
        print(f"No '{args.label}' with score >= {args.min_score} in {args.seconds:.0f} s. Is it in view, upright, and not hidden? Also seen: {seen}")
        return
    r = np.array(readings)
    line = (f"{args.label}: {len(r)} readings, median {np.median(r):.2f} m "
            f"(10-90%: {np.percentile(r, 10):.2f}-{np.percentile(r, 90):.2f} m), detector score {np.median(scores):.2f}, {np.median(heights):.0%} of frame height")
    if args.true:
        err = np.median(r) - args.true
        line += f" | true {args.true:.2f} m -> error {err:+.2f} m ({100 * err / args.true:+.0f}%)"
    print(line)
    if args.calibrate and args.true:
        factor = args.scale * args.true / float(np.median(r))
        print(f"Correction factor for this camera: --scale {factor:.3f}  (check it at other distances before trusting it)")
    if seen:
        print(f"Also in view (keep people and other {args.label}s out for a clean test): {seen}")


if __name__ == "__main__":
    main()
