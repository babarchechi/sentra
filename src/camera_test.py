import argparse
import cv2
from ultralytics import YOLO


def parse_source(raw: str):
    return int(raw) if raw.isdecimal() else raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentra basic YOLO pose camera test")
    parser.add_argument("--source", default="0", help="Webcam index (0) or approved camera URL")
    parser.add_argument("--track", action="store_true", help="Enable ByteTrack person tracking")
    args = parser.parse_args()

    source = parse_source(args.source)
    model = YOLO("yolo26n-pose.pt")
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit("Cannot open camera source. Check permissions, source, and network.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Unable to read a frame; ending camera test.")
                break
            if args.track:
                results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
            else:
                results = model.predict(frame, verbose=False)
            annotated = results[0].plot()
            cv2.imshow("Sentra - Pose Test (press Q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
