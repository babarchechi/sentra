"""Run the dashboard on fake events, with no camera or model. Track C, owner: Swechha03.

Run from the repo root:  python src/ui_demo.py            (add --privacy to start in skeleton-only mode)

Person #1 falls on a loop and stays down until you answer the alert: click the window, then press c (confirmed) or
f (false alarm). Every third loop they get up on their own first, which is logged as "recovered". Person #2 always
stays upright. Press p to switch privacy mode and q to quit. Alerts are logged to outputs/alerts_<time>.csv.
"""
from __future__ import annotations

import argparse
import logging

from sentra.ui.cv_dashboard import CvDashboard
from sentra.ui.fake_events import fake_updates


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentra dashboard demo on fake events (no camera or model)")
    parser.add_argument("--privacy", action="store_true", help="Start in skeleton-only privacy mode")
    parser.add_argument("--fps", type=float, default=10.0, help="Fake frames per second (default 10)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("sentra").setLevel(logging.INFO)

    sink = CvDashboard(privacy=args.privacy)
    try:
        for update in fake_updates(args.fps, alert_answered=lambda track_id: not sink.has_open_alert(track_id)):
            sink.publish(update)
            if sink.should_stop():
                break
    except KeyboardInterrupt:
        pass
    finally:
        sink.close()


if __name__ == "__main__":
    main()
