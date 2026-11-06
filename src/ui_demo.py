"""Run the dashboard on fake events, with no camera or model. Track C, owner: Swechha03.

Run from the repo root:  python src/ui_demo.py
"""
from __future__ import annotations

from sentra.ui.cv_dashboard import CvDashboard
from sentra.ui.fake_events import fake_updates


def main() -> None:
    sink = CvDashboard()
    try:
        for update in fake_updates():
            sink.publish(update)
            if sink.should_stop():
                break
    finally:
        sink.close()


if __name__ == "__main__":
    main()
