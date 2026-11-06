"""Tests for posture analysis. Track A, owner: iesa242. See docs/PHASE1_PLAN.md, Section 6.4.

These starter tests should keep passing once the real analyzer is in. Add the scenario tests from 6.4.
"""
import numpy as np

from sentra.events import PersonPose, PoseFrame, PostureState
from sentra.posture import PostureAnalyzer


def make_frame(*people: PersonPose, timestamp: float = 0.0) -> PoseFrame:
    return PoseFrame(timestamp=timestamp, frame_width=640, frame_height=480, people=people)


def test_one_status_per_person_and_invisible_keypoints_are_unknown():
    invisible = PersonPose(
        track_id=7,
        keypoints=np.zeros((17, 3), np.float32),  # every keypoint has confidence 0
        box=(0.0, 0.0, 50.0, 120.0),
        box_confidence=0.9,
    )
    statuses = PostureAnalyzer().update(make_frame(invisible))
    assert [s.track_id for s in statuses] == [7]
    assert statuses[0].state is PostureState.UNKNOWN


def test_empty_frame_returns_no_statuses():
    assert PostureAnalyzer().update(make_frame()) == ()
