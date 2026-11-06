"""Fake StatusUpdate generator. Track C, owner: Swechha03. See docs/PHASE1_PLAN.md, Section 8.1 step 1.

Lets the dashboard be built and tested with no camera, model, or analyzer.
"""
from __future__ import annotations

from collections.abc import Iterator

from sentra.events import StatusUpdate


def fake_updates(fps: float = 10.0) -> Iterator[StatusUpdate]:
    """Yield scripted StatusUpdates forever at roughly `fps`.

    One stick figure stands, falls and lies down on a loop; a second figure always stays upright.
    """
    # TODO(Swechha03): draw stick figures from synthetic keypoints with events.SKELETON and script the states.
    raise NotImplementedError("fake_updates is not built yet (docs/PHASE1_PLAN.md, Section 8.1)")
    yield  # unreachable; makes this a generator so the signature is already correct
