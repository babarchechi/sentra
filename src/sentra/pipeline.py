"""Processing pipeline. Track B, owner: babarchechi. See docs/PHASE1_PLAN.md, Section 7.1 step 3."""
from __future__ import annotations

from sentra.capture import FrameSource
from sentra.events import StatusSink
from sentra.posture import PostureAnalyzer

DEFAULT_MODEL = "yolo26n-pose.pt"


class Pipeline:
    """Camera -> YOLO pose + ByteTrack -> PoseFrame -> PostureAnalyzer -> StatusUpdate -> StatusSink."""

    def __init__(
        self,
        source: FrameSource,
        analyzer: PostureAnalyzer,
        sink: StatusSink,
        model_path: str = DEFAULT_MODEL,
    ) -> None:
        self.source = source
        self.analyzer = analyzer
        self.sink = sink
        self.model_path = model_path

    def run(self) -> None:
        """Process frames until sink.should_stop().

        Writes outputs/transitions_<YYYYmmdd_HHMMSS>.csv with state changes only, never images.
        """
        # TODO(babarchechi): load YOLO(self.model_path); loop read -> track -> PoseFrame -> analyze -> publish.
        raise NotImplementedError("Pipeline.run is not built yet (docs/PHASE1_PLAN.md, Section 7.1)")
