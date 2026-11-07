"""Decide what is worth saying, and when. The point is restraint: few, short, relevant cues.

Rules (all tunable in AlertConfig):
- Only object classes in `labels` are announced (a curated obstacle list by default).
- Tiny boxes (far away or small things) are ignored.
- A label + direction must be seen in `persist_frames` of the last `persist_window` frames before it is said,
  so one-frame flickers are never spoken.
- The same label + direction is not repeated for `repeat_after_s`, and nothing is said within `min_gap_s` of
  the previous announcement. At most one announcement per frame: the most important one.
- Importance: severity (central and large first), then confident over uncertain, then size, then score.
- A low-confidence detection that is large and central becomes "Possible obstacle ahead." The label is not
  spoken because the model isn't sure of it.
- Frames that were already older than `max_frame_age_s` when processed never trigger speech.
- A distance is spoken only if it comes from a validated depth source (DepthEstimate.speakable).
- The words "clear" and "safe" are never produced: missing detections do not mean a free path.

Caution cues (danger warnings), on unless `caution=False`:
- Something in the way is a group with a HIGH-severity member: in the central band and at least
  SpatialConfig.high_view_height of the frame tall, so big or near (box size is not a distance).
- It must be HIGH in `persist_frames` of the last `persist_window` frames too. Then it is said as
  "Caution, chair ahead." (or "Caution, possible obstacle ahead."), ahead of any other cue.
- A caution may follow any earlier cue after `caution_min_gap_s` (shorter than `min_gap_s`) and repeats every
  `caution_repeat_after_s` while the thing stays in the way. Something already announced normally that comes close
  gets its caution straight away.
- A caution has GuideAlert.caution=True: the laptop voice says it before anything waiting, the phone page vibrates
  and interrupts other speech, and the debug view shows it in red. There is no "safe" counterpart: no caution never
  means nothing is in the way.
"""
from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from sentra.guide.person_check import UNCONFIRMED_PERSON
from sentra.guide.schema import (
    SEVERITY_RANK,
    DetectionFrame,
    GuideAlert,
    Region,
    Severity,
    SpatialObject,
)

# Labels that are never spoken by name, not even with --all-labels. When large and central they can still be a
# "Possible obstacle ahead.". UNCONFIRMED_PERSON: a "person" box the pose model found no body in (a hand, an arm,
# or a real person seen too close), see sentra.guide.person_check.
OBSTACLE_ONLY_LABELS: frozenset[str] = frozenset({UNCONFIRMED_PERSON})

# COCO classes worth mentioning for someone moving through a room or along a path. Small tabletop things
# (cups, phones, keyboards, ...) are left out on purpose: they would make the guide chatty.
DEFAULT_ANNOUNCE_LABELS: frozenset[str] = frozenset({
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "fire hydrant", "stop sign", "parking meter",
    "bench", "dog", "cat", "backpack", "umbrella", "suitcase", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "refrigerator",
})

SPOKEN_NAMES = {
    "dining table": "table",
    "potted plant": "plant",
    "tv": "TV",
    "motorcycle": "motorbike",
    "cell phone": "phone",
    "sports ball": "ball",
}
_IRREGULAR_PLURALS = {"person": "people", "bus": "buses", "couch": "couches", "bench": "benches"}

PLACE_WORDS = {Region.CENTER: "ahead", Region.LEFT: "to your left", Region.RIGHT: "to your right"}
OBSTACLE = "obstacle"

GroupKey = tuple[str, Region]  # (spoken name, region)


@dataclass(frozen=True)
class AlertConfig:
    announce_confidence: float = 0.5        # detector score needed to say the object's name
    uncertain_confidence: float = 0.3       # between this and announce_confidence: maybe "possible obstacle"
    uncertain_min_view_height: float = 0.35  # ...and only if central and at least this tall
    min_view_height: float = 0.1            # ignore boxes shorter than this share of the frame height
    persist_frames: int = 3
    persist_window: int = 5
    repeat_after_s: float = 10.0
    min_gap_s: float = 2.5
    max_frame_age_s: float = 1.0
    labels: frozenset[str] | None = field(default=DEFAULT_ANNOUNCE_LABELS)  # None = every class
    caution: bool = True                    # warn "Caution, ... ahead." about big things in the way
    caution_repeat_after_s: float = 5.0     # repeat a caution this often while the thing stays in the way
    caution_min_gap_s: float = 1.0          # a caution may follow any earlier cue after this long

    def __post_init__(self) -> None:
        if not 1 <= self.persist_frames <= self.persist_window:
            raise ValueError("Need 1 <= persist_frames <= persist_window")
        if self.uncertain_confidence > self.announce_confidence:
            raise ValueError("uncertain_confidence must not exceed announce_confidence")


def spoken_name(label: str) -> str:
    return SPOKEN_NAMES.get(label, label)


def plural(name: str) -> str:
    if name in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[name]
    if name.endswith(("s", "sh", "ch", "x")):
        return name + "es"
    return name + "s"


def alert_text(name: str, count: int, region: Region, meters: float | None = None, *, caution: bool = False) -> str:
    """E.g. "Chair ahead.", "2 people to your right.", "Chair ahead, approximately 1.4 meters.",
    and with `caution`: "Caution, chair ahead.", "Caution, possible obstacle ahead."."""
    if name == OBSTACLE:
        subject = "possible obstacle"
    else:
        subject = f"{count} {plural(name)}" if count > 1 else name
    text = f"{subject} {PLACE_WORDS[region]}"
    if meters is not None:
        text += f", approximately {meters:.1f} meters"
    if caution:
        return f"Caution, {text}."
    return text[0].upper() + text[1:] + "."


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class AlertPolicy:
    def __init__(self, config: AlertConfig | None = None, clock=time.perf_counter) -> None:
        """`clock` gives "now" in perf_counter seconds; tests pass a fake to control frame age."""
        self.config = config or AlertConfig()
        self._clock = clock
        self._history: deque[set[GroupKey]] = deque(maxlen=self.config.persist_window)
        self._in_the_way: deque[set[GroupKey]] = deque(maxlen=self.config.persist_window)  # HIGH groups per frame
        self._last_said: dict[GroupKey, float] = {}
        self._last_caution: dict[GroupKey, float] = {}
        self._last_any = float("-inf")

    def is_confident(self, obj: SpatialObject) -> bool:
        cfg = self.config
        return (
            obj.detection.confidence >= cfg.announce_confidence
            and obj.view_height >= cfg.min_view_height
            and obj.detection.label not in OBSTACLE_ONLY_LABELS
            and (cfg.labels is None or obj.detection.label in cfg.labels)
        )

    def group(self, objects: Iterable[SpatialObject]) -> dict[GroupKey, list[SpatialObject]]:
        """Group announceable objects by (spoken name, region). Uncertain ones go under ("obstacle", CENTER)."""
        cfg = self.config
        objects = list(objects)
        confident = [o for o in objects if self.is_confident(o)]
        confident_ids = {id(o) for o in confident}
        groups: dict[GroupKey, list[SpatialObject]] = {}
        for obj in confident:
            groups.setdefault((spoken_name(obj.detection.label), obj.region), []).append(obj)
        for obj in objects:
            if (
                id(obj) not in confident_ids
                and obj.region is Region.CENTER
                and cfg.uncertain_confidence <= obj.detection.confidence
                and (obj.detection.confidence < cfg.announce_confidence
                     or obj.detection.label in OBSTACLE_ONLY_LABELS)
                and obj.view_height >= cfg.uncertain_min_view_height
                # the same thing detected twice (e.g. chair 0.8 and couch 0.4) is not a second obstacle
                and not any(_iou(obj.detection.box, c.detection.box) > 0.3 for c in confident)
            ):
                groups.setdefault((OBSTACLE, Region.CENTER), []).append(obj)
        return groups

    def update(self, objects: Iterable[SpatialObject], frame: DetectionFrame) -> tuple[GuideAlert, ...]:
        """Feed one frame's objects. Returns at most one alert to speak now."""
        cfg = self.config
        now_frame = frame.timestamp
        groups = self.group(objects)
        in_the_way = {key for key, members in groups.items()
                      if cfg.caution and any(o.severity is Severity.HIGH for o in members)}
        self._history.append(set(groups))
        self._in_the_way.append(in_the_way)

        age_s = self._clock() - frame.timestamp
        if age_s > cfg.max_frame_age_s:
            return ()
        since_last = now_frame - self._last_any

        # 1. Cautions first: something big in the middle of the view, for several frames.
        cautions = [
            key for key in in_the_way
            if self._in_the_way_count(key) >= cfg.persist_frames
            and now_frame - self._last_caution.get(key, float("-inf")) >= cfg.caution_repeat_after_s
        ]
        if cautions and since_last >= min(cfg.caution_min_gap_s, cfg.min_gap_s):
            best = max(cautions, key=lambda key: self._priority(key, groups[key]))
            alert = self._make_alert(best, groups[best], frame, age_s, caution=True)
            self._last_caution[best] = self._last_said[best] = self._last_any = now_frame
            return (alert,)

        # 2. Ordinary cues. Things in the way are left to the caution above (no "Chair ahead." just before
        #    "Caution, chair ahead.", and no plain repeat while its caution is cooling down).
        if since_last < cfg.min_gap_s:
            return ()
        candidates = [
            key for key in groups
            if key not in in_the_way
            and self._seen_count(key) >= cfg.persist_frames
            and now_frame - self._last_said.get(key, float("-inf")) >= cfg.repeat_after_s
        ]
        if not candidates:
            return ()
        best = max(candidates, key=lambda key: self._priority(key, groups[key]))
        alert = self._make_alert(best, groups[best], frame, age_s)
        self._last_said[best] = now_frame
        self._last_any = now_frame
        return (alert,)

    def reset(self) -> None:
        """Forget recent frames and cooldowns, e.g. after switching cameras: the new view starts fresh."""
        self._history.clear()
        self._in_the_way.clear()
        self._last_said.clear()
        self._last_caution.clear()
        self._last_any = float("-inf")

    def _seen_count(self, key: GroupKey) -> int:
        return sum(key in keys for keys in self._history)

    def _in_the_way_count(self, key: GroupKey) -> int:
        return sum(key in keys for keys in self._in_the_way)

    @staticmethod
    def _priority(key: GroupKey, members: list[SpatialObject]):
        severity = max(SEVERITY_RANK[o.severity] for o in members)
        return (
            severity,
            key[0] != OBSTACLE,
            max(o.view_height for o in members),
            max(o.detection.confidence for o in members),
        )

    def _make_alert(
        self, key: GroupKey, members: list[SpatialObject], frame: DetectionFrame, age_s: float, *,
        caution: bool = False,
    ) -> GuideAlert:
        name, region = key
        count = 1 if name == OBSTACLE else len(members)
        severity = max((o.severity for o in members), key=SEVERITY_RANK.__getitem__)
        confidence = max(o.detection.confidence for o in members)
        speakable = [o.depth for o in members if o.depth.speakable]
        nearest = min(speakable, key=lambda d: d.meters) if speakable else None
        meters = round(nearest.meters, 1) if nearest is not None else None
        track_id = members[0].detection.track_id if len(members) == 1 else None
        what = name
        if name == OBSTACLE:
            unconfirmed = any(o.detection.label in OBSTACLE_ONLY_LABELS for o in members)
            what = "person with no body seen (hand, arm or too close)" if unconfirmed else "uncertain detection"
        if caution:
            reason = (
                f"in the way: {what} (score {confidence:.2f}, {severity.value}) {region.value}, big in view "
                f"in {self._in_the_way_count(key)}/{len(self._in_the_way)} recent frames; "
                f"no caution for it in the last {self.config.caution_repeat_after_s:.0f} s"
            )
        else:
            reason = (
                f"{what} (score {confidence:.2f}, {severity.value}) {region.value} "
                f"in {self._seen_count(key)}/{len(self._history)} recent frames; "
                f"not said in the last {self.config.repeat_after_s:.0f} s"
            )
        if any(o.depth.meters is not None and not o.depth.validated for o in members):
            reason += "; unvalidated distance not spoken"
        return GuideAlert(
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
            label=name,
            count=count,
            region=region,
            severity=severity,
            confidence=round(confidence, 3),
            text=alert_text(name, count, region, meters, caution=caution),
            reason=reason,
            track_id=track_id,
            meters=meters,
            depth_source=nearest.source if nearest else "none",
            depth_confidence=round(nearest.confidence, 2) if nearest else 0.0,
            latency_ms=round(age_s * 1000.0, 1),
            source=frame.source,
            caution=caution,
        )
