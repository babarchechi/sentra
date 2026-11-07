"""Posture analysis. Track A, owner: iesa242. See docs/PHASE1_PLAN.md, Section 6.

Turns each PoseFrame into one PersonStatus per person. POSSIBLE_FALL is raised only after a fast hip drop
followed by staying on the ground (or, with lower confidence, a fast drop followed by losing sight of the person,
or lying still for a long time with no drop seen).

How it works, per track_id:
1. Features (6.1): torso angle from vertical, box aspect, torso length and hip drop speed in torso-lengths per
   second, using only keypoints with confidence >= min_keypoint_conf. Time always comes from pose.timestamp.
2. A debounced posture: "upright", "lying" or "unknown". A new posture must be seen for state_hold_frames frames
   in a row before it replaces the old one (angles between the upright and lying thresholds keep the old posture).
3. Fall logic on top of the debounced posture (6.2): a fast drop opens a window; lying that starts inside the window
   starts the on-ground timer; staying down for on_ground_alert_s raises POSSIBLE_FALL. Only being upright for
   recovery_s clears it. Lying down slowly gives ON_GROUND, never an alert (unless the long-stillness stretch rule).

Known limits: works best with the camera at roughly chest height, side-on. A camera looking straight down makes the
torso angle unreliable. A person who leaves the frame entirely after a drop can't be reported yet: the events.py
contract only allows statuses for people in pose.people (plan 6.2, uncertain item 2).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from sentra.events import KP, PersonPose, PersonStatus, PoseFrame, PostureState

UPRIGHT, LYING, HIDDEN = "upright", "lying", "unknown"
TORSO_POINTS = (KP["left_shoulder"], KP["right_shoulder"], KP["left_hip"], KP["right_hip"])


@dataclass
class PostureConfig:
    """State-machine thresholds. These are starting guesses to tune (plan Section 6.3)."""

    min_keypoint_conf: float = 0.5        # ignore keypoints below this confidence
    min_torso_px: float = 10.0            # skeletons with a shorter torso are too small to judge -> UNKNOWN
    upright_max_angle_deg: float = 35.0   # torso within this of vertical counts as upright
    lying_min_angle_deg: float = 60.0     # torso beyond this counts as lying
    lying_min_aspect: float = 1.2         # box width/height above this supports "lying" for in-between angles
    descent_speed: float = 1.5            # torso-lengths per second of hip drop that counts as "fast"
    descent_window_s: float = 2.0         # lying must start within this long after the fast drop
    on_ground_alert_s: float = 5.0        # seconds on the ground before POSSIBLE_FALL (short for the demo)
    recovery_s: float = 2.0               # upright this long clears POSSIBLE_FALL
    state_hold_frames: int = 3            # frames a new posture must persist before switching
    track_timeout_s: float = 3.0          # forget a track not seen for this long
    lost_memory_s: float = 10.0           # keep a track that vanished after a fast drop this long, to re-match it
    lost_match_iou: float = 0.3           # box overlap needed to treat a new track ID as that lost person
    lost_after_drop_alert_s: float = 5.0  # hidden (UNKNOWN) this long after a fast drop -> low-confidence alert
    uncertain_confidence_max: float = 0.5  # highest confidence for alerts raised from uncertain evidence
    still_motion_max: float = 0.05        # keypoint movement (torso-lengths per second) that still counts as "still"
    still_long_alert_s: float = 20.0      # stretch: lying still this long with no drop seen -> low-confidence alert
    speed_window_s: float = 0.4           # hip drop speed is measured over roughly this much recent time
    still_window_s: float = 1.5           # stillness is measured over roughly this much recent time


@dataclass
class _Features:
    angle: float | None       # torso degrees from vertical, None if hips or shoulders are hidden
    hip_y: float | None       # mid-hip y in px
    torso_len: float | None   # mid-shoulder <-> mid-hip in px
    aspect: float             # box width / height
    conf: float               # mean confidence of the visible torso keypoints
    xy: np.ndarray            # (17, 2) keypoint positions
    visible: np.ndarray       # (17,) bool mask


@dataclass
class _Track:
    posture: str | None = None
    posture_since: float = 0.0
    pending: str | None = None
    pending_since: float = 0.0
    pending_count: int = 0
    hips: deque = field(default_factory=lambda: deque(maxlen=64))          # (t, mid-hip y)
    torso_samples: deque = field(default_factory=lambda: deque(maxlen=30))  # torso lengths seen while upright
    motion: deque = field(default_factory=lambda: deque(maxlen=64))        # (t, xy, visible) while lying
    drop_at: float | None = None          # time of the most recent fast drop not yet resolved
    drop_speed: float = 0.0
    fall_lying_since: float | None = None  # lying started inside the window after a fast drop
    still_since: float | None = None
    alert_since: float | None = None
    alert_kind: str = ""                  # "fall", "hidden" or "still"
    state: PostureState = PostureState.UNKNOWN
    since: float = 0.0
    last_seen: float = 0.0
    last_box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def in_fall_context(self) -> bool:
        """True while a fast drop or an alert is unresolved, so this person is worth re-matching if their ID changes."""
        return self.alert_since is not None or self.drop_at is not None


class PostureAnalyzer:
    """Keeps a short history per track_id and returns one PersonStatus per person each frame."""

    def __init__(self, config: PostureConfig | None = None) -> None:
        self.config = config or PostureConfig()
        self._tracks: dict[int, _Track] = {}
        self._lost: list[_Track] = []  # tracks that vanished soon after a fast drop

    @property
    def tracked_ids(self) -> tuple[int, ...]:
        """IDs currently remembered (for tests and debugging)."""
        return tuple(self._tracks)

    def update(self, pose: PoseFrame) -> tuple[PersonStatus, ...]:
        """Call once per frame. Returns one PersonStatus per person in pose.people, in the same order."""
        t = pose.timestamp
        present = {person.track_id for person in pose.people}
        statuses = []
        for person in pose.people:
            track = self._tracks.get(person.track_id)
            if track is None:
                track = self._adopt_lost(person.box, t, present) or _Track(since=t)
                self._tracks[person.track_id] = track
            statuses.append(self._step(person, track, t))
        self._forget(t, present)
        return tuple(statuses)

    # Tracks ---------------------------------------------------------------------------------------

    def _adopt_lost(self, box, t: float, present: set[int]) -> _Track | None:
        """A new track_id whose box overlaps someone who vanished after a fast drop inherits their history (6.2, 1)."""
        cfg = self.config
        self._lost = [lost for lost in self._lost if t - lost.last_seen <= cfg.lost_memory_s]
        candidates = [(None, lost) for lost in self._lost]
        candidates += [
            (track_id, track) for track_id, track in self._tracks.items()
            if track_id not in present and track.in_fall_context()
        ]
        best, best_iou = None, cfg.lost_match_iou
        for track_id, track in candidates:
            overlap = _iou(box, track.last_box)
            if overlap >= best_iou:
                best, best_iou = (track_id, track), overlap
        if best is None:
            return None
        track_id, track = best
        if track_id is None:
            self._lost.remove(track)
        else:
            del self._tracks[track_id]
        return track

    def _forget(self, t: float, present: set[int]) -> None:
        cfg = self.config
        for track_id in [i for i in self._tracks if i not in present]:
            track = self._tracks[track_id]
            if t - track.last_seen > cfg.track_timeout_s:
                del self._tracks[track_id]
                if track.in_fall_context():
                    self._lost.append(track)
        self._lost = [lost for lost in self._lost if t - lost.last_seen <= cfg.lost_memory_s]

    # One person, one frame ---------------------------------------------------------------------

    def _step(self, person: PersonPose, track: _Track, t: float) -> PersonStatus:
        cfg = self.config
        feats = self._features(person)
        track.last_seen, track.last_box = t, person.box

        raw = self._raw_posture(feats)
        if raw == UPRIGHT and feats.torso_len:
            track.torso_samples.append(feats.torso_len)
        scale = self._torso_scale(track, feats)
        if feats.hip_y is not None:
            track.hips.append((t, feats.hip_y))
        self._debounce(track, raw, t)
        posture = track.posture

        speed = self._hip_speed(track, t, scale)
        if (speed is not None and speed >= cfg.descent_speed and posture != LYING
                and track.alert_since is None and track.fall_lying_since is None):
            if track.drop_at is None or t - track.drop_at > cfg.descent_window_s:
                track.drop_at, track.drop_speed = t, speed
            track.drop_speed = max(track.drop_speed, speed)

        motion = self._motion(track, feats, t, scale) if posture == LYING else None
        if posture != LYING:
            track.motion.clear()
            track.still_since = None
        elif motion is not None:
            if motion < cfg.still_motion_max:
                if track.still_since is None:
                    track.still_since = t
            else:
                track.still_since = None

        drop_recent = track.drop_at is not None and t - track.drop_at <= cfg.descent_window_s

        if posture == UPRIGHT:
            upright_for = t - track.posture_since
            if track.alert_since is not None and upright_for >= cfg.recovery_s:
                self._clear_fall(track)
            elif track.alert_since is None:
                track.fall_lying_since = None  # got back up before the alert
                if not drop_recent:
                    track.drop_at = None
        elif posture == LYING:
            if (track.fall_lying_since is None and track.drop_at is not None
                    and track.posture_since - track.drop_at <= cfg.descent_window_s):
                track.fall_lying_since = track.posture_since

        if track.alert_since is None:
            self._maybe_alert(track, posture, t)

        state, confidence, reason = self._describe(track, posture, feats, t, speed, drop_recent)
        if state is not track.state:
            track.state, track.since = state, t
        return PersonStatus(
            track_id=person.track_id,
            state=state,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            reason=reason,
            since=track.since,
        )

    def _maybe_alert(self, track: _Track, posture: str | None, t: float) -> None:
        cfg = self.config
        if posture == LYING and track.fall_lying_since is not None:
            if t - track.fall_lying_since >= cfg.on_ground_alert_s:
                track.alert_since, track.alert_kind = t, "fall"
        elif posture == HIDDEN and track.drop_at is not None:
            hidden_after_drop = (track.posture_since - track.drop_at <= cfg.descent_window_s
                                 or track.fall_lying_since is not None)
            if hidden_after_drop and t - track.posture_since >= cfg.lost_after_drop_alert_s:
                track.alert_since, track.alert_kind = t, "hidden"
        elif posture == LYING and track.still_since is not None:
            if t - track.still_since >= cfg.still_long_alert_s:
                track.alert_since, track.alert_kind = t, "still"

    @staticmethod
    def _clear_fall(track: _Track) -> None:
        track.alert_since, track.alert_kind = None, ""
        track.drop_at, track.drop_speed = None, 0.0
        track.fall_lying_since = None
        track.still_since = None

    def _describe(self, track: _Track, posture, feats: _Features, t: float, speed, drop_recent: bool):
        """Final (state, confidence, reason) for this frame."""
        cfg = self.config
        angle = feats.angle
        still = track.still_since is not None
        still_text = f", still for {t - track.still_since:.0f} s" if still else ""

        if track.alert_since is not None:
            if track.alert_kind == "hidden":
                conf = cfg.uncertain_confidence_max * 0.8
                hidden_for = t - track.posture_since if posture == HIDDEN else 0.0
                reason = (f"fast drop, then hips/shoulders hidden for {hidden_for:.0f} s" if posture == HIDDEN
                          else "fast drop, then lost from view")
            elif track.alert_kind == "still":
                conf = min(cfg.uncertain_confidence_max, feats.conf * 0.6)
                reason = f"lying still for {t - (track.still_since or track.alert_since):.0f} s, no fall seen"
            else:
                conf = feats.conf * (0.5 + 0.3 * _lying_margin(angle, cfg) + 0.2 * still)
                down_for = t - track.fall_lying_since if track.fall_lying_since is not None else 0.0
                reason = (f"torso {angle:.0f}° for {down_for:.1f} s after fast drop{still_text}"
                          if angle is not None else "on the ground after fast drop")
            if posture == UPRIGHT:
                reason = f"possible fall; upright again for {t - track.posture_since:.1f} s"
            return PostureState.POSSIBLE_FALL, conf, reason

        if posture == HIDDEN or posture is None:
            if posture == HIDDEN and track.drop_at is not None:
                return PostureState.UNKNOWN, 0.0, f"fast drop, then hidden for {t - track.posture_since:.1f} s"
            return PostureState.UNKNOWN, 0.0, "hips or shoulders not visible"

        if posture == LYING:
            margin = _lying_margin(angle, cfg)
            if track.fall_lying_since is not None:
                down_for = t - track.fall_lying_since
                return (PostureState.DESCENDING, feats.conf * (0.6 + 0.4 * margin),
                        f"on the ground {down_for:.1f} s after fast drop")
            angle_text = f"torso {angle:.0f}°" if angle is not None else "lying"
            return (PostureState.ON_GROUND, feats.conf * (0.6 + 0.4 * margin),
                    f"{angle_text}, no fast drop seen{still_text}")

        if drop_recent:
            return (PostureState.DESCENDING, feats.conf * min(1.0, track.drop_speed / (2 * cfg.descent_speed) + 0.5),
                    f"hips dropping {track.drop_speed:.1f} torso-lengths/s")

        clarity = 1.0 if angle is None else float(np.clip(1.0 - angle / (2 * cfg.lying_min_angle_deg), 0.0, 1.0))
        angle_text = f"torso {angle:.0f}° from vertical" if angle is not None else "upright"
        return PostureState.UPRIGHT, feats.conf * (0.5 + 0.5 * clarity), angle_text

    # Features ------------------------------------------------------------------------------------

    def _features(self, person: PersonPose) -> _Features:
        kps = np.asarray(person.keypoints, dtype=np.float32).reshape(-1, 3)
        xy, confs = kps[:, :2], kps[:, 2]
        visible = confs >= self.config.min_keypoint_conf
        shoulder = _midpoint(xy, visible, KP["left_shoulder"], KP["right_shoulder"])
        hip = _midpoint(xy, visible, KP["left_hip"], KP["right_hip"])
        x1, y1, x2, y2 = person.box
        aspect = (x2 - x1) / max(y2 - y1, 1e-6)
        torso_visible = [confs[i] for i in TORSO_POINTS if visible[i]]
        conf = float(np.mean(torso_visible)) if torso_visible else 0.0

        angle = torso_len = None
        if shoulder is not None and hip is not None:
            dx, dy = shoulder[0] - hip[0], shoulder[1] - hip[1]
            torso_len = math.hypot(dx, dy)
            if torso_len >= self.config.min_torso_px:
                angle = math.degrees(math.atan2(abs(dx), -dy))  # image y grows downward
            else:  # far away or a bad detection: too small for angles or speeds to mean anything
                torso_len = hip = None
        return _Features(angle, hip[1] if hip is not None else None, torso_len, aspect, conf, xy, visible)

    def _raw_posture(self, feats: _Features) -> str | None:
        """Posture seen in this frame alone; None for in-between angles (keep the previous posture)."""
        cfg = self.config
        if feats.angle is None:
            return HIDDEN
        if feats.angle <= cfg.upright_max_angle_deg:
            return UPRIGHT
        if feats.angle >= cfg.lying_min_angle_deg or feats.aspect >= cfg.lying_min_aspect:
            return LYING
        return None

    def _debounce(self, track: _Track, raw: str | None, t: float) -> None:
        if raw is None:
            return
        if track.posture is None:  # first sighting: nothing to flicker from
            track.posture, track.posture_since = raw, t
            return
        if raw == track.posture:
            track.pending, track.pending_count = None, 0
            return
        if raw != track.pending:
            track.pending, track.pending_since, track.pending_count = raw, t, 0
        track.pending_count += 1
        if track.pending_count >= self.config.state_hold_frames:
            track.posture, track.posture_since = raw, track.pending_since
            track.pending, track.pending_count = None, 0

    @staticmethod
    def _torso_scale(track: _Track, feats: _Features) -> float:
        """Torso length to normalise speeds: the median while upright, else the current one."""
        if track.torso_samples:
            return max(float(np.median(track.torso_samples)), 1.0)
        if feats.torso_len:
            return max(feats.torso_len, 1.0)
        return 100.0

    def _hip_speed(self, track: _Track, t: float, scale: float) -> float | None:
        """Mid-hip drop speed in torso-lengths per second over ~speed_window_s (positive = moving down)."""
        while track.hips and t - track.hips[0][0] > 2.0:
            track.hips.popleft()
        if len(track.hips) < 2 or track.hips[-1][0] != t:
            return None
        target = t - self.config.speed_window_s
        ref = track.hips[0]
        for sample in track.hips:
            if sample[0] <= target:
                ref = sample
            else:
                break
        dt = t - ref[0]
        if dt < 0.15:
            return None
        return (track.hips[-1][1] - ref[1]) / scale / dt

    def _motion(self, track: _Track, feats: _Features, t: float, scale: float) -> float | None:
        """How much visible keypoints moved over ~still_window_s, in torso-lengths per second."""
        window = self.config.still_window_s
        track.motion.append((t, feats.xy.copy(), feats.visible.copy()))
        while track.motion and t - track.motion[0][0] > window:
            track.motion.popleft()
        if len(track.motion) < 2 or t - track.motion[0][0] < 0.75 * window:
            return None
        _, old_xy, old_vis = track.motion[0]
        both = old_vis & feats.visible
        if not both.any():
            return None
        shift = np.linalg.norm(feats.xy[both] - old_xy[both], axis=1).mean()
        return float(shift) / scale / (t - track.motion[0][0])


def _midpoint(xy: np.ndarray, visible: np.ndarray, left: int, right: int) -> np.ndarray | None:
    """Average of a left/right keypoint pair, falling back to whichever side is visible."""
    if visible[left] and visible[right]:
        return (xy[left] + xy[right]) / 2
    if visible[left]:
        return xy[left]
    if visible[right]:
        return xy[right]
    return None


def _lying_margin(angle: float | None, cfg: PostureConfig) -> float:
    """0 at the lying threshold, 1 at fully horizontal."""
    if angle is None:
        return 0.0
    return float(np.clip((angle - cfg.lying_min_angle_deg) / max(90.0 - cfg.lying_min_angle_deg, 1e-6), 0.0, 1.0))


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
