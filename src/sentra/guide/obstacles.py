"""Unknown-obstacle layer (experimental, off by default): warn about things YOLO can't name, from where the floor stops.

YOLO26n names 80 everyday classes, so poles, bollards, boxes, bins and unusual furniture get no label and no cue.
This layer asks a different question: "how far does the walkable floor reach in front of me?" A semantic
segmentation model (indoor: ADE20K, 150 classes; outdoor: Cityscapes) labels every pixel; in the central band (the
walking-corridor stand-in) each column is scanned from the bottom of the image up until a run of non-floor pixels.
That height is the column's "free space". See docs/OBSTACLE_DETECTION.md for the feasibility check behind it.

Decisions, per scan:
- Wall-labelled pixels fill the central view: confirm across scans and keep cautioning even without visible floor.
- Nearly featureless image without floor evidence: repeated "Caution, view unclear." (not a wall claim).
- Otherwise little floor visible in the lower half of the corridor: "can't judge", on screen only.
- Enough corridor columns where the floor stops close (free space < `near_free` of the image height) and no confident
  YOLO box explains it: a possible obstruction. After 2 of the last 3 scans agree, it is added to the frame as an
  "unknown obstacle" detection, which the alert policy only ever says as "Possible obstacle ahead." (a caution when
  it is very close). It is never named, even when the segmentation class is "pole" or "box".
- Otherwise "no near obstruction found": on screen only, with "not proof the way is clear".

Honesty notes: segmentation errors (a rug edge, a shadow, a glossy floor) cause false warnings; glass, drop-offs and
low-contrast obstacles are missed. Open-looking floor is not proof that it is walkable. Nothing here is validated
against a tape measure, so no distance is ever spoken. The per-third "open-looking" numbers are for the developer
view only and must never become "move left" / "move right" commands.

Runs on a background thread on a downscaled frame (~30-70 ms per scan on our laptops), at most every
`min_interval_s` (~4 a second), so YOLO, tracking and speech keep most of their speed; the newest scan is used while
it is fresh (`max_age_s`).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from statistics import median
from typing import Any

import numpy as np

from sentra.guide.schema import Box, Detection, DetectionFrame, Region

log = logging.getLogger(__name__)

UNKNOWN_OBSTACLE = "unknown obstacle"  # label of the detections this layer adds (never spoken by name)
WALL_OBSTACLE = "wall obstruction"  # internal provenance; spoken as "possible obstacle", not a distance claim
VIEW_UNCLEAR = "view unclear"

# environment -> (weights, walkable class names). Stairs and steps are deliberately NOT walkable here.
ENVIRONMENTS: dict[str, tuple[str, frozenset[str]]] = {
    "indoor": ("yolo26n-sem-ade20k.pt", frozenset({
        "floor", "rug", "path", "sidewalk", "road", "grass", "earth", "field", "sand", "dirt track", "land"})),
    "outdoor": ("yolo26n-sem.pt", frozenset({"road", "sidewalk", "terrain"})),
}

CANT_JUDGE = "can't judge (little floor visible)"
OBSTRUCTION = "possible obstruction ahead"
NONE_FOUND = "no near obstruction found (not proof the way is clear)"


@dataclass(frozen=True)
class ObstacleConfig:
    scan_width: int = 360            # frames are downscaled to this width before segmenting (speed)
    imgsz: int = 320                 # segmentation model input size
    left_edge: float = 0.35          # the central band (same as SpatialConfig)
    right_edge: float = 0.65
    min_block_run: float = 0.03      # a run of non-floor at least this share of the height blocks a column
    near_free: float = 0.3           # floor reaching less than this share of the height up the image: close
    min_blocked_share: float = 0.10  # share of corridor columns that must be blocked (a pole is enough)
    min_walkable_share: float = 0.15  # floor needed in the lower half of the corridor, else "can't judge"
    wall_column_share: float = 0.85  # wall must fill this fraction of BOTH halves of a column
    wall_corridor_share: float = 0.80  # require broad wall coverage, not a small patch or side wall
    featureless_std: float = 5.0     # max per-channel spatial std on 0..255 image; visibility warning only
    explain_conf: float = 0.5        # a YOLO box at least this sure explains the columns it covers
    confirm_scans: int = 2           # an obstruction must be found in this many of the last `confirm_window` scans
    confirm_window: int = 3
    max_age_s: float = 0.8           # don't use a scan older than this
    min_interval_s: float = 0.25     # at most ~4 scans a second, so YOLO keeps most of the CPU


@dataclass(frozen=True)
class ScanResult:
    """One segmentation scan, in the downscaled class map's pixels."""

    timestamp: float                 # perf_counter of the frame that was scanned
    width: int
    height: int
    free: np.ndarray                 # per column: share of the image height that is floor from the bottom up (0..1)
    walkable_share: float            # floor in the lower half of the corridor (0..1)
    blocked: np.ndarray              # per column: corridor column whose floor stops close
    blocker: str                     # most common non-floor class just above the blocking point (screen and log)
    scan_ms: float = 0.0
    wall_columns: np.ndarray | None = None  # central columns dominated by the semantic wall class
    featureless: bool = False

    def thirds(self, cfg: ObstacleConfig) -> dict[Region, float]:
        """Mean free space in the left, central and right parts (the experimental open-space view; display only)."""
        a, b = int(cfg.left_edge * self.width), int(cfg.right_edge * self.width)
        parts = {Region.LEFT: self.free[:a], Region.CENTER: self.free[a:b], Region.RIGHT: self.free[b:]}
        return {r: float(p.mean()) if p.size else 0.0 for r, p in parts.items()}


@dataclass
class Assessment:
    """What the newest scan means for this frame, after YOLO boxes have explained what they can."""

    status: str                      # CANT_JUDGE, OBSTRUCTION or NONE_FOUND
    obstacle: Detection | None = None
    unexplained: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # per column: blocked, unexplained


def free_space(walkable: np.ndarray, min_run_px: int) -> np.ndarray:
    """Per column, the share of the height that is walkable from the bottom up before a non-walkable run starts."""
    height = walkable.shape[0]
    blocked = ~walkable[::-1]  # bottom row first
    run = max(1, min(min_run_px, height))
    cs = np.vstack([np.zeros((1, blocked.shape[1]), np.int32), np.cumsum(blocked, axis=0, dtype=np.int32)])
    window = cs[run:] - cs[:-run]          # non-walkable pixels in each run-long window, per start row
    full = window >= run
    has = full.any(axis=0)
    first = full.argmax(axis=0)
    return np.where(has, first / height, 1.0).astype(np.float32)


class ObstacleScanner:
    """Segments frames on a background thread and turns "the floor stops close ahead" into an unknown-obstacle box.

    `model` lets tests pass a fake; `background=False` scans synchronously in submit() (tests, benchmarks).
    """

    def __init__(
        self,
        environment: str = "indoor",
        *,
        model: Any | None = None,
        config: ObstacleConfig | None = None,
        background: bool = True,
        clock=time.perf_counter,
    ) -> None:
        if environment not in ENVIRONMENTS:
            raise ValueError(f"environment must be one of {sorted(ENVIRONMENTS)}")
        self.environment = environment
        self.model_path, self.walkable_names = ENVIRONMENTS[environment]
        self.model = model
        self.config = config or ObstacleConfig()
        self.background = background
        self._clock = clock
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._pending: tuple[np.ndarray, float, int] | None = None
        self._generation = 0
        self._assessed: ScanResult | None = None
        self._latest: ScanResult | None = None
        self._recent: deque[bool] = deque(maxlen=self.config.confirm_window)  # obstruction found, per scan
        self._wall_recent: deque[bool] = deque(maxlen=self.config.confirm_window)
        self._unclear_recent: deque[bool] = deque(maxlen=self.config.confirm_window)
        self._thread: threading.Thread | None = None
        self._walkable_ids: np.ndarray | None = None
        self.last: Assessment = Assessment("starting...")
        self.scan_ms: list[float] = []

    # Model and thread -------------------------------------------------------------------------------------

    def load(self) -> Any:
        if self.model is None:
            from ultralytics import YOLO  # imported here so importing this module stays fast

            model = YOLO(self.model_path)
            if model.task != "semantic":
                raise ValueError(f"{self.model_path} is a '{model.task}' model; the obstacle layer needs a semantic "
                                 "segmentation model")
            self.model = model
        if self._walkable_ids is None:
            names = self.model.names
            self._walkable_ids = np.array(sorted(i for i, n in names.items() if n in self.walkable_names), np.int64)
        if self.background and self._thread is None:
            self._thread = threading.Thread(target=self._run, name="ObstacleScanner", daemon=True)
            self._thread.start()
        return self.model

    def warmup(self) -> None:
        cfg = self.config
        self.load().predict(np.zeros((cfg.imgsz, cfg.imgsz, 3), np.uint8), imgsz=cfg.imgsz, verbose=False)

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def reset(self) -> None:
        """Forget scans, e.g. after switching cameras or rotating."""
        with self._lock:
            self._generation += 1
            self._assessed = None
            self._pending, self._latest = None, None
            self._recent.clear()
            self._wall_recent.clear()
            self._unclear_recent.clear()
            self.last = Assessment("starting...")

    # Feeding frames -----------------------------------------------------------------------------------------

    def submit(self, image: np.ndarray, timestamp: float) -> None:
        """Offer the newest frame. In the background, frames that arrive while a scan runs replace each other."""
        self.load()
        with self._lock:
            generation = self._generation
        if not self.background:
            self._store(self.scan(image, timestamp), generation)
            return
        with self._lock:
            self._pending = (image, timestamp, generation)
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(0.5)
            self._wake.clear()
            with self._lock:
                job, self._pending = self._pending, None
            if job is None:
                continue
            started = time.perf_counter()
            try:
                image, timestamp, generation = job
                self._store(self.scan(image, timestamp), generation)
            except Exception as exc:  # a broken scan must never take the app down
                log.warning("Obstacle scan failed: %s", exc)
            self._stop.wait(max(0.0, self.config.min_interval_s - (time.perf_counter() - started)))

    def _store(self, result: ScanResult, generation: int) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._latest = result
        self.scan_ms.append(result.scan_ms)

    def latest(self) -> ScanResult | None:
        """The newest scan if it's still fresh, else None."""
        with self._lock:
            result = self._latest
        if result is None or self._clock() - result.timestamp > self.config.max_age_s:
            return None
        return result

    # Scanning ---------------------------------------------------------------------------------------------

    def scan(self, image: np.ndarray, timestamp: float) -> ScanResult:
        import cv2

        cfg = self.config
        start = time.perf_counter()
        h, w = image.shape[:2]
        small_w = min(cfg.scan_width, w)
        small = cv2.resize(image, (small_w, max(1, round(h * small_w / w))), interpolation=cv2.INTER_AREA)
        result = self.load().predict(small, imgsz=cfg.imgsz, verbose=False)[0]
        classes = result.semantic_mask.data
        classes = classes.cpu().numpy() if hasattr(classes, "cpu") else np.asarray(classes)
        featureless = bool(np.std(small.astype(np.float32), axis=(0, 1)).max() <= cfg.featureless_std)
        return replace(self.analyse(classes, timestamp, (time.perf_counter() - start) * 1000.0),
                       featureless=featureless)

    def analyse(self, classes: np.ndarray, timestamp: float, scan_ms: float = 0.0) -> ScanResult:
        """Free space per column, floor coverage and blocked corridor columns, from a class map (H x W ids)."""
        cfg = self.config
        if self._walkable_ids is None:
            self.load()
        height, width = classes.shape[:2]
        walkable = np.isin(classes, self._walkable_ids)
        free = free_space(walkable, round(cfg.min_block_run * height))
        a, b = int(cfg.left_edge * width), max(int(cfg.right_edge * width), int(cfg.left_edge * width) + 1)
        lower = walkable[height // 2:, a:b]
        walkable_share = float(lower.mean()) if lower.size else 0.0
        blocked = np.zeros(width, bool)
        if walkable_share >= cfg.min_walkable_share:
            blocked[a:b] = free[a:b] < cfg.near_free
        blocker = self._blocker(classes, free, blocked)
        # Use semantic evidence, never flat colour or absent floor alone. Requiring
        # both upper and lower coverage excludes a normal wall above open floor.
        wall_ids = [i for i, name in self.model.names.items() if name == "wall"]
        wall = np.isin(classes, wall_ids)
        wall_columns = np.zeros(width, bool)
        middle = height // 2
        if middle > 0:
            wall_columns[a:b] = ((wall[:middle, a:b].mean(axis=0) >= cfg.wall_column_share)
                                 & (wall[middle:, a:b].mean(axis=0) >= cfg.wall_column_share))
        return ScanResult(timestamp, width, height, free, walkable_share, blocked, blocker, scan_ms, wall_columns)

    def _blocker(self, classes: np.ndarray, free: np.ndarray, blocked: np.ndarray) -> str:
        if not blocked.any():
            return ""
        height = classes.shape[0]
        cols = np.flatnonzero(blocked)
        rows = np.clip((height - 1 - free[cols] * height - 2).astype(int), 0, height - 1)
        ids, counts = np.unique(classes[rows, cols], return_counts=True)
        return str(self.model.names.get(int(ids[counts.argmax()]), "?")) if ids.size else ""

    # Using a scan ---------------------------------------------------------------------------------------------

    def augment(self, frame: DetectionFrame, *, policy=None) -> DetectionFrame:
        """Add an "unknown obstacle" detection to `frame` if the newest scan shows an unexplained obstruction."""
        self.last = self.assess(frame, policy=policy)
        if self.last.obstacle is None:
            return frame
        return replace(frame, detections=frame.detections + (self.last.obstacle,))

    def assess(self, frame: DetectionFrame, *, policy=None) -> Assessment:
        cfg = self.config
        with self._lock:
            generation = self._generation
        scan = self.latest()
        if scan is None:
            return Assessment("waiting for a scan" if self._latest is None else "scan too old")
        explained = self._explained(scan, frame, policy=policy)
        unexplained = scan.blocked & ~explained
        a, b = int(cfg.left_edge * scan.width), int(cfg.right_edge * scan.width)
        corridor_cols = max(b - a, 1)
        qualifies = (scan.walkable_share >= cfg.min_walkable_share
                     and unexplained.sum() / corridor_cols >= cfg.min_blocked_share)
        wall_unexplained = (scan.wall_columns & ~explained if scan.wall_columns is not None
                            else np.zeros(scan.width, bool))
        wall_qualifies = wall_unexplained.sum() / corridor_cols >= cfg.wall_corridor_share
        unclear = scan.featureless and scan.walkable_share < cfg.min_walkable_share and not wall_qualifies
        with self._lock:
            # A newer scan does not invalidate this fresh snapshot, but a reset does.
            if self._generation != generation:
                return Assessment("waiting for a scan")
            if self._assessed is not scan:
                if self._assessed is not None and scan.timestamp - self._assessed.timestamp > cfg.max_age_s:
                    self._recent.clear()
                    self._wall_recent.clear()
                    self._unclear_recent.clear()
                self._recent.append(bool(qualifies))
                self._wall_recent.append(bool(wall_qualifies))
                self._unclear_recent.append(bool(unclear))
                self._assessed = scan
            confirmed = sum(self._recent) >= cfg.confirm_scans
            wall_confirmed = sum(self._wall_recent) >= cfg.confirm_scans
            unclear_confirmed = sum(self._unclear_recent) >= cfg.confirm_scans
        if unclear:
            if not unclear_confirmed:
                return Assessment("featureless view (confirming)")
            box = (a * frame.frame_width / scan.width, 0.0,
                   b * frame.frame_width / scan.width, float(frame.frame_height))
            return Assessment("view unclear (featureless image; distance unknown)",
                              Detection(VIEW_UNCLEAR, -1, 0.9, box, None))
        if wall_qualifies:
            if not wall_confirmed:
                return Assessment("wall fills central view (confirming)", unexplained=wall_unexplained)
            # Full-height central extent triggers the existing repeated caution
            # policy. It represents wall coverage in the view, not measured range.
            box = (a * frame.frame_width / scan.width, 0.0,
                   b * frame.frame_width / scan.width, float(frame.frame_height))
            obstacle = Detection(WALL_OBSTACLE, -1, 0.9, box, None)
            return Assessment("wall fills central view (distance unknown)", obstacle, wall_unexplained)
        if scan.walkable_share < cfg.min_walkable_share:
            return Assessment(CANT_JUDGE)
        if not qualifies or not confirmed:
            return Assessment(NONE_FOUND, unexplained=unexplained)
        x1, x2 = _longest_run(unexplained)
        nearest_free = float(scan.free[x1:x2].min())
        sx, sy = frame.frame_width / scan.width, frame.frame_height / scan.height
        # Box size encodes nearness the way the rest of the pipeline reads it: free space 0 -> full height (very
        # close, a caution), free space at `near_free` -> 40% of the height (a plain "Possible obstacle ahead.").
        view_height = float(np.clip(1.0 - 0.6 * nearest_free / cfg.near_free, 0.4, 1.0))
        box: Box = (x1 * sx, frame.frame_height * (1.0 - view_height), x2 * sx, float(frame.frame_height))
        share = (x2 - x1) / corridor_cols
        confidence = round(min(0.9, 0.4 + share), 2)
        obstacle = Detection(UNKNOWN_OBSTACLE, -1, confidence, box, None)
        return Assessment(f"{OBSTRUCTION} ({scan.blocker or 'unlabelled'})", obstacle, unexplained)

    def _explained(self, scan: ScanResult, frame: DetectionFrame, *, policy=None) -> np.ndarray:
        """Columns where a confident YOLO box stands where the floor stops: YOLO will name that thing itself."""
        explained = np.zeros(scan.width, bool)
        # Import here to avoid the alerts -> obstacles module cycle. Use the active
        # policy so custom label/confidence/path settings also govern suppression.
        from sentra.guide.alerts import AlertPolicy
        from sentra.guide.spatial import SpatialEstimator

        policy = policy if policy is not None else AlertPolicy()
        sx = scan.width / max(frame.frame_width, 1)
        for obj in SpatialEstimator().locate(frame):
            det = obj.detection
            if not policy.is_confident(obj) or det.confidence < self.config.explain_conf:
                continue
            x1, y1, x2, y2 = det.box
            c1, c2 = max(0, int(x1 * sx)), min(scan.width, int(np.ceil(x2 * sx)))
            if c2 <= c1:
                continue
            stop = 1.0 - scan.free[c1:c2]                 # where the floor stops, as a share from the top
            base = y2 / max(frame.frame_height, 1)
            explained[c1:c2] |= base >= stop - 0.1        # the box reaches down to where the floor stops
        return explained

    def status(self) -> str:
        return f"Floor scan ({self.environment}, experimental): {self.last.status}"

    @property
    def median_scan_ms(self) -> float:
        return median(self.scan_ms[-200:]) if self.scan_ms else 0.0


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    """[start, end) of the longest run of True."""
    best, start = (0, 0), None
    for i, on in enumerate(np.append(mask, False)):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best
