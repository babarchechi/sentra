# Sentra — Phase 1 Team Plan

> Capital Hacks 5.0 · Fri Nov 6 – Sun Nov 8, 2026
> Read this whole file once, then jump to **your track**. If anything here is unclear or wrong, say so in the team
> chat before building around it.

## 1. Who does what

| Person | Track | One-line goal | Branch |
| --- | --- | --- | --- |
| **babarchechi** | **B — Camera & pipeline** (+ integration lead) | Camera → pose model → analyzer → dashboard, running end to end | `feature/pipeline` |
| **iesa242** | **A — Posture analysis** | Turn 17 body keypoints into a per-person "possible fall" state | `feature/posture-analysis` |
| **Swechha03** | **C — Dashboard & alerts** (+ pitch) | Show the live view, per-person status, a "please check" alert, and privacy mode | `feature/ui` |

Every **code** change goes through a pull request that **another teammate reviews and approves** before it is
merged; docs and instructions can go straight to `main` (Section 9). Once a PR is approved, any teammate can merge
it. babarchechi, as integration lead, makes sure `main` keeps running after each merge.

**What we are building:** a demo that watches a consenting person on a camera and, if they drop quickly to the floor
and stay there, shows **"Possible fall — please check"** so a human can confirm or dismiss it.
It is a hackathon prototype, **not a medical device**.

## 2. Tonight (Friday) — everyone

1. **Set up and run the Phase 0 test** (README → *Setup*). Use **Python 3.11/3.12 from python.org**, not the Microsoft
   Store build (Windows blocks the Store build from the camera).

   ```powershell
   git clone https://github.com/babarchechi/sentra.git
   cd sentra
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   python src/camera_test.py --source 0 --track
   ```

   macOS: `python3 -m venv .venv` then `source .venv/bin/activate`, and allow camera access for your terminal app in
   *System Settings → Privacy & Security → Camera*. If `pip install` fails on torch, tell the team.

   **Done when:** you see the skeleton with an `id:` label on yourself and `q` quits.

2. **Read Section 4 (the shared contract) and agree on it** in the team chat. It's the only thing all three tracks depend on.
3. ✅ **Done:** the package skeleton (Section 5) is merged into `main` (PR #1). Every file you own already exists with
   a `TODO` showing where to start.
4. **Everyone:** `git pull`, `python -m pip install -r requirements.txt`, then create your branch (Section 9).

## 3. How the pieces fit

```text
Camera (webcam index or phone URL)
   │   capture.py ............................ B   latest frame only, reconnects
   ▼
YOLO26n-Pose + ByteTrack   pipeline.py ....... B
   │   PoseFrame  (each person: track_id, 17 keypoints, box)
   ▼
PostureAnalyzer   posture.py ................. A
   │   PersonStatus per person  (state, confidence, reason)
   ▼
StatusUpdate ──► StatusSink   ui/ ............ C   dashboard, alert, privacy mode
```

Each track can be built and tested **without the others**, using stand-ins:

| Track | Builds against | Stand-in until the real thing is merged |
| --- | --- | --- |
| A | `PoseFrame` in, `PersonStatus` out | Synthetic keypoints generated in tests (no camera needed) |
| B | Real camera + model | Stub analyzer that returns `UNKNOWN`/`UPRIGHT` |
| C | `StatusUpdate` in | `fake_events.py` — a scripted stick figure that stands, falls, lies down |

## 4. The shared contract — `src/sentra/events.py`

**Owner: everyone.** Changing it needs a PR and a 👍 from all three people, because it breaks the other tracks.
babarchechi commits this exact file in the skeleton PR.

```python
"""Shared data contract between pipeline (B), posture analysis (A) and UI (C).

Changing anything in this file needs agreement from all three track owners.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

# COCO-17 keypoint order produced by YOLO pose models.
KEYPOINT_NAMES: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
KP: dict[str, int] = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# Pairs of keypoint indices to connect when drawing a skeleton (0-indexed).
SKELETON: tuple[tuple[int, int], ...] = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
)


@dataclass(frozen=True)
class PersonPose:
    track_id: int                              # temporary ByteTrack ID — NOT an identity
    keypoints: np.ndarray                      # shape (17, 3) float32: x px, y px, confidence 0..1
    box: tuple[float, float, float, float]     # x1, y1, x2, y2 in pixels
    box_confidence: float                      # 0..1


@dataclass(frozen=True)
class PoseFrame:
    timestamp: float                           # seconds from time.perf_counter() — use this, never frame counts
    frame_width: int
    frame_height: int
    people: tuple[PersonPose, ...]             # empty tuple when nobody is tracked


class PostureState(str, Enum):
    UNKNOWN = "unknown"              # not enough visible keypoints to judge
    UPRIGHT = "upright"              # standing / walking / sitting upright
    DESCENDING = "descending"        # hips dropping fast right now
    ON_GROUND = "on_ground"          # lying / horizontal, no fast drop seen before it
    POSSIBLE_FALL = "possible_fall"  # fast drop, then on the ground for N seconds → alert


@dataclass(frozen=True)
class PersonStatus:
    track_id: int
    state: PostureState
    confidence: float                # 0..1, how sure the analyzer is about `state`
    reason: str                      # short text for the UI, e.g. "torso 78° for 6.1 s after fast drop"
    since: float                     # timestamp when this person entered `state`
                                     # (time in state = time.perf_counter() - since)


@dataclass(frozen=True)
class StatusUpdate:
    pose: PoseFrame
    statuses: tuple[PersonStatus, ...]   # one per person in pose.people — match on track_id, not order
    frame: np.ndarray                    # raw BGR camera frame (never written to disk)
    annotated: np.ndarray                # BGR frame with the Ultralytics skeleton overlay
    fps: float                           # pipeline frames per second


class StatusSink(Protocol):
    """Anything that displays StatusUpdates (the dashboard). Implemented by Track C."""

    def publish(self, update: StatusUpdate) -> None: ...   # called once per processed frame
    def should_stop(self) -> bool: ...                     # True when the user quit (e.g. pressed q)
    def close(self) -> None: ...                           # release windows / servers
```

The analyzer's public interface (lives in `posture.py`, owned by A, but other tracks call it, so don't change it
without telling the team):

```python
class PostureAnalyzer:
    def __init__(self, config: PostureConfig | None = None) -> None: ...
    def update(self, pose: PoseFrame) -> tuple[PersonStatus, ...]: ...   # call once per frame
```

## 5. Repo layout and file ownership

Only edit files you own. Shared files go through a PR **and** a message in the team chat.

```text
sentra/
├── README.md                    shared
├── requirements.txt             shared — add pinned deps via PR, tell the team to re-run pip install
├── pyproject.toml               B      (pytest config: pythonpath = ["src"], testpaths = ["tests"])
├── docs/
│   ├── PHASE1_PLAN.md           shared — this file
│   └── evaluation.md            B      test-session results
├── src/
│   ├── camera_test.py           Phase 0 — leave as is
│   ├── main.py                  B      real app entry point
│   ├── ui_demo.py               C      dashboard on fake events, no camera or model
│   └── sentra/
│       ├── __init__.py          B
│       ├── events.py            shared — Section 4
│       ├── capture.py           B
│       ├── pipeline.py          B
│       ├── posture.py           A
│       └── ui/
│           ├── __init__.py      C
│           ├── cv_dashboard.py  C      Level 1 dashboard (OpenCV window)
│           ├── web_dashboard.py C      Level 2 dashboard (stretch)
│           └── fake_events.py   C
└── tests/
    └── test_posture.py          A
```

**Skeleton PR (babarchechi, tonight):** create every file above with the real `events.py` and minimal stubs, so
nobody creates the same new file on two branches:

- `posture.py` — `PostureConfig` dataclass and a `PostureAnalyzer` whose `update()` returns `UNKNOWN` for everyone
- `cv_dashboard.py` — shows `update.annotated` in a window, `should_stop()` is True after `q`
- `fake_events.py`, `ui_demo.py`, `capture.py`, `pipeline.py`, `main.py` — empty functions with docstrings
- `tests/test_posture.py` — one placeholder test that imports `PostureAnalyzer`

Scripts run as `python src/main.py` / `python src/ui_demo.py` from the repo root; that puts `src/` on the import path
so `from sentra.events import PoseFrame` works.

---

## 6. Track A — Posture analysis · **iesa242**

**Goal:** given a stream of `PoseFrame`s, keep a short history per `track_id` and output a `PersonStatus` for each
person every frame. Raise `POSSIBLE_FALL` only after a **fast drop followed by staying on the ground**.

**Your files:** `src/sentra/posture.py`, `tests/test_posture.py` (and any helper script under `src/tools/` you want
for live tuning).

**You don't need a camera to start.** Build and test everything on synthetic keypoints first.

### 6.1 Features to compute per person, per frame

Use only keypoints with confidence ≥ `min_keypoint_conf`. Useful indices (`KP[...]` in `events.py`):
shoulders 5/6, hips 11/12, knees 13/14, ankles 15/16, nose 0.

| Feature | How | Why |
| --- | --- | --- |
| Mid-shoulder, mid-hip | Average of left/right (fall back to whichever side is visible) | Stable body centre points |
| **Torso angle** | Vector mid-hip → mid-shoulder; angle from vertical = `degrees(atan2(abs(dx), -dy))` (image y grows downward) | ~0–30° standing, ~60–90° lying |
| **Box aspect** | `(x2 - x1) / (y2 - y1)` from `box` | Tall box when standing, wide box when lying |
| **Torso length** | Distance mid-shoulder ↔ mid-hip; remember the median while UPRIGHT | Scale factor, so speed doesn't depend on distance from camera |
| **Hip drop speed** | Change in mid-hip y over the last ~0.3–0.5 s of timestamps ÷ torso length ÷ seconds (positive = moving down) | A fall drops the hips fast; sitting down is slow |

Always use `pose.timestamp` for time. CPU frame rate varies (roughly 8–15 fps), so never count frames as time.

### 6.2 State machine (per track)

```text
            fast hip drop                      lying within descent_window_s
UPRIGHT ───────────────────► DESCENDING ───────────────────────────────► (lying, after drop)
   ▲                             │ no lying within window                     │ stays lying ≥ on_ground_alert_s
   │                             ▼                                            ▼
   │                          UPRIGHT                                   POSSIBLE_FALL
   │                                                                          │ upright for ≥ recovery_s
   └──────────────────────────────────────────────────────────────────────────┘

lying WITHOUT a fast drop first (lying down slowly) ──► ON_GROUND   (no alert)
too few keypoints (hips/shoulders not visible) ───────► UNKNOWN     (never alert from UNKNOWN)
```

- Before the alert fires, the "lying after a drop" period can report `DESCENDING` or `ON_GROUND`, whichever is
  clearer. The UI only reacts to `POSSIBLE_FALL`.
- Add **hysteresis**: a new state must hold for `state_hold_frames` frames before you switch, so jitter doesn't flicker.
- **Forget** tracks not seen for `track_timeout_s`. Exception: a track that disappears soon after a fast drop goes
  on a "recently lost" list instead (see below).
- `confidence`: keep it simple, e.g. mean keypoint confidence × how clearly the thresholds were exceeded, clipped to 0..1.
- `reason`: short and human, e.g. `"torso 81° for 5.4 s after fast drop"`. The dashboard shows it.

#### When the picture is uncertain

Rule of thumb: **uncertain after a fast drop → a lower-confidence `POSSIBLE_FALL` for a human to check. Uncertain
with no drop → no alert.** Build the core state machine above first, then add these in order:

1. **Track ID changes mid-fall** (should-have). ByteTrack can give someone a new ID when they drop suddenly or are
   briefly hidden, and the new ID has no history. When a track disappears within a few seconds of a fast drop, put
   its last box and drop time on a **"recently lost"** list for `lost_memory_s`. If a new `track_id` appears whose box
   overlaps that last box (IoU ≥ `lost_match_iou`), copy the old history onto the new ID. The drop still counts and
   the on-ground timer keeps running.
2. **Fast drop, then can't be seen** (should-have). After a fast drop the person turns `UNKNOWN` (hips or shoulders
   hidden) or leaves the frame entirely (fell behind a bed, out of shot). If they don't reappear upright within
   `lost_after_drop_alert_s`, raise `POSSIBLE_FALL` with confidence ≤ `uncertain_confidence_max` and a reason like
   `"fast drop, then lost from view for 6 s"`. Reappearing upright cancels it.
   - The `UNKNOWN`-after-drop case works with the current contract, because that person is still in `pose.people`.
   - ⚠️ **The fully-out-of-frame case needs a contract change.** `events.py` says statuses are "one per person in
     pose.people", and a person who's out of frame isn't in it. It needs that rule relaxed so a `PersonStatus` can be
     sent for a recently lost `track_id`, and the dashboard has to show an alert with no skeleton. Agree it with
     babarchechi and Swechha03 and change `events.py` in a PR that all three approve before building on it.
3. **Stillness on the ground** (should-have for confidence; stretch for alerting). While someone is lying, measure
   how much their visible keypoints move over the last ~1–2 s, in torso-lengths per second.
   - Still (below `still_motion_max`): raise the `POSSIBLE_FALL` confidence and say so in `reason`
     (`"... still for 6 s"`).
   - Moving does **not** cancel an alert, because someone struggling after a fall has still fallen. Only getting back
     upright does (`recovery_s`).
   - *Stretch:* lying still for `still_long_alert_s` with **no** drop seen → low-confidence `POSSIBLE_FALL`. This
     catches slow collapses. Lying down slowly and moving around (e.g. stretching on a mat) stays `ON_GROUND`.

### 6.3 Starting thresholds — put them in a `PostureConfig` dataclass

These are **guesses to tune**, not known-good values. Keep them all in one dataclass with these defaults (no YAML needed).

| Setting | Start | Meaning |
| --- | --- | --- |
| `min_keypoint_conf` | 0.5 | Ignore keypoints below this |
| `upright_max_angle_deg` | 35 | Torso within this of vertical counts as upright |
| `lying_min_angle_deg` | 60 | Torso beyond this counts as lying |
| `lying_min_aspect` | 1.2 | Box width/height above this supports "lying" |
| `descent_speed` | 1.5 | Torso-lengths per second of hip drop that counts as "fast" |
| `descent_window_s` | 2.0 | Lying must start within this long after the fast drop |
| `on_ground_alert_s` | 5.0 | Seconds on the ground before `POSSIBLE_FALL` (short for the demo) |
| `recovery_s` | 2.0 | Upright this long clears `POSSIBLE_FALL` |
| `state_hold_frames` | 3 | Frames a new state must persist before switching |
| `track_timeout_s` | 3.0 | Forget a track not seen for this long |
| `lost_memory_s` | 10.0 | How long to keep a track that disappeared soon after a fast drop |
| `lost_match_iou` | 0.3 | Box overlap needed to treat a new track ID as that lost person |
| `lost_after_drop_alert_s` | 5.0 | Unseen or `UNKNOWN` this long after a fast drop → low-confidence `POSSIBLE_FALL` |
| `uncertain_confidence_max` | 0.5 | Highest confidence allowed for alerts raised from uncertain evidence |
| `still_motion_max` | 0.05 | Keypoint movement (torso-lengths per second) that still counts as "still" |
| `still_long_alert_s` | 20.0 | *Stretch:* still on the ground this long with no drop seen → low-confidence alert |

### 6.4 Tests (`tests/test_posture.py`, run with `python -m pytest -q`)

`pytest` is already in `requirements.txt` (added by the skeleton PR). Write a helper that builds a
`(17, 3)` keypoint array for a person with a given torso angle, hip position and scale, plus a helper that turns a
timeline into `PoseFrame`s at ~10 fps. Minimum cases:

| Scenario (synthetic) | Expected |
| --- | --- |
| Standing still 5 s | `UPRIGHT` throughout, never `POSSIBLE_FALL` |
| Stand 2 s → drop to lying in 0.5 s → lie 6 s | `POSSIBLE_FALL` after ~5 s of lying |
| Same, but stands back up after 2 s of lying | No `POSSIBLE_FALL` |
| Lie down slowly over 3 s → lie 8 s | `ON_GROUND`, no `POSSIBLE_FALL` |
| Sit down (hips drop slowly, torso stays upright) | No `POSSIBLE_FALL` |
| `POSSIBLE_FALL` then upright 3 s | Back to `UPRIGHT` |
| Hip and shoulder keypoints low-confidence | `UNKNOWN` |
| Two people, only one falls | Alert only for that `track_id` |
| Person disappears for 5 s | Track forgotten, no crash |
| Uneven frame timing (5–15 fps jitter) | Same outcomes as above |
| Fast drop, then the track ID changes while lying (new ID, overlapping box) | `POSSIBLE_FALL` for the new ID, on time |
| Fast drop, then hips and shoulders go low-confidence for 6 s | `POSSIBLE_FALL` with confidence ≤ 0.5 |
| Fast drop, disappear, reappear upright within 3 s | No alert |
| Fast drop, then out of frame for 6 s *(only after the contract change)* | Low-confidence `POSSIBLE_FALL` for the old ID |
| Fast drop, then lying still vs. lying and moving | Both `POSSIBLE_FALL`; the still one has higher confidence |
| Lie down slowly, keep moving for 25 s | `ON_GROUND`, no alert |
| *Stretch:* lie down slowly, stay still for 25 s | Low-confidence `POSSIBLE_FALL` |

### 6.5 Definition of done (Track A)

- [ ] All the test cases above pass on `python -m pytest -q`
- [ ] `PostureAnalyzer.update()` matches the interface in Section 4 and never crashes on empty or partial poses
- [ ] Thresholds live in `PostureConfig` with comments
- [ ] "When the picture is uncertain" items 1–3 handled (the out-of-frame part of item 2 only after the contract change)
- [ ] Tuned once against a live camera with babarchechi during the evaluation session (Section 11)

**Known limits — list them, don't hide them:** works best with the camera at roughly chest height, side-on to the
person. A camera looking straight down makes torso angle unreliable. Track ID changes and people lost from view after
a fast drop become low-confidence alerts (6.2), but a slow collapse with no fast drop is only caught by the
long-stillness stretch rule.

---

## 7. Track B — Camera & pipeline · **babarchechi**

**Goal:** `python src/main.py --source 0` runs the whole system end to end. You also own integration: checking that
`main` still runs after each merge (anyone can merge an approved PR), and running the evaluation sessions.

**Your files:** `capture.py`, `pipeline.py`, `main.py`, `sentra/__init__.py`, `pyproject.toml`, `docs/evaluation.md`.

### 7.1 Tasks, in order

1. **Tonight — skeleton PR** (Section 5), with the real `events.py`. Merge it, tell the team to pull.
2. **`capture.py` → `FrameSource`**
   - Takes a webcam index or a URL (reuse `parse_source` from `camera_test.py`).
   - A background thread reads continuously and keeps **only the latest frame**, so slow inference never builds a
     backlog of stale frames.
   - `read() -> (ok, frame, timestamp)` with `time.perf_counter()`; `close()` stops the thread.
     (Not `time.monotonic()`: on Python 3.12 for Windows it only ticks every ~15.6 ms, so frame timestamps collide.)
   - URL sources: if reads fail, reconnect with a short backoff (1 s, 2 s, 4 s, max 5 s) instead of crashing.
3. **`pipeline.py` → `Pipeline(source, analyzer, sink, model_path="yolo26n-pose.pt")`**
   - Loop: `frame = source.read()` → `model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)` →
     build a `PoseFrame` → `analyzer.update(pose)` → build a `StatusUpdate` → `sink.publish(update)` → stop when
     `sink.should_stop()`.
   - Building `PoseFrame` from `r = results[0]`:
     - `r.boxes.id` is `None` until ByteTrack assigns IDs. Treat that as nobody tracked (empty `people`).
     - Keypoints: `r.keypoints.data.cpu().numpy()` gives shape `(N, 17, 3)`.
     - Boxes: `r.boxes.xyxy` and `r.boxes.conf`. Frame size: `r.orig_shape` is `(height, width)`.
   - `annotated = r.plot()`; measure fps over a rolling window.
   - **Transition log:** write `outputs/transitions_<YYYYmmdd_HHMMSS>.csv` with columns
     `timestamp, track_id, from_state, to_state, confidence, reason`. Write only events; **never save images**.
     (`outputs/` is already git-ignored.)
4. **`main.py`** — CLI: `--source` (default `0`), `--rotate 0|90|180|270` (turn frames upright, e.g. `90` for a phone
   in portrait), `--ui cv|web` (default `cv`), `--privacy` (start in skeleton-only
   mode), `--model` (default `yolo26n-pose.pt`). Builds `FrameSource`, `PostureAnalyzer`, the chosen dashboard and
   `Pipeline`, and always calls `close()` on everything in a `finally`.
5. **Phone camera** — test an IP-camera app stream on a private hotspot (see README). Record the result in
   `docs/evaluation.md`. The laptop webcam is the demo fallback if the phone fails.
6. **Integration and evaluation** (Sections 10–11).

### 7.2 Definition of done (Track B)

- [ ] `python src/main.py --source 0` shows real tracked poses with statuses and fps in the dashboard
- [ ] Handles 0 people, many people, and `boxes.id is None` without crashing
- [ ] A URL source that drops out reconnects instead of crashing
- [ ] The transition CSV is written; no images or video saved anywhere
- [ ] `main` runs after every merge

---

## 8. Track C — Dashboard & alerts · **Swechha03**

**Goal:** a dashboard that shows the live view and each person's status, raises a clear **"Possible fall — please
check"** alert that a human must confirm or dismiss, and has a **privacy mode** that hides the video and shows only
the skeleton. You also own the pitch deck and the live-demo script.

**Your files:** `src/sentra/ui/` (all of it), `src/ui_demo.py`.

**You don't need the model or the analyzer to start.** Build on `fake_events.py` first.

### 8.1 Tasks, in order

1. **`fake_events.py`** — a generator of `StatusUpdate`s with no camera or model:
   - Draw a stick figure on a 640×480 black canvas from synthetic keypoints (use `SKELETON`).
   - Script it on a loop: `UPRIGHT` 5 s → `DESCENDING` 0.5 s → `ON_GROUND` 5 s → `POSSIBLE_FALL` until acknowledged →
     `UPRIGHT`.
   - Add a second figure that always stays `UPRIGHT`, so you can test multi-person display.
   - Fill `frame` and `annotated` with the canvas, and put realistic text in `reason`.
2. **`ui_demo.py`** — runs your dashboard on the fake events: `python src/ui_demo.py`.
3. **Level 1 (must-have): `cv_dashboard.py` → `CvDashboard` implementing `StatusSink`**
   - **Video panel:** `update.annotated` normally. In privacy mode, draw only the skeleton from `update.pose` on black.
   - **Side panel:** one row per person with `id`, state, how long they've been in it (`now - since`), and `reason`.
     Colours: green `UPRIGHT`, grey `UNKNOWN`, amber `DESCENDING`/`ON_GROUND`, **red `POSSIBLE_FALL`**.
   - **Alert:** when a `track_id` first enters `POSSIBLE_FALL`, show a big red banner
     `Possible fall — person #<id> — please check` and play a sound **without freezing the loop**. On Windows:
     `winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)`. Wrap it in `try/except` so it
     doesn't crash on macOS.
   - The alert **stays until a human responds**: `c` = confirmed (I checked, they needed help), `f` = false alarm. If
     the person gets up before anyone responds, mark it "recovered" but keep the alert in the log.
   - **Low-confidence alerts** (confidence ≤ 0.5, from Section 6.2 "When the picture is uncertain") still show the
     full alert and still need `c`/`f`. Add a small "uncertain" tag and show the `reason`. If the team adopts the
     out-of-frame status, an alert can arrive for a `track_id` with no skeleton on screen; show it anyway.
   - **Keys:** `c` confirm · `f` false alarm · `p` privacy mode on/off · `q` quit (`should_stop()` returns True).
   - Show fps and a small "PRIVACY MODE" or "LIVE" label.
   - **Alert log:** `outputs/alerts_<YYYYmmdd_HHMMSS>.csv` with columns
     `alert_time, track_id, reason, response (confirmed|false_alarm|recovered), response_time`. No images.
4. **Plug into the real app** once B's pipeline is merged: `python src/main.py --source 0` uses your `CvDashboard`.
5. **Level 2 (stretch, only after Level 1 works with the real pipeline): `web_dashboard.py`**
   - Recommended: **Flask** (it's simple with a background pipeline thread). Pin it in `requirements.txt` via your PR.
   - `publish()` stores the latest update under a `threading.Lock`. Routes: `/` page, `/video` MJPEG stream of the
     latest frame (respecting privacy mode), `/status` JSON (poll every ~500 ms), `POST /ack` for confirm/false alarm.
   - Bind to **`127.0.0.1` only**, never `0.0.0.0`. The video must not be reachable from the venue Wi-Fi.
   - Same features as Level 1. If you'd rather use another framework, keep the `StatusSink` interface.
6. **Pitch and demo:** draft the deck by **Saturday evening**. Include problem, how it works (Section 3 diagram),
   privacy-by-design (no recording, skeleton-only mode, local only, consent), evaluation numbers from Section 11, the
   limitations, and that a human always confirms. Write a 2-minute live-demo script with a backup plan.

### 8.2 Wording rules (the UI and pitch)

- Say **"Possible fall — please check"**. Never "Fall detected", "Emergency", or "Calling 911".
- Never show or imply identity ("person #3", not a name). Track IDs are temporary.
- Show the disclaimer somewhere on screen: *Prototype — not a medical device.*

### 8.3 Definition of done (Track C)

- [ ] `python src/ui_demo.py` shows the dashboard on fake events with all keys working
- [ ] The alert banner and sound fire once per `POSSIBLE_FALL`, stay until `c`/`f`, and are logged to CSV
- [ ] Privacy mode shows skeleton only, no video pixels
- [ ] Works as the real app's dashboard via `python src/main.py`
- [ ] Pitch deck and demo script ready for the Sunday rehearsal

---

## 9. Git workflow (everyone)

```powershell
# start of every work session
git checkout main
git pull origin main

# first time only: create your branch
git checkout -b feature/posture-analysis     # or feature/pipeline / feature/ui

# later sessions: bring main's updates into your branch
git checkout feature/posture-analysis
git merge main

# save work (add only your own files — never `git add .`)
git add src/sentra/posture.py tests/test_posture.py
git commit -m "feat(posture): add torso-angle and hip-speed features"
git push -u origin feature/posture-analysis   # -u only needed the first time
```

Then open a pull request on GitHub (the yellow **Compare & pull request** button, or `gh pr create`) into `main`,
pick a teammate under **Reviewers**, and post the link in the team chat.

### Review rule

- **Code** (anything in `src/` or `tests/`, plus `requirements.txt` and `pyproject.toml`): goes through a pull
  request and needs an **approving review from someone other than the author** before it's merged. That includes
  babarchechi's own PRs. Changes to the shared `src/sentra/events.py` need all three to approve.
- **Docs and instructions** (`docs/`, `README.md`): commit straight to `main` and push, no PR needed. Pull first to
  avoid conflicts, and post a one-line note in the team chat saying what changed.

### Merging

- **Anyone can merge** a code PR once it has the approvals it needs: the author, the reviewer, or anyone else.
- **New commits need a new approval.** If commits are pushed after someone approved, the PR needs a fresh approval
  before merging. On our free plan GitHub **keeps showing "Approved"** even after new commits, so before you click
  Merge, check on the *Conversation* tab that the latest approval comes **after** the latest commit.
- **`events.py` changes need all three.** If a PR touches `src/sentra/events.py`, wait for approval from both
  people who didn't write it.
- After merging, click **Delete branch**, then post "merged #N" in the team chat so everyone pulls.

GitHub can't enforce any of this on our free private repo, so it's a team agreement: **don't click Merge on a code
PR that doesn't have the approvals above.**

### How to review a PR

Try to respond within about 30 minutes during work sessions; a slow review blocks a teammate.

1. Open the PR on GitHub and go to the **Files changed** tab.
2. Check:
   - Does it only touch the author's own files (Section 5)? Changes to shared files need everyone's OK.
   - Does it still match `events.py`?
   - Nothing committed that shouldn't be: `.venv/`, `*.pt`, videos, `outputs/`, stream URLs, passwords.
   - Does it run? To try it locally:

     ```powershell
     git fetch origin
     git checkout <their-branch-name>
     python -m pytest -q          # plus that track's run command
     git checkout <your-branch>   # go back to your own work
     ```

3. Comment on specific lines if something needs changing, then **Review changes** → **Approve** or
   **Request changes**. Small style nits shouldn't block an approval during a hackathon.
4. Once it has the approvals it needs (see *Merging* above), anyone can click **Merge pull request** and then
   **Delete branch**. Everyone then pulls `main` and merges it into their own branch.

### Habits

- **Small PRs, often.** Two or three per person per day is better than one giant PR on Saturday night, and they're
  much faster to review.
- Commit prefixes: `feat:`, `fix:`, `test:`, `docs:`, `chore:`.
- **Never commit** `.venv/`, `*.pt` weights, videos, `outputs/`, `.env`, stream URLs or passwords.
- Never commit code directly to `main`, and never force-push it. If a merge conflict is in someone else's file, ask
  them instead of guessing.
- New dependency? Add the pinned line to `requirements.txt` in your PR and tell everyone to re-run
  `python -m pip install -r requirements.txt` after it merges.

## 10. Timeline and checkpoints

Adjust times to the official hackathon schedule. Fill in the submission deadline here: **________**

| When | Checkpoint | A (iesa242) | B (babarchechi) | C (Swechha03) |
| --- | --- | --- | --- | --- |
| **Fri night** | Everyone set up | Phase 0 test passes; read Section 6 | Phase 0 passes; **skeleton PR merged** | Phase 0 passes; read Section 8 |
| **Sat 12:00** | Each track works on its own | Features + state machine; first tests pass | `FrameSource` + `Pipeline` with stub analyzer, real camera | Level 1 dashboard on fake events |
| **Sat ~15:00** | **Integration 1** — all merged | PR merged | Merge all three; `main.py` runs end to end | PR merged |
| **Sat evening** | Evaluation + tuning | Tune thresholds from session results | Run evaluation (Section 11), phone stream test | Alert log; pitch deck draft; Level 2 if time |
| **Sun morning** | **Feature freeze** (≥ 3 h before deadline) | Bug fixes only | Final merge, tag `v0.1-demo` | Rehearse demo ×3 with backup plan |

At each checkpoint, post a short update in the team chat: done / next / blocked.

## 11. Evaluation session (Saturday evening, led by babarchechi)

Only **consenting teammates** on camera. Use a mat or cushions, and **never do a real, hard fall**: a controlled
kneel-then-lie-on-side is enough. Nobody records video; we use the transition and alert CSVs only.

Camera at roughly chest height, side-on, 2–4 m away. **Before starting, check that people appear upright on screen.**
A phone in portrait needs `--rotate 90` (or `270`), otherwise everyone looks like they're lying down. Repeat each
scenario 5 times:

| # | Scenario | Expected |
| --- | --- | --- |
| 1 | Walk across the frame | No alert |
| 2 | Sit down on a chair normally | No alert |
| 3 | Bend down to pick something up | No alert |
| 4 | Sit on the floor slowly | No alert |
| 5 | Lie down slowly on the mat | `ON_GROUND`, no alert |
| 6 | Controlled quick drop onto the mat, stay down 8 s | Alert after ~5 s |
| 7 | Quick drop, get up within 2 s | No alert |
| 8 | Two people, one does #6 | Alert for that person's ID only |
| 9 | Walk partly out of frame / behind a chair | No alert (note what happens) |

Record in `docs/evaluation.md`: for each scenario, how many correct alerts, false alarms and misses, plus fps and any
odd behaviour. These numbers go in the pitch, and honest numbers beat "it works".

## 12. Ground rules

- **Out of scope this weekend:** SMS/email/phone alerts, cloud upload, face or identity recognition, video recording,
  training a custom model, buying hardware or APIs.
- **Consent:** only teammates who agreed. No footage of strangers, patients or care-home residents.
- **Privacy by default:** frames stay in memory; nothing is saved except event CSVs; web dashboard on `127.0.0.1` only.
- **Honest framing:** prototype that flags *possible* falls for a human to check. Not a medical or emergency device.
- **Stuck for more than 30 minutes?** Post in the team chat with what you tried. Don't stay blocked.

## 13. Decisions still open

- [ ] Confirm Section 4 (`events.py`) tonight. It's already merged in PR #1, so any changes go in a new PR that all three approve
- [ ] **Swechha03:** confirm the `events.py` clock change merged in PR #3 without your review: timestamps now come from
  `time.perf_counter()`, so "time in state" is `time.perf_counter() - status.since`
- [ ] Is the phone camera a must-have for the demo, or a bonus? (Default: bonus; laptop webcam is the safe path)
- [ ] Level 2 web dashboard: go or no-go at the Saturday evening checkpoint
