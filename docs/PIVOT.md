# Pivot: Sentra (fall detection) → Sentra Guide (supplementary spatial awareness)

> Capital Hacks 5.0 · Health + Accessibility Technology · Nov 6–8, 2026
> Sentra Guide is an **experimental supplementary awareness prototype**. It is not a navigation aid, not a replacement
> for a cane or guide dog, and it never says a path is clear or safe.

This file records what the repository contained when we pivoted, what actually works, and what happens to each part.
The plan for who builds what next is in [TEAM_PLAN.md](TEAM_PLAN.md).

## 1. Baseline at the pivot (audited 2026-11-06, ~18:15)

| Item | Value |
| --- | --- |
| Remote | `origin` = https://github.com/babarchechi/sentra (private) |
| Baseline commit | `main` @ `7de069b` "Merge pull request #7 from babarchechi/feature/ui" |
| Merged PRs | #1 skeleton, #2 review workflow, #3 pipeline, #4 Windows camera, #5 eval report, #6 posture (fall state machine), #7 dashboard |
| Open PRs | none |
| Working tree | clean (no uncommitted teammate edits) |
| Tests at baseline | `python -m pytest -q` → **116 passed** (Windows 11, Python 3.12.10, CPU) |
| Stack | ultralytics 8.4.203, torch 2.15.0+cpu, opencv-python 5.0.0.93, lap 0.5.14, winrt-* (Windows camera) |
| Pivot branch | `feature/sentra-guide-pivot` (from `7de069b`) |

There was no frontend/backend split, no web server, no websocket API and no speech. The app was a single Python
process with an OpenCV window.

## 2. What works: verified vs. not verified

"Verified" means someone ran it on real hardware and saw the result, with the date.

| Capability | Status | Evidence |
| --- | --- | --- |
| Laptop webcam → frames (`FrameSource`, newest frame only) | ✅ Verified | Phase 0/1 runs, and two Sentra Guide runs on 2026-11-06 at ~25 fps |
| iPhone camera → laptop (DroidCam "PC Webcam" MJPEG over Personal Hotspot) | ✅ Verified with the **fall-detection** app (`src/main.py`), landscape and portrait (`--rotate 90`) on 2026-11-06 | docs/evaluation.md. Not yet run with `src/guide.py`, which uses the same `FrameSource`. |
| Android camera → laptop (IP Webcam) | ⏳ Not verified | evaluation.md still says "pending" |
| Windows camera API source (`windows:<name>`, Phone Link) | 🧪 Partly | `windows:HP 5MP` worked (iesa242); Phone Link only showed a spinner |
| Automatic reconnect on a real phone stream | ⏳ Not verified | covered by fake-camera tests only |
| YOLO26n-Pose + ByteTrack, multi-person IDs | ✅ Verified | webcam + iPhone, 2026-11-06 |
| Fall state machine (`posture.py`) and dashboard (`ui/`) | ✅ Unit-tested, merged | 116 tests. A full live evaluation session (docs/evaluation.md) was not run. |
| **YOLO26n object detection (`yolo26n.pt`)** | ✅ Verified (new) | sample image: bus + 4 people found, ~29 ms/frame on CPU after warm-up |
| **Sentra Guide: camera → detection → left/center/right → throttled speech** | ✅ Verified on the laptop webcam (new) | 2026-11-06: Babar heard the cues and confirmed directions matched the screen. 25 fps, median 29–30 ms model time, 39–44 ms capture-to-decision |
| Laptop text-to-speech (Windows System.Speech) | ✅ Verified (new) | heard live; automated round-trip test passes |
| Speech on the phone / browser | ❌ Not built | planned (TEAM_PLAN.md, Swechha) |
| Depth: Depth Anything V2 / Record3D LiDAR | ❌ Not built, not installed | DEPTH_OPTIONS.md |
| Websocket/API, web UI | ❌ Not built | `ui/web_dashboard.py` is a stub |

## 3. Keep / adapt / archive / remove-later

Nothing was deleted, moved or renamed. "Archive" means it **stays on `main` and keeps running**, but gets no new work.

| Component | Decision | Why / how |
| --- | --- | --- |
| `src/sentra/capture.py` (FrameSource) | **Keep** + small extension | Proven camera path for webcams and phone streams. Added: local video files play at their own frame rate and end cleanly (for repeatable tests on consented clips), and file names (not folders) appear in logs. |
| `src/sentra/windows_camera.py` | **Keep** (experimental) | Opt-in Windows camera source; unchanged |
| `src/sentra/pipeline.py` (pose loop) | **Keep** for fall mode; **reuse** | `FpsMeter` is reused by the guide pipeline; the "newest frame, never images on disk" pattern is copied |
| `src/sentra/events.py` (pose contract) | **Archive** | Still used by fall mode. Sentra Guide has its own contract: `src/sentra/guide/schema.py` |
| `src/sentra/posture.py` (fall state machine) | **Archive** | Works and is tested. Not part of the guide demo. Could come back later as an optional "person fell nearby" cue, which would need its own validation. |
| `src/sentra/ui/cv_dashboard.py`, `fake_events.py`, `src/ui_demo.py` | **Archive** | Fall dashboard. Ideas reused (privacy mode, ASCII-only on-screen text, closing the window stops the app) |
| `src/sentra/ui/web_dashboard.py` (stub) | **Adapt later** | Starting point for the phone/browser speech page (Swechha) |
| `src/sentra/evaluation.py`, `src/eval_report.py` | **Archive** | Fall-scenario scoring. Its CSV-to-table approach is the model for guide validation (VALIDATION.md) |
| `src/main.py` | **Keep** | Still launches fall detection, unchanged |
| `src/camera_test.py` | **Keep** | Phase 0 camera check, unchanged |
| `docs/PHASE1_PLAN.md`, `docs/evaluation.md` | **Archive** (banner added) | Fall-detection plan and results. The phone-stream setup notes in evaluation.md remain valid |
| Pose model `yolo26n-pose.pt` | **Keep, optional** | Not loaded by Sentra Guide. Only needed for fall mode |
| — | **Remove later** | Nothing is scheduled for removal. After the hackathon, the team can decide whether fall mode moves to its own branch; that needs everyone's OK |

**Preservation:** fall detection keeps working on `main` (`python src/main.py`), and its 116 tests still pass alongside
the new ones. To mark the exact pre-pivot state, the team can optionally tag it (non-destructive):

```powershell
git tag fall-detection-v1 7de069b
git push origin fall-detection-v1
```

## 4. What the pivot added (first milestone)

```text
src/guide.py                      entry point (CLI)
src/sentra/guide/
  schema.py                       versioned data contract "sentra.guide/1"
  detector.py                     ObjectDetector: yolo26n.pt + ByteTrack, runtime stats
  spatial.py                      SpatialEstimator: left/center/right, central band, severity, edge hysteresis
  depth.py                        DepthProvider interface, NoDepth (default), box_depth() helper for adapters
  alerts.py                       AlertPolicy: persistence, cooldown, priority, wording, honesty rules
  speech.py                       Speaker queue (drops stale/backlog), Windows voice, print fallback
  event_log.py                    JSON-lines alert log + session summary (no images)
  pipeline.py                     GuidePipeline: ties it together, measures fps/latency
  view.py                         OpenCV debug view (regions, boxes, last cue, mute/privacy keys)
tests/test_guide_*.py, guide_helpers.py   41 new tests (157 total)
```

## 5. Missing dependencies and open questions

| Need | Status | Owner |
| --- | --- | --- |
| `yolo26n.pt` (5.3 MB, Ultralytics v8.4.0 release) | ✅ Downloads automatically on first run; git-ignored | — |
| Laptop voice | ✅ Built into Windows (System.Speech via Windows PowerShell 5.1); no new package | — |
| Phone-side speech | ❌ Needs a small local web page (stdlib HTTP + Server-Sent Events, or a websocket library). On iOS, DroidCam must stay in the foreground to stream, so speech on the **same** iPhone is unlikely to work; options: a second phone, Android (IP Webcam can run in the background), or Bluetooth earbuds paired with the laptop | Swechha |
| Android stream with Sentra Guide | ⏳ Untested | Iesa |
| Swechha's iPhone model (LiDAR or not) | ❓ Unknown. LiDAR iPhones: 12 Pro / Pro Max and later Pro models | Swechha |
| Depth Anything V2 Small (metric, indoor) | ❌ Would add `transformers` + ~100 MB weights. Optional | Iesa |
| Record3D (iPhone LiDAR over USB) | ❌ Paid in-app purchase (needs team approval), iTunes on Windows, `pip install record3d` (a Python 3.12 Windows wheel exists). Optional | Babar |
| Submission deadline | ❓ Fill in TEAM_PLAN.md, Section 8 | Babar |
