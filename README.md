# Sentra Guide

> **Status: hackathon prototype (Capital Hacks 5.0, Health + Accessibility Technology, Nov 6–8, 2026).**
> **Experimental supplementary awareness only. Not a navigation aid, not a replacement for a cane or guide dog, and it
> never tells anyone a path is clear or safe.**

Sentra Guide uses a forward-facing phone camera to notice common objects and people, works out whether each is to the
left, ahead, or to the right, and speaks short, rate-limited cues such as **"Chair ahead."** or **"Person to your
right."** It speaks a distance only if a depth source has passed tape-measure tests. None has yet, so today it never
says a number.

The project started as **Sentra**, a fall-detection prototype. That app still works and is kept
(`python src/main.py`); see [docs/PIVOT.md](docs/PIVOT.md).

## What works right now

| Part | State |
| --- | --- |
| Laptop webcam → YOLO26n objects → left/ahead/right → laptop voice | ✅ Verified 2026-11-06: cues heard, directions correct, ~25 fps, ~40 ms capture-to-decision |
| Cooldown, persistence, priority, "possible obstacle", stale-frame rule, no invented distances | ✅ Unit-tested (41 new tests, 157 total) |
| Debug window (regions, boxes, last cue, mute, privacy) and JSON-lines event log | ✅ |
| Consented local video files as input (repeatable tests) | ✅ |
| iPhone stream (DroidCam) | ✅ Verified with Sentra Guide 2026-11-06 (portrait + `--rotate 90`, campus Wi-Fi): cues and directions correct, 24–29 fps |
| Android stream (IP Webcam) | ⏳ Not tested yet |
| Speech on the phone (browser page, `--web`) | ✅ Verified 2026-11-07 on an iPhone (Safari, iPhone hotspot): cues spoken on the phone. Delay not yet timed; Android not tested |
| Depth: Depth Anything V2 (estimated) / Record3D iPhone LiDAR | ❌ Optional experiments, not started ([docs/DEPTH_OPTIONS.md](docs/DEPTH_OPTIONS.md)) |

## Quick start (Windows)

Use Python 3.11/3.12 from python.org (not the Microsoft Store build). Full instructions, phone setup and
troubleshooting: **[docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md)**.

```powershell
git clone https://github.com/babarchechi/sentra.git
cd sentra
python -m venv $HOME\.venvs\sentra
& $HOME\.venvs\sentra\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python src/guide.py --source 0                                        # laptop webcam
python src/guide.py --source "http://<DroidCam WiFi IP>:4747/video" --rotate 90   # iPhone held upright
```

`yolo26n.pt` (5.3 MB) downloads on the first run. Click the window, then **m** mute, **p** privacy, **q** quit.
Directions are from the camera's point of view.

## How each teammate starts

Full plan with tasks, acceptance tests and file ownership: **[docs/TEAM_PLAN.md](docs/TEAM_PLAN.md)**.

| Who | Branch | Start with |
| --- | --- | --- |
| Babar | `feature/phone-and-depth` | Run Sentra Guide on the iPhone stream; disconnect test; Record3D preflight (no purchase without OK) |
| Iesa | `feature/object-perception` | Run on the Android stream; `src/guide_bench.py` FPS/latency benchmark |
| Swechha | `feature/audio-and-validation` | Run on her iPhone; alert-quality pass and tuning; phone speech page |

Code goes through a PR approved by another teammate; docs go straight to `main`. Changes to
`src/sentra/guide/schema.py` (the shared contract) need both other teammates.

## How it works

```text
phone / webcam ──► FrameSource (newest frame only, reconnects)
               ──► YOLO26n + ByteTrack (yolo26n.pt)
               ──► depth (none by default) ──► left / ahead / right
               ──► AlertPolicy (persistence, cooldown, priority, no "clear"/"safe")
               ──► laptop voice + JSON-lines log + debug window
```

Details and the event schema: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Docs

| File | What's in it |
| --- | --- |
| [docs/TEAM_PLAN.md](docs/TEAM_PLAN.md) | Who does what, Today/Next/Stretch, acceptance tests, merge rules, timeline |
| [docs/PIVOT.md](docs/PIVOT.md) | Repo audit at the pivot, verified vs. unverified, keep/adapt/archive matrix |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Pipeline, threads, alert rules, `sentra.guide/1` schema |
| [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md) | Tested setup and run commands, iPhone/Android steps, troubleshooting |
| [docs/DEPTH_OPTIONS.md](docs/DEPTH_OPTIONS.md) | Camera-only vs. Depth Anything V2 vs. Record3D, with a decision rule |
| [docs/VALIDATION.md](docs/VALIDATION.md) | Metrics, measured results so far, test scenes, user-feedback log |
| [docs/SAFETY_AND_PRIVACY.md](docs/SAFETY_AND_PRIVACY.md) | Claims we make and don't make, testing rules, privacy |
| [docs/PHASE1_PLAN.md](docs/PHASE1_PLAN.md), [docs/evaluation.md](docs/evaluation.md) | Archived fall-detection plan and results |

## Limitations

- The model knows 80 everyday classes. It **does not see** glass doors, steps, curbs, drop-offs, poles or overhanging
  objects. Silence never means the way is clear.
- Left/ahead/right are screen regions, not a measured walking path. A tilted or turned phone changes them.
- On a handheld, moving phone, movement in the image is not reliable evidence of anything approaching.
- The laptop speaks, not the phone, which is fine for a controlled demo but not for real use.
- No user testing yet. See [docs/VALIDATION.md](docs/VALIDATION.md).

## Next steps

Verify all three phones → tune the cues with logged evidence → measure fps/latency properly → phone-side speech →
optional depth only if it passes the tape test → validation session → demo rehearsal.

## Project layout

```text
src/
  guide.py                 Sentra Guide entry point
  main.py                  fall detection (archived, still works)
  camera_test.py           Phase 0 camera check
  sentra/
    capture.py             camera input (shared)
    guide/                 Sentra Guide: schema, detector, spatial, depth, alerts, speech, event_log, pipeline, view
    pipeline.py, posture.py, events.py, evaluation.py, ui/    fall detection (archived)
tests/                     157 tests (python -m pytest -q)
docs/
```

## Privacy and licensing

Frames are processed in memory on the laptop and never saved or uploaded; logs contain labels and timing only. Test
only with people who have agreed to be on camera. More in [docs/SAFETY_AND_PRIVACY.md](docs/SAFETY_AND_PRIVACY.md).
Ultralytics YOLO is AGPL-3.0. Review licensing before any use beyond this hackathon.
