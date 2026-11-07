# Sentra evaluation results

Owner: babarchechi. Protocol: docs/PHASE1_PLAN.md, Section 11. Consenting teammates only, mat for every drop,
no video recorded (CSV event logs only).

Record each run with `python src/main.py --source 0 --scenario <N>`, then paste the output of
`python src/eval_report.py` over the table below.

**Session date/time:** ____  **Camera position:** ____  **Average fps:** ____  **Thresholds (commit):** ____

| # | Scenario | Runs | Correct alerts | False alarms | Misses | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Walk across the frame (expect no alert) | | | | | |
| 2 | Sit down on a chair (expect no alert) | | | | | |
| 3 | Bend to pick something up (expect no alert) | | | | | |
| 4 | Sit on the floor slowly (expect no alert) | | | | | |
| 5 | Lie down slowly on the mat (expect ON_GROUND, no alert) | | | | | |
| 6 | Controlled quick drop, stay down 8 s (expect alert after ~5 s) | | | | | |
| 7 | Quick drop, get up within 2 s (expect no alert) | | | | | |
| 8 | Two people, one does #6 (expect alert for that ID only) | | | | | |
| 9 | Partly out of frame / behind a chair (expect no alert) | | | | | |

## Phone / IP camera stream

| Date | Phone / app | Network | Command | Result |
| --- | --- | --- | --- | --- |
| 2026-11-06 | iPhone, **DroidCam** ("PC Webcam" mode; no Windows client needed) | iPhone Personal Hotspot (laptop at 172.20.10.2) | `python src/main.py --source "http://172.20.10.1:4747/video"` | ✅ Skeleton and tracking ID work. The stream is MJPEG at 640×480 (free version). |
| 2026-11-06 | Same, phone held in **portrait** | Same | add `--rotate 90` (PR #3) | ✅ People upright with skeletons |
| 2026-11-06 | Android, **IP Webcam** (tested by iesa242) | | | Pending: add results here |

How to set up the iPhone stream:
1. In DroidCam, choose **PC Webcam** and ignore the prompt to install the client.
2. Turn on *Personal Hotspot* with **Maximize Compatibility**, and connect the laptop to it.
3. Use `http://172.20.10.1:4747/video`. That's the iPhone's address on its own hotspot; the "Wi-Fi IP" DroidCam shows also works but changes with the network.

**Not tested yet:** automatic reconnect after the phone app is closed and reopened. It's covered only by fake-camera tests.

## USB camera

Status: not tested yet (camera not available on 2026-11-06). Plug it in and try `--source 1`.

## Observations and known limits

- **Orientation matters for fall detection.** A phone in portrait streams sideways frames, so a standing person looks
  horizontal, as if lying down. Always check people appear upright on screen, and use `--rotate 90` or `270` if not.
- **eduroam** likely blocks phone-to-laptop streams. Use the phone's own hotspot.
- **Keep the app awake:** on the iPhone, DroidCam must stay on screen with Auto-Lock off. iOS pauses the camera when
  the app goes into the background.
