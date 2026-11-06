# Sentra

> **Status: Hackathon prototype (Capital Hacks 5.0) — initial pose-tracking setup only.**
> **Sentra is not a medical device and not an emergency-response or life-safety system.**
> Nothing in this repository detects falls yet. The pose model outputs body keypoints, not medical-grade fall detection.

Sentra is a consent-based visual safety-monitoring prototype. Phase 0 (this commit) only sets up the development
environment and a small camera test that draws YOLO26n-Pose skeletal keypoints on a live webcam feed.

## Tested setup (Phase 0, 2026-11-06)

| Item | Version / result |
| --- | --- |
| OS | Windows 11 Home (build 26200), built-in laptop webcam (640×480) |
| Python | 3.12.10, 64-bit, **python.org installer** (see Troubleshooting) |
| ultralytics | 8.4.203 |
| torch | 2.15.0+cpu (no GPU / CUDA used) |
| opencv-python | 5.0.0.93 |
| lap (needed by ByteTrack) | 0.5.14 |
| Pose model | `yolo26n-pose.pt` — downloads automatically on first run (7.5 MB) |

| Test | Result |
| --- | --- |
| `import ultralytics, cv2, torch` | OK |
| Model load + sample image (`bus.jpg`) | OK — task `pose`, 5 people, 17 keypoints each, ~69 ms/frame on CPU |
| Webcam capture (index 0) | OK — 640×480, ~33 fps raw |
| `python src/camera_test.py --source 0` | OK — skeleton overlay visible, `q` quits cleanly |
| `python src/camera_test.py --source 0 --track` | OK — per-person tracking IDs shown, `q` quits cleanly |
| Phone / IP camera stream | **Not tested yet (pending)** |

## Setup

Use **Python 3.11 or 3.12 (64-bit) from https://www.python.org/downloads/** — not the Microsoft Store build (see Troubleshooting).

```powershell
git clone https://github.com/babarchechi/sentra.git
cd sentra
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you have the `py` launcher you can use `py -3.12 -m venv .venv` instead. If PowerShell blocks activation, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or skip activation and use `.\.venv\Scripts\python.exe`
in place of `python`.

`requirements.txt` is a `pip freeze` from the Windows/CPU machine above. On macOS/Linux or with an NVIDIA GPU the
PyTorch pins may need adjusting.

## Run the camera test

```powershell
python src/camera_test.py --source 0            # local webcam, pose keypoints
python src/camera_test.py --source 0 --track    # plus ByteTrack per-person IDs
```

Click the video window and press `q` to quit. CPU inference is slower than the raw camera rate; that is expected.

Tracking IDs are temporary and can change when people overlap or leave and re-enter the frame. Sentra does not
identify individuals.

### Optional: phone or IP camera (only after the webcam works)

Use an IP-camera app on a phone you own, on the same trusted Wi-Fi, and pass the exact URL the app shows:

```powershell
python src/camera_test.py --source "http://PHONE_IP:PORT/ACTUAL_VIDEO_PATH"
python src/camera_test.py --source "rtsp://PHONE_IP:PORT/ACTUAL_STREAM_PATH"
```

These are placeholders. Never commit stream URLs or credentials, and never expose a camera to the public internet.

## Privacy and consent

- Test only with cameras you are authorized to use and people who have agreed to be on camera.
- No real patient, resident, or other non-consenting footage.
- The camera test does **not** record, save, or upload video; frames are processed in memory and discarded.
- `.gitignore` blocks videos, model weights, `.env` files, and virtual environments from being committed.

## Troubleshooting

- **`Cannot open camera source` / OpenCV warns `can't be used to capture by index`:** if your Python came from the
  Microsoft Store, Windows can block it from the camera even when camera access is on. Install Python from python.org
  and recreate the venv. Also check *Settings → Privacy & security → Camera* and close Zoom/Teams.
- **Project folder is inside OneDrive:** OneDrive will try to sync the ~1 GB venv. Create the venv outside OneDrive
  instead, e.g. `python -m venv $HOME\.venvs\sentra`, and activate it from there.
- **`--track` first run says `lap` not found:** run `python -m pip install -r requirements.txt` again; `lap` is pinned there.

## Project layout

```text
sentra/
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   └── camera_test.py   # webcam / stream pose test (no recording, no fall detection)
└── docs/
```

## Next-phase backlog (planned, not implemented)

- Posture and motion analysis from keypoints (e.g. candidate "person on floor" signals) — prototype only, no medical claims
- Phone / IP camera validation
- Evaluation plan with consenting participants and staged scenarios
- Alerting design (who is notified, false-positive handling, human confirmation)
- Simple local UI / dashboard

Team practice: keep `main` stable and work on branches such as `feature/camera-input`, `feature/posture-analysis`,
`feature/ui`, merged through pull requests.

## Licensing note

Ultralytics YOLO is licensed under AGPL-3.0 (enterprise licenses are also available). Review licensing, privacy, and
security before any commercial use or deployment with real care facilities.
