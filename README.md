# Sentra

> **Status: Hackathon prototype (Capital Hacks 5.0). Phase 1 in progress.**
> **Sentra is not a medical device and not an emergency-response or life-safety system.**
> Fall detection is still being built. Today the app tracks people and their body keypoints; it does not flag falls yet.

Sentra is a consent-based visual safety-monitoring prototype. It watches consenting people through a webcam or phone
camera and tracks their body pose. When Phase 1 is finished, it will show **"Possible fall — please check"** when someone
drops quickly and stays down, so a human can confirm or dismiss it.

- **Team plan** (who does what, the shared data contract, Git workflow): [docs/PHASE1_PLAN.md](docs/PHASE1_PLAN.md)
- **Test results:** [docs/evaluation.md](docs/evaluation.md)

## What works today

| Part | Owner | State |
| --- | --- | --- |
| Camera input and pose pipeline (`capture.py`, `pipeline.py`, `src/main.py`) | babarchechi | ✅ Webcams, phone streams, `--rotate`, auto-reconnect, state-change log |
| Windows virtual cameras: Phone Link, DroidCam (`windows_camera.py`) | iesa242 | 🧪 Experimental, opt-in |
| Posture analysis (`posture.py`) | iesa242 | 🚧 In progress. The stub reports everyone as `unknown` |
| Dashboard and alerts (`src/sentra/ui/`) | Swechha03 | 🚧 In progress. The stub shows the video; `q` quits |

```text
Camera (webcam, phone stream, or Windows camera)
   │   capture.py: newest frame only, reconnects, optional rotation
   ▼
YOLO26n-Pose + ByteTrack   pipeline.py
   │   PoseFrame: each person's track ID, 17 keypoints, box
   ▼
PostureAnalyzer   posture.py
   │   PersonStatus: state, confidence, reason
   ▼
Dashboard   ui/: live view, "possible fall" alert, privacy mode
```

## Setup

Use **Python 3.11 or 3.12 (64-bit) from https://www.python.org/downloads/**, not the Microsoft Store build (see
Troubleshooting).

```powershell
git clone https://github.com/babarchechi/sentra.git
cd sentra
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

- **`py` launcher:** if you have it, `py -3.12 -m venv .venv` also works.
- **Activation blocked:** if PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
  first, or skip activation and use `.\.venv\Scripts\python.exe` in place of `python`.
- **Other platforms:** `requirements.txt` is a `pip freeze` from a Windows/CPU machine. The `winrt-*` packages install
  only on Windows. On macOS/Linux or with an NVIDIA GPU, the PyTorch pins may need adjusting.

After `git pull`, run `python -m pip install -r requirements.txt` again in case new packages were added.

## Run the app

From the repo root:

```powershell
python src/main.py --source 0
```

Click the video window and press `q` to quit (or Ctrl+C in the terminal). The pose model (`yolo26n-pose.pt`, 7.5 MB)
downloads automatically the first time.

| Option | What it does |
| --- | --- |
| `--source 0` | Camera to use: a webcam number (`0` built-in, `1` usually a USB camera), a stream URL, or on Windows `phone` / `windows:<name>` (below) |
| `--rotate 90` | Turn frames upright: `0`, `90`, `180` or `270` degrees clockwise. **A phone held in portrait needs 90 or 270.** |
| `--privacy` | Start in skeleton-only privacy mode (the stub shows a black screen for now) |
| `--model` | Pose model file (default `yolo26n-pose.pt`) |

**Always check that people appear upright on screen.** A sideways image makes a standing person look like they're
lying down, which posture analysis will read as a fall.

**Logging:** each run writes state changes (no images) to `outputs/transitions_<date>_<time>.csv`. The `outputs/`
folder is git-ignored.

Tracking IDs are temporary and can change when people overlap or leave and re-enter the frame. Sentra does not identify
individuals.

### Phone camera (tested path: an IP camera app over the phone's hotspot)

Campus Wi-Fi such as eduroam usually blocks phone-to-laptop connections, so **connect the laptop to the phone's own
hotspot**. That also keeps the stream private.

| Phone | App | Command |
| --- | --- | --- |
| iPhone | **DroidCam**: choose *PC Webcam*; no Windows client needed. Turn on Personal Hotspot with **Maximize Compatibility**, and keep the app on screen with Auto-Lock off | `python src/main.py --source "http://172.20.10.1:4747/video" --rotate 90` (drop `--rotate` in landscape) |
| Android | **IP Webcam**: tap *Start server* and use the address it shows | `python src/main.py --source "http://<address it shows>/video"` |

- **The address:** `172.20.10.1` is the iPhone's address on its own hotspot. Otherwise, use the exact address your app
  shows. Before running Sentra, check it in a browser, then close the tab, because these apps often serve only one
  viewer at a time.
- **Keep URLs and passwords private:** never commit stream URLs or passwords, and never expose a camera to the public
  internet.

### USB camera

Plug it in and try `python src/main.py --source 1`. If that's still the built-in camera, try `2`.

### Optional, Windows only: Phone Link or DroidCam as a Windows camera (experimental)

OpenCV can't open Windows virtual cameras such as Phone Link's *Use as a connected camera* or DroidCam's Windows client.
`src/main.py` can read them through the Windows camera API when you ask for them by name. Numbered webcams and URLs are
unaffected.

```powershell
cd src; python -m sentra.windows_camera; cd ..    # list camera names
python src/main.py --source phone                 # Phone Link connected camera
python src/main.py --source "windows:DroidCam"    # any camera, by part of its name
```

Not yet shown working end to end with live video: Phone Link has only delivered its loading spinner in testing. The IP
camera route above is the supported phone path.

### Phase 0 quick check

`python src/camera_test.py --source 0` (add `--track` for IDs) is the original standalone pose test. It's handy for
checking a camera without the rest of the app.

## Tests

```powershell
python -m pytest -q
```

35 tests, all passing on Windows. Without the `winrt` packages (macOS/Linux), the Windows camera tests are skipped. The
tests use fake cameras and fake model results, plus one test that runs the real model on a sample image if
`yolo26n-pose.pt` has been downloaded.

## Tested setup

| Item | Version |
| --- | --- |
| OS | Windows 11 Home (build 26200) |
| Python | 3.12.10, 64-bit, python.org installer |
| ultralytics / torch / opencv-python | 8.4.203 / 2.15.0+cpu (no GPU) / 5.0.0.93 |

| Check (2026-11-06) | Result |
| --- | --- |
| Built-in webcam, `python src/main.py --source 0` | ✅ Skeleton and tracking ID, `q` quits, CSV written |
| iPhone + DroidCam over Personal Hotspot | ✅ Landscape; ✅ portrait with `--rotate 90` |
| Windows camera reader, `windows:HP 5MP` (by iesa242) | ✅ 1280×720, about 26 fps |
| Android + IP Webcam | ⏳ Results pending |
| USB camera | ⏳ Not tested yet |
| Auto-reconnect on a real phone stream | ⏳ Covered by tests with fake cameras only |
| Phone Link connected camera | ⏳ Only the loading spinner so far |

Details in [docs/evaluation.md](docs/evaluation.md).

## Privacy and consent

- Test only with cameras you are authorized to use and people who have agreed to be on camera.
- No real patient, resident, or other non-consenting footage.
- Sentra does **not** record, save, or upload video. Frames are processed in memory and discarded; only state-change
  CSVs are written, to `outputs/`.
- `.gitignore` blocks videos, model weights, `outputs/`, `.env` files, and virtual environments from being committed.

## Troubleshooting

- **`Cannot open camera source`, or OpenCV warns `can't be used to capture by index`:** if your Python came from the
  Microsoft Store, Windows can block it from the camera even when camera access is on. Install Python from python.org
  and recreate the venv. Also check *Settings → Privacy & security → Camera* and close Zoom/Teams.
- **People look sideways or lying down:** add `--rotate 90` (or `270` if they come out upside down).
- **Phone stream won't connect:** use the phone's hotspot rather than campus Wi-Fi, and close any browser tab showing
  the stream. On iPhone, keep DroidCam on screen.
- **Project folder is inside OneDrive:** OneDrive will try to sync the ~1 GB venv. Create the venv outside OneDrive
  instead, e.g. `python -m venv $HOME\.venvs\sentra`, and activate it from there.
- **A test is skipped, or `--track` says `lap` not found:** run `python -m pip install -r requirements.txt` again.

## Project layout

```text
sentra/
├── README.md
├── requirements.txt
├── pyproject.toml               pytest settings
├── docs/
│   ├── PHASE1_PLAN.md           team plan: tracks, contract, workflow, timeline
│   └── evaluation.md            test results
├── src/
│   ├── main.py                  the app
│   ├── ui_demo.py               dashboard on fake events (Track C)
│   ├── camera_test.py           Phase 0 standalone pose test
│   └── sentra/
│       ├── events.py            shared data contract
│       ├── capture.py           camera input
│       ├── windows_camera.py    Windows camera reader (experimental)
│       ├── pipeline.py          pose model + tracking loop, state-change log
│       ├── posture.py           posture analysis (Track A)
│       └── ui/                  dashboards (Track C)
└── tests/
```

## Licensing note

Ultralytics YOLO is licensed under AGPL-3.0 (enterprise licenses are also available). Review licensing, privacy, and
security before any commercial use or deployment with real care facilities.
