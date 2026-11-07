# Windows setup and run commands (Sentra Guide)

Each step says whether it was **tested** on our hardware and when. Tested machine: Windows 11 Home (build 26200),
Python 3.12.10 (python.org, 64-bit), CPU only, ultralytics 8.4.203, torch 2.15.0+cpu, opencv-python 5.0.0.93.

## 1. Python environment (tested 2026-11-06)

Use **Python 3.11 or 3.12 (64-bit) from python.org**, not the Microsoft Store build: Windows can silently block the
Store build from the camera.

```powershell
git clone https://github.com/babarchechi/sentra.git
cd sentra
python -m venv $HOME\.venvs\sentra          # outside OneDrive, so it doesn't sync ~1 GB of packages
& $HOME\.venvs\sentra\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q                          # expect all tests to pass (157 on 2026-11-06)
```

If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, or skip activation and use
`$HOME\.venvs\sentra\Scripts\python.exe` instead of `python`.

After every `git pull`, run `python -m pip install -r requirements.txt` again.

## 2. Model (tested 2026-11-06)

`yolo26n.pt` (5.3 MB) downloads automatically from the official Ultralytics release on the first run of
`src/guide.py`, into the repo folder. It is git-ignored. To fetch it ahead of time (e.g. before going offline):

```powershell
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
```

`--person-check` (experimental) also needs the pose model, `yolo26n-pose.pt` (7.5 MB, the one fall detection uses).
It downloads the same way on first use; fetch it too before going offline:

```powershell
python -c "from ultralytics import YOLO; YOLO('yolo26n-pose.pt')"
```

After that, Sentra Guide needs no internet.

## 3. Camera permissions

*Settings → Privacy & security → Camera*: turn on **Camera access** and **Let desktop apps access your camera**.
Close Zoom/Teams/the Camera app first, since a webcam can usually be used by only one app at a time.

## 4. Laptop voice check (tested 2026-11-06)

Sentra Guide speaks through Windows' built-in voice (no extra package). To check sound on its own:

```powershell
powershell -NoProfile -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('Sentra voice check')"
```

Turn the volume up. Bluetooth earbuds paired with the laptop work too and bring the voice closer to the user.

## 5. Run on the laptop webcam (tested 2026-11-06)

```powershell
python src/guide.py --source 0
```

A cue needs the object in 3 of the last 5 frames, so at 25 fps it can come about 0.1–0.2 s after the object appears
(worked out from the settings, not measured), unless a cooldown is running. Measured on 2026-11-06: 25 fps,
median 29–30 ms model time, 39–44 ms from capture to decision. Click the window, then: **m** mute/unmute, **p** privacy (hide the image),
**q** quit. Each run writes `outputs/guide_events_<time>.jsonl` (alerts and a summary, no images) and prints a summary
line at the end.

Directions are from the camera's point of view. With the webcam facing you, your own right shows up as "to your left".

| Option | What it does |
| --- | --- |
| `--source` | `0` webcam, `1` USB camera, a stream URL, or a consented local video file |
| `--rotate 90` | Turn frames upright (phone in portrait: `90` or `270`). Left/right are wrong on a sideways image |
| `--voice print` / `off` | Print cues instead of speaking them / no speech |
| `--muted`, `--privacy` | Start muted / with the image hidden |
| `--no-view --max-seconds 60` | No window, stop after 60 s (cues go to the console) |
| `--repeat-after 10`, `--min-gap 2.5` | Cooldown for the same cue / minimum gap between any two cues (seconds) |
| `--announce-conf 0.5`, `--conf 0.25` | Score needed to name an object / detector threshold |
| `--all-labels` | Announce every class the model knows, not just the obstacle list |
| `--imgsz 480`, `--no-track` | Faster model input / no tracking IDs |
| `--label t3` | Add a tag to the log file name (for VALIDATION.md runs) |

## 6. Phone as the camera

**Network:** on 2026-11-06 the iPhone stream worked over **campus eduroam** using the "WiFi IP" DroidCam shows
(phone and laptop were on different 10.x subnets). Other networks, including eduroam at other times or places, may
block it. The phone's **own hotspot** is the reliable fallback and is more private: on a campus network, other devices
that can reach the phone could open the stream too. Before running Sentra, open the stream URL in a browser to check
it, then **close that tab**: these apps usually serve one viewer at a time.

### iPhone: Babar's 14 Pro Max, Swechha's iPhone (DroidCam)

Tested 2026-11-06 with Sentra Guide on Babar's iPhone 14 Pro Max: phone in portrait + `--rotate 90`, over eduroam,
24–29 fps, ~40 ms capture-to-decision, cues and directions confirmed. (Also tested earlier with the fall app over the
iPhone hotspot.)

1. Install **DroidCam** from the App Store and choose **PC Webcam**. Ignore the prompt to install the Windows client;
   it isn't needed.
2. Either put the laptop on the same Wi-Fi as the phone (worked on eduroam), or turn on the iPhone's Personal Hotspot
   with **Maximize Compatibility** and connect the laptop to it.
3. Keep DroidCam **on screen**, with Auto-Lock off (Settings → Display & Brightness). iOS pauses the camera in the
   background.
4. Run:

   ```powershell
   python src/guide.py --source "http://<WiFi IP DroidCam shows>:4747/video" --rotate 90  # phone upright (portrait)
   python src/guide.py --source "http://172.20.10.1:4747/video" --rotate 90               # same, on the iPhone hotspot
   ```

   Drop `--rotate 90` if you hold the phone in landscape. **Holding it upright without `--rotate 90` gives a sideways
   picture** (seen 2026-11-06), and then left/right are meaningless. `172.20.10.1` is the iPhone's address on its own
   hotspot. The free DroidCam stream is 640×480 MJPEG.

### Android: Iesa (IP Webcam)

**Not yet tested by the team** (TEAM_PLAN.md, Iesa task 1).

1. Install **IP Webcam** (Pavel Khlebovich) from Google Play, then tap **Start server** at the bottom.
2. Turn on the phone's hotspot and connect the laptop to it (or use a network that doesn't isolate devices).
3. Use the address the app shows, plus `/video`:

   ```powershell
   python src/guide.py --source "http://<address it shows>:8080/video"
   ```

4. If the image is sideways, add `--rotate 90` (or `270`). Lower the app's video resolution to 640×480 if it lags.

### LiDAR depth (optional, not set up)

See [DEPTH_OPTIONS.md](DEPTH_OPTIONS.md). Nothing about depth is needed for the demo.

## 7. Other ways to run

| Command | Status |
| --- | --- |
| `python src/guide.py --source my_clip.mp4 --no-view --voice print` (a **consented** recording, kept out of git) | Tested 2026-11-06 on a synthetic clip made from Ultralytics' sample image: plays at the clip's frame rate, then stops |
| `python src/guide.py --source 1` (USB camera) | Not tested (no USB camera yet) |
| `python src/main.py --source 0` (fall detection, archived) | Tested 2026-11-06, unchanged |
| `python src/guide.py --source 0 --voice off --web --web-host <laptop IP on the phone hotspot>`, then open `http://<that IP>:8765/` on a phone and tap **Tap to turn on sound** | Phone speech page. Tested 2026-11-07 on an iPhone in Safari over its own hotspot: cues spoken (VALIDATION.md, Section 3). If the phone can't load the page, allow Python through Windows Firewall for Private networks. Run `ipconfig` to find the laptop's hotspot IP (iPhone hotspot: `172.20.10.x`). Without `--web` Sentra Guide opens no ports |

## 8. Troubleshooting

| Problem | Fix |
| --- | --- |
| `Cannot open camera source` | Wrong index/URL, camera busy, or Microsoft Store Python (see Section 1). For phones: same network? browser tab closed? app on screen? |
| No voice | Run the voice check in Section 4. If it fails, use `--voice print`. The app logs "Voice ... unavailable; printing alerts instead" and keeps running |
| Left/right seem wrong | Make sure the image is upright (`--rotate`). With a webcam facing you, directions are mirrored relative to you |
| Talks too much | `--repeat-after 20 --min-gap 4`, or press **m**. Report what was noisy in VALIDATION.md |
| "LAGGING" on screen, or low fps | Try `--imgsz 480`. Close other heavy apps. A stale frame never triggers a cue |
| "Waiting for frames from the camera..." | The stream stalled. It reconnects automatically with backoff (1, 2, 4, then every 5 s). Each attempt on an unreachable stream takes ~14 s, so after a 10 s outage recovery took 9 s in testing and can take up to ~20 s. Keep the app running |
| `yolo26n.pt` download fails | You're offline. Download it once while online (Section 2) |
| Project inside OneDrive is slow | Keep the venv outside OneDrive (Section 1) |
