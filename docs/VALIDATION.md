# Sentra Guide validation

Owner: Swechha (everyone adds their own measurements). **Record only what was actually measured or said.** Blank is
better than a guess. Each run's `outputs/guide_events_*.jsonl` ends with a `session_summary` line (frames, fps,
median latency, alert count): copy numbers from there.

Rules for every test: consenting teammates only, seated or standing still or walking slowly across a **fixed** camera
in a cleared indoor space. No blindfolds, no walking guided by the app. No video is recorded unless everyone in frame
agreed and it's needed; recordings stay out of git (`outputs/`, `*.mp4` are ignored).

## 1. Metrics

| Metric | How it's measured |
| --- | --- |
| Detection rate | Per scene: objects present vs. objects detected (box shown with the right label) |
| Direction correctness | Cues whose left/ahead/right matches where the object really was, relative to the camera |
| Capture-to-decision latency | `median_latency_ms` in the session summary (camera frame → cue decided) |
| Speech delay | Stopwatch or screen recording: cue logged → voice starts (not automated yet) |
| fps | `fps` in the session summary |
| Duplicate cues | The same text for the same object within the cooldown (should be 0), plus a subjective "too chatty?" |
| Missed cues | Something relevant in the central band for > 3 s with no cue |
| False cues | A cue for something that isn't there, or with the wrong label |
| Depth error (optional) | Estimated vs. tape-measured meters, per distance (Section 4) |

## 2. Performance on our laptop

Laptop: Windows 11 Home, CPU only. Model `yolo26n.pt`, `--imgsz 640`, ByteTrack on, laptop voice.

| Date | Source | Who | Frames / s | fps | Median model ms | Median capture→decision ms | Cues | Stale frames | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-06 | Synthetic 15 fps clip from Ultralytics `bus.jpg` (panning) | Claude | 179 / 12.0 | 14.9 (clip-limited) | 29.5 | 29.7 | 5 | 0 | `--repeat-after 4`. Pipeline smoke test, not a real scene |
| 2026-11-06 | Laptop webcam, Babar moving around | Babar | 1689 / 66.4 | 25.4 | 29.7 | 44.1 | 18 | 0 | Cues included person left/ahead/right, "TV" (a screen), "Possible obstacle ahead." ×1 |
| 2026-11-06 | Laptop webcam, voice check | Babar | 1247 / 49.5 | 25.2 | 29.2 | 39.2 | 16 | 0 | Babar heard the cues; directions matched the screen. "Possible obstacle ahead." ×2. Gaps between cues 2.5–5.3 s while moving (chatty, see Section 5) |
| 2026-11-06 | iPhone 14 Pro Max, DroidCam over campus eduroam, landscape image but phone in portrait (sideways) | Babar | 1490 / 52.0 | 28.7 | 27 | 39 | 15 | 0 | Stream fine but the picture was sideways, so directions weren't meaningful: rerun with `--rotate 90` |
| 2026-11-06 | iPhone 14 Pro Max, DroidCam over eduroam, portrait, `--rotate 90` | Babar | 1323 / 56.1 | 23.6 | 26.8 | 42.9 | 15 | 0 | ✅ Upright; Babar confirmed cues and directions. Chair announced ahead, left and right; people incl. "2 people ahead/left/right"; "Possible obstacle ahead." ×1; "TV to your left." ×1 |
| | Android (IP Webcam) | Iesa | | | | | | | |
| 2026-11-06 | `src/guide_bench.py` at 640 / 480 / 320, synthetic 30 fps `bus.jpg` clip, **laptop on battery** | Iesa | 200 per setting | 15–16 (640), 19–21 (480), 29–30 (320, clip-limited) | 61–65 (640), 48–51 (480), 33–35 (320) | 78–81 (640), 65–71 (480), 42–52 (320) | n/a | 0 | Full table and notes in Section 2.1. Plugged-in run still to do |
| 2026-11-07 | `src/guide_bench.py` at 640 / 480 / 320, synthetic 30 fps `bus.jpg` clip, **demo laptop (Babar's), plugged in** | Claude | 200 per setting | 29.1–30 (all sizes clip-limited) | 31–33 (640), 25 (480), 17 (320) | 34–42 (640), 25 (480), 17 (320) | n/a | 0 | Full table in Section 2.1. 640 with tracking keeps up with a 30 fps source on this laptop |
| 2026-11-07 | Laptop webcam, 30 s headless check of `main` after merging #10 (`--no-view --voice print --max-seconds 30`) | Claude | 856 / 30.0 | 28.5 | 31 | 32 | 3 | 0 | "Person ahead." ×3, 10 s apart (cooldown). Plugged in |

### 2.1 Benchmark: model size and tracking (`src/guide_bench.py`)

2026-11-06, Iesa's laptop: Windows 11 Home, Intel64 Family 6 Model 189, CPU only, Python 3.12.10, **on battery**
(Balanced power plan). Source: a synthetic 40 s, 30 fps clip made from Ultralytics' `bus.jpg` (810 × 1080, slowly
panning; a bus and 4 people), so fps can't go above 30. Each row: 20 warm-up frames, then 200 measured frames through
the same loop as `src/guide.py` (no voice, no window).

Command: `python src/guide_bench.py --source bus_30fps_40s.mp4`

| imgsz | Tracking | Frames / s | fps | Model ms median / p90 | Capture→decision ms median / p90 | Objects / frame | Stale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 640 | on | 200 / 13.28 | 15 | 64.7 / 75.9 | 81 / 96.8 | 4.7 | 0 |
| 640 | off | 200 / 12.37 | 16.1 | 61.4 / 65.7 | 77.8 / 91.5 | 4.8 | 0 |
| 480 | on | 200 / 10.42 | 19.1 | 51 / 57.9 | 70.5 / 85.5 | 4.5 | 0 |
| 480 | off | 200 / 9.69 | 20.5 | 48.2 / 51.3 | 65 / 83.9 | 4.7 | 0 |
| 320 | on | 200 / 6.98 | 28.5 | 34.5 / 38 | 51.9 / 64 | 3.8 | 0 |
| 320 | off | 200 / 6.68 | 29.8 | 32.9 / 35.4 | 41.5 / 57.9 | 4 | 0 |

What it shows:
- **Model size is the main lever.** 320 is about twice as fast as 640 and keeps up with a 30 fps source. Tracking
  costs only 1–3 ms per frame.
- **Battery makes it slow and unpredictable.** Five minutes before this run, the same 640 setting measured 101 ms
  median (and 102 ms on a 10 fps clip), against ~30 ms earlier the same day (power state not recorded) and Babar's
  29 ms. **Plug the laptop in for the demo.** If it must run on battery, `--imgsz 320` is the safer choice.
- **320 finds fewer objects** (3.8–4 per frame vs 4.7–4.8 at 640 on this clip): small or far things drop out. Check on
  real scenes before making it the default.
- No stale frames in any setting.

Still to do: the same run plugged in, and on the Android stream (camera tests postponed).

#### Demo laptop (Babar's), plugged in

2026-11-07, run by Claude on Babar's laptop (the one that runs the demo): Windows 11 Home, AMD64 Family 25 Model 68,
CPU only, Python 3.12.10, **on AC power** (Balanced power plan). Source: the same kind of clip as above (40 s, 30 fps
slow pan over `bus.jpg`, 810 × 1080), rebuilt locally, so fps can't go above 30. 20 warm-up + 200 measured frames per
row, `src/guide_bench.py` as merged in #10.

Command: `python src/guide_bench.py --source bus_30fps_40s.mp4 --model yolo26n.pt`

| imgsz | Tracking | Frames / s | fps | Model ms median / p90 | Capture→decision ms median / p90 | Objects / frame | Stale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 640 | on | 200 / 6.84 | 29.1 | 33.3 / 37 | 42.2 / 60.6 | 4.4 | 0 |
| 640 | off | 200 / 6.62 | 30 | 31.4 / 33.6 | 33.9 / 40.3 | 4.5 | 0 |
| 480 | on | 200 / 6.63 | 30 | 25.1 / 28.4 | 25.3 / 30 | 4.3 | 0 |
| 480 | off | 200 / 6.64 | 30 | 25 / 27.4 | 25.5 / 28.1 | 4.4 | 0 |
| 320 | on | 200 / 6.62 | 30 | 16.8 / 19.3 | 17 / 19.6 | 3.7 | 0 |
| 320 | off | 200 / 6.64 | 30 | 16.6 / 18.7 | 16.9 / 18.9 | 3.8 | 0 |

What it shows:
- **Plugged in, this laptop is about twice as fast as Iesa's battery run**, and matches the ~27–30 ms model time in the
  Section 2 webcam and iPhone runs.
- **640 with tracking keeps up with a 30 fps camera here** (33 ms per frame), so it stays the demo default. With every
  setting at the 30 fps cap, this clip can't show how much faster 480 or 320 would run.
- Capture→decision at 640 with tracking (42 ms median, 61 ms p90) is higher than without (34 / 40 ms). Unconfirmed
  guess: at ~33 ms per frame the loop is right at the camera's frame interval, so a frame sometimes waits for the
  previous one to finish.
- Battery fallback stays `--imgsz 320`, as above.

### 2.2 Cost of `--person-check` (pose model confirms people)

Same 30 fps `bus.jpg` clip, 640, 20 + 200 frames (Iesa's laptop, 2026-11-07).

| Run | Power | Tracking | fps | Capture→decision median | Pose model ran on |
| --- | --- | --- | --- | --- | --- |
| no check | plugged in | on | 27.6 | 50 ms | – |
| `--person-check` | plugged in | on | 27.8 | 34 ms | 13 of 220 frames (36 ms each) |
| no check | battery | off | 14.6 | 87 ms | – |
| `--person-check`, first version (checked every frame) | battery | off | 7.2 | 154 ms | 220 of 220 |
| `--person-check`, reusing verdicts for boxes in nearly the same place | battery | off | 12.4 | 85 ms | 56 of 220 |

With tracking it's free (babar measured the same on the demo laptop: 29 fps, 9 of 220 frames). Without tracking
(`--no-track`, the fallback) it costs about 15% fps instead of half. Rows on battery only compare with each other.
A rejected "person" (a hand, or someone cut off at the edge) is re-checked every 0.5 s, so with one in view the pose
model runs about twice a second.

## 3. Devices

| Date | Phone | App / connection | Orientation | Result |
| --- | --- | --- | --- | --- |
| 2026-11-06 | Laptop webcam | built-in | n/a | ✅ Cues heard, directions correct (Babar) |
| 2026-11-06 | Babar, iPhone 14 Pro Max | DroidCam PC Webcam, iPhone hotspot | landscape + portrait | ✅ with the fall app (`src/main.py`) |
| 2026-11-06 | Babar, iPhone 14 Pro Max | DroidCam PC Webcam, **campus eduroam** (phone's Wi-Fi IP, port 4747) | portrait + `--rotate 90` | ✅ Sentra Guide: cues heard, directions correct. Without `--rotate` the picture is sideways |
| | Iesa, Android (model: ____) | IP Webcam | | ⏳ |
| | Swechha, iPhone (model: ____, LiDAR: ____) | DroidCam | | ⏳ |
| 2026-11-06 | Disconnect test, **simulated on the laptop**: a relay between the app and Babar's iPhone stream cut the connection and refused new ones for 10 s | DroidCam over eduroam, via `127.0.0.1` relay | portrait + `--rotate 90` | ✅ No crash. Outage noticed within 1 s; **reconnected 9 s after the stream came back** (19 s after the cut); frames resumed at full rate. ⚠ Each failed reopen attempt took ~14 s, so recovery can take 10–20 s. No new cues after the reconnect (likely nothing new in view; not checked) |
| | Disconnect test on the phone itself (close DroidCam 10 s) and Wi-Fi dropping out (no clean disconnect) | | | ⏳ Not tested |
| 2026-11-07 | **Phone speech page, laptop only:** `python src/guide.py --source 0 --no-view --voice off --web` for 75 s, page opened in a browser on the same laptop (`http://127.0.0.1:8765/`) | laptop webcam, page on 127.0.0.1 | n/a | ✅ All 4 cues after the page connected appeared on it **in the same second** as in the event log (log `wall_time` vs page clock, 1 s resolution); the cue from before it connected showed as "earlier", unspoken. Page said "No signal from the laptop" within ~4 s of the app stopping. 21 fps, 61 ms median capture-to-decision with the page on. Speech was triggered after "Tap to turn on sound" but **not listened to** (Swechha) |
| 2026-11-07 | **Phone speech page on a phone:** Swechha's iPhone (model not noted), page in Safari over the iPhone's own Personal Hotspot; laptop at 172.20.10.2 running `python src/guide.py --source 0 --voice off --web --web-host 172.20.10.2` | laptop webcam as the camera; page at `http://172.20.10.2:8765/` | n/a | ✅ "Tap to turn on sound" worked and **the iPhone spoke the cues** (Swechha). Page connected from 172.20.10.1 and stayed connected ~1.5 min; 24 cues sent in that time. No hotspot warning (172.20.10.x is recognised). 17.7 fps, 66 ms median capture-to-decision with the page on. ⚠ Delay from laptop log to speech **not timed** with a stopwatch; screen-lock behaviour not tried. Chatty: 24 cues in ~97 s (≈ 1 per 4 s), incl. "Refrigerator to your left." ×5, likely a false detection: input for the alert-quality pass (Section 5) |
| | Phone speech page on Android Chrome, and with DroidCam as the camera (page on a second phone) | | | ⏳ Not tested |

## 4. Depth error (only if a depth adapter exists)

Tape-measure from the camera lens to the front of the object. 3 readings at each distance, object in the centre and
then at the edge.

| Date | Adapter | Object | True m | Estimated m (3 readings) | Error | Valid depth % | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-07 | yolo26n-depth, **uncalibrated**, box median | chair (Babar's iPhone, DroidCam, portrait, eduroam) | 0.99 | 5.4–6.2 (medians of 3 runs) | **+450–530%** | ~100 | First runs picked far-away chairs; with only the test chair in view still 6.0 m. Floor just in front of the phone read 2.4–3.2 m (should be ~1–1.5) |
| 2026-11-07 | same, nearest 20% of box | chair | 0.99 | 6.22, then **3.05** after the phone was moved | +529% / +208% | ~100 | The correction factor this implies changed from 0.159 to 0.325 just by moving the phone: **calibration only holds for one setup** |
| 2026-11-07 | same, **calibrated** at 0.99 m (`--scale 0.325`), phone fixed | chair | 2.03 (80 in) | 2.41 (10–90%: 2.23–2.57) | **+0.38 m (+19%)** | ~100 | |
| 2026-11-07 | same, calibrated, phone fixed | chair | 3.00 (118 in) | 3.42 (10–90%: 3.22–3.66) | **+0.42 m (+14%)** | ~100 | Consistently ~0.4 m too far |
| | | person | 2.0 | | | | |

**Verdict (2026-11-07):** uncalibrated estimates are unusable (3–6x too far). With a 1-point calibration for that exact
phone position, errors were 14–19% at 2–3 m: good enough to **show on screen as "est."**, not to speak (the spoken
bar is about 10 cm or 10%). Recalibrate whenever the phone or scene changes:
`python src/depth_check.py --source <camera> --rotate 90 --true <meters> --calibrate`, then run Sentra Guide with
`--depth estimate --depth-scale <factor>`. Readings are the median of ~70 frames over 8 s, one object in view.

Pass/fail thresholds are in DEPTH_OPTIONS.md ("Decision rule"). Agree on them before measuring.

## 5. Controlled scenes (fixed camera on a stand or table, ~1 m high, landscape)

Run each with `python src/guide.py --source <camera> --label <scene>` so its log is easy to find.

| # | Scene | Expected | Runs | Detected | Direction correct | Duplicates | Missed | False | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | Chair in the centre, 2 m | "Chair ahead." once, then at most every 10 s | | | | | | | |
| S2 | Move the chair from centre to left, then right | "Chair to your left." / "...right." after each move | | | | | | | |
| S3 | Person walks slowly left → right across the frame | "Person to your left." → "...ahead." → "...right." (some may be skipped by the 2.5 s gap) | | | | | | | |
| S4 | Two people, one ahead and one to the side | Ahead first | | | | | | | |
| S5 | Person sitting on a chair, centre | Person and/or chair, not a flood of cues | | | | | | | |
| S6 | Box/backpack/table (things outside or at the edge of the class list) | Note what is and isn't said | | | | | | | |
| S7 | Poster or photo of a person | Note whether it's announced as a person (expected limitation) | | | | | | | |
| S8 | Object at the very edge of the frame | Note behaviour | | | | | | | |
| S9 | Dim light / backlight from a window | Note detection drop | | | | | | | |
| S10 | Camera moving slowly (handheld pan), static room | Note false "new" cues from movement | | | | | | | |
| S11 | Depth off: check the screen says "Distance: off" and no cue has a number | No numbers | | | | | | | |
| S12 | Stop the phone stream for 10 s | No crash, reconnects, cues resume | | | | | | | |

**Alert-quality log** (Swechha task 2): for each cue that felt wrong, write the time, the text, and why (late,
repeated, irrelevant, wrong side, wrong label). Then list the `AlertConfig` change it led to.

| Time | Cue | Problem | Change made |
| --- | --- | --- | --- |
| 2026-11-06 | (webcam, someone moving around) | A cue every 2.5–5 s while moving: likely too chatty for real use | To review: longer `min_gap_s` while lots is changing? Only announce sides once? |

## 6. User and expert feedback

Ask for feedback on a **seated tabletop or recorded demo**, never on walking with the app. Tell people it's an
experimental prototype, not a mobility aid. Ask permission before writing down anything identifying; first names or
"participant 1" are enough.

Questions to ask:
1. What information about your surroundings would actually help, and what would be noise?
2. How often is too often for spoken cues? Would tones, vibration, or a different voice be better?
3. Left/ahead/right: useful? Would clock directions ("chair at 2 o'clock") or distances matter more?
4. Where should the phone be (hand, chest mount, lanyard)? Where should the sound come from (phone speaker, earbuds,
   bone conduction)?
5. Privacy: how do you feel about a camera pointed at bystanders, even with nothing stored?
6. What would make you trust it less or more?

| Date | Who (role, with consent) | Demo shown | What they said (close to their words) | What we'll change |
| --- | --- | --- | --- | --- |
| | | | | |

Keep "interested in the idea" separate from evidence that it helps people get around. We have no evidence of the
second.

## 7. What the model can't see

YOLO26n detects the 80 COCO classes. It **does not** detect: glass doors and walls, steps, stairs, curbs and
drop-offs, poles and posts (except the few named classes, like fire hydrant and stop sign), low-hanging branches and
signs, wet floors, holes, cables, or most furniture outside its list. It can also mistake screens for a TV, posters for
people, and so on. **Silence never means the way is clear.** That limitation goes in the pitch.

**Hands and arms as "person"** (2026-11-07, Iesa): the object model calls the camera holder's own hands and arms
"person" with scores as high as a real person's (0.51–0.87 vs 0.86–0.92), so "Person to your right." can be your own
hand. `--person-check` (experimental, off by default) asks the pose model for a head, shoulders, hips or legs before
saying "person". A box with no body seen is never named: off to the side it says nothing, and large and central it
becomes "Possible obstacle ahead.". Known limits with the check on:
- someone almost entirely out of frame (only an arm at the edge) isn't named;
- someone right in front of the camera with only chest to waist in view may have no body found, so they get
  "Possible obstacle ahead." rather than "Person ahead." (babar's `bus.jpg` crop: score 0.41–0.73 → obstacle);
- not yet shown live on a real hand (the 2026-11-07 laptop run had no hand detected as a person at all).
