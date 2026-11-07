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
| | iPhone (DroidCam) | Babar | | | | | | | |
| | Android (IP Webcam) | Iesa | | | | | | | |
| | `src/guide_bench.py` at 640 / 480 / 320 | Iesa | | | | | | | |

## 3. Devices

| Date | Phone | App / connection | Orientation | Result |
| --- | --- | --- | --- | --- |
| 2026-11-06 | Laptop webcam | built-in | n/a | ✅ Cues heard, directions correct (Babar) |
| 2026-11-06 | Babar, iPhone 14 Pro Max | DroidCam PC Webcam, iPhone hotspot | landscape + portrait | ✅ with the fall app (`src/main.py`). ⏳ not yet with `src/guide.py` |
| | Iesa, Android (model: ____) | IP Webcam | | ⏳ |
| | Swechha, iPhone (model: ____, LiDAR: ____) | DroidCam | | ⏳ |
| | Disconnect test: phone stream stopped 10 s, then restarted | | | ⏳ |

## 4. Depth error (only if a depth adapter exists)

Tape-measure from the camera lens to the front of the object. 3 readings at each distance, object in the centre and
then at the edge.

| Date | Adapter | Object | True m | Estimated m (3 readings) | Error | Valid depth % | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | chair | 1.0 | | | | |
| | | chair | 2.0 | | | | |
| | | chair | 3.0 | | | | |
| | | person | 2.0 | | | | |

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
