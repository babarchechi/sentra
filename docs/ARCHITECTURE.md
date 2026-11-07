# Sentra Guide architecture

> Experimental supplementary awareness prototype. Not a navigation aid.

Everything runs in one Python process on the Windows laptop. The phone is only a camera: it streams video to the
laptop over the phone's own hotspot. Nothing is uploaded to a server, and no video or images are written to disk.

## 1. Pipeline (as built)

```text
 Babar's iPhone (DroidCam)    Iesa's Android (IP Webcam)    laptop webcam    consented .mp4 (tests)
            \                          |                         |                /
             +------- MJPEG over the phone hotspot (LAN only) ---+---------------+
                                       |
   capture.py      FrameSource         reader thread keeps ONLY the newest frame; reconnects with backoff;
                                       optional --rotate; video files paced at their own fps
                                       |  (ok, BGR frame, perf_counter timestamp)
   guide/detector.py  ObjectDetector   YOLO26n (yolo26n.pt, 80 COCO classes) + ByteTrack IDs
                                       |  DetectionFrame
   guide/depth.py     DepthProvider    NoDepth (default)  |  [Depth Anything V2: planned]  |  [Record3D LiDAR: planned]
                                       |  one DepthEstimate per detection (meters may be None)
   guide/spatial.py   SpatialEstimator left / center / right, central-band overlap, severity, edge hysteresis
                                       |  SpatialObject[]
   guide/alerts.py    AlertPolicy      label filter, size filter, 3-of-5-frame persistence, per-cue cooldown,
                                       minimum gap, one cue per frame, stale-frame rule, honest wording
                                       |  GuideAlert (0 or 1 per frame)
          +----------------------------+-----------------------------+
          |                            |                             |
   guide/speech.py Speaker      guide/event_log.py            guide/view.py GuideView
   queue (max 2, newest wins,   outputs/guide_events_*.jsonl  OpenCV window: regions, boxes,
   drops cues older than 2 s)   (alerts + session summary,    last cue, fps/latency, depth mode,
   -> Windows voice process     never images)                 m mute / p privacy / q quit
```

`src/guide.py` builds this from command-line options. `GuidePipeline.run()` (in `guide/pipeline.py`) is the loop.

The fall-detection app (`src/main.py` → `sentra/pipeline.py` → `posture.py` → `ui/`) shares only `capture.py` and
`FpsMeter`. It is unchanged.

## 2. Threads and queues

| Thread / process | Does | Bounded by |
| --- | --- | --- |
| `FrameSource` reader thread | Reads the camera as fast as it delivers | Holds 1 frame; older ones are overwritten, never queued |
| Main thread | Detection → depth → spatial → policy → log → view | Processes whichever frame is newest when it's ready, so a slow model drops frames instead of lagging |
| `Speaker` thread | Sends one cue at a time to the voice | Queue of 2 (newest wins); a cue older than 2 s when the voice is free is dropped |
| `powershell.exe` child process | Windows System.Speech voice, started once | One line in, "done" out; restarted never. On failure, cues are printed instead |

**Stale-frame rule:** if a frame is more than `max_frame_age_s` (1.0 s) old by the time it has been analyzed, it can't
trigger a cue. The view shows "LAGGING". Measured on 2026-11-06: median capture-to-decision latency 39–44 ms on the
laptop webcam, with 0 stale frames.

**Timing:** all timestamps are `time.perf_counter()` seconds (`time.monotonic()` only ticks every ~15.6 ms on
Python 3.12 for Windows).

## 3. Alert rules (defaults in `AlertConfig`)

| Rule | Default | Why |
| --- | --- | --- |
| Announced classes | person, chair, couch, bench, dining table, bed, potted plant, toilet, tv, refrigerator, bicycle, car, motorcycle, bus, truck, dog, cat, backpack, umbrella, suitcase, fire hydrant, stop sign, parking meter (`--all-labels` for every class) | Small tabletop objects would make it chatty |
| Name a class only at score | ≥ 0.5 (`--announce-conf`) | Below that the label is a guess |
| "Possible obstacle ahead." | score 0.3–0.5, central, ≥ 35% of frame height, not overlapping a confident box | Something big is in front but the model isn't sure what |
| Ignore boxes shorter than | 10% of the frame height | Far away or tiny |
| Persistence | seen in 3 of the last 5 frames | No one-frame flickers |
| Same cue again | Not while it stays in view. Again only if it comes back after ≥ 3 s out of view, more of them appear (1 → 2 chairs), or as a reminder every 30 s for central objects (`--remind-after`, 0 = never). Never sooner than 10 s (`--repeat-after`). A new direction is a new cue | Say new information, not the same thing again |
| Any two cues | ≥ 2.5 s apart (`--min-gap`) | Room to think between cues |
| Per frame | at most 1 cue, the most important | Importance: central and large → central → touching the centre → side; then confident before uncertain, bigger, higher score |
| Distances | spoken only if `DepthEstimate.validated` | No invented meters |
| **Caution** (danger warning) | something announced (or a "possible obstacle") in the central band and ≥ 50% of the frame height (`SpatialConfig.high_view_height`) in 3 of the last 5 frames → "Caution, chair ahead." (`--no-caution` turns it off) | Big in the middle of the view means big or near, in the way. Box size is not a distance |
| Caution timing | said before any other cue; may follow any cue after 1 s; unlike ordinary cues (said once), repeats every 5 s while it stays in the way (`--caution-every`) | Something in the way can't wait behind "Person to your left." |
| Caution delivery | `GuideAlert.caution = true`: the laptop voice drops waiting cues; the phone page vibrates (Android; iPhones can't vibrate from a web page), interrupts an ordinary cue and shows it white on red; the debug view frames the video in red | Felt as well as heard |
| Words never produced | "clear", "safe" (enforced by a test) | Missing detections don't mean a free path. There is **no "safe to keep walking" signal**: no caution never means nothing is in the way |

Cue wording: "Chair ahead.", "Person to your left.", "2 people to your right.", "Possible obstacle ahead.",
"Caution, chair ahead.", "Caution, possible obstacle ahead.", and only with validated depth: "Chair ahead,
approximately 1.4 meters."

**Directions are from the camera's point of view.** With a phone held out in front, that is the user's left and right.
With a laptop webcam facing you, it's mirrored relative to you.

## 4. Data contract: `src/sentra/guide/schema.py` (schema `sentra.guide/1`)

| Type | Fields |
| --- | --- |
| `Detection` | `label`, `class_id`, `confidence`, `box` (x1, y1, x2, y2 px), `track_id` (or None; never an identity) |
| `DetectionFrame` | `frame_id`, `timestamp`, `frame_width`, `frame_height`, `detections`, `source`, `inference_ms` |
| `DepthEstimate` | `meters` (None = unknown), `confidence`, `source` (`none` / `depth_anything_v2` / `record3d`), `validated`; `speakable` = has meters **and** validated |
| `SpatialObject` | `detection`, `region` (left/center/right), `center_x` (0–1), `in_corridor`, `view_height` (size cue, not distance), `severity` (low/medium/high), `depth` |
| `GuideAlert` | `schema`, `type` = "alert", `timestamp`, `frame_id`, `label`, `count`, `region`, `severity`, `confidence`, `text`, `reason`, `track_id`, `meters`, `depth_source`, `depth_confidence`, `latency_ms`, `source`, `caution` (optional, default false: a danger warning) |
| `GuideUpdate` | what a UI receives each frame: `frame`, `objects`, `alerts`, `image` (never saved), `fps`, `latency_ms`, `depth_status`, `stale` |
| `GuideSink` | protocol: `publish(update)`, `should_stop()`, `close()`. The OpenCV view and a future web page both implement it |

A real alert from the event log (laptop webcam, 2026-11-06). The log adds `wall_time` and `spoken`:

```json
{"schema": "sentra.guide/1", "type": "alert", "timestamp": 520754.5497016, "frame_id": 3, "label": "person",
 "count": 1, "region": "left", "severity": "low", "confidence": 0.602, "text": "Person to your left.",
 "reason": "person (score 0.60, low) left in 3/3 recent frames; not said in the last 10 s", "track_id": 1,
 "meters": null, "depth_source": "none", "depth_confidence": 0.0, "latency_ms": 31.7, "source": "webcam 0",
 "wall_time": "2026-11-06T18:30:34.716", "spoken": true}
```

Each run ends with a summary line, which is what VALIDATION.md asks for:

```json
{"schema": "sentra.guide/1", "type": "session_summary", "frames": 1247, "seconds": 49.5, "fps": 25.2,
 "median_latency_ms": 39.2, "median_inference_ms": 29.2, "alerts": 16, "stale_frames": 0}
```

The phone speech page (planned) should send `GuideAlert.to_dict()` as-is. That is the contract between Iesa's
pipeline and Swechha's page.

## 5. Depth (optional, never required)

`DepthProvider.estimate(image, frame)` returns one `DepthEstimate` per detection. Rules every adapter follows (and
`box_depth()` implements):

- Keep RGB and depth paired by frame; drop mismatched or stale pairs.
- Scale boxes from RGB pixels to the depth map's resolution, and check alignment with a known object first.
- Use the median of valid, confidence-filtered pixels in the central half of the box, never a single pixel.
- `validated=False` until VALIDATION.md, Section 4 shows acceptable error against a tape measure. Unvalidated
  distances may be shown on screen as "est." and are never spoken.

Options and current status: [DEPTH_OPTIONS.md](DEPTH_OPTIONS.md).

## 6. Network and privacy

- Without `--web`, Sentra Guide opens **no network ports**. It only connects out to the phone's stream URL.
- With `--web`, `guide/web.py` serves the phone speech page on one address: 127.0.0.1 by default, or the laptop's
  IP on the phone hotspot (`--web-host`). Listening on every network (0.0.0.0) and public addresses are refused.
  It sends only `GuideAlert.to_dict()` and a status heartbeat over Server-Sent Events; there is no route that
  returns a frame, and the page's Content-Security-Policy allows no images or media.
- Use the phone's own hotspot (campus Wi-Fi often isolates devices, and the hotspot keeps the stream private).
- Stream URLs are printed without credentials or query strings; video file names are printed without folders.
- More in [SAFETY_AND_PRIVACY.md](SAFETY_AND_PRIVACY.md).
