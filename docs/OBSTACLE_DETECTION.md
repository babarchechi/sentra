# Object-independent obstacle awareness: feasibility check and build plan

> Experimental. The goal is to *attempt* to warn about things YOLO can't name. It will still miss things. It never
> means the way is clear.

## 1. The problem

YOLO26n names only 80 everyday classes. Poles, bollards, traffic cones, boxes, trash cans, and unusual furniture get no
label, so Sentra Guide stays **silent** about them. Our current "Possible obstacle ahead." only fires when YOLO is
unsure what a *large* object is, so it is still recognition-based.

Idea: instead of asking "what is this?", ask **"is the ground in front of me walkable, and where does it stop?"**
Anything that interrupts the walkable ground low in the image, close to the user, is a possible obstacle, whatever it
is.

## 2. Feasibility check (2026-11-07, Babar + Claude, ~1 hour)

**Models** (all in Ultralytics 8.4.203, downloaded automatically, git-ignored):

| Model | What it does | Classes that matter | File |
| --- | --- | --- | --- |
| `yolo26n-sem.pt` | Labels every pixel; trained on Cityscapes (street scenes from a car) | road, sidewalk, terrain (walkable); pole, traffic sign, fence, wall, vegetation | 3.5 MB |
| `yolo26n-sem-ade20k.pt` | Labels every pixel; trained on ADE20K (indoor + outdoor, 150 classes) | floor, rug, path, sidewalk, road, grass, earth (walkable); wall, door, column, pole, box, railing, stairs, step | 3.5 MB |
| `yolo26n-depth.pt` | Estimated depth in meters per pixel (indoor + outdoor training mix) | n/a. Unvalidated, so display only | 13 MB |

**Speed on the demo laptop's CPU** (median, warm):

| Model | Frame | Model input | ms / frame |
| --- | --- | --- | --- |
| YOLO26n objects (what runs today) | 640×480 | 640 | 29 |
| Street segmentation (Cityscapes) | 640×480 | 640 | 36 |
| Indoor segmentation (ADE20K) | 720×1280 (today's DroidCam stream, rotated) | 384 | 317 (≈3 fps, too laggy) |
| Indoor segmentation (ADE20K) | downscaled to 360×640 | 320 | **91** |
| Indoor segmentation (ADE20K) | downscaled to 270×480 | 320 | **65** |
| Depth | 640×480 | 512 | 62 |

The ADE20K slowdown comes from scaling 150 class maps back up to the full frame, so **downscale the frame before
segmenting**.

**Results:**

- **Street sample photo** (Ultralytics `bus.jpg`): both models found the pavement. The scan traced the base of the
  people, the bus, and the **bollards**, which YOLO does not detect at all (it found only the people and the bus).
  This is the core idea working on a real photo.
- **Live, Babar's iPhone indoors, phone facing a desk:** ADE20K's labels made sense (desk, computer, chair, box,
  floor). It flagged "possible obstacle" in all 74 frames. The desk *was* in front, but we didn't get a clean
  open-floor test, so **accuracy is unknown**. At 1280×720 it ran at about 3 fps, which felt laggy.
- **The street model is useless indoors:** it labelled a desk as "car". Use it outdoors only.
- **Prototype bug:** when no floor is visible it said "obstacle"; it must say "can't judge" instead.

**Decision: go, time-boxed.** Build it as an **optional layer, off by default** until it passes the tests in
Section 4. If it isn't good enough by Saturday evening, demo it as an on-screen overlay only and present the measured
misses honestly.

## 3. How to build it

New module `src/sentra/guide/obstacles.py` (Iesa), used by `GuidePipeline` only when `--obstacles indoor|outdoor` is
given (indoor = ADE20K, outdoor = Cityscapes).

1. **Every 2nd–3rd frame, on a background thread** (so YOLO, tracking, and speech keep their full speed), downscale
   the frame to about 360×640 and segment it.
2. **Walkable mask:** pixels whose class is in the walkable list for that model.
3. **Free-space scan:** in the central band (same edges as `SpatialConfig`, 35–65% of the width), for each column,
   start at the bottom row and move up until you hit a run of non-walkable pixels at least ~3% of the image height
   tall. That height (as a share of the image) is the column's free space.
4. **Decision (updated for wall-filled views):**
   - Wall class covers at least 85% of each half of a column, across at least 80% of central columns not explained
     by announceable objects: confirm in 2 of the last 3 assessed scans, then feed a full-height central candidate
     to the existing caution policy. This does not require visible floor. Say "Caution, possible obstacle ahead."
     every two seconds while confirmed, fresh wall evidence persists. Wall coverage is not distance.
   - Low-texture image (per-channel RMS residual at most 5 on the 0–255 scale after subtracting a Gaussian blur
     with sigma 3 at scan resolution), with too little floor evidence and no qualifying wall: independently
     confirm across scans and repeat "Caution, camera can't see." Gradual lighting gradients are tolerated;
     sharp edges and random texture retain residual energy.
     This can be a plain surface, covered lens or featureless ceiling. It is a visibility warning, not object recognition.
   - The two conditions have separate confirmation histories from floor obstacles. Reusing a scan does not count
     again; reset or a gap longer than the scan freshness limit clears history. New contrary evidence stops cues.
   - If walkable ground covers < ~15% of the scanned area: **can't judge**. Show it on screen, no cue, and never
     "clear", unless one of the wall/visibility rules above applies.
   - If ≥ ~10% of central columns have free space < ~30% of the height: **possible obstacle ahead**. Name it only if
     a separate object detector can name it under the active speech policy. Otherwise say "Possible obstacle ahead."
   - Otherwise, show "no near obstruction found (not proof the way is clear)" on screen only, and say nothing.
5. **Feed it to `AlertPolicy`** as a candidate under the existing `("obstacle", CENTER)` key, so it gets the same
   persistence (3 of 5 checks), cooldown, and minimum gap. If YOLO already named a confident object whose box covers
   the blocked columns ("Chair ahead."), skip the generic cue.
6. **Depth (optional):** `yolo26n-depth.pt` can order obstacles (nearer first) and show "est. 1.8 m" on screen. Never
   speak it unless it passes the tape test (DEPTH_OPTIONS.md).
7. **Tests:** synthetic masks (open floor → quiet; box in the band → obstacle; no floor → can't judge; thin pole →
   obstacle), plus a check that the background thread never blocks the main loop.

The core of the scan, from the prototype:

```python
def free_space(mask, walkable_ids, left=0.35, right=0.65, scan_top=0.35, run_share=0.03):
    """Per central column: share of image height from the bottom up to the first blockage."""
    h, w = mask.shape
    x0, x1, top = int(left * w), int(right * w), int(scan_top * h)
    blocked = ~np.isin(mask[top:, x0:x1], walkable_ids)[::-1]        # bottom row first
    run = max(2, int(run_share * h))
    runs = cv2.filter2D(blocked.astype(np.float32), -1, np.ones((run, 1), np.float32),
                        anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    hit = runs >= run - 0.5                                             # `run` blocked pixels in a row
    first = np.where(hit.any(axis=0), hit.argmax(axis=0), blocked.shape[0])
    return first / h
```

**Known limits:** camera tilt changes what "near" means, so hold the phone at chest height, tilted slightly down.
Glass, thin poles at a distance, and dark or unusual floors can fool segmentation. Shadows and floor patterns may cause
false alarms; Section 4 measures that.

**Owners:** Iesa builds `obstacles.py` and its tests. Swechha owns the wording, how it combines with YOLO cues, and
rate limits. Babar picks the test spots, records labelled results, and decides whether it goes in the demo.

## 4. Test protocol (controlled; results go in VALIDATION.md)

Sighted teammates only, not relying on the app. Camera held still or moved slowly at chest height. Indoors, or a
campus walkway **away from vehicles**. Don't record people who haven't agreed; if recording, only teammates in frame.

| # | Scene | Expect | What to count |
| --- | --- | --- | --- |
| I1 | Open floor ahead, nothing in the way | Quiet | False alarms per minute |
| I2 | Box / bag / trash can in the middle at 1, 2, 3 m | "Possible obstacle ahead." | Caught at each distance? |
| I3 | Wall or closed door ahead | Obstacle (or "wall"/"door") | Caught? |
| I4 | Chair in the middle | "Chair ahead." only, not a second generic cue | Double cues |
| I5 | Floor with a rug, tile pattern, or strong shadow | Quiet | False alarms |
| I6 | Phone pointed up, no floor visible | On screen "can't judge", no cue | Wrong "obstacle" cues |
| O1 | Pole / bollard / sign post, from 3 positions | Obstacle (or "pole") | Caught per position |
| O2 | Teammate walks across | "Person ..." from YOLO | Correct, not doubled |
| O3 | Empty walkway with shadows and paving lines | Quiet | False alarms per minute |
| O4 | Traffic cone, bench, bike rack | Obstacle | Caught? |

**Report misses, not just hits:** "caught X of Y obstacles; Z false alarms in N minutes of empty floor/walkway".
