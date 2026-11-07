# Safety and privacy

## What Sentra Guide is, and isn't

- It **is** an experimental, supplementary awareness prototype: it names some common objects and people that a phone
  camera sees, says roughly which side they're on, and keeps quiet most of the time.
- It **is not** a navigation aid, a mobility-aid replacement (cane, guide dog, O&M training), a collision-avoidance
  system, a fall-prevention device, or anything medical or safety-certified.
- It **never** says "clear", "safe", or "go ahead". Not hearing a cue does not mean nothing is there (see
  VALIDATION.md, Section 7 for what the model can't detect).
- Distances are spoken only from a depth source that passed tape-measure testing. Today there is none, so no
  distances are spoken.

Use this wording in the pitch, the UI and the README. Don't present user interest as proof that it improves mobility.

## Testing safely

- Consenting teammates only. Seated, standing, or walking slowly across a **fixed** camera in a cleared indoor space.
- No blindfolded testers, no one walking guided by the app.
- Outdoors: controlled engineering tests only (docs/OBSTACLE_DETECTION.md, Section 4). Sighted teammates who aren't
  relying on the app, camera held still or moved slowly, on a campus walkway away from vehicles. No streets.
- Feedback sessions with blind or low-vision people or professionals use a **tabletop or recorded demo**, not walking.

## Privacy

- **Camera on:** the debug window shows a red "CAMERA ON" dot whenever frames are processed; **p** hides the image.
- **Opt-in:** the camera starts only when someone runs the app with a source.
- **Local only:** frames are processed in memory on the laptop and discarded. Nothing is uploaded. No cloud API.
- **No storage of images:** logs (`outputs/guide_events_*.jsonl`) contain labels, directions, scores and timing, never
  images. `outputs/` is git-ignored.
- **No identification:** no face recognition. Tracking IDs are temporary numbers that can change, never identities.
- **Bystanders:** avoid pointing the camera at people who haven't agreed. Don't record them.
- **Network:** stream over the phone's own hotspot. Never expose a camera stream or the phone page to the public
  internet. Never commit stream URLs, passwords, `.env` files, recordings or model weights.
- **Phone page (`--web`):** sends cue text and a status line only, never video or images. It listens on one address:
  this laptop only by default, or the laptop's hotspot IP when given with `--web-host`. It refuses 0.0.0.0 and
  public addresses. Anyone on the same hotspot who opens the page can read the cues, so keep the hotspot to the
  team. It shows the "not a navigation aid" line and says clearly when cues are not coming through.
- **Recordings for tests:** only with the consent of everyone in frame, only when needed, kept locally, deleted after
  the hackathon.

## Licences to keep in mind

- Ultralytics YOLO: AGPL-3.0 (enterprise licences exist). Fine for a hackathon. Review before any commercial use.
- Depth Anything V2: only the Small model is Apache-2.0; larger ones are non-commercial (DEPTH_OPTIONS.md).
- Record3D Python client: LGPL-2.1. The app itself is a paid product.
