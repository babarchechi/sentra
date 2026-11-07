# Sentra Guide: team plan

> Capital Hacks 5.0 · Fri Nov 6 – Sun Nov 8, 2026 · Health + Accessibility Technology
> **Experimental supplementary awareness prototype. Not a navigation aid. Never say "path clear" or "safe".**
> Replaces [PHASE1_PLAN.md](PHASE1_PLAN.md) (fall detection), which stays for reference. Why we pivoted and what we
> kept: [PIVOT.md](PIVOT.md). How it fits together: [ARCHITECTURE.md](ARCHITECTURE.md).

## 1. The goal: a reliable 60–90 second demo

The first working version is already built and running on the laptop webcam. **Every task below makes that demo
better, more honest, or better tested. None may break it.** Run it:

```powershell
python src/guide.py --source 0
```

**Demo script** (Babar runs it; Swechha is backup operator):

1. Start `src/guide.py` with Babar's iPhone as the camera (DroidCam over the iPhone hotspot). Hold the phone still.
2. A chair in the middle → "Chair ahead." Box and AHEAD label on screen.
3. Move the chair to the side → "Chair to your left." Show that it isn't repeated every second (cooldown).
4. A consenting teammate walks across the fixed camera → "Person to your right." / "...ahead."
5. Point out **"Distance: off (camera only)"**: no numbers are invented. (Only if a depth adapter passes the tape
   test: show its overlay, marked approximate.)
6. Switch the source to Iesa's Android (and Swechha's iPhone if there's time) and show the same cues.
7. Turn off the phone app's stream for ~10 s: the app keeps running, then reconnects.

**Done means:** phone stream runs; at least two kinds of objects/people detected in controlled tests; directions match
the test scenes; cues follow the cooldown; missing depth is handled gracefully; fps and alert latency are measured and
written in [VALIDATION.md](VALIDATION.md).

## 2. Who does what

| Person | Role | Branch | First thing tonight |
| --- | --- | --- | --- |
| **Babar** (babarchechi) | Product lead, camera integration, optional LiDAR experiment, pitch + live demo, merge order | `feature/phone-and-depth` | Run Sentra Guide on the iPhone stream; record the result |
| **Iesa** (iesa242) | AI perception + backend: detection, tracking, direction, pipeline, benchmarks, optional Depth Anything | `feature/object-perception` | Run it on the Android stream; write the benchmark script |
| **Swechha** (Swechha03) | Audio UX, accessibility, validation: alert wording and timing, speech, debug view, phone speech page, user feedback | `feature/audio-and-validation` | Run it on her iPhone; start the alert-quality log in VALIDATION.md |

## 3. File ownership

Edit only your own files. To change someone else's, ask them, or put it in your PR and request their review.

| File | Owner | Notes |
| --- | --- | --- |
| `src/sentra/guide/schema.py` | **Shared**, Iesa maintains | The contract. Changes need approval from **both** other teammates. Adding an optional field is fine; changing or removing one bumps `SCHEMA_VERSION`. |
| `tests/guide_helpers.py` | **Shared** | Test fakes. Tell the team before changing. |
| `src/sentra/capture.py`, `src/sentra/windows_camera.py` | Babar | Camera input |
| `src/guide.py` | Babar | CLI. Others: add your option in your PR, and Babar reviews it |
| `src/sentra/guide/record3d.py` (new, optional) | Babar | Record3D LiDAR `DepthProvider` |
| `src/record3d_probe.py` (new, optional) | Babar | Checks RGB + depth arrive, prints shapes/units |
| `tests/test_capture.py`, `tests/test_guide_record3d.py` (new) | Babar | |
| `src/sentra/guide/detector.py`, `spatial.py`, `pipeline.py`, `depth.py` | Iesa | Perception + loop |
| `src/sentra/guide/depth_anything.py` (new, optional) | Iesa | Depth Anything V2 `DepthProvider` |
| `src/guide_bench.py` (new) | Iesa | FPS/latency benchmark |
| `tests/test_guide_pipeline.py`, `tests/test_guide_spatial.py` | Iesa | |
| `src/sentra/guide/alerts.py`, `speech.py`, `view.py`, `event_log.py` | Swechha | What is said, when, how it sounds and looks |
| `src/sentra/guide/web.py` + `src/sentra/guide/static/` (new) | Swechha | Phone/browser speech page |
| `tests/test_guide_alerts.py`, `tests/test_guide_speech_view.py`, `tests/test_guide_web.py` (new) | Swechha | |
| `docs/VALIDATION.md`, `docs/SAFETY_AND_PRIVACY.md` | Swechha | Everyone adds their own measurements |
| `docs/TEAM_PLAN.md`, `PIVOT.md`, `ARCHITECTURE.md`, `SETUP_WINDOWS.md`, `DEPTH_OPTIONS.md`, `README.md` | Babar | Anyone may fix mistakes; docs go straight to `main` |
| Fall-detection files (`main.py`, `pipeline.py`, `posture.py`, `events.py`, `ui/`, `evaluation.py`) | Archived | No new work. Keep them passing their tests. |

## 4. Tasks: Today / Next / Stretch

Priorities: **P0** = needed for the demo, **P1** = makes it clearly better, **P2** = only if everything else is done.

### Babar: camera integration, depth experiment, demo

**Today (Fri night)**
1. **P0** Review and merge the pivot PR once Iesa or Swechha has approved it (`schema.py` is new, so ideally both).
2. ✅ **Done 2026-11-06** (portrait + `--rotate 90`, over eduroam; results in VALIDATION.md). Run Sentra Guide on the iPhone: DroidCam *PC Webcam*, laptop on the iPhone hotspot,
   `python src/guide.py --source "http://172.20.10.1:4747/video"` (landscape), then portrait with `--rotate 90`.
   *Accept:* boxes upright; a chair in the middle → "Chair ahead."; moved left → "Chair to your left."; heard on the laptop.
   Record it in VALIDATION.md, Section 3.
3. **P0** Disconnect test (demo step 7): stop the DroidCam stream for 10 s, restart it.
   *Accept:* no crash, "Reconnected" in the terminal, cues resume. Record it either way.

**Next (Sat morning)**
4. **P1** Record3D preflight, **no purchase without a teammate's explicit OK** ([DEPTH_OPTIONS.md](DEPTH_OPTIONS.md), option C).
   Check the App Store price and what the purchase includes, install iTunes, `python -m pip install record3d` in a
   **separate** test venv, and only after an approved purchase run the upstream `demo-main.py`.
   *Accept:* written findings in DEPTH_OPTIONS.md (works / blocked, and why). **Stop** if it costs unapproved money,
   doesn't stream depth on Windows, or takes more than ~2 hours.
5. **P0** Write the demo run sheet (commands, phone settings, who stands where) and rehearse it twice.

**Stretch**
6. **P2** `record3d.py` `DepthProvider` using `box_depth()` with `validated=False`; tape-measure tests with Swechha
   (VALIDATION.md, Section 4); flip to validated only if the errors are acceptable.

**Fallback:** if the phone stream fails during the demo, use the laptop webcam (verified). If Record3D is blocked,
present camera-only plus the honest blocker.

**Run/test:** `python -m pytest -q tests/test_capture.py` · `python src/guide.py --source <url> [--rotate 90]`

### Iesa: perception and backend

**Today (Fri night)**
1. **P0** Pull the pivot branch (or `main` after merge), `python -m pytest -q`, then run on the Android IP Webcam stream:
   `python src/guide.py --source "http://<phone-ip>:8080/video"`.
   *Accept:* cues and directions correct for a chair and a person; record it in VALIDATION.md, Section 3.
2. **P0** `src/guide_bench.py`: time N frames from a source at `--imgsz 640/480/320`, tracking on/off. Report median and
   p90 model time, end-to-end latency, fps. *Accept:* numbers from this laptop in VALIDATION.md, Section 2, with the
   exact command.

**Next (Sat)**
3. **P1** Perception quality: check direction correctness on the VALIDATION.md test scenes, tune region edges and
   hysteresis (`SpatialConfig`), and check that a slow frame never causes a cue (stale-frame rule). Add tests for any
   change.
4. **P1** Optional `depth_anything.py`: Depth Anything V2 **Small, metric indoor** checkpoint behind the
   `DepthProvider` interface, `validated=False`, run every Nth frame if slow. New packages go in a separate
   `requirements-depth.txt` (not the main file), so camera-only installs stay small.
   *Accept:* measured ms/frame on this laptop, plus a measured error table at 1/2/3 m before anyone talks about meters.

**Stretch**
5. **P2** Motion cues in a **fixed-camera** scene only (for example "Person crossing ahead") from tracked box movement.
   No approach/collision claims on a handheld phone.

**Fallback:** if tracking is unstable, run with `--no-track` (directions don't need IDs). If depth is too slow, leave
`NoDepth`.

**Run/test:** `python -m pytest -q tests/test_guide_pipeline.py tests/test_guide_spatial.py` · `python src/guide_bench.py ...`

### Swechha: audio UX, accessibility, validation

**Today (Fri night)**
1. **P0** Find out your iPhone model (Settings → General → About). Run Sentra Guide on it via DroidCam, same steps as
   Babar. *Accept:* result recorded in VALIDATION.md, Section 3.
2. **P0** Alert-quality pass: run the three short scenes in VALIDATION.md, Section 5 and log every cue that was
   unhelpful, duplicated, late, or wrong. Then tune `AlertConfig` defaults (persistence, cooldown, gap, label list,
   wording) and back each change with the log. *Accept:* before/after counts in VALIDATION.md; tests updated.

**Next (Sat)**
3. **P1** Phone/browser speech page (`web.py` + `static/`): the laptop serves a small page on the hotspot network; the
   page receives `GuideAlert` JSON (Server-Sent Events from Python's standard library is enough) and speaks it with the
   browser's `speechSynthesis`. Large high-contrast text, connection status, voice on/off, a "tap to enable audio"
   button (iOS needs a tap before speaking). **Never send video to the page**, only alert JSON.
   *Accept:* cues heard from a phone browser within ~1 s of appearing in the laptop log, and the page says so when
   disconnected. Note: an iPhone streaming with DroidCam must keep DroidCam in the foreground, so test the page on a
   second phone or on Android.
4. **P1** User feedback: reach out (campus disability services, accessibility clubs, O&M instructors) and ask for
   feedback on a **seated tabletop or recorded demo**, not walking. Use the interview template in VALIDATION.md,
   Section 6. Write down what people actually said, including "this wouldn't help me".

**Stretch**
5. **P2** Earcons (a short tone per direction), haptics on Android (`navigator.vibrate`), a user-adjustable "quiet"
   level.

**Fallback:** the laptop voice works today. Bluetooth earbuds paired with the laptop bring it closer to the user.

**Run/test:** `python -m pytest -q tests/test_guide_alerts.py tests/test_guide_speech_view.py` · `python src/guide.py --source 0`

## 5. Dependencies and integration order

```text
pivot PR merged (schema.py contract)
 ├─► Babar: iPhone run ─┐
 ├─► Iesa: Android run ─┼─► all three phones verified ─► demo rehearsal (Sat evening)
 ├─► Swechha: iPhone run┘                                   ▲
 ├─► Iesa: benchmark ──► (optional) Depth Anything ─────────┤  only if it passes the tape test
 ├─► Swechha: alert tuning ──► phone speech page ───────────┤
 └─► Babar: Record3D preflight ──► (optional) record3d.py ──┘  only if approved and it works
```

Merge order: (1) pivot PR → (2) small device-test doc updates (straight to `main`) → (3) Swechha's alert tuning and
Iesa's benchmark, in either order → (4) phone speech page → (5) depth adapters last, only if they pass validation.
**Feature freeze: Sunday morning.** After that, only fixes for bugs found in rehearsal.

## 6. Review and merge rules (unchanged from Phase 1)

- **Code** (`src/`, `tests/`, `requirements*.txt`, `pyproject.toml`): branch → PR → **approval from someone other than
  the author** → anyone may merge. Commits pushed after an approval need a **fresh** approval (GitHub keeps showing
  "Approved", so compare the dates).
- **`schema.py` changes need both other teammates.**
- **Docs** (`docs/`, `README.md`): commit straight to `main` and push; post a one-line note in the chat.
- Never force-push, never `git add .` (add your files by name), never commit `.venv/`, `*.pt`, videos, `outputs/`,
  stream URLs or passwords.
- **Babar owns the merge order** and checks `main` still runs after each merge (`python -m pytest -q` +
  a 30-second webcam run). **Babar runs the live demo**; Swechha is the backup operator.

```powershell
git checkout main; git pull origin main
git checkout -b feature/object-perception        # your branch name, first time only
# ...work...
git add src/sentra/guide/detector.py tests/test_guide_pipeline.py
git commit -m "feat(guide): ..."
git push -u origin feature/object-perception
gh pr create --base main                          # or the Compare & pull request button on GitHub
```

## 7. Hard rules

- No Mac/Xcode/Swift. Windows + phone video streams only.
- No purchases (including Record3D) without a teammate's explicit OK in the chat.
- No accounts, payments, custom model training, or 3D mapping.
- Don't delete the fall-detection code to make the repo look cleaner.
- No blindfold tests, no unsupervised walking tests, no testing on people who haven't agreed to be on camera.
- Never speak a distance that hasn't passed the tape-measure test. Never say "clear" or "safe".
- Don't claim something works (device, depth, GitHub) until someone has run it and written it down.

## 8. Timeline

Submission deadline: **________** (Babar: fill in from the official schedule).

| When | Checkpoint |
| --- | --- |
| Fri night | Pivot PR merged; all three phones tried with Sentra Guide; benchmark script started |
| Sat midday | Alert tuning merged; benchmark numbers in VALIDATION.md; Record3D go/no-go decided |
| Sat evening | Validation session (VALIDATION.md scenes), phone speech page if ready, first full demo rehearsal |
| Sun morning | Feature freeze; pitch (4 min) + Q&A (3 min) rehearsed twice; demo runs **offline and without depth** |
