# Depth options: none / Depth Anything V2 / Record3D LiDAR

> **The demo must work with option A.** B and C are optional experiments. Neither may block or slow down the
> camera-only demo. No purchase is made without a teammate's explicit OK in the chat.

Status on 2026-11-06: **only option A exists in code.** `src/sentra/guide/depth.py` defines the `DepthProvider`
interface, `NoDepth`, and a `box_depth()` helper that B and C should both use.

## Option A: camera only (default, built, verified)

- `NoDepth` returns "unknown" for every object. The screen shows **"Distance: off (camera only)"** and cues never
  contain numbers ("Chair ahead.").
- Box size (`view_height`) is used only to order cues (a big central box first), never as a distance.
- Cost: $0. Status: ✅ verified on the laptop webcam, 2026-11-06.

## Option B: Depth Anything V2 Small, metric indoor (free, estimated from the RGB image)

| | |
| --- | --- |
| What | A neural network that estimates a depth map from a normal camera image. **Relative** models give near/far only; **metric** models are fine-tuned to output meters. For meters, use the **metric indoor (Hypersim)** checkpoint, not the default relative one. |
| Checkpoints | Original repo: `depth_anything_v2_metric_hypersim_vits.pth` (Small, 24.8M params, indoor, `max_depth` 20). Hugging Face transformers: `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` |
| Licence | The repo says *"Depth-Anything-V2-Small model is under the Apache-2.0 license"*, and Base/Large/Giant are CC-BY-NC-4.0. The metric-indoor model card doesn't state a licence, so confirm it before any use beyond this hackathon. **Use Small only.** |
| Install | Not installed. Easiest Windows path: `transformers` (pipeline `"depth-estimation"`), in a separate `requirements-depth.txt`. Adds roughly 100 MB of weights plus the library |
| Speed | **Unknown on our laptop.** Must be measured (Iesa). If slow: run every Nth frame, use a smaller input, or drop it |
| Accuracy | Model-estimated. A phone image isn't what it was trained on (synthetic indoor scenes), so errors can be large. **Never spoken until tape-tested.** |
| Status | ❌ Not started. Owner: Iesa (`src/sentra/guide/depth_anything.py`), only after the camera-only pipeline is solid |

Sources: https://github.com/DepthAnything/Depth-Anything-V2 and its `metric_depth/README.md`.

## Option C: Record3D, real iPhone LiDAR over USB to Windows (paid app, no Mac needed)

| | |
| --- | --- |
| What | The Record3D iOS app streams RGB + LiDAR depth (and, per the library's changelog, confidence maps and camera pose) from Babar's iPhone 14 Pro Max over a USB cable to a computer. An open-source Python client receives it. |
| Cost | Record3D's features page says: *"To use the feature, you need to purchase Record3D."* No price is listed there. Wi-Fi streaming is a **separate** paid "Extension Pack". **Check the App Store price; buy only with a teammate's explicit OK.** USB is the path to try first. |
| Windows needs | **iTunes** installed (for the iPhone USB connection), a data-capable USB cable, and `python -m pip install record3d`. PyPI has `record3d` 1.4.1 with a ready-made **Windows wheel for Python 3.12** (`cp312-win_amd64`), so no CMake build should be needed. Licence: LGPL-2.1 |
| Sample | `demo-main.py` in https://github.com/marek-simonik/record3d. Enable *USB Streaming mode* in the app's Settings, then press record in the app |
| Unknowns | Not tested on our machines. Depth resolution, units and the confidence scale must be checked by printing them. Depth is a lower resolution than RGB, and orientation/crop may differ, so alignment has to be checked with a known object |
| Status | ❌ Not started. Owner: Babar (`src/record3d_probe.py`, then `src/sentra/guide/record3d.py`) |

**Preflight, in order. Stop at the first failure:**

1. Read the current App Store price and what the purchase includes (USB streaming vs. Wi-Fi pack). Post it in the chat.
2. Get a teammate's explicit OK before buying anything.
3. Install iTunes; plug the iPhone in with a data cable; check that Windows sees it (trust the computer on the phone).
4. In a **separate** venv: `python -m pip install record3d` and `python -c "import record3d"`.
5. Run the upstream `demo-main.py`; confirm **both** RGB and depth frames arrive live.
6. Print the depth dtype, units, shape and confidence values; check orientation against the RGB frame; place a box at a
   tape-measured 1.0 m and see whether `box_depth()` reads about 1.0.

**Stop condition:** stop and stay on option A if Record3D needs an unapproved purchase, won't install, doesn't stream
depth on Windows, or takes more than about 2 hours. Write down how far you got.

## Decision rule

| Situation | Use |
| --- | --- |
| Default, and always for the demo's main run | **A** |
| B measured at ≥ ~5 fps on the laptop (or acceptable every Nth frame) **and** tape-test median error ≤ ~25% at 1–3 m | B on screen as "est.", **still not spoken** unless it passes the stricter test below |
| C working end to end **and** tape-test error ≤ ~10 cm or 10% at 0.5–3 m, with ≥ 80% valid depth pixels on the test objects | C, with `validated=True` → cues may include "approximately X meters" |
| Anything else | A, and present the measured results honestly |

Thresholds are starting points for the team to agree on before testing, not after. How to run the tape test:
[VALIDATION.md](VALIDATION.md), Section 4.

## Rules for any depth adapter

- Return one `DepthEstimate` per detection, in order. `meters=None` when unsure.
- Pair RGB and depth by frame; drop stale or mismatched pairs.
- Use `box_depth()`: central half of the box, confidence-filtered, median, `None` if under 30% valid.
- `validated=False` by default. It becomes `True` only after the tape test passes, set by a reviewed PR.
- Keep it in its own module and import it lazily, so camera-only installs and runs never depend on it.
