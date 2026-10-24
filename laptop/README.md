# Foot Piano - laptop side

Plays real piano keys (through the Pico + servos) from body movement seen by a webcam.
Which body parts play is a **profile**; the layout, calibration and even the pose model adapt to it.

## Install and run

```
pip install -r requirements.txt
python foot_piano.py --port COM7 --profile two_feet          # Windows
python foot_piano.py --no-pico --profile hand_foot           # no hardware: prints P/R commands
python foot_piano.py --list-presets
```

The first run downloads the RTMPose weights (rtmlib does this itself; needs internet once).
`--model-mode lightweight|balanced|performance` trades speed for accuracy (default `lightweight`); `--device cuda` for an NVIDIA GPU (needs `onnxruntime-gpu`).
With `--no-pico` the laptop plays the notes itself through a simple software piano (`--mute` to silence it, `--sound` to also use it with the Pico). A short C-E-G chime plays at start-up; if you don't hear it, sound is the problem, not the camera. `python foot_piano.py --test-sound` checks sound on its own and says what's wrong (most often: `pip install sounddevice` was not run in this virtual environment).

Camera: 1-1.5 m away, raised and angled down so the feet (and hands, if used) are in view.
At start-up rest feet flat / hands open and hold still ~1 s; calibration is automatic.

## Camera set-up (the app tells you what to do)

While it calibrates, the screen shows a `SET-UP:` line when something is wrong: a foot out of view, feet cut off at the
bottom edge, a foot at the edge of the picture, or "too far / too close" with a rough distance to move
("step back about 40 cm"). It judges by the toe-to-heel length on screen: roughly 14-32% of the frame height is OK.
The centimetre figures assume a 26 cm foot and a webcam that sees about 45 degrees top-to-bottom, so treat them as
roughly +-30%; a wide-angle webcam or a very small/large foot will shift them. The advice disappears once the set-up
is acceptable. It can't tell if the camera is angled well, so check the picture yourself.

## Playing real keys with a working hand

If a player can press piano keys with a hand or fingers, they don't need the camera for that hand: leave the hand out
of the profile (feet only) and let them play the real keyboard with it. Hand tracking exists for people who can move
but not press (e.g. a finger that can bend but not press a key), and it is the least reliable part of the system.
Feet-only profiles also use the smaller, faster model.

## Speed (if it is laggy)

Tap detection needs roughly 20+ frames per second; below ~15 fps taps are missed and the picture lags. The pose
models are the cost. What to try, in order:

1. `python foot_piano.py --bench --profile two_feet` times the model sizes on your camera and prints ms/frame and
   fps for each. Pick the slowest one that still reaches ~20 fps. (I could not measure this on your machine.)
2. `--model-mode lightweight` (now the default) uses a tiny person detector and the small pose model.
3. The person detector is the heaviest part, so it runs on a background thread (`--det-every 30` = refresh about
   once a second) and never blocks a frame; the pose model reuses the box from the previous frame's keypoints. The
   detector only runs in line when the app has lost you (start-up, stepping out of view). `--det-every 1` runs it in
   line on every frame, which is much slower. On one test machine (CPU only, lightweight models, 720p) the pose model
   alone took ~27 ms/frame and the detector ~160 ms, which is why this matters.
4. Lower the camera size: `--width 640 --height 480` (feet-only profiles don't need 720p; hand profiles do better with it).
5. NVIDIA GPU: `pip install onnxruntime-gpu` (instead of `onnxruntime`), then `--device cuda`.
6. Hand profiles add up to two small hand-model runs per frame; `--no-hand-refine` removes that cost.
7. Close other heavy programs; laptops on battery saver throttle the CPU.

The on-screen line "model N ms/frame" shows the model time live.

## Model

| Profile uses | Model (rtmlib) | Keypoints used |
|---|---|---|
| feet only | `BodyWithFeet` (RTMPose, Halpe-26) | ankle, big toe, heel per foot |
| any hand / fingers | `Wholebody` (RTMW, 133 pts) | feet as above + wrist, 5 knuckles, 5 fingertips per hand |

It switches automatically when the profile changes.

**Finger refinement.** The whole-body model squeezes the entire person into a 192x256 image, so a hand that is
~25 px wide in the camera frame is read from only a few pixels. When a hand is in the profile, a second
stage crops each hand out of the *full-resolution* frame and runs a hand-only model (RTMPose-m, hand5, 256x256
input) on it. The crop box follows the whole-body hand points, is smoothed, and is held for 0.3 s if the
whole-body model loses the hand. The refined points are used only if the hand model is itself confident
(mean score >= 0.35); otherwise the whole-body points are used as before. A finger's flexion is also rescaled by
the current wrist-to-palm distance, so moving the hand toward or away from the camera is not read as bending a finger.
Cost: up to two extra small-model inferences per frame (not measured here). `--no-hand-refine` turns it off.
Not measured on real hands: how much better it is than the whole-body model alone. MediaPipe is no longer used: its pose detector uses the
face as a proxy for the person and assumes the head is visible, which a floor-level, feet-only camera violates.

## Profiles (what the frontend writes)

`--profile` takes a preset name or a JSON file. If it is a file, the app **re-reads it whenever it changes**, so a
frontend can switch how someone plays without restarting. Invalid or half-written files are ignored.
Presets as JSON are in `profiles/` (`python foot_piano.py --dump-presets DIR` regenerates them).

```json
{
  "name": "right hand (3 fingers) + left foot",
  "effectors": [
    {"id": "L_FOOT", "zones": 4},
    {"id": "R_HAND", "fingers": ["index", "middle", "ring"]}
  ],
  "mode": "FREE",
  "chords": "L_FOOT",
  "sensitivity": 1.0,
  "zone_width": 0.10,
  "sustain": false,
  "tap_len": 0.25
}
```

| field | meaning |
|---|---|
| `effectors[].id` | `L_FOOT`, `R_FOOT`, `L_HAND`, `R_HAND` |
| `effectors[].zones` | feet / whole-hand pad: how many notes it chooses between (feet default 4, hand pad 3) |
| `effectors[].fingers` | hands only: which of `thumb index middle ring pinky` can be used; each is one key. Omit for a whole-hand pad |
| `mode` | start mode: `FREE`, `GUIDED`, `CHORD`, `SPLIT` |
| `chords` | `SPLIT`: which effector plays chords (default: first foot) |
| `sensitivity` | 0.2-5. Higher = smaller movements count |
| `zone_width` | one zone's width as a fraction of frame width |
| `sustain` | `false`: a tap plays a fixed-length note. `true`: the note lasts while held |
| `tap_len` | seconds, when `sustain` is false |
| `meta` | anything the frontend wants to keep; ignored |

**Layout adapts to the slot count** (a slot = one foot zone or one finger). Slots are ordered low to high across
the body (left before right, foot before hand; right hand thumb-to-pinky rises in pitch, left hand pinky-to-thumb).
8 slots get all 8 keys; 7 get C-B; 6 add high C; 5 get the pentatonic C D E G A; 4 get C E G high-C; 3 get C E G;
more than 8 share keys. `CHORD` hands out I, V, IV, vi first. `GUIDED` makes any slot play the next song note,
so it works for any profile, even a single slot.

Left/right feet and hands are tracked **by position**, not by the model's own left/right labels, which can flip
when only part of the body is visible.

## Hands-free controls

Hold any tracked body part over a button in the top strip for ~1.2 s: MODE, SONG, PROFILE (next preset), RECAL, STOP.
Notes are never played from inside that strip. The keyboard still works for a helper: C M N P R Q.
Whether a foot can reach the strip depends on camera framing - check it with each player.

## Recording a session and replaying it (tuning without a dataset)

```
python foot_piano.py --no-pico --profile two_feet --record session1.jsonl     # play normally, then quit
python tools/replay.py session1.jsonl                                          # summary: notes per zone, bursts, thresholds
python tools/replay.py session1.jsonl --sens 0.7 --zone-width 0.14             # same footage, other settings
python tools/replay.py session1.jsonl --csv lift.csv                           # per-frame lift signal for plotting
```
The recording holds keypoint coordinates only (no video, no images). Replaying it runs the identical tap logic, so
you can try settings on one recording instead of re-doing the movement each time, and you can send the file to
someone else to debug.

**Zone width.** Zones keep the width set by `zone_width` (default 0.10 of the frame width). Left and right lanes may
overlap in floor space - each foot is tracked separately, so that cannot cause a wrong-foot note - and are then drawn in
two rows. Raise `zone_width` (try 0.13-0.15) if your foot placement is not precise enough for the zones.

## Robustness and security notes

- A profile file written by a frontend is validated strictly (types, ranges, size <= 64 KB, names stripped to
  printable text). Anything invalid is ignored and the current profile keeps playing.
- If the Pico is unplugged mid-play the app stays up and shows "PICO LINK PROBLEM" instead of crashing.
- Only the player is tracked once calibrated: with several people in view, the person nearest the calibrated
  position is followed, so a helper walking in does not take over. **During the ~1 s calibration only the
  player should be in view** (it picks the most confident person then).
- The Pico drops over-long serial lines whole, refuses servo angles outside 0-180, survives I2C errors, and
  lets go of any key held longer than 4 s. Serial commands are not authenticated: anything that can write to the
  USB port can move the servos, which is fine for a USB device on your own desk.
- rtmlib downloads model files over HTTPS from OpenMMLab / Hugging Face and does **not** verify a checksum.
  The versions are pinned in `requirements.txt`; for a shared or clinical deployment, download the models once,
  check their hashes yourself, and keep them offline.

## Tests and measurements

```
cd laptop && python -m pytest tests -q          # 203 tests, no camera or model needed
python tests/bench_vs_legacy.py                 # old tap logic vs new, on simulated foot motion
cd ../pico && python -m pytest -q               # 33 tests: firmware motion maths against a mocked I2C
```

**These test the logic on simulated keypoints. They say nothing about how well the pose model tracks real feet
or real hands.** To measure that, record the target pose with a phone in slow motion beside the webcam, label ~300 tap
events, run each model on the same clips, and compare missed taps, false taps per minute while idle, wrong-zone rate and
onset timing error.

Simulation results (3-5 seeds, 30 fps, keypoint jitter ~3 px unless noted):

- toe taps, still foot: 100% detected, 0 false. Small toe taps (0.3 foot-lengths): 100%.
- 1-3 taps/s: 100%. 4 taps/s: ~94%.
- depth drift of +-0.06 frame-heights while tapping: 100% detected, 0 false (old logic: 2 of 12, 1 false).
- idle with drift, sideways slides, or 3x jitter: 0 false notes.
- **Known limit:** 2x jitter (~6 px) *and* continuous large drift: about 3 false notes per 100 s, and with taps
  on top only ~40% are detected (it misses rather than invents). Use better light, a closer camera, or
  `--model-mode performance`.

## Known limits (read before relying on it)

- Nothing here has been run on a real camera, the real models, or the real Pico/servos.
- **Menu strip vs. hands:** the top 9% of the picture is the button strip. Notes are suppressed and the dwell
  menu can trigger if a *hand* is held up there. Frame the camera so hands play below the strip, or use the keyboard.
- Finger tracking at a distance, or on partial/residual digits, is unvalidated. Fingers shorter than a quarter of
  the palm are disabled with an on-screen warning.
- If the hand-only model has not seen a hand well (e.g. very unusual anatomy), it can be confidently wrong;
  the confidence gate only catches the cases where it knows it is unsure.
- Calibration assumes a relaxed, still pose. If the player cannot hold still for ~1 s, calibration will not finish.

## Tuning knobs

`pianolib/detect.py` (`BASE_LIFT_HI`, `TRIG_FRAC`, `V_MIN`, `REFRACTORY_S`), `pianolib/rig.py` (`MIN_SCORE`,
`LOST_GRACE_S`, learning gates), `pianolib/filters.py` (`DEFAULT_MIN_CUTOFF`, `DEFAULT_BETA`).
The defaults were tuned on simulated motion; expect to retune once real video is available.
