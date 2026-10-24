"""
foot_piano.py - runs on the laptop.

Watches the player with a webcam (RTMPose pose model) and presses real piano keys through the Pico +
servos. Which body parts play is chosen by a PROFILE (feet, one hand + one foot, specific fingers, ...);
the key layout, calibration and even the model adapt to it.

How you play
  - Camera 1-1.5 m away, raised and angled down so it sees the feet (and hands, if the profile uses them).
  - Hold still for ~1 s when the app starts: it calibrates itself (feet flat, hands open).
  - Each foot owns a lane of zones around where it rested. Tap a toe (or step) in a zone -> that note.
  - A finger owns one key. Bend the finger -> that note.
  - A tap plays a fixed-length note by default; set "sustain": true in the profile to hold while held.

Modes (MODE button / M)
  FREE    each slot plays its own note (layout adapts: 8 slots -> 8 keys, 4 -> C E G C, 3 -> C E G ...)
  GUIDED  ANY slot plays the next note of the song, with your timing
  CHORD   each slot plays a 3-note chord (I, V, IV, vi first)
  SPLIT   one effector (default: first foot) plays chords, the others play the melody

Hands-free: hold any tracked body part over a button in the top strip for ~1.2 s.
Keyboard (for a helper): C recalibrate  M mode  N song  P next preset  R release all  Q/Esc quit

Run (see requirements.txt):
  python foot_piano.py --port COM7 --profile two_feet
  python foot_piano.py --port COM7 --profile my_profile.json      (a frontend can rewrite this file live)
  python foot_piano.py --no-pico                                    (no hardware: prints commands)
  python foot_piano.py --list-presets
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pianolib.pose import extract_observations                        # noqa: E402
from pianolib.player import PianoLink, Player                         # noqa: E402
from pianolib.profile import (PRESET_ORDER, PRESETS, ProfileFile,    # noqa: E402
                              load_profile)
from pianolib.rig import Rig                                          # noqa: E402
from pianolib.ui import MENU_H, DwellMenu                             # noqa: E402


class App:
    """Everything except the camera window, so it can be driven by a test."""

    def __init__(self, profile, link, backend_factory, aspect=16 / 9.0):
        self.link = link
        self.backend_factory = backend_factory
        self.backend = None
        self.menu = DwellMenu()
        self.preset_i = 0
        self.player = Player(link, profile)
        self.infer_ms = 0.0                      # smoothed model time per frame, shown on screen
        self.last_obs = {}
        self.aspect = aspect
        self.set_profile(profile)

    def set_profile(self, profile):
        # Build the (possibly new) model first: if loading it fails, the old profile keeps working.
        backend = self.backend
        if backend is None or backend.need_hands != profile.needs_hands:
            backend = self.backend_factory(profile.needs_hands)        # hands need the whole-body model
        if backend is not self.backend and self.backend is not None and hasattr(self.backend, "close"):
            self.backend.close()                                       # stop the old model's background thread
        self.backend = backend
        self.profile = profile
        self.player.set_profile(profile)
        self.rig = Rig(profile, self.aspect)
        if getattr(self.backend, "refiner", None) is not None:
            self.backend.refiner.reset()
        print("[profile] %s -> %s model" % (profile.name, self.backend.layout))

    def process(self, frame, tcap):
        """One frame in -> note events out (already forwarded to the player)."""
        H, W = frame.shape[:2]
        self.aspect = self.rig.aspect = W / float(H)
        t0 = time.perf_counter()
        kpts, scores = self.backend.infer(frame)
        self.infer_ms = 0.9 * self.infer_ms + 0.1 * (time.perf_counter() - t0) * 1000.0
        ref = getattr(self.backend, "refiner", None)
        refine = (lambda k, s: ref.refine(frame, k, s, tcap)) if ref is not None else None
        obs = extract_observations(kpts, scores, self.backend.layout, W, H, self.profile,
                                   self.rig.last_x(), self.rig.calibrating, refine=refine)
        self.last_obs = obs
        events = self.rig.update(tcap, obs, menu_y=MENU_H)
        for eid, slot, kind in events:
            self.player.on_event(eid, slot, kind, tcap)
        self.player.tick(time.monotonic())
        return events

    def pointers_px(self, H):
        return [(p[0] * H, p[1] * H) for p in self.rig.pointers()]

    def press_button(self, name):
        if name == "MODE":
            self.player.next_mode()
        elif name == "SONG":
            self.player.next_song()
        elif name == "RECAL":
            self.player.stop_all()
            self.rig.recalibrate()
        elif name == "STOP":
            self.player.stop_all()
        elif name == "PROFILE":
            nxt = (self.preset_i + 1) % len(PRESET_ORDER)
            try:
                self.set_profile(PRESETS[PRESET_ORDER[nxt]])
                self.preset_i = nxt
            except Exception as e:                    # e.g. the hand model could not be loaded
                print("[profile] could not switch:", e)


def tap_readout(rig):
    """One HUD line per tap effector: current lift vs the lift needed (both in % of foot length)."""
    out = []
    for e in rig.effs.values():
        det = getattr(e, "det", None)
        if det is None or not getattr(e, "calibrated", False) or not hasattr(e, "last_lift"):
            continue
        out.append("%s lift %3.0f%% / needs %3.0f%% (noise %.0f%%)"
                   % (e.id, max(0.0, e.last_lift) * 100, det.hi * 100, getattr(e, "sigma", 0.0) * 100))
    return out


def run_bench(args, profile):
    """Time each model size on real camera frames (stand in view while it runs)."""
    from pianolib.pose import LatestFrameCamera, PoseBackend
    cam = LatestFrameCamera(args.camera, args.width, args.height)
    try:
        for _ in range(10):                       # let the camera settle
            frame, _t = cam.read(timeout=2.0)
        if frame is None:
            print("[bench] no camera frames - check --camera")
            return
        print("[bench] profile %s, %dx%d frames, device %s. Stay in view.\n" %
              (profile.name, frame.shape[1], frame.shape[0], args.device))
        for mode in ("lightweight", "balanced"):
            for det_every in (1, max(2, args.det_every)):          # 1 = in line every frame; else background
                be = PoseBackend(profile.needs_hands, mode=mode, device=args.device,
                                 refine_hands=False, det_every=det_every)
                for _ in range(3):
                    be.infer(frame)               # warm-up
                ts = []
                for _ in range(40):
                    f, _t = cam.read(timeout=2.0)
                    t0 = time.perf_counter()
                    be.infer(f if f is not None else frame)
                    ts.append((time.perf_counter() - t0) * 1000.0)
                ts.sort()
                be.close()
                print("%-12s detector %-18s: mean %5.0f ms  p95 %5.0f ms  -> about %4.1f fps (model only)"
                      % (mode, "every frame" if det_every == 1 else "in background", sum(ts) / len(ts), ts[int(0.95 * len(ts)) - 1], 1000.0 / (sum(ts) / len(ts))))
        print("\nTap detection wants roughly 20+ fps. Pick the slowest option that still gets there.")
    finally:
        cam.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="Pico serial port, e.g. COM7 or /dev/cu.usbmodem1101")
    ap.add_argument("--no-pico", action="store_true", help="run without hardware")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--profile", default="two_feet", help="preset name or path to a profile JSON")
    ap.add_argument("--model-mode", default="lightweight", choices=["lightweight", "balanced", "performance"],
                    help="rtmlib model size: lightweight is fastest (default), performance most accurate")
    ap.add_argument("--det-every", type=int, default=30,
                    help="refresh the person detector in the background every N frames "
                         "(1 = run it in line every frame: much slower)")
    ap.add_argument("--sound", action="store_true", help="play notes on the laptop speakers (default with --no-pico)")
    ap.add_argument("--mute", action="store_true", help="never play sound")
    ap.add_argument("--record", metavar="FILE",
                    help="save the tracked keypoints (no video, no images) to FILE for tools/replay.py")
    ap.add_argument("--test-sound", action="store_true",
                    help="play a test tune through the laptop speakers and exit (checks sound without a camera)")
    ap.add_argument("--bench", action="store_true",
                    help="time the pose models on your camera, print ms/frame, and exit")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--no-hand-refine", action="store_true",
                    help="skip the hand-only crop model (faster, but fingers are less reliable)")
    ap.add_argument("--sens", type=float, metavar="X",
                    help="override the profile's sensitivity (0.2-5). Higher = smaller foot lifts count as taps")
    ap.add_argument("--list-presets", action="store_true")
    ap.add_argument("--dump-presets", metavar="DIR", help="write every preset to DIR/*.json and exit")
    args = ap.parse_args()

    if args.list_presets:
        for k in PRESET_ORDER:
            print("%-18s %s" % (k, PRESETS[k].name))
        return
    if args.dump_presets:
        os.makedirs(args.dump_presets, exist_ok=True)
        for k in PRESET_ORDER:
            with open(os.path.join(args.dump_presets, k + ".json"), "w", encoding="utf-8") as f:
                json.dump(PRESETS[k].to_dict(), f, indent=2)
        print("wrote", len(PRESET_ORDER), "presets to", args.dump_presets)
        return

    import cv2
    from pianolib.pose import LatestFrameCamera, PoseBackend
    from pianolib.ui import draw

    if args.test_sound:
        from pianolib.audio import self_test
        print(self_test())
        return
    if args.sens is not None and not 0.2 <= args.sens <= 5.0:
        raise SystemExit("--sens must be between 0.2 and 5")
    profile = load_profile(args.profile)
    if args.sens is not None:
        profile.sensitivity = args.sens
    if args.bench:
        return run_bench(args, profile)
    watcher = ProfileFile(args.profile if os.path.exists(args.profile) else None)
    link = PianoLink(None if args.no_pico else args.port)
    audio = None
    if (args.no_pico or args.sound) and not args.mute:
        try:
            from pianolib.audio import AudioOut
            audio = AudioOut()
            link.sinks.append(audio)
            audio.chime()                         # you should hear C-E-G now; if not, sound is the problem
            print("[audio] software piano on (%s, %d Hz) - you should hear a short C-E-G chime"
                  % (audio.device_name, audio.synth.sr))
        except Exception as e:                    # no sound must not stop the rest
            print("[audio] could not start sound (%s). Try:  pip install sounddevice" % e)

    def make_backend(need_hands):
        print("[model] loading RTMPose (%s) - the first run downloads the weights..." % args.model_mode)
        return PoseBackend(need_hands, mode=args.model_mode, device=args.device,
                           refine_hands=not args.no_hand_refine, det_every=args.det_every)

    app = cam = rec = None
    try:
        app = App(profile, link, make_backend, args.width / float(args.height))
        cam = LatestFrameCamera(args.camera, args.width, args.height)
        if args.record:
            try:
                rec = open(args.record, "x", encoding="utf-8")        # never overwrite an existing file
            except FileExistsError:
                raise SystemExit("%s already exists - choose another name for --record." % args.record)
            rec.write(json.dumps({"profile": profile.to_dict(), "aspect": app.aspect, "model": args.model_mode,
                                  "width": args.width, "height": args.height}) + "\n")
            print("[record] saving keypoints to", args.record)
        last, fps, lat_ms, nframes, last_err = time.monotonic(), 0.0, 0.0, 0, 0.0
        while True:
            frame, tcap = cam.read(timeout=2.0)
            if frame is None:
                print("[camera] no frames - check --camera")
                break

            newp = watcher.poll()
            if newp is not None:
                try:
                    app.player.stop_all()
                    if args.sens is not None:
                        newp.sensitivity = args.sens
                    app.set_profile(newp)
                except Exception as e:                # e.g. the new model could not be loaded
                    print("[profile] could not switch, keeping the current one:", e)

            try:
                events = app.process(frame, tcap)
            except Exception as e:                    # one bad frame (model hiccup) must not end the performance
                events = []
                app.player.stop_all()
                if time.monotonic() - last_err > 2.0:
                    print("[error] frame skipped:", repr(e))
                    last_err = time.monotonic()
            if rec is not None:
                rec.write(json.dumps({"t": round(tcap, 4), "ev": [list(e) for e in events], "obs": {
                    eid: {n: [round(float(v[0]), 4), round(float(v[1]), 4), round(float(v[2]), 3)]
                          for n, v in pts.items()} for eid, pts in app.last_obs.items()}}) + "\n")
            H, W = frame.shape[:2]
            btn = app.menu.update(time.monotonic(), app.pointers_px(H), W, H)
            if btn:
                app.press_button(btn)

            now = time.monotonic()
            lat_ms = 0.9 * lat_ms + 0.1 * (now - tcap) * 1000.0
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-3))
            last = now
            nframes += 1

            disp = cv2.flip(frame, 1)                 # mirror: your left is on screen-left
            slow = "   (slow - see README: Speed)" if nframes > 60 and fps < 15 else ""
            draw(disp, cv2, app.rig, app.player, link, app.menu, fps, lat_ms,
                 hud_extra=["model %.0f ms/frame%s" % (app.infer_ms, slow)] + tap_readout(app.rig))
            cv2.imshow("Foot Piano", disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty("Foot Piano", cv2.WND_PROP_VISIBLE) < 1:
                break                                 # the window's X button
            elif key == ord("c"):
                app.press_button("RECAL")
            elif key == ord("m"):
                app.press_button("MODE")
            elif key == ord("n"):
                app.press_button("SONG")
            elif key == ord("p"):
                app.press_button("PROFILE")
            elif key == ord("r"):
                app.press_button("STOP")
    finally:
        if rec is not None:
            rec.close()
        if app is not None and app.backend is not None and hasattr(app.backend, "close"):
            app.backend.close()
        link.close()
        if audio is not None:
            audio.close()
        if cam is not None:
            cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
