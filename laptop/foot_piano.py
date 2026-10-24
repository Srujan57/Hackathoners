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
        self.aspect = aspect
        self.set_profile(profile)

    def set_profile(self, profile):
        # Build the (possibly new) model first: if loading it fails, the old profile keeps working.
        backend = self.backend
        if backend is None or backend.need_hands != profile.needs_hands:
            backend = self.backend_factory(profile.needs_hands)        # hands need the whole-body model
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
        kpts, scores = self.backend.infer(frame)
        ref = getattr(self.backend, "refiner", None)
        refine = (lambda k, s: ref.refine(frame, k, s, tcap)) if ref is not None else None
        obs = extract_observations(kpts, scores, self.backend.layout, W, H, self.profile,
                                   self.rig.last_x(), self.rig.calibrating, refine=refine)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="Pico serial port, e.g. COM7 or /dev/cu.usbmodem1101")
    ap.add_argument("--no-pico", action="store_true", help="run without hardware")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--profile", default="two_feet", help="preset name or path to a profile JSON")
    ap.add_argument("--model-mode", default="balanced", choices=["lightweight", "balanced", "performance"],
                    help="rtmlib model size: lightweight is fastest, performance most accurate")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--no-hand-refine", action="store_true",
                    help="skip the hand-only crop model (faster, but fingers are less reliable)")
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

    profile = load_profile(args.profile)
    watcher = ProfileFile(args.profile if os.path.exists(args.profile) else None)
    link = PianoLink(None if args.no_pico else args.port)

    def make_backend(need_hands):
        print("[model] loading RTMPose (%s) - the first run downloads the weights..." % args.model_mode)
        return PoseBackend(need_hands, mode=args.model_mode, device=args.device,
                           refine_hands=not args.no_hand_refine)

    app = App(profile, link, make_backend, args.width / float(args.height))
    cam = LatestFrameCamera(args.camera, args.width, args.height)

    last, fps, lat_ms = time.monotonic(), 0.0, 0.0
    try:
        while True:
            frame, tcap = cam.read(timeout=2.0)
            if frame is None:
                print("[camera] no frames - check --camera")
                break

            newp = watcher.poll()
            if newp is not None:
                try:
                    app.player.stop_all()
                    app.set_profile(newp)
                except Exception as e:                # e.g. the new model could not be loaded
                    print("[profile] could not switch, keeping the current one:", e)

            app.process(frame, tcap)
            H, W = frame.shape[:2]
            btn = app.menu.update(time.monotonic(), app.pointers_px(H), W, H)
            if btn:
                app.press_button(btn)

            now = time.monotonic()
            lat_ms = 0.9 * lat_ms + 0.1 * (now - tcap) * 1000.0
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-3))
            last = now

            disp = cv2.flip(frame, 1)                 # mirror: your left is on screen-left
            draw(disp, cv2, app.rig, app.player, link, app.menu, fps, lat_ms)
            cv2.imshow("Foot Piano", disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
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
        link.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
