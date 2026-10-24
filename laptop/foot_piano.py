"""
foot_piano.py - runs on the laptop.

Watches your FEET with the webcam (MediaPipe Pose) and presses real piano keys
through the Pico + servos. No extra hardware: same Pico, PCA9685 and SG90s.

How you play:
  - Camera sits on the floor (or a low tripod) 1-1.5 m in front of you, looking at your feet.
  - The screen is split into 8 zones: C D E F G A B C.
  - Lift a foot, move it over a zone, put it down -> that key presses and holds.
  - Lift the foot -> key releases.
  - Both feet work at the same time (two-note chords, melody + bass, etc).

Modes (press M to switch):
  FREE    each zone plays its own note
  GUIDED  ANY foot-down plays the next note of the song, with your timing.
          One foot, one movement, a whole song.
  CHORD   each zone plays a 3-note chord built on that note

Keys:
  C  calibrate the floor (stand/sit with both feet flat for ~1 second)
  M  next mode            N  next song           R  release everything
  H  switch feet <-> hands (hands = original fist mode, as a backup)
  Q / Esc quit

Run:
  pip install opencv-python mediapipe==0.10.9 numpy pyserial
  python foot_piano.py --port COM7          (Windows)
  python foot_piano.py --port /dev/cu.usbmodem1101   (Mac)
  python foot_piano.py --no-pico            (test without hardware)
"""

import argparse
import time

NUM_KEYS = 8
NOTE_NAMES = ["C", "D", "E", "F", "G", "A", "B", "C"]

# Songs use key index: 0=C 1=D 2=E 3=F 4=G 5=A 6=B 7=high C
SONGS = [
    ("Ode to Joy", [2, 2, 3, 4, 4, 3, 2, 1, 0, 0, 1, 2, 2, 1, 1,
                    2, 2, 3, 4, 4, 3, 2, 1, 0, 0, 1, 2, 1, 0, 0]),
    ("Mary Had a Little Lamb", [2, 1, 0, 1, 2, 2, 2, 1, 1, 1, 2, 4, 4,
                                2, 1, 0, 1, 2, 2, 2, 2, 1, 1, 2, 1, 0]),
    ("Twinkle Twinkle", [0, 0, 4, 4, 5, 5, 4, 3, 3, 2, 2, 1, 1, 0,
                         4, 4, 3, 3, 2, 2, 1, 4, 4, 3, 3, 2, 2, 1,
                         0, 0, 4, 4, 5, 5, 4, 3, 3, 2, 2, 1, 1, 0]),
]

MODES = ["FREE", "GUIDED", "CHORD"]

# ---- foot detection tuning (fractions of frame height) ----
LIFT_THRESH = 0.045   # foot this far above its floor line = "lifted"
LAND_THRESH = 0.020   # back within this of the floor line = "down"
MIN_VIS = 0.5         # MediaPipe visibility below this = foot not seen
MAX_NOTE_S = 3.0      # auto-release a held note after this long (0 = never)
CAL_FRAMES = 25       # frames used to learn where the floor is


# --------------------------------------------------------------------------
# Talking to the Pico
# --------------------------------------------------------------------------
class PianoLink:
    """Sends P/R/A commands. Counts holders per key so overlapping chords work."""

    def __init__(self, port=None, baud=115200):
        self.ser = None
        self.holders = [0] * NUM_KEYS
        if port:
            import serial
            self.ser = serial.Serial(port, baud, timeout=0)
            time.sleep(1.5)            # Pico resets/boots when the port opens
            self.ser.reset_input_buffer()
        self.log = []                  # recent commands, shown on screen

    def _send(self, msg):
        self.log = (self.log + [msg])[-6:]
        if self.ser:
            self.ser.write((msg + "\n").encode())
        else:
            print("[pico]", msg)

    def press(self, k):
        if not 0 <= k < NUM_KEYS:
            return
        self.holders[k] += 1
        if self.holders[k] == 1:
            self._send(f"P{k}")

    def release(self, k):
        if not 0 <= k < NUM_KEYS or self.holders[k] == 0:
            return
        self.holders[k] -= 1
        if self.holders[k] == 0:
            self._send(f"R{k}")

    def release_all(self):
        self.holders = [0] * NUM_KEYS
        self._send("A")

    def held(self, k):
        return self.holders[k] > 0

    def close(self):
        self.release_all()
        if self.ser:
            self.ser.close()


# --------------------------------------------------------------------------
# Modes: turn "a foot came down in zone Z" into keys to press
# --------------------------------------------------------------------------
class Player:
    def __init__(self, link):
        self.link = link
        self.mode = 0
        self.song = 0
        self.pos = 0
        self.active = {}               # source id -> list of keys it holds

    def keys_for(self, zone):
        m = MODES[self.mode]
        if m == "FREE":
            return [zone]
        if m == "GUIDED":
            notes = SONGS[self.song][1]
            k = notes[self.pos]
            self.pos = (self.pos + 1) % len(notes)
            return [k]
        # CHORD: triad on the scale degree (stays inside the one octave)
        d = zone % 7
        root = 7 if zone == 7 else d
        return [root, (d + 2) % 7, (d + 4) % 7]

    def start(self, src, zone):
        self.stop(src)
        keys = self.keys_for(zone)
        self.active[src] = keys
        for k in keys:
            self.link.press(k)

    def stop(self, src):
        for k in self.active.pop(src, []):
            self.link.release(k)

    def stop_all(self):
        self.active.clear()
        self.link.release_all()

    def next_mode(self):
        self.stop_all()
        self.mode = (self.mode + 1) % len(MODES)
        self.pos = 0

    def next_song(self):
        self.stop_all()
        self.song = (self.song + 1) % len(SONGS)
        self.pos = 0

    def next_note_name(self):
        if MODES[self.mode] != "GUIDED":
            return None
        return NOTE_NAMES[SONGS[self.song][1][self.pos]]

    def progress(self):
        notes = SONGS[self.song][1]
        return self.pos, len(notes)


def zone_of(x):
    return max(0, min(NUM_KEYS - 1, int(x * NUM_KEYS)))


# --------------------------------------------------------------------------
# Foot logic (no camera code here, so it can be tested with fake numbers)
# --------------------------------------------------------------------------
class FootState:
    def __init__(self, name):
        self.name = name
        self.floor_y = None            # learned floor line (normalized y)
        self.cal = []
        self.down = True               # assume planted until we see a lift
        self.armed = False             # only play after a real lift (no ghost notes)
        self.note_since = None
        self.x = self.y = None
        self.visible = False


class FootTracker:
    """
    A note fires when a foot goes LIFTED -> DOWN. It holds while the foot stays down
    and releases when the foot lifts again (or after MAX_NOTE_S).
    """

    def __init__(self, player):
        self.player = player
        self.feet = {"L": FootState("L"), "R": FootState("R")}
        self.calibrating = False

    def start_calibration(self):
        self.player.stop_all()
        self.calibrating = True
        for f in self.feet.values():
            f.cal = []
            f.floor_y = None
            f.down, f.armed, f.note_since = True, False, None

    def calibrated(self):
        return all(f.floor_y is not None for f in self.feet.values())

    def update(self, name, x, y, vis, now=None):
        now = time.time() if now is None else now
        f = self.feet[name]
        f.visible = vis >= MIN_VIS
        if not f.visible:
            if f.note_since is not None:
                self.player.stop(name)
                f.note_since = None
            return
        f.x, f.y = x, y

        if self.calibrating:
            f.cal.append(y)
            if len(f.cal) >= CAL_FRAMES:
                f.floor_y = sorted(f.cal)[len(f.cal) // 2]
            if self.calibrated():
                self.calibrating = False
            return
        if f.floor_y is None:
            return

        height = f.floor_y - y          # positive = foot is above the floor line
        if f.down and height > LIFT_THRESH:
            f.down = False
            f.armed = True
            if f.note_since is not None:
                self.player.stop(name)
                f.note_since = None
        elif not f.down and height < LAND_THRESH:
            f.down = True
            if f.armed:
                self.player.start(name, zone_of(x))
                f.note_since = now
                f.armed = False

        if f.note_since is not None and MAX_NOTE_S and now - f.note_since > MAX_NOTE_S:
            self.player.stop(name)
            f.note_since = None


# --------------------------------------------------------------------------
# Hands (backup mode, same idea as the original air_piano.py: fist = press)
# --------------------------------------------------------------------------
def is_fist(lm):
    # fingertip below its middle knuckle (in image y) = curled
    return all(lm[tip].y > lm[pip].y for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)))


class HandTracker:
    def __init__(self, player):
        self.player = player
        self.state = {}                # hand id -> (pressed, zone)

    def update(self, hand_id, x, fist):
        pressed, zone = self.state.get(hand_id, (False, None))
        if fist and not pressed:
            z = zone_of(x)
            self.player.start(hand_id, z)
            self.state[hand_id] = (True, z)
        elif not fist and pressed:
            self.player.stop(hand_id)
            self.state[hand_id] = (False, None)

    def lost(self, hand_id):
        if self.state.get(hand_id, (False,))[0]:
            self.player.stop(hand_id)
        self.state.pop(hand_id, None)


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def draw(frame, cv2, player, link, feet, input_mode, fps):
    h, w = frame.shape[:2]
    zw = w / NUM_KEYS
    overlay = frame.copy()
    for k in range(NUM_KEYS):
        x0, x1 = int(k * zw), int((k + 1) * zw)
        held = link.held(k)
        color = (60, 200, 90) if held else ((70, 70, 70) if k % 2 else (100, 100, 100))
        cv2.rectangle(overlay, (x0, int(h * 0.80)), (x1, h), color, -1)
        cv2.line(frame, (x0, 0), (x0, h), (200, 200, 200), 1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for k in range(NUM_KEYS):
        cx = int((k + 0.5) * zw)
        cv2.putText(frame, NOTE_NAMES[k], (cx - 12, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if input_mode == "feet":
        for f in feet.feet.values():
            if f.floor_y is not None:
                y_floor = int(f.floor_y * h)
                y_lift = int((f.floor_y - LIFT_THRESH) * h)
                cv2.line(frame, (0, y_floor), (w, y_floor), (0, 180, 255), 1)
                cv2.line(frame, (0, y_lift), (w, y_lift), (0, 90, 255), 1)
            if f.visible and f.x is not None:
                c = (60, 220, 90) if f.note_since else ((0, 200, 255) if not f.down else (255, 255, 255))
                p = (int(f.x * w), int(f.y * h))
                cv2.circle(frame, p, 14, c, -1)
                cv2.putText(frame, f.name, (p[0] - 7, p[1] + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    lines = [f"Mode: {MODES[player.mode]}   Input: {input_mode.upper()}   {fps:4.1f} fps"]
    if MODES[player.mode] == "GUIDED":
        i, n = player.progress()
        lines.append(f"Song: {SONGS[player.song][0]}  ({i}/{n})   Next: {player.next_note_name()}")
    if input_mode == "feet" and feet.calibrating:
        lines.append("CALIBRATING - keep both feet flat...")
    elif input_mode == "feet" and not feet.calibrated():
        lines.append("Press C with both feet flat on the floor to calibrate")
    lines.append("C calibrate  M mode  N song  R release  H feet/hands  Q quit")
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, t, (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="Pico serial port, e.g. COM7 or /dev/cu.usbmodem1101")
    ap.add_argument("--no-pico", action="store_true", help="run without hardware")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--hands", action="store_true", help="start in hands mode")
    args = ap.parse_args()

    import cv2
    import mediapipe as mp

    link = PianoLink(None if args.no_pico else args.port)
    player = Player(link)
    feet = FootTracker(player)
    hands_tr = HandTracker(player)
    input_mode = "hands" if args.hands else "feet"

    pose = mp.solutions.pose.Pose(model_complexity=0, smooth_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=0,
                                     min_detection_confidence=0.6, min_tracking_confidence=0.5)
    P = mp.solutions.pose.PoseLandmark

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if input_mode == "feet":
        feet.start_calibration()

    last, fps = time.time(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)         # mirror so left/right match the player
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if input_mode == "feet":
                res = pose.process(rgb)
                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark
                    # after mirroring, the person's left foot shows on screen-left
                    for name, idx in (("L", P.RIGHT_FOOT_INDEX), ("R", P.LEFT_FOOT_INDEX)):
                        p = lm[idx]
                        feet.update(name, p.x, p.y, p.visibility)
                else:
                    for name in ("L", "R"):
                        feet.update(name, 0, 0, 0.0)
            else:
                res = hands.process(rgb)
                seen = set()
                if res.multi_hand_landmarks:
                    for i, hl in enumerate(res.multi_hand_landmarks):
                        hid = f"hand{i}"
                        seen.add(hid)
                        hands_tr.update(hid, hl.landmark[9].x, is_fist(hl.landmark))
                for hid in list(hands_tr.state):
                    if hid not in seen:
                        hands_tr.lost(hid)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-3))
            last = now
            draw(frame, cv2, player, link, feet, input_mode, fps)
            cv2.imshow("Foot Piano", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                feet.start_calibration()
            elif key == ord("m"):
                player.next_mode()
            elif key == ord("n"):
                player.next_song()
            elif key == ord("r"):
                player.stop_all()
            elif key == ord("h"):
                player.stop_all()
                input_mode = "hands" if input_mode == "feet" else "feet"
                if input_mode == "feet":
                    feet.start_calibration()
    finally:
        link.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
