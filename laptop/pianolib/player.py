"""Songs, modes and the serial link to the Pico."""

import time

from .layout import NOTE_NAMES, NUM_KEYS, available_modes, build_slotmap

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

MAX_NOTE_S = 3.0      # safety: a sustained note never lasts longer than this


class PianoLink:
    """Sends P/R/A commands to the Pico. Counts holders per key so overlapping chords work."""

    def __init__(self, port=None, baud=115200):
        self.ser = None
        self.error = ""                # non-empty = the link is broken (shown on screen)
        self.holders = [0] * NUM_KEYS
        if port:
            import serial
            try:
                self.ser = serial.Serial(port, baud, timeout=0, write_timeout=0.2)
            except (serial.SerialException, OSError) as e:
                raise SystemExit("Cannot open the Pico on %s: %s\n(is the port right, and is Thonny or another "
                                 "program holding it?)" % (port, e))
            time.sleep(1.5)            # Pico resets/boots when the port opens
            self.ser.reset_input_buffer()
        self.log = []                  # recent commands, shown on screen

    def _send(self, msg):
        self.log = (self.log + [msg])[-6:]
        if self.ser is None:
            print("[pico]", msg)
            return
        try:
            if self.ser.in_waiting:            # the Pico's prints must be drained or its USB buffer can fill
                self.ser.read(self.ser.in_waiting)
            self.ser.write((msg + "\n").encode())
            self.error = ""
        except Exception as e:                 # unplugged / stalled: keep the app alive and say so
            if not self.error:
                print("[pico] link problem:", e)
            self.error = str(e)[:60] or "write failed"

    def press(self, k):
        if not 0 <= k < NUM_KEYS:
            return
        self.holders[k] += 1
        if self.holders[k] == 1:
            self._send("P%d" % k)

    def release(self, k):
        if not 0 <= k < NUM_KEYS or self.holders[k] == 0:
            return
        self.holders[k] -= 1
        if self.holders[k] == 0:
            self._send("R%d" % k)

    def release_all(self):
        self.holders = [0] * NUM_KEYS
        self._send("A")

    def held(self, k):
        return self.holders[k] > 0

    def close(self):
        self.release_all()
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass


class Player:
    """Turns slot events (a foot zone / a finger was triggered) into key presses."""

    def __init__(self, link, profile):
        self.link = link
        self.song = 0
        self.pos = 0
        self.active = {}               # source (effector, slot) -> keys it holds
        self.release_at = {}           # source -> time to let go
        self.set_profile(profile)

    # ---- profile / mode -------------------------------------------------
    def set_profile(self, profile):
        self.stop_all()
        self.profile = profile
        self.modes = available_modes(profile)
        self.mode = profile.mode if profile.mode in self.modes else "FREE"
        self.pos = 0
        self._rebuild()

    def _rebuild(self):
        self.slotmap = build_slotmap(self.profile, self.mode)

    def next_mode(self):
        self.stop_all()
        self.mode = self.modes[(self.modes.index(self.mode) + 1) % len(self.modes)]
        self.pos = 0
        self._rebuild()

    def next_song(self):
        self.stop_all()
        self.song = (self.song + 1) % len(SONGS)
        self.pos = 0

    def label_for(self, src):
        a = self.slotmap.get(src)
        return a.label if a else ""

    # ---- playing --------------------------------------------------------
    def on_event(self, eid, slot, kind, t):
        src = (eid, slot)
        if kind == "down":
            self.start(src, t)
        elif kind == "up" and self.profile.sustain:
            self.stop(src)

    def start(self, src, t):
        act = self.slotmap.get(src)
        if act is None:
            return
        self.stop(src)
        if act.is_next:
            notes = SONGS[self.song][1]
            keys = [notes[self.pos]]
            self.pos = (self.pos + 1) % len(notes)
        else:
            keys = list(act.keys)
        self.active[src] = keys
        for k in keys:
            self.link.press(k)
        self.release_at[src] = t + (MAX_NOTE_S if self.profile.sustain else self.profile.tap_len)

    def stop(self, src):
        self.release_at.pop(src, None)
        for k in self.active.pop(src, []):
            self.link.release(k)

    def stop_all(self):
        self.active.clear()
        self.release_at.clear()
        self.link.release_all()

    def tick(self, t):
        """Call every frame: ends notes whose time is up."""
        for src, when in list(self.release_at.items()):
            if t >= when:
                self.stop(src)

    # ---- display helpers ------------------------------------------------
    def next_note_name(self):
        if self.mode != "GUIDED":
            return None
        return NOTE_NAMES[SONGS[self.song][1][self.pos]]

    def progress(self):
        return self.pos, len(SONGS[self.song][1])
