# main.py - runs on the Raspberry Pi Pico 2W (MicroPython)
# Receives commands from the laptop over USB serial and moves the servos SMOOTHLY.
#
# Commands (one per line) - unchanged, so the laptop code needs no changes:
#   P<k>                 press key k and hold it
#   R<k>                 release key k
#   A                    release all keys
#   T<k>                 quick tap on key k (non-blocking)
#   C<k> <rest> <press>  set calibration angles for key k (live, not saved)
#   S                    print current calibration (paste it back into this file)
#
# What changed from the first version, and why
#   1. Moves are EASED, not instant. A servo told to jump to an angle runs flat-out, draws a current
#      spike, overshoots and rattles. Each move is now a smoothstep ramp (zero speed at both ends),
#      updated every LOOP_MS. Attack is fast, release is gentler so keys don't slam.
#   2. PWM pulses are STAGGERED. Every channel used to start its pulse at count 0, so all servo signals
#      rose together. The PCA9685 has a programmable ON delay meant for exactly this ("minimize
#      current surges"); each channel now starts at channel x 256 counts.
#   3. Several servos starting at once (a chord) are started STAGGER_MS apart instead of together.
#   4. T<k> no longer blocks the whole program for 150 ms.
#   5. Keys live in a table (KEY_MAP), so more keys / a second PCA9685 is a data change, not a rewrite.
#
# Wiring (same as the README):
#   GP4 -> SDA, GP5 -> SCL, 3V3 -> VCC, GND -> GND
#   Servo power: EXTERNAL 5V 3A+ supply into PCA9685 V+, grounds tied together, with a big
#   electrolytic capacitor across V+/GND at the board. USB VBUS can't feed 8 servos pressing at once.

import sys
import time
import uselect
from machine import Pin, I2C

# ---------------------------------------------------------------- hardware map (edit to scale)
PCA_ADDRS = (0x40,)                           # one entry per PCA9685 board (address pins A0-A5)
KEY_MAP = [(0, ch) for ch in range(8)]        # key k -> (board index, channel on that board)
NOTE_NAMES = ["C", "D", "E", "F", "G", "A", "B", "C2"]

# Per-key angles. Tune with C<k> <rest> <press>, then type S and paste the output here.
# Tip: put REST just ABOVE the key surface - a shorter stroke is a faster, quieter key.
REST_DEG = [90, 90, 90, 90, 90, 90, 90, 90]
PRESS_DEG = [140, 140, 140, 140, 140, 140, 140, 140]

# PCA9685 internal oscillator is "25 MHz typical", not exact. Measure your board's PWM frequency
# (logic analyser / scope) and put the measured oscillator value here for accurate angles.
OSC_HZ = 25000000

# ---------------------------------------------------------------- motion tuning
ATTACK_DPS = 360        # average degrees/second pressing (peak of a smoothstep is 1.5x this)
RELEASE_DPS = 240       # average degrees/second letting go (gentler: no slam, less bounce)
MIN_MOVE_MS = 20        # even a tiny move takes at least this long
STAGGER_MS = 6          # servos that start together are started this far apart
LOOP_MS = 4             # motion update period
TAP_MS = 150            # how long T<k> holds the key
MAX_HOLD_MS = 4000      # safety: never hold a key longer (protects servos, frees keys if laptop dies)
SERVO_US_MIN, SERVO_US_MAX = 1000, 2000  # pulse for 0 and 180 degrees. Same as the original Air Piano firmware
                                         # (205-410 PCA9685 ticks at 50 Hz), so REST/PRESS angles match it.

NUM_KEYS = len(KEY_MAP)


class PCA9685:
    def __init__(self, i2c, addr=0x40, osc_hz=OSC_HZ):
        self.i2c = i2c
        self.addr = addr
        self._w(0x00, 0x00)                       # MODE1 reset
        prescale = int(osc_hz / 4096.0 / 50 + 0.5) - 1
        old = self._r(0x00)
        self._w(0x00, (old & 0x7F) | 0x10)        # sleep to change the prescaler
        self._w(0xFE, prescale)
        self._w(0x00, old)
        time.sleep_us(500)
        self._w(0x00, old | 0xA1)                 # restart + auto-increment
        # real PWM period for this prescaler, used to turn microseconds into counts
        self.period_us = 1000000.0 * 4096 * (prescale + 1) / osc_hz

    def _w(self, reg, val):
        self.i2c.writeto_mem(self.addr, reg, bytes([val]))

    def _r(self, reg):
        return self.i2c.readfrom_mem(self.addr, reg, 1)[0]

    def ticks_for(self, us):
        t = int(us * 4096 / self.period_us + 0.5)
        return max(1, min(4095, t))

    def set_ticks(self, ch, ticks):
        on = (ch * 256) & 0xFFF                   # staggered start: channels don't all rise together
        off = (on + ticks) & 0xFFF                # wraps past the end of the period; the chip allows it
        self.i2c.writeto_mem(self.addr, 0x06 + 4 * ch,
                             bytes([on & 0xFF, on >> 8, off & 0xFF, off >> 8]))


def angle_to_us(deg):
    deg = max(0.0, min(180.0, deg))
    return SERVO_US_MIN + deg * (SERVO_US_MAX - SERVO_US_MIN) / 180.0


class Key:
    """One servo with an eased trajectory (smoothstep) between two angles."""

    def __init__(self, pca, ch, rest, press):
        self.pca, self.ch = pca, ch
        self.rest, self.press = rest, press
        self.cur = float(rest)
        self.a0 = self.a1 = float(rest)
        self.t0 = 0
        self.dur = MIN_MOVE_MS
        self.moving = False
        self.last_ticks = -1
        self.held_since = 0                       # 0 = not held, else ticks_ms when pressed

    def move_to(self, target, start_ms, dps):
        self.a0, self.a1 = self.cur, float(target)
        d = abs(self.a1 - self.a0)
        self.dur = max(MIN_MOVE_MS, int(d * 1000.0 / dps))
        self.t0 = start_ms
        self.moving = d > 0.01

    def jump_to(self, deg):
        self.cur = self.a0 = self.a1 = float(deg)
        self.moving = False
        self.write(force=True)

    def update(self, now):
        if not self.moving:
            return
        e = time.ticks_diff(now, self.t0)
        if e < 0:                                 # staggered start not reached yet
            return
        if e >= self.dur:
            self.cur, self.moving = self.a1, False
        else:
            x = e / self.dur
            self.cur = self.a0 + (self.a1 - self.a0) * x * x * (3.0 - 2.0 * x)
        self.write()

    def write(self, force=False):
        ticks = self.pca.ticks_for(angle_to_us(self.cur))
        if force or ticks != self.last_ticks:     # I2C only when the pulse actually changes
            self.pca.set_ticks(self.ch, ticks)
            self.last_ticks = ticks


class Engine:
    def __init__(self, keys, led=None):
        self.keys = keys
        self.led = led
        self.tap_release = {}
        self.next_slot = time.ticks_ms()

    def _slot(self, now):
        """Start time for a new motion: now, or STAGGER_MS after the previous start if that is later."""
        t = now
        if time.ticks_diff(self.next_slot, t) > 0:
            t = self.next_slot
        self.next_slot = time.ticks_add(t, STAGGER_MS)
        return t

    def _led(self):
        if self.led is not None:
            if any(k.held_since for k in self.keys):
                self.led.on()
            else:
                self.led.off()

    def press(self, k, now):
        key = self.keys[k]
        key.move_to(key.press, self._slot(now), ATTACK_DPS)
        key.held_since = now or 1
        self._led()

    def release(self, k, now):
        key = self.keys[k]
        key.move_to(key.rest, self._slot(now), RELEASE_DPS)
        key.held_since = 0
        self.tap_release.pop(k, None)
        self._led()

    def release_all(self, now):
        for k in range(len(self.keys)):
            self.release(k, now)

    def tap(self, k, now):
        self.press(k, now)
        self.tap_release[k] = time.ticks_add(now, TAP_MS)

    def update(self, now):
        for k, key in enumerate(self.keys):
            if key.held_since and time.ticks_diff(now, key.held_since) > MAX_HOLD_MS:
                self.release(k, now)
        for k, when in list(self.tap_release.items()):
            if time.ticks_diff(now, when) >= 0:
                self.release(k, now)
        for key in self.keys:
            key.update(now)

    def calibrate(self, k, rest, press, now):
        key = self.keys[k]
        REST_DEG[k], PRESS_DEG[k] = rest, press
        key.rest, key.press = rest, press
        key.move_to(rest, now, RELEASE_DPS)

    def handle(self, line, now):
        line = line.strip()
        if not line:
            return
        cmd = line[0].upper()
        arg = line[1:].strip()
        try:
            if cmd == "P":
                k = int(arg)
                if 0 <= k < len(self.keys):
                    self.press(k, now)
            elif cmd == "R":
                k = int(arg)
                if 0 <= k < len(self.keys):
                    self.release(k, now)
            elif cmd == "A":
                self.release_all(now)
            elif cmd == "T":
                k = int(arg)
                if 0 <= k < len(self.keys):
                    self.tap(k, now)
            elif cmd == "C":
                k, r, p = [int(x) for x in arg.split()]
                if not (0 <= r <= 180 and 0 <= p <= 180):
                    print("angles must be 0-180:", line)
                elif 0 <= k < len(self.keys):
                    self.calibrate(k, r, p, now)
                    print("key", k, NOTE_NAMES[k] if k < len(NOTE_NAMES) else "", "rest", r, "press", p)
            elif cmd == "S":
                print("REST_DEG = " + str(REST_DEG))
                print("PRESS_DEG = " + str(PRESS_DEG))
        except Exception as e:
            print("bad command:", line, e)


def build_engine():
    i2c = I2C(0, sda=Pin(4), scl=Pin(5), freq=400000)
    boards = [PCA9685(i2c, a) for a in PCA_ADDRS]
    if not (len(KEY_MAP) == len(REST_DEG) == len(PRESS_DEG)):
        raise ValueError("KEY_MAP, REST_DEG and PRESS_DEG must have the same length")
    keys = [Key(boards[b], ch, REST_DEG[i], PRESS_DEG[i]) for i, (b, ch) in enumerate(KEY_MAP)]
    return Engine(keys, Pin("LED", Pin.OUT))


class LineReader:
    """Collects serial characters into lines. A line longer than MAX_LINE is dropped ENTIRELY (the old loop
    dropped only its first 40 characters and then ran the tail as a command)."""

    MAX_LINE = 40

    def __init__(self):
        self.buf = ""
        self.skipping = False

    def feed(self, ch):
        """Returns a finished line, or None."""
        if ch in ("\n", "\r"):
            line = None if self.skipping else self.buf
            self.buf, self.skipping = "", False
            return line
        if not self.skipping:
            self.buf += ch
            if len(self.buf) > self.MAX_LINE:
                self.buf, self.skipping = "", True
        return None


def run():
    eng = build_engine()
    for key in eng.keys:                          # one servo at a time so the supply doesn't brown out
        key.jump_to(key.rest)
        time.sleep_ms(80)
    print("All servos at rest. Waiting for commands...")

    poll = uselect.poll()
    poll.register(sys.stdin, uselect.POLLIN)
    reader = LineReader()
    while True:
        while poll.poll(0):
            line = reader.feed(sys.stdin.read(1))
            if line is not None:
                eng.handle(line, time.ticks_ms())
        try:
            eng.update(time.ticks_ms())
        except OSError as e:                      # I2C trouble (loose wire / brown-out): keep running
            print("i2c error:", e)
            time.sleep_ms(50)
        time.sleep_ms(LOOP_MS)


if __name__ == "__main__":
    run()
