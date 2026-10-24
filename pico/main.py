# main.py - runs on the Raspberry Pi Pico 2W (MicroPython)
# Receives commands from the laptop over USB serial and moves the servos.
#
# Commands (one per line):
#   P<k>                 press key k (0-7) and hold it
#   R<k>                 release key k
#   A                    release all keys
#   T<k>                 quick tap on key k
#   C<k> <rest> <press>  set calibration angles for key k (live, not saved)
#   S                    print current calibration (paste it back into this file)
#
# Wiring (same as the README):
#   GP4 -> SDA, GP5 -> SCL, 3V3 -> VCC, GND -> GND
#   Servo power: use an EXTERNAL 5V 3A+ supply into PCA9685 V+, grounds tied together.
#   (VBUS from USB can't feed 8 SG90s pressing at once - you'll get resets/jitter.)

import sys
import time
import uselect
from machine import Pin, I2C

NUM_KEYS = 8
NOTE_NAMES = ["C", "D", "E", "F", "G", "A", "B", "C2"]

# Per-key angles. Tune with C<k> <rest> <press>, then type S and paste the output here.
REST_DEG = [90, 90, 90, 90, 90, 90, 90, 90]
PRESS_DEG = [120, 120, 120, 120, 120, 120, 120, 120]

# Safety: never hold a key longer than this (protects SG90s from stalling hot,
# and frees a key if the laptop crashes mid-note).
MAX_HOLD_MS = 4000


class PCA9685:
    def __init__(self, i2c, addr=0x40):
        self.i2c = i2c
        self.addr = addr
        self._w(0x00, 0x00)          # MODE1 reset
        self.set_freq(50)

    def _w(self, reg, val):
        self.i2c.writeto_mem(self.addr, reg, bytes([val]))

    def _r(self, reg):
        return self.i2c.readfrom_mem(self.addr, reg, 1)[0]

    def set_freq(self, freq):
        prescale = int(25000000.0 / 4096.0 / freq + 0.5) - 1
        old = self._r(0x00)
        self._w(0x00, (old & 0x7F) | 0x10)   # sleep
        self._w(0xFE, prescale)
        self._w(0x00, old)
        time.sleep_us(500)
        self._w(0x00, old | 0xA1)            # restart + auto-increment

    def set_pwm(self, ch, on, off):
        self.i2c.writeto_mem(self.addr, 0x06 + 4 * ch,
                             bytes([on & 0xFF, on >> 8, off & 0xFF, off >> 8]))

    def angle(self, ch, deg):
        deg = max(0, min(180, deg))
        pulse_us = 500 + deg * 2000 / 180          # 0.5ms - 2.5ms
        self.set_pwm(ch, 0, int(pulse_us * 4096 / 20000))


i2c = I2C(0, sda=Pin(4), scl=Pin(5), freq=400000)
pca = PCA9685(i2c)
led = Pin("LED", Pin.OUT)

held_since = [0] * NUM_KEYS     # 0 = not held, otherwise ticks_ms when pressed


def press(k):
    pca.angle(k, PRESS_DEG[k])
    held_since[k] = time.ticks_ms() or 1
    led.on()


def release(k):
    pca.angle(k, REST_DEG[k])
    held_since[k] = 0
    if not any(held_since):
        led.off()


def release_all():
    for k in range(NUM_KEYS):
        release(k)


def tap(k):
    press(k)
    time.sleep_ms(150)
    release(k)


def print_cal():
    print("REST_DEG = " + str(REST_DEG))
    print("PRESS_DEG = " + str(PRESS_DEG))


def handle(line):
    line = line.strip()
    if not line:
        return
    cmd = line[0].upper()
    arg = line[1:].strip()
    try:
        if cmd == "P":
            k = int(arg)
            if 0 <= k < NUM_KEYS:
                press(k)
        elif cmd == "R":
            k = int(arg)
            if 0 <= k < NUM_KEYS:
                release(k)
        elif cmd == "A":
            release_all()
        elif cmd == "T":
            k = int(arg)
            if 0 <= k < NUM_KEYS:
                tap(k)
        elif cmd == "C":
            k, r, p = [int(x) for x in arg.split()]
            if 0 <= k < NUM_KEYS:
                REST_DEG[k] = r
                PRESS_DEG[k] = p
                pca.angle(k, r)
                print("key", k, NOTE_NAMES[k], "rest", r, "press", p)
        elif cmd == "S":
            print_cal()
    except Exception as e:
        print("bad command:", line, e)


# Move to rest one servo at a time so the supply doesn't brown out
for k in range(NUM_KEYS):
    pca.angle(k, REST_DEG[k])
    time.sleep_ms(80)
print("All servos at rest. Waiting for commands...")

poll = uselect.poll()
poll.register(sys.stdin, uselect.POLLIN)
buf = ""

while True:
    while poll.poll(0):
        ch = sys.stdin.read(1)
        if ch in ("\n", "\r"):
            handle(buf)
            buf = ""
        else:
            buf += ch
            if len(buf) > 40:      # junk on the line, drop it
                buf = ""

    now = time.ticks_ms()
    for k in range(NUM_KEYS):
        if held_since[k] and time.ticks_diff(now, held_since[k]) > MAX_HOLD_MS:
            release(k)

    time.sleep_ms(2)
