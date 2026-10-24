"""Tap / press detection. Pure functions of numbers, so it can be tested without a camera.

Why this replaces the old FootTracker
--------------------------------------
The old logic compared a foot's image-y against ONE fixed "floor line" per foot, in fractions of the
frame height. A single camera cannot tell "foot lifted" from "foot moved away from the camera" (both
shift image-y), so drifting your foot back looked like a lift and returning looked like a landing -> ghost
notes. Thresholds also didn't scale with foot size or camera distance.

TapDetector instead works on a *lift* signal measured against an ADAPTIVE baseline:
  - the baseline follows the foot while it is planted and still, so slow drift (depth, posture) is absorbed;
  - a lift is a fast excursion above the baseline, measured in units of the user's own foot length;
  - the note fires on the DOWN-stroke (descending fast and past a fraction of the peak), i.e. slightly
    BEFORE contact, which gives back some of the camera/servo latency. If the foot is lowered slowly,
    the note fires on landing instead.
"""

import math
import statistics

# ---- tuning (all lengths are in units of the user's calibrated reference length) ----
BASE_LIFT_HI = 0.12      # lift needed to count as a real lift (before sensitivity / noise floor)
NOISE_K = 6.0            # ...but never less than this many sigma of the calibration noise
LIFT_LO_FRAC = 0.5       # "landed" when lift drops below this fraction of lift_hi
TRIG_FRAC = 0.5          # fire when descending below this fraction of the peak lift
V_MIN = 0.5              # reference-lengths/second of descent needed for the early trigger
REFRACTORY_S = 0.12      # minimum time between two notes from one effector
SETTLE_S = 0.05          # foot must be down and still this long before the baseline learns
V_SETTLE = 0.8           # baseline only follows the signal when speed is below this (taps are > 2)
BASELINE_TAU_S = 0.25    # time constant of the drift-tracking baseline


class DriftBaseline:
    """Tracks where a planted point sits in the image, INCLUDING slow drift (alpha-beta filter).

    A plain moving average lags a steady drift by (speed x time-constant), and that lag looks like a
    permanent lift. This filter also estimates the drift velocity, so a foot that slowly slides toward
    or away from the camera produces no lift signal, and during a quick tap it keeps coasting along the
    drift instead of being dragged by the tap.
    """

    MAX_V = 0.15       # frame-heights per second; drift faster than this is not drift

    def __init__(self, y0, tau=BASELINE_TAU_S):
        self.b = y0
        self.v = 0.0
        self.tau = tau

    def predict(self, dt):
        self.b += self.v * dt
        self.v *= math.exp(-dt / 5.0)      # with no recent evidence, the drift estimate fades

    def correct(self, y, dt):
        if dt <= 0:
            return
        a = 1.0 - math.exp(-dt / self.tau)
        beta = a * a / (2.0 - a)
        err = y - self.b
        self.b += a * err
        self.v = max(-self.MAX_V, min(self.MAX_V, self.v + beta * err / dt))


class TapDetector:
    """lift (>0 = raised above baseline) -> 'down' (note starts) / 'up' (the next lift begins)."""

    def __init__(self, lift_hi=None, trig_frac=None, v_min=None, refractory=None):
        self.hi = BASE_LIFT_HI if lift_hi is None else lift_hi
        self.lo = self.hi * LIFT_LO_FRAC
        self.trig_frac = TRIG_FRAC if trig_frac is None else trig_frac
        self.v_min = V_MIN if v_min is None else v_min
        self.refractory = REFRACTORY_S if refractory is None else refractory
        self.reset()

    def reset(self):
        self.state = "DOWN"
        self.peak = 0.0
        self.vel = 0.0
        self.fired = False
        self.holding = False
        self.trough = 0.0
        self.last_fire = -1e9
        self._t = None
        self._lift = 0.0

    def update(self, t, lift):
        events = []
        if self._t is not None and t > self._t:
            raw = (lift - self._lift) / (t - self._t)
            self.vel = 0.5 * self.vel + 0.5 * raw
        self._t, self._lift = t, lift

        if self.state == "DOWN":
            if lift > self.hi:
                self.state, self.peak, self.fired = "LIFTED", lift, False
                if self.holding:
                    events.append("up")
                    self.holding = False
        else:
            self.peak = max(self.peak, lift)
            if self.fired:
                # A new lift can start before the foot ever returns to baseline (fast repeated taps).
                self.trough = min(self.trough, lift)
                if lift > self.hi and lift > self.trough + 0.75 * self.hi:
                    if self.holding:
                        events.append("up")
                        self.holding = False
                    self.peak, self.fired, self.trough = lift, False, lift
            if not self.fired and t - self.last_fire >= self.refractory:
                early = self.vel < -self.v_min and lift <= self.trig_frac * self.peak
                landed = lift < self.lo
                if early or landed:
                    events.append("down")
                    self.fired = True
                    self.holding = True
                    self.trough = lift
                    self.last_fire = t
            if lift < self.lo:
                self.state = "DOWN"
        return events


class FlexDetector:
    """Level trigger with hysteresis for finger flexion: pressed above hi, released below lo."""

    def __init__(self, hi=0.35, lo=0.2, refractory=0.10):
        self.hi, self.lo, self.refractory = hi, lo, refractory
        self.reset()

    def reset(self):
        self.pressed = False
        self.last_up = -1e9

    def update(self, t, flex):
        if not self.pressed and flex > self.hi and t - self.last_up >= self.refractory:
            self.pressed = True
            return ["down"]
        if self.pressed and flex < self.lo:
            self.pressed = False
            self.last_up = t
            return ["up"]
        return []


class ZoneMap:
    """n equal zones of `width` centred on `center` (all in frame-height units)."""

    def __init__(self, center, n, width):
        self.center, self.n, self.width = center, n, width

    @property
    def left(self):
        return self.center - self.n * self.width / 2.0

    def zone_center(self, i):
        return self.left + (i + 0.5) * self.width

    def zone_for(self, x, prev=None, margin=0.25):
        z = int(math.floor((x - self.left) / self.width))
        z = max(0, min(self.n - 1, z))
        if prev is not None and prev != z:
            # hysteresis: stay in the previous zone unless clearly closer to the new one
            if abs(x - self.zone_center(prev)) - abs(x - self.zone_center(z)) < margin * self.width:
                z = prev
        return z


def median(vals):
    return statistics.median(vals) if vals else 0.0


def stdev(vals):
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def detrended_std(ts, vals):
    """Std of the residuals after removing a straight line. Slow drift during calibration is not noise, and
    counting it as noise would raise the tap threshold for no reason."""
    n = len(vals)
    if n < 3:
        return stdev(vals)
    mt, mv = sum(ts) / n, sum(vals) / n
    den = sum((t - mt) ** 2 for t in ts)
    slope = sum((t - mt) * (v - mv) for t, v in zip(ts, vals)) / den if den > 1e-12 else 0.0
    return stdev([v - (mv + slope * (t - mt)) for t, v in zip(ts, vals)])
