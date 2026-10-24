"""Adaptive learning: the piano tunes itself to each player while they play.

Built on top of detect.py / rig.py (drop-in subclasses, same units, same state machines):

  STRENGTH  AdaptiveTapDetector / AdaptiveFlexDetector learn how big this person's movements are,
            including "almost taps" that never crossed the threshold, and move the threshold to half
            of their typical movement. Weak players start getting notes; strong players stop getting
            accidental ones. Never drops below the noise floor (calibration noise or tremor seen live).
  TREMOR    while a foot/finger rests, the live shake is measured and raises the noise floor.
  REACH     AdaptiveZoneMap shrinks zone width when the player never reaches the outer zones, so
            every note becomes reachable with the movement they actually have.
  AIM       a quick re-tap on the neighbouring zone right after landing near a boundary = probably a
            miss; that boundary moves a little toward where they really land.
  FATIGUE   taps getting much weaker over the session -> a "take a break" warning.
  MEMORY    everything is saved to users/<user>.json and restored on the next calibration.

Units: lift / flex are the same units the detectors already use (fractions of foot length / finger
length), so learned values carry over between sessions even if the camera moves.
"""

import json
import os
from bisect import bisect_right
from collections import deque

from .detect import LIFT_LO_FRAC, NOISE_K, FlexDetector, TapDetector, ZoneMap, median

MIN_SAMPLES = 6          # movements seen before thresholds start moving
WINDOW = 15              # recent movements used for "typical"
TARGET_FRAC = 0.5        # threshold = half of the player's typical movement
RATE = 0.3               # how fast the threshold moves toward the target
ATTEMPT_MAX_S = 0.8      # a sub-threshold excursion shorter than this counts as an attempt
TREMOR_MAX_S = 0.15      # ...but one shorter than this is a shake, not a movement
AUTOSAVE_EVERY = 10      # taps between saves
MIN_W = 0.03             # narrowest zone (same as rig.MIN_ZONE_W)
MAX_SHRINK = 0.5         # zones never shrink below half their calibrated width
MAX_NUDGE = 0.3          # a boundary never moves more than 0.3 zone widths
CORRECT_GAP_S = 0.45     # adjacent re-tap this fast after landing near a boundary = a miss
NEAR_EDGE = 0.35         # "near a boundary" = within this many zone widths
USERS_DIR = os.environ.get("PIANO_USERS_DIR", "users")


class StrengthStats:
    """How big one effector's movements are. Shared by the tap and flex detectors."""

    def __init__(self, floor_abs, ceil_abs=None, ceil_mult=2.0):
        self.floor_abs = floor_abs
        self.ceil_abs = ceil_abs
        self.ceil_mult = ceil_mult
        self.recent = deque(maxlen=WINDOW)
        self.session = []
        self.hi = None
        self.cal_hi = None
        self.cal_sigma = 0.0
        self._m = 0.0
        self._v = 0.0
        self.live_sigma = 0.0
        self.taps = 0
        self.attempts = 0

    def bind(self, cal_hi, cal_sigma):
        self.cal_hi, self.cal_sigma = cal_hi, cal_sigma
        if self.hi is not None:
            self.hi = self._clamp(self.hi)

    def observe_rest(self, v):
        d = v - self._m
        self._m += 0.05 * d
        self._v = 0.95 * (self._v + 0.05 * d * d)
        self.live_sigma = self._v ** 0.5

    def noise_floor(self):
        floor = min(self.floor_abs, self.cal_hi) if self.cal_hi else self.floor_abs
        return max(floor, NOISE_K * max(self.cal_sigma, self.live_sigma))

    def attempt_floor(self, hi):
        return max(self.noise_floor(), 0.25 * hi)

    def _ceiling(self):
        c = self.ceil_mult * (self.cal_hi or 0.0)
        return min(c, self.ceil_abs) if self.ceil_abs else c

    def _clamp(self, h):
        return max(self.noise_floor(), min(self._ceiling(), h))

    def add(self, peak, success):
        self.recent.append(peak)
        if success:
            self.taps += 1
            self.session.append(peak)
        else:
            self.attempts += 1

    def current(self):
        """The threshold this effector should use right now."""
        hi = self.hi if self.hi is not None else self.cal_hi
        if len(self.recent) >= MIN_SAMPLES:
            target = TARGET_FRAC * median(list(self.recent))
            hi = hi + RATE * (target - hi)
        self.hi = self._clamp(hi)
        return self.hi

    def tired(self):
        s = self.session
        return len(s) >= 20 and median(s[:10]) > 0 and median(s[-10:]) / median(s[:10]) < 0.7

    def to_dict(self):
        return {"hi": self.hi, "live_sigma": self.live_sigma, "recent": list(self.recent),
                "taps": self.taps, "attempts": self.attempts}

    def load(self, d):
        self.hi = d.get("hi")
        self.live_sigma = d.get("live_sigma", 0.0)
        self._v = self.live_sigma ** 2
        self.recent.extend(d.get("recent", []))
        self.taps, self.attempts = d.get("taps", 0), d.get("attempts", 0)


class AdaptiveTapDetector(TapDetector):
    def __init__(self, lift_hi, noise_sigma=0.0, stats=None, **kw):
        super().__init__(lift_hi=lift_hi, **kw)
        self.stats = stats if stats is not None else StrengthStats(floor_abs=0.03)
        self.stats.bind(lift_hi, noise_sigma)
        if self.stats.hi is not None:
            self._set(self.stats.hi)

    def reset(self):
        super().reset()
        self._exc_t = None
        self._exc_peak = 0.0
        self._exc_fast = False
        self._exc_vals = []
        self._cand_t = None
        self._cand_vals = []
        self._quiet_from = -1e9

    def _set(self, hi):
        self.hi, self.lo = hi, hi * LIFT_LO_FRAC

    def update(self, t, lift):
        was_down = self.state == "DOWN"
        events = super().update(t, lift)
        st = self.stats
        if self.state == "DOWN" and was_down:
            floor = st.attempt_floor(self.hi)
            if self._exc_t is None:                            # time movements from where they start
                if lift > 0.5 * floor:
                    if self._cand_t is None:
                        self._cand_t, self._cand_vals = t, []
                    self._cand_vals.append(lift)
                else:
                    self._cand_t = None
            if lift > floor:                                   # an excursion that hasn't crossed hi
                if self._exc_t is None:
                    t0 = self._cand_t if self._cand_t is not None else t
                    self._exc_t, self._exc_peak, self._exc_fast = t0, lift, False
                    self._exc_vals = list(self._cand_vals)
                    self._cand_t = None
                self._exc_peak = max(self._exc_peak, lift)
            if self._exc_t is not None:
                self._exc_vals.append(lift)
                if self.vel < -0.5 * self.v_min:               # came back down quickly = a tap
                    self._exc_fast = True
                if lift < 0.5 * floor:
                    dur = t - self._exc_t
                    if dur < TREMOR_MAX_S:                      # a blip, not a movement: it's tremor
                        for v in self._exc_vals:
                            st.observe_rest(v)
                    elif self._exc_fast and dur < ATTEMPT_MAX_S:
                        st.add(self._exc_peak, success=False)   # they tried; it was too small
                        self._set(st.current())
                        self._quiet_from = t + 0.4
                    self._exc_t = None
            elif t >= self._quiet_from and self._cand_t is None:
                st.observe_rest(lift)                           # resting: measure tremor
            if self.hi < st.noise_floor():                      # shaking got worse: back off now
                self._set(st.noise_floor())
                st.hi = self.hi
        else:
            self._exc_t = None
            self._quiet_from = t + 0.4
        if "down" in events:
            st.add(self.peak, success=True)
            self._set(st.current())
        return events


class AdaptiveFlexDetector(FlexDetector):
    def __init__(self, hi=0.35, lo=0.2, refractory=0.10, noise_sigma=0.0, stats=None):
        super().__init__(hi=hi, lo=lo, refractory=refractory)
        self.lo_frac = lo / hi if hi else 0.6
        self.stats = stats if stats is not None else StrengthStats(floor_abs=0.08, ceil_abs=0.7)
        self.stats.bind(hi, noise_sigma)
        if self.stats.hi is not None:
            self._set(self.stats.hi)

    def reset(self):
        super().reset()
        self._exc_t = None
        self._exc_peak = 0.0
        self._press_peak = 0.0

    def _set(self, hi):
        self.hi, self.lo = hi, hi * self.lo_frac

    def update(self, t, flex):
        was = self.pressed
        events = super().update(t, flex)
        st = self.stats
        if self.pressed:
            self._press_peak = max(self._press_peak, flex) if was else flex
        elif "up" in events:
            st.add(self._press_peak, success=True)
            self._set(st.current())
        elif not was:
            floor = st.attempt_floor(self.hi)
            if flex > floor:
                if self._exc_t is None:
                    self._exc_t, self._exc_peak = t, flex
                self._exc_peak = max(self._exc_peak, flex)
            elif self._exc_t is not None:
                if flex < 0.5 * floor:
                    if t - self._exc_t < ATTEMPT_MAX_S:
                        st.add(self._exc_peak, success=False)
                        self._set(st.current())
                    self._exc_t = None
            elif abs(flex) < self.lo:
                st.observe_rest(flex)
        return events


class ZoneState:
    """What we've learned about one effector's zones (saved between sessions)."""

    def __init__(self):
        self.w_ratio = 1.0          # learned width / calibrated width
        self.nudge = {}             # boundary index -> offset in zone widths
        self.reach = deque(maxlen=40)
        self.fixes = 0

    def to_dict(self):
        return {"w_ratio": self.w_ratio, "nudge": {str(k): v for k, v in self.nudge.items()},
                "fixes": self.fixes}

    def load(self, d):
        self.w_ratio = d.get("w_ratio", 1.0)
        self.nudge = {int(k): v for k, v in d.get("nudge", {}).items()}
        self.fixes = d.get("fixes", 0)


class AdaptiveZoneMap(ZoneMap):
    """ZoneMap whose zone width and boundaries adapt. Same zone_for / zone_center / left / width API."""

    def __init__(self, center, n, width, rest_x=None, home=None, state=None):
        super().__init__(center, n, width)
        self.fit_w = width
        self.rest_x = center if rest_x is None else rest_x
        self.home = n // 2 if home is None else home
        # rig may have shifted the lane to keep it on screen; remember that shift
        self.shift = (center - n * width / 2.0) - (self.rest_x - (self.home + 0.5) * width)
        self.state = state if state is not None else ZoneState()
        self.state.w_ratio = max(MAX_SHRINK, min(1.0, self.state.w_ratio))
        self._last = None
        self._rebuild()

    def _rebuild(self):
        w = max(MIN_W, self.fit_w * self.state.w_ratio)
        if self.fit_w < MIN_W:
            w = self.fit_w
        left = self.rest_x - (self.home + 0.5) * w + self.shift * (w / self.fit_w)
        e = [left + k * w for k in range(self.n + 1)]
        for k, d in self.state.nudge.items():
            if 0 < k < self.n:
                e[k] += max(-MAX_NUDGE, min(MAX_NUDGE, d)) * w
        for k in range(1, self.n + 1):                 # keep every zone at least half-width
            e[k] = max(e[k], e[k - 1] + 0.5 * w)
        self.edges, self.width = e, w
        self.center = (e[0] + e[-1]) / 2.0

    @property
    def left(self):
        return self.edges[0]

    def zone_center(self, i):
        return (self.edges[i] + self.edges[i + 1]) / 2.0

    def bounds(self, i):
        return self.edges[i], self.edges[i + 1]

    def zone_for(self, x, prev=None, margin=0.25):
        z = max(0, min(self.n - 1, bisect_right(self.edges, x) - 1))
        if prev is not None and prev != z and 0 <= prev < self.n:
            if abs(x - self.zone_center(prev)) - abs(x - self.zone_center(z)) < margin * self.width:
                z = prev
        return z

    def observe_tap(self, t, x, z):
        st = self.state
        # AIM: landed near a boundary, then quickly re-tapped the neighbour -> move that boundary
        if self._last is not None:
            t0, x0, z0 = self._last
            if t - t0 < CORRECT_GAP_S and abs(z - z0) == 1:
                k = max(z, z0)
                if abs(x0 - self.edges[k]) < NEAR_EDGE * self.width:
                    base = self.edges[k] - st.nudge.get(k, 0.0) * self.width
                    want = x0 - 0.1 * self.width if z > z0 else x0 + 0.1 * self.width
                    d = st.nudge.get(k, 0.0)
                    st.nudge[k] = max(-MAX_NUDGE, min(MAX_NUDGE, d + 0.4 * ((want - base) / self.width - d)))
                    st.fixes += 1
        self._last = (t, x, z)
        # REACH: if they never get out to the outer zones, shrink zones so they can
        st.reach.append(x - self.rest_x)
        if len(st.reach) >= 12:
            r = sorted(st.reach)
            lo_r = -r[max(0, int(0.05 * (len(r) - 1)))]
            hi_r = r[int(0.95 * (len(r) - 1))]
            fits = []
            if self.home > 0:
                fits.append(1.15 * max(lo_r, 0.0) / self.home)
            if self.n - 1 - self.home > 0:
                fits.append(1.15 * max(hi_r, 0.0) / (self.n - 1 - self.home))
            if fits:
                target = max(MAX_SHRINK, min(1.0, min(fits) / self.fit_w))
                st.w_ratio += 0.2 * (target - st.w_ratio)
        self._rebuild()


def zone_bounds(zones, i):
    """(left, right) of zone i for any ZoneMap, adaptive or not."""
    if hasattr(zones, "bounds"):
        return zones.bounds(i)
    return zones.left + i * zones.width, zones.left + (i + 1) * zones.width


class Learner:
    """Owns everything learned about one player and saves it to users/<user>.json."""

    def __init__(self, user="default", folder=None):
        safe = "".join(c for c in str(user) if c.isalnum() or c in "-_") or "default"
        self.user = safe
        self.path = os.path.join(folder or USERS_DIR, safe + ".json")
        self._strength, self._zones = {}, {}
        self._taps = 0
        self._warned = set()
        self.load()

    def strength(self, key):
        if key not in self._strength:
            hand = "/" in key
            self._strength[key] = (StrengthStats(floor_abs=0.08, ceil_abs=0.7) if hand
                                   else StrengthStats(floor_abs=0.03))
            saved = self._saved.get("strength", {}).get(key)
            if saved:
                self._strength[key].load(saved)
        return self._strength[key]

    def zones(self, eid):
        if eid not in self._zones:
            self._zones[eid] = ZoneState()
            saved = self._saved.get("zones", {}).get(eid)
            if saved:
                self._zones[eid].load(saved)
        return self._zones[eid]

    def on_events(self, t, events, warnings):
        downs = sum(1 for e in events if e[2] == "down")
        if not downs:
            return
        self._taps += downs
        for key, st in self._strength.items():
            if st.tired() and key not in self._warned:
                self._warned.add(key)
                warnings.append("%s: movements are getting weaker - maybe take a short break" % key)
        if self._taps % AUTOSAVE_EVERY < downs:
            self.save()

    def status(self):
        lines = []
        for key, st in sorted(self._strength.items()):
            if st.hi is None:
                continue
            state = "learning" if len(st.recent) < MIN_SAMPLES else "adapted"
            line = "%s: %s, threshold %.2f (calibrated %.2f)" % (key, state, st.hi, st.cal_hi or 0)
            z = self._zones.get(key)
            if z is not None and (z.w_ratio < 0.99 or z.fixes):
                line += ", zones %d%%" % round(100 * z.w_ratio)
                if z.fixes:
                    line += ", %d aim fixes" % z.fixes
            lines.append(line)
        return lines

    def save(self):
        data = {"user": self.user,
                "strength": {k: v.to_dict() for k, v in self._strength.items()},
                "zones": {k: v.to_dict() for k, v in self._zones.items()}}
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=1)
            os.replace(tmp, self.path)
        except OSError as e:
            print("[adapt] could not save:", e)

    def load(self):
        self._saved = {}
        try:
            with open(self.path) as f:
                self._saved = json.load(f)
        except (OSError, ValueError):
            pass

    def forget(self):
        self._strength, self._zones, self._saved = {}, {}, {}
        try:
            os.remove(self.path)
        except OSError:
            pass
