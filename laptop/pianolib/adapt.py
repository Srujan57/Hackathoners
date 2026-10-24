"""
pianolib/adapt.py - adaptive learning: the piano tunes itself to each player.

What it learns, per limb (left foot, right hand, etc.):
  1. REACH      - where this person can actually move. Zones stretch to fit their range,
                  so someone who can only slide a foot 20 cm still gets the full octave.
  2. STRENGTH   - how high they usually lift / tap. The "lifted" threshold follows them,
                  so weaker movements still play notes and strong ones don't double-fire.
  3. STEADINESS - how much they shake when resting (tremor). Thresholds stay above that
                  noise and smoothing gets stronger, so tremor doesn't trigger notes.
  4. AIM        - where they actually land when aiming at each key (warm-up game), and
                  quick "oops" corrections in free play. Key boundaries move to match.
  5. FATIGUE    - if their taps get weaker over a session, it notices and flags it.

Everything is saved to a per-user JSON file, so next session starts already tuned.

Units: x is 0-1 across the frame, h is height above that limb's floor/rest line
(0-1 of frame height, positive = lifted). Use whatever your detector already has.

Hook it up (3 calls):
  model.observe(limb, t, x, h)                    every frame the limb is visible
  model.on_tap(limb, t, x_land, peak_h, duration) when a tap/press finishes
  model.zone_of(limb, x), model.thresholds(limb)  read these instead of constants
"""

import json
import statistics
from collections import deque

NUM_KEYS = 8

# Safety rails: learning can never push values outside these.
LIFT_MIN, LIFT_MAX = 0.012, 0.09
MIN_ZONE_WIDTH = 0.02          # no key can shrink below 2% of the frame width
MIN_TAPS_TO_ADAPT = 8          # stay on defaults until we've seen this many taps
CORRECTION_GAP_S = 0.5         # an adjacent re-tap this fast after...
CORRECTION_MAX_DUR_S = 0.18    # ...a very short note = probably a miss
CORRECTION_RATE = 0.35


def _pct(values, p):
    s = sorted(values)
    if not s:
        return None
    i = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[i]


class LimbModel:
    def __init__(self, name, default_lift=0.045, default_land=0.02):
        self.name = name
        self.default_lift = default_lift
        self.default_land = default_land
        self.peaks = deque(maxlen=60)        # recent tap heights
        self.session_peaks = []              # this session only, for fatigue
        self.durations = deque(maxlen=60)
        self.landings = deque(maxlen=200)    # x where taps landed
        self.noise = 0.0                     # tremor while resting (EMA of abs deviation)
        self._rest_h = None
        self.boundaries = None               # 9 edges for 8 zones, or None = uniform
        self.warmup = {}                     # zone -> list of landing x
        self.last_tap = None                 # (t, zone, x, duration)
        self.corrections = 0
        self.taps = 0

    # ---------- learning inputs ----------
    def observe(self, t, x, h):
        land = self.thresholds()["land"]
        if h < land:                         # resting: learn tremor size
            if self._rest_h is None:
                self._rest_h = h
            self._rest_h = 0.95 * self._rest_h + 0.05 * h
            dev = abs(h - self._rest_h)
            self.noise = 0.97 * self.noise + 0.03 * dev

    def on_tap(self, t, x, peak_h, duration, allow_corrections=True):
        self.taps += 1
        self.peaks.append(peak_h)
        self.session_peaks.append(peak_h)
        self.durations.append(duration)
        self.landings.append(x)

        zone = self.zone_of(x)
        if allow_corrections and self.last_tap:
            pt, pz, px, pdur = self.last_tap
            if (abs(zone - pz) == 1 and t - pt < CORRECTION_GAP_S
                    and pdur < CORRECTION_MAX_DUR_S):
                self._nudge_boundary(pz, zone, px)
                self.corrections += 1
        self.last_tap = (t, zone, x, duration)
        return zone

    def on_warmup_hit(self, target_zone, x):
        self.warmup.setdefault(target_zone, []).append(x)
        self.landings.append(x)

    # ---------- learned outputs ----------
    def thresholds(self):
        lift, land = self.default_lift, self.default_land
        if len(self.peaks) >= MIN_TAPS_TO_ADAPT:
            typical = statistics.median(self.peaks)
            lift = 0.45 * typical            # half their normal lift counts as a lift
        lift = max(lift, 3.0 * self.noise)   # always clear the tremor
        lift = min(max(lift, LIFT_MIN), LIFT_MAX)
        if len(self.peaks) >= MIN_TAPS_TO_ADAPT or self.noise > 0:
            land = min(max(0.45 * lift, 1.5 * self.noise, 0.006), 0.8 * lift)
        # smoothing: shakier = smoother (weight of the newest sample)
        alpha = min(0.6, max(0.2, 0.6 - 20.0 * self.noise))
        return {"lift": round(lift, 4), "land": round(land, 4), "smooth_alpha": round(alpha, 3)}

    def reach(self):
        if len(self.landings) < MIN_TAPS_TO_ADAPT:
            return 0.0, 1.0
        lo, hi = _pct(self.landings, 3), _pct(self.landings, 97)
        pad = 0.05 * max(hi - lo, 0.1)
        lo, hi = max(0.0, lo - pad), min(1.0, hi + pad)
        if hi - lo < NUM_KEYS * MIN_ZONE_WIDTH:
            mid = (lo + hi) / 2
            half = NUM_KEYS * MIN_ZONE_WIDTH / 2
            lo, hi = max(0.0, mid - half), min(1.0, mid + half)
        return lo, hi

    def edges(self):
        if self.boundaries:
            return self.boundaries
        lo, hi = self.reach()
        w = (hi - lo) / NUM_KEYS
        return [lo + i * w for i in range(NUM_KEYS + 1)]

    def zone_of(self, x):
        e = self.edges()
        for k in range(NUM_KEYS):
            if x < e[k + 1]:
                return k
        return NUM_KEYS - 1

    def fit_warmup(self):
        """Place key boundaries halfway between where they actually land on each key."""
        if any(len(self.warmup.get(k, [])) == 0 for k in range(NUM_KEYS)):
            return False
        centers = [statistics.median(self.warmup[k]) for k in range(NUM_KEYS)]
        if any(centers[k + 1] - centers[k] < MIN_ZONE_WIDTH / 2 for k in range(NUM_KEYS - 1)):
            return False                     # aim too inconsistent; keep reach-based zones
        inner = [(centers[k] + centers[k + 1]) / 2 for k in range(NUM_KEYS - 1)]
        first = max(0.0, centers[0] - (inner[0] - centers[0]))
        last = min(1.0, centers[-1] + (centers[-1] - inner[-1]))
        self.boundaries = [first] + inner + [last]
        self._enforce_widths()
        return True

    def fatigue(self):
        s = self.session_peaks
        if len(s) < 20:
            return False
        early = statistics.median(s[:10])
        late = statistics.median(s[-10:])
        return early > 0 and late / early < 0.7

    # ---------- internals ----------
    def _nudge_boundary(self, wrong_zone, right_zone, x_prev):
        e = list(self.edges())
        b = max(wrong_zone, right_zone)      # edge index between the two zones
        target = x_prev + (0.01 if right_zone < wrong_zone else -0.01)
        e[b] = e[b] + CORRECTION_RATE * (target - e[b])
        self.boundaries = e
        self._enforce_widths()

    def _enforce_widths(self):
        e = self.boundaries
        for k in range(1, len(e)):
            if e[k] - e[k - 1] < MIN_ZONE_WIDTH:
                e[k] = e[k - 1] + MIN_ZONE_WIDTH
        if e[-1] > 1.0:                      # pushed off the right side: squeeze back
            over = e[-1] - 1.0
            e[:] = [v - over for v in e]
            for k in range(len(e) - 2, -1, -1):
                if e[k + 1] - e[k] < MIN_ZONE_WIDTH:
                    e[k] = e[k + 1] - MIN_ZONE_WIDTH

    # ---------- save / load ----------
    def to_dict(self):
        return {"peaks": list(self.peaks), "durations": list(self.durations),
                "landings": list(self.landings), "noise": self.noise,
                "boundaries": self.boundaries, "taps": self.taps}

    def load_dict(self, d):
        self.peaks.extend(d.get("peaks", []))
        self.durations.extend(d.get("durations", []))
        self.landings.extend(d.get("landings", []))
        self.noise = d.get("noise", 0.0)
        self.boundaries = d.get("boundaries")
        self.taps = d.get("taps", 0)

    def summary(self):
        th = self.thresholds()
        lo, hi = self.reach()
        state = "learning" if self.taps < MIN_TAPS_TO_ADAPT else "adapted"
        s = (f"{self.name}: {state} ({self.taps} taps) | reach {lo:.2f}-{hi:.2f} | "
             f"lift>{th['lift']:.3f} | tremor {self.noise:.3f}")
        if self.corrections:
            s += f" | {self.corrections} aim fixes"
        if self.fatigue():
            s += " | TIRED - take a break?"
        return s


class AdaptiveModel:
    """One LimbModel per limb, saved together as one user file."""

    def __init__(self, user="default", path=None):
        self.user = user
        self.path = path or f"users/{user}.json"
        self.limbs = {}

    def limb(self, name):
        if name not in self.limbs:
            self.limbs[name] = LimbModel(name)
        return self.limbs[name]

    def observe(self, name, t, x, h):
        self.limb(name).observe(t, x, h)

    def on_tap(self, name, t, x, peak_h, duration, allow_corrections=True):
        return self.limb(name).on_tap(t, x, peak_h, duration, allow_corrections)

    def zone_of(self, name, x):
        return self.limb(name).zone_of(x)

    def thresholds(self, name):
        return self.limb(name).thresholds()

    def reset(self, name=None):
        for n in ([name] if name else list(self.limbs)):
            self.limbs[n] = LimbModel(n)

    def save(self):
        import os
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"user": self.user,
                       "limbs": {n: m.to_dict() for n, m in self.limbs.items()}}, f, indent=1)

    def load(self):
        try:
            with open(self.path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return False
        for n, d in data.get("limbs", {}).items():
            self.limb(n).load_dict(d)
        return True


class Warmup:
    """
    30-second aiming game. Lights up a target key, player taps it, we record where they
    landed. Two rounds through all 8 keys in a shuffled order, then fit the boundaries.
    """

    def __init__(self, model, limb, rounds=2, seed=7):
        import random
        rng = random.Random(seed)
        self.model, self.limb = model, limb
        self.order = []
        for _ in range(rounds):
            r = list(range(NUM_KEYS))
            rng.shuffle(r)
            self.order += r
        self.i = 0

    def target(self):
        return self.order[self.i] if self.i < len(self.order) else None

    def hit(self, x):
        t = self.target()
        if t is None:
            return None
        self.model.limb(self.limb).on_warmup_hit(t, x)
        self.i += 1
        if self.i == len(self.order):
            return self.model.limb(self.limb).fit_warmup()
        return None

    def done(self):
        return self.i >= len(self.order)

    def progress(self):
        return self.i, len(self.order)
