"""Synthetic foot motion for testing the tap logic without a camera or model.

Coordinates are frame-height units (pixels / frame height), like the real pipeline.
A "foot" is three points (toe, heel, ankle). Noise, slow depth drift and several tap styles can be mixed.
"""

import math
import random

FPS = 30.0


def raised_cos(u):
    """0 -> 1 -> 0 over u in [0, 1]."""
    return 0.5 - 0.5 * math.cos(2 * math.pi * min(max(u, 0.0), 1.0))


class FootScript:
    """Ground truth + trajectory for one foot.

    taps: list of dicts {t, amp, kind} - 'toe' lifts only the toe, 'step' lifts the whole foot.
    Each tap rises for `rise` s and falls for `fall` s; contact (ground truth) is at t + rise + fall.
    drift: function t -> y offset (frame heights), models moving toward/away from the camera.
    slides: list of (t0, t1, dx): the planted foot slides sideways (not a tap).
    """

    def __init__(self, x0=0.35, y0=0.78, taps=(), drift=None, slides=(), noise=0.004,
                 rise=0.12, fall=0.10, seed=1):
        self.x0, self.y0 = x0, y0
        self.taps = list(taps)
        self.drift = drift or (lambda t: 0.0)
        self.slides = list(slides)
        self.noise = noise
        self.rise, self.fall = rise, fall
        self.rng = random.Random(seed)

    def contacts(self):
        return [tp["t"] + self.rise + self.fall for tp in self.taps]

    def _lift(self, t):
        toe = step = 0.0
        for tp in self.taps:
            u = t - tp["t"]
            total = self.rise + self.fall
            if 0 <= u <= total:
                prof = raised_cos(u / total) if False else (
                    raised_cos(0.5 * u / self.rise) if u < self.rise
                    else raised_cos(0.5 + 0.5 * (u - self.rise) / self.fall))
                if tp.get("kind", "toe") == "toe":
                    toe = max(toe, tp["amp"] * prof)
                else:
                    step = max(step, tp["amp"] * prof)
        return toe, step

    def _x(self, t):
        x = self.x0
        for t0, t1, dx in self.slides:
            if t >= t1:
                x += dx
            elif t > t0:
                x += dx * (t - t0) / (t1 - t0)
        return x

    def sample(self, t):
        toe_lift, step = self._lift(t)
        x = self._x(t)
        d = self.drift(t)
        n = lambda: self.rng.gauss(0.0, self.noise)
        base_y = self.y0 + d
        toe = (x + 0.06 + n(), base_y - toe_lift - step + n(), 0.9)
        heel = (x - 0.06 + n(), base_y + 0.01 - step + n(), 0.9)
        ankle = (x - 0.04 + n(), base_y - 0.05 - 0.5 * step - 0.2 * toe_lift + n(), 0.9)
        return {"toe": toe, "heel": heel, "ankle": ankle}


def frames(duration, fps=FPS, start=0.0):
    n = int(duration * fps)
    return [start + i / fps for i in range(n)]


def match_events(contacts, fires, early=0.30, late=0.35):
    """Greedy match of detections to ground-truth contacts. Returns (hits, misses, false, latencies)."""
    fires = sorted(fires)
    used = set()
    lat = []
    hits = 0
    for c in contacts:
        cand = [i for i, f in enumerate(fires) if i not in used and c - early <= f <= c + late]
        if cand:
            i = min(cand, key=lambda j: abs(fires[j] - c))
            used.add(i)
            hits += 1
            lat.append(fires[i] - c)
    return hits, len(contacts) - hits, len(fires) - len(used), lat


# ---- the OLD logic, copied from the original foot_piano.py, for comparison ----
class LegacyFoot:
    LIFT_THRESH, LAND_THRESH, CAL_FRAMES = 0.045, 0.020, 25

    def __init__(self):
        self.floor_y = None
        self.cal = []
        self.down = True
        self.armed = False
        self.fires = []

    def update(self, t, y):
        if self.floor_y is None:
            self.cal.append(y)
            if len(self.cal) >= self.CAL_FRAMES:
                self.floor_y = sorted(self.cal)[len(self.cal) // 2]
            return
        height = self.floor_y - y
        if self.down and height > self.LIFT_THRESH:
            self.down, self.armed = False, True
        elif not self.down and height < self.LAND_THRESH:
            self.down = True
            if self.armed:
                self.fires.append(t)
                self.armed = False
