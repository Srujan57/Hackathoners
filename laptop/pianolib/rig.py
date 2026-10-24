"""Effectors (foot / hand / fingers), automatic calibration and lane fitting.

Input to every update is a dict of raw points, in frame-height units:
    {"toe": (x, y, score), "heel": (...), "ankle": (...)}                      for a foot
    {"wrist": ..., "palm": ..., "tip_index": ..., "mcp_index": ..., ...}      for a hand
Output is a list of (effector_id, slot, "down"|"up"). `slot` is a zone number for feet / hand-pads and a
finger name for fingers, matching EffectorSpec.slot_ids().
"""

import math
from collections import deque

from .detect import (BASE_LIFT_HI, NOISE_K, SETTLE_S, V_SETTLE, DriftBaseline, FlexDetector, TapDetector,
                     ZoneMap, median, stdev)
from .filters import OneEuro
from .adapt import (AdaptiveFlexDetector, AdaptiveTapDetector, AdaptiveZoneMap, Learner,
                    zone_bounds)

MIN_SCORE = 0.35       # keypoints below this confidence are ignored
LOST_GRACE_S = 0.25    # an effector unseen this long releases its notes
REACQUIRE_S = 1.0      # unseen longer than this -> baseline is re-learned on return
CAL_MIN_S = 0.8        # calibration needs this long of stillness
CAL_WINDOW_S = 1.2
CAL_MOVE = 0.025       # max wobble (frame heights) allowed during calibration
OFF_S = 0.6            # still but far from the baseline this long -> the baseline is wrong, relearn
NEAR_GATE = False      # (tested both ways: ungated tracks drift best; see tests/bench_vs_legacy.py)
V_LEARN = 0.35         # ...learn only while moving slower than this (reference lengths / s)
V_RISING = 0.8         # ...and not rising faster than this
MIN_REF = 0.04         # smallest allowed reference length (frame heights)
MIN_ZONE_W = 0.03      # narrowest allowed zone (frame heights)
FINGER_MIN_LEN = 0.25  # finger shorter than this x palm length is too short to track reliably


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class BaseEffector:
    ANCHOR = ()

    def __init__(self, spec, sensitivity, min_score=MIN_SCORE):
        self.spec = spec
        self.id = spec.id
        self.sens = sensitivity
        self.min_score = min_score
        self._f = {}
        self.calibrated = False
        self.cal = []
        self.cal_state = "missing"
        self.visible = False
        self.last_seen = None
        self._lost_since = None
        self.pts = {}
        self.warnings = []
        self.learner = None

    # ---- helpers ----------------------------------------------------------
    def _filter(self, raw, t):
        out = {}
        for name, (x, y, s) in raw.items():
            if s < self.min_score or not (math.isfinite(x) and math.isfinite(y)):
                continue               # a NaN would poison the baseline for good
            fx = self._f.setdefault((name, "x"), OneEuro())
            fy = self._f.setdefault((name, "y"), OneEuro())
            if fx.last_time is not None and t - fx.last_time > 0.3:   # stale filter after a gap
                fx.reset()
                fy.reset()
            out[name] = (fx(x, t), fy(y, t))
        return out

    @property
    def anchor(self):
        for n in self.ANCHOR:
            if n in self.pts:
                return self.pts[n]
        return None

    # ---- calibration ------------------------------------------------------
    def begin_calibration(self):
        self.calibrated = False
        self.cal = []
        self.cal_state = "missing"
        self.pts = {}
        self.warnings = []
        self.reset_runtime()

    def cal_feed(self, t, raw):
        f = self._filter(raw, t)
        self.pts = f
        if not self._cal_ok(f):
            self.cal, self.cal_state = [], "missing"
            return False
        self.cal.append((t, f))
        while self.cal and t - self.cal[0][0] > CAL_WINDOW_S:
            self.cal.pop(0)
        if t - self.cal[0][0] < CAL_MIN_S or len(self.cal) < 8:
            self.cal_state = "hold"
            return False
        if self._wobble() > CAL_MOVE:
            self.cal_state = "moving"
            self.cal = self.cal[-3:]
            return False
        self.cal_state = "ok"
        return True

    def _wobble(self):
        worst = 0.0
        for name in self.CAL_TRACK:
            xs = [s[1][name][0] for s in self.cal if name in s[1]]
            ys = [s[1][name][1] for s in self.cal if name in s[1]]
            if len(xs) >= 4:
                worst = max(worst, max(xs) - min(xs), max(ys) - min(ys))
        return worst

    def _medians(self):
        need = 0.6 * len(self.cal)
        out = {}
        for name in {n for _, f in self.cal for n in f}:
            xs = [f[name][0] for _, f in self.cal if name in f]
            ys = [f[name][1] for _, f in self.cal if name in f]
            if len(xs) >= need:
                out[name] = (median(xs), median(ys), xs, ys)
        return out

    # subclasses implement: reset_runtime, _cal_ok, finish_calibration, update


class TapEffector(BaseEffector):
    """A foot (toe/heel tap or step) or a whole hand used as a pad. Picks one of n zones by x."""

    LIFT = ("toe", "heel")
    FALLBACK = ("ankle",)
    ANCHOR = ("ankle", "toe")
    REF_PAIRS = (("toe", "heel"), ("ankle", "toe"))
    CAL_TRACK = ("toe", "heel", "ankle")

    def __init__(self, spec, sensitivity, min_score=MIN_SCORE):
        super().__init__(spec, sensitivity, min_score)
        self.n = spec.n_zones
        # resting zone: left side keeps more zones to its left (outward), right side to its right
        self.home = self.n // 2 if spec.side == "L" else (self.n - 1) // 2
        self.det = TapDetector()
        self.zones = ZoneMap(0.5, self.n, 0.1)
        self.rest_x = 0.5
        self.ref = 0.15
        self.base = {}
        self.reset_runtime()

    def reset_runtime(self):
        self.det.reset()
        self.held = None
        self.prev_zone = None
        self.xs = deque(maxlen=4)
        self._t_prev = None
        self._still = 0.0
        self._off = 0.0

    def _cal_ok(self, f):
        return any(n in f for n in self.LIFT + self.FALLBACK)

    def finish_calibration(self):
        med = self._medians()
        pos = {n: (v[0], v[1]) for n, v in med.items()}
        refs = [_dist(pos[a], pos[b]) for a, b in self.REF_PAIRS if a in pos and b in pos]
        self.ref = max([MIN_REF] + refs)
        self.base = {n: DriftBaseline(pos[n][1]) for n in pos}
        lift_names = [n for n in self.LIFT if n in med] or [n for n in self.FALLBACK if n in med]
        sigma = max([stdev([(y - pos[n][1]) / self.ref for y in med[n][3]]) for n in lift_names] or [0.0])
        hi = max(BASE_LIFT_HI / self.sens, NOISE_K * sigma)
        self.det = (AdaptiveTapDetector(lift_hi=hi, noise_sigma=sigma, stats=self.learner.strength(self.id))
                    if self.learner else TapDetector(lift_hi=hi))
        anchor = next((pos[n] for n in self.ANCHOR if n in pos), None)
        self.rest_x = anchor[0] if anchor else 0.5
        self.sigma = sigma
        if hi > 0.5:
            self.warnings.append("%s: tracking is noisy (needs lifts of %.0f%% of foot length)"
                                 % (self.id, hi * 100))
        self.reset_runtime()
        self.calibrated = True

    def update(self, t, raw):
        f = self._filter(raw, t)
        self.pts = f
        names = [n for n in self.LIFT if n in f] or [n for n in self.FALLBACK if n in f]
        if not names:
            return self._lost(t)
        if self._lost_since is not None:
            if t - self._lost_since > REACQUIRE_S:
                for n in names:
                    self.base[n] = DriftBaseline(f[n][1])
            self._lost_since = None
        self.visible, self.last_seen = True, t
        for n in names:                                  # first sight of a point: assume planted
            if n not in self.base:
                self.base[n] = DriftBaseline(f[n][1])

        dt = (t - self._t_prev) if self._t_prev is not None else 0.0
        self._t_prev = t
        for n in names:
            self.base[n].predict(dt)                     # coast along the estimated drift

        lift = max((self.base[n].b - f[n][1]) / self.ref for n in names)

        # Learn the planted position only from a foot that is down, still, not rising, and (optionally) near
        # the baseline - so neither the tail of a tap nor the start of the next lift is mistaken for floor.
        calm = abs(self.det.vel) < V_SETTLE
        near = abs(lift) < self.det.lo
        learn = (self.det.state == "DOWN" and abs(self.det.vel) < V_LEARN and self.det.vel < V_RISING
                 and (near or not NEAR_GATE))
        self._still = self._still + dt if learn else 0.0
        if self._still >= SETTLE_S:
            for n in names:
                self.base[n].correct(f[n][1], dt)
        # Self-heal, in ANY state: motionless but away from the baseline for OFF_S means the baseline is
        # wrong (or the player rested the foot up). Re-plant it here instead of staying stuck, and do it
        # silently - a re-plant must never produce a note.
        self._off = self._off + dt if (calm and not near) else 0.0
        if self._off >= OFF_S:
            for n in names:
                self.base[n].b = f[n][1]
            self.det.reset()
            self._off = 0.0
            lift = 0.0
            if self.held is not None:
                held, self.held = self.held, None
                return [(held, "up")]

        a_pt = self.anchor
        if a_pt is not None:
            self.xs.append(a_pt[0])

        out = []
        for ev in self.det.update(t, lift):
            if ev == "down":
                x = median(list(self.xs)) if self.xs else self.rest_x
                z = self.zones.zone_for(x, self.prev_zone)
                if hasattr(self.zones, "observe_tap"):
                    self.zones.observe_tap(t, x, z)
                self.prev_zone = z
                self.held = z
                out.append((z, "down"))
            elif self.held is not None:
                out.append((self.held, "up"))
                self.held = None
        return out

    def _lost(self, t):
        out = []
        if self.last_seen is None or t - self.last_seen > LOST_GRACE_S:
            if self.visible:
                self.visible = False
                self._lost_since = t
                self.det.reset()
                self.xs.clear()
            if self.held is not None:
                out.append((self.held, "up"))
                self.held = None
        return out


class FootEffector(TapEffector):
    pass


class HandPointEffector(TapEffector):
    LIFT = ("wrist",)
    FALLBACK = ("palm",)
    ANCHOR = ("wrist", "palm")
    REF_PAIRS = (("wrist", "palm"),)
    CAL_TRACK = ("wrist", "palm")


class FingerEffector(BaseEffector):
    """A hand where each chosen finger is its own key. A finger 'presses' when its tip comes toward its
    knuckle (flexion), measured against that finger's own relaxed length."""

    ANCHOR = ("palm", "wrist")
    CAL_TRACK = ("wrist", "palm")

    def __init__(self, spec, sensitivity, min_score=MIN_SCORE):
        super().__init__(spec, sensitivity, min_score)
        self.fingers = list(spec.fingers)
        self.e0 = {}
        self.dets = {}
        self.ok = {}
        self.ref = 0.1
        self.reset_runtime()

    def reset_runtime(self):
        self.dets = {f: FlexDetector() for f in self.fingers}
        self._seen = {f: None for f in self.fingers}
        self._scale = 1.0            # hand size now / hand size at calibration (moving toward/away from camera)

    def _cal_ok(self, f):
        return "wrist" in f and "palm" in f and any(
            ("tip_" + g) in f and ("mcp_" + g) in f for g in self.fingers)

    def finish_calibration(self):
        med = self._medians()
        pos = {n: (v[0], v[1]) for n, v in med.items()}
        self.ref = max(MIN_REF, _dist(pos["wrist"], pos["palm"]) if "wrist" in pos and "palm" in pos else 0)
        self.e0, self.ok, dets = {}, {}, {}
        for g in self.fingers:
            lens = [_dist(f["tip_" + g], f["mcp_" + g]) for _, f in self.cal
                    if ("tip_" + g) in f and ("mcp_" + g) in f]
            if len(lens) < 0.6 * len(self.cal):
                self.ok[g] = False
                self.warnings.append("%s %s: not seen during calibration (finger disabled)" % (self.id, g))
                continue
            e0 = median(lens)
            if e0 < FINGER_MIN_LEN * self.ref:
                self.ok[g] = False
                self.warnings.append("%s %s: too short to track reliably (finger disabled)" % (self.id, g))
                continue
            self.e0[g], self.ok[g] = e0, True
            sigma = stdev([(e0 - l) / e0 for l in lens])
            hi = min(0.7, max(0.35 / self.sens, NOISE_K * sigma))
            dets[g] = (AdaptiveFlexDetector(hi=hi, lo=0.6 * hi, noise_sigma=sigma,
                                            stats=self.learner.strength(self.id + "/" + g))
                       if self.learner else FlexDetector(hi=hi, lo=0.6 * hi))
        self.reset_runtime()
        self.dets.update(dets)
        self.calibrated = True

    def update(self, t, raw):
        f = self._filter(raw, t)
        self.pts = f
        out = []
        w, pm = f.get("wrist"), f.get("palm")
        if w is not None and pm is not None:
            # A hand that moves away from the camera makes every finger look shorter, which would read as
            # flexion. The wrist-to-palm distance shrinks the same way, so use it to rescale the finger lengths.
            ratio = min(1.6, max(0.6, _dist(w, pm) / self.ref))
            self._scale += 0.2 * (ratio - self._scale)
        for g in self.fingers:
            det = self.dets.get(g)
            if det is None or not self.ok.get(g):
                continue
            tip, mcp = f.get("tip_" + g), f.get("mcp_" + g)
            if tip is None or mcp is None:
                last = self._seen[g]
                if last is None or t - last > LOST_GRACE_S:
                    if det.pressed:
                        out.append((g, "up"))
                    det.reset()
                continue
            self._seen[g] = t
            e0 = self.e0[g] * self._scale
            flex = (e0 - _dist(tip, mcp)) / e0
            out += [(g, ev) for ev in det.update(t, flex)]
        self.visible = any(n in f for n in ("wrist", "palm"))
        if self.visible:
            self.last_seen = t
        return out


def make_effector(spec, sensitivity, min_score=MIN_SCORE):
    if spec.kind == "foot":
        return FootEffector(spec, sensitivity, min_score)
    if spec.uses_fingers:
        return FingerEffector(spec, sensitivity, min_score)
    return HandPointEffector(spec, sensitivity, min_score)


class Rig:
    """All of a player's effectors: auto-calibrates, then turns observations into slot events."""

    def __init__(self, profile, aspect=16 / 9.0, min_score=MIN_SCORE, user=None):
        self.profile = profile
        self.aspect = aspect
        self.effs = {e.id: make_effector(e, profile.sensitivity, min_score) for e in profile.effectors}
        self.learner = Learner(user or profile.meta.get("user") or "default")
        for e in self.effs.values():
            e.learner = self.learner
        self.warnings = []
        self.recalibrate()

    def recalibrate(self):
        self.state = "CAL"
        self.warnings = []
        for e in self.effs.values():
            e.begin_calibration()

    @property
    def calibrating(self):
        return self.state == "CAL"

    def update(self, t, obs, menu_y=0.0):
        if self.state == "CAL":
            ready = [e.cal_feed(t, obs.get(e.id, {})) for e in self.effs.values()]
            if all(ready):
                for e in self.effs.values():
                    e.finish_calibration()
                self._fit_lanes()
                self.state = "READY"
            return []
        events = []
        for eid, e in self.effs.items():
            evs = e.update(t, obs.get(eid, {}))
            a = e.anchor
            if menu_y and a is not None and a[1] < menu_y:    # in the menu strip: never play a note
                evs = [x for x in evs if x[1] != "down"]
            events += [(eid, slot, kind) for slot, kind in evs]
        self.learner.on_events(t, events, self.warnings)
        return events

    def _fit_lanes(self):
        """Place each lane so the resting position is the CENTRE of one zone (a tap in place always plays
        the same note), with more zones on the outward side, then shrink zones if lanes would overlap."""
        lanes = sorted((e for e in self.effs.values() if isinstance(e, TapEffector)),
                       key=lambda e: e.rest_x)
        base_w = self.profile.zone_width * self.aspect
        w = base_w
        for a, b in zip(lanes, lanes[1:]):
            room = a.n - a.home + b.home                      # zones between the two resting positions
            w = min(w, (b.rest_x - a.rest_x) / max(room, 1e-6))
        if w < base_w - 1e-6:
            self.warnings.append("zones narrowed to fit - move your feet/hands further apart for bigger zones")
        if w < MIN_ZONE_W:
            self.warnings.append("zones are very small - increase the distance between the two sides")
            w = MIN_ZONE_W
        for e in lanes:
            left = e.rest_x - (e.home + 0.5) * w
            left = min(max(left, 0.0), max(self.aspect - e.n * w, 0.0))
            e.zones = AdaptiveZoneMap(left + e.n * w / 2.0, e.n, w, rest_x=e.rest_x, home=e.home,
                                      state=self.learner.zones(e.id))
        for e in self.effs.values():
            self.warnings += e.warnings

    # ---- queries for the UI / pose association -------------------------------
    def zone_rects(self):
        return [(e.id, i) + tuple(zone_bounds(e.zones, i))
                for e in self.effs.values() if isinstance(e, TapEffector) for i in range(e.n)]

    def pointers(self):
        return [e.anchor for e in self.effs.values() if e.anchor is not None]

    def last_x(self):
        out = {}
        for eid, e in self.effs.items():
            a = e.anchor
            if a is not None:
                out[eid] = a[0]
            elif isinstance(e, TapEffector) and e.calibrated:
                out[eid] = e.rest_x
        return out

    def status(self):
        if self.state == "READY":
            return ""
        msgs = []
        for eid, e in self.effs.items():
            if e.cal_state == "missing":
                msgs.append("%s: not visible" % eid)
            elif e.cal_state == "moving":
                msgs.append("%s: hold still" % eid)
        return "Calibrating - rest feet flat / hands open.  " + ("; ".join(msgs) if msgs else "hold still...")

    def learn_status(self):
        """One line per effector describing what has been learned (for the UI)."""
        return self.learner.status()

    def save_learning(self):
        self.learner.save()
