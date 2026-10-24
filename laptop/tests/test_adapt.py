"""End-to-end tests for adaptive learning, run through the real Rig with simulated feet / fingers.

    cd laptop && python tests/test_adapt.py
"""
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_tmp = tempfile.mkdtemp()
os.environ["PIANO_USERS_DIR"] = _tmp

from pianolib import adapt  # noqa: E402
adapt.USERS_DIR = _tmp
from pianolib.profile import PRESETS  # noqa: E402
from pianolib.rig import Rig  # noqa: E402

FPS = 30.0
FOOT = 0.10          # toe-heel distance in frame heights (= the reference length)


def foot_pts(x, lift_frac, rng, noise):
    """Foot with its ANKLE (what rig aims with) at x, raised by lift_frac x foot length."""
    y = 0.85 - lift_frac * FOOT
    n = lambda: rng.gauss(0, noise)
    return {"toe": (x + 0.05 + n(), y + n(), 0.9),
            "heel": (x + 0.05 - FOOT + n(), y + n(), 0.9),
            "ankle": (x + n(), y - 0.06 + n(), 0.9)}


class Sim:
    def __init__(self, preset="one_foot", user=None, noise=0.0004, seed=0):
        self.rig = Rig(PRESETS[preset], user=user or "u%d" % random.random().__hash__())
        self.rng = random.Random(seed)
        self.noise = noise
        self.t = 0.0
        self.eid = PRESETS[preset].effectors[0].id
        self.x = 0.9
        self.notes = []

    def step(self, lift, x=None):
        if x is not None:
            self.x = x
        obs = {self.eid: foot_pts(self.x, lift, self.rng, self.noise)}
        for ev in self.rig.update(self.t, obs):
            if ev[2] == "down":
                self.notes.append((self.t, ev[1]))
        self.t += 1.0 / FPS

    def rest(self, s):
        for _ in range(int(s * FPS)):
            self.step(0.0)

    def calibrate(self):
        self.rest(2.0)
        assert not self.rig.calibrating, self.rig.status()

    def tap(self, height, x=None, rise=0.15, fall=0.1, hold=0.35):
        if x is not None:
            self.x = x
        n_up, n_dn = int(rise * FPS), int(fall * FPS)
        for i in range(1, n_up + 1):
            self.step(height * i / n_up)
        for i in range(n_dn - 1, -1, -1):
            self.step(height * i / n_dn)
        self.rest(hold)

    def det(self):
        return self.rig.effs[self.eid].det

    def zones(self):
        return self.rig.effs[self.eid].zones


def test_weak_player_starts_getting_notes():
    s = Sim()
    s.calibrate()
    hi0 = s.det().hi
    for _ in range(4):
        s.tap(0.08)                    # 8% of foot length: under the default 12% threshold
    before = len(s.notes)
    for _ in range(14):
        s.tap(0.08)
    assert before == 0, "weak taps should not play at first (default threshold)"
    assert s.det().hi < hi0
    assert len(s.notes) >= 6, "after learning, weak taps play notes (got %d)" % len(s.notes)


def test_strong_player_keeps_default_threshold_and_plays_every_tap():
    s = Sim()
    s.calibrate()
    hi0 = s.det().hi
    for _ in range(15):
        s.tap(0.6)
    assert s.det().hi <= hi0 + 1e-9, "learning never makes it harder than the tuned default"
    assert len(s.notes) == 15


def test_tremor_stops_playing_notes():
    import math
    for amp in (0.08, 0.12, 0.2, 0.35):        # from a small shake to one bigger than their taps
        s = Sim(noise=0.0004)
        s.calibrate()
        for _ in range(14):
            s.tap(0.08)                         # teach it a low threshold first
        s.noise = 0.0008
        mark = None
        for i in range(int(6 * FPS)):           # 5 Hz tremor while resting
            s.step(amp * max(0.0, math.sin(2 * math.pi * 5 * s.t)))
            if i == int(1.5 * FPS):
                mark = len(s.notes)
        late = len(s.notes) - mark
        assert late == 0, "amp %.2f: after 1.5 s of shaking, tremor must not make notes (got %d)" % (amp, late)


def test_playing_again_after_tremor_stops():
    import math
    s = Sim(noise=0.0004)
    s.calibrate()
    for _ in range(14):
        s.tap(0.08)
    s.noise = 0.0008
    for _ in range(int(3 * FPS)):
        s.step(0.2 * max(0.0, math.sin(2 * math.pi * 5 * s.t)))
    s.noise = 0.0004
    s.rest(4.0)                                  # shaking stops
    n = len(s.notes)
    for _ in range(6):
        s.tap(0.08)
    assert len(s.notes) - n >= 5, "weak taps work again once the shaking stops"


def test_threshold_never_below_noise_floor():
    s = Sim()
    s.calibrate()
    for _ in range(30):
        s.tap(0.035)                   # tiny movements
    st = s.det().stats
    assert s.det().hi >= st.noise_floor() - 1e-9


def test_small_reach_shrinks_zones():
    s = Sim()
    s.calibrate()
    z = s.zones()
    w0 = z.width
    rest = z.rest_x
    xs = [rest - 0.25 * w0, rest, rest + 0.25 * w0, rest + 0.5 * w0]   # never reaches the outer zones
    for i in range(24):
        s.tap(0.3, x=xs[i % len(xs)])
    z = s.zones()
    assert z.width < 0.9 * w0
    assert z.width >= max(adapt.MIN_W, adapt.MAX_SHRINK * w0) - 1e-9
    reached = {z.zone_for(x) for x in [rest - 0.3 * w0, rest + 0.6 * w0]}
    assert len(reached) == 2


def test_rest_position_stays_on_home_key():
    s = Sim()
    s.calibrate()
    e = s.rig.effs[s.eid]
    for i in range(24):
        s.tap(0.3, x=e.zones.rest_x + (0.1 if i % 2 else -0.1) * e.zones.fit_w)
    assert e.zones.zone_for(e.rest_x) == e.home


def test_aim_correction_moves_boundary():
    s = Sim()
    s.calibrate()
    z = s.zones()
    k = z.home + 1
    edge0 = z.edges[k]
    for _ in range(5):
        s.tap(0.3, x=edge0 - 0.1 * z.width, hold=0.05)   # lands just left of the boundary
        s.tap(0.3, x=edge0 + 0.6 * z.width, hold=0.4)    # fixes it right away
    z = s.zones()
    assert z.state.fixes >= 1
    assert z.edges[k] < edge0
    assert abs(z.edges[k] - edge0) <= adapt.MAX_NUDGE * z.width + 1e-9


def test_normal_play_is_not_a_correction():
    s = Sim()
    s.calibrate()
    z = s.zones()
    for i in range(10):                            # adjacent notes, but landed in zone centres
        s.tap(0.3, x=z.zone_center(i % z.n), hold=0.5)
    assert s.zones().state.fixes == 0


def test_learning_survives_restart():
    s = Sim(user="maria")
    s.calibrate()
    for _ in range(15):
        s.tap(0.08)
    s.rig.save_learning()
    learned = s.det().hi
    s2 = Sim(user="maria")
    s2.calibrate()
    assert abs(s2.det().hi - learned) < 1e-9, "next session starts already tuned"
    assert os.path.exists(os.path.join(_tmp, "maria.json"))


def test_fatigue_warning():
    s = Sim()
    s.calibrate()
    for _ in range(12):
        s.tap(0.6)
    for _ in range(14):
        s.tap(0.3)
    assert any("weaker" in w for w in s.rig.warnings)


def test_status_lines():
    s = Sim()
    s.calibrate()
    for _ in range(10):
        s.tap(0.3)
    lines = s.rig.learn_status()
    assert lines and "adapted" in lines[0]


def test_fingers_learn_too():
    rig = Rig(PRESETS["one_hand_fingers"], user="fingers")
    rng = random.Random(3)

    def hand(t, bend):                   # bend = how far the index tip curls toward its knuckle
        pts = {"wrist": (1.0, 0.6, 0.9), "palm": (1.0, 0.5, 0.9)}
        for i, g in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
            mx = 0.94 + 0.03 * i
            length = 0.08 * (1 - (bend if g == "index" else 0.0))
            pts["mcp_" + g] = (mx + rng.gauss(0, 0.0003), 0.45, 0.9)
            pts["tip_" + g] = (mx + rng.gauss(0, 0.0003), 0.45 - length, 0.9)
        return {"R_HAND": pts}

    t, notes = 0.0, 0
    for _ in range(60):
        rig.update(t, hand(t, 0.0)); t += 1 / FPS
    assert not rig.calibrating
    det = rig.effs["R_HAND"].dets["index"]
    hi0 = det.hi
    for _ in range(14):                  # weak curls: 25% (default needs 35%)
        for b in [0.08, 0.17, 0.25, 0.17, 0.08, 0.0]:
            notes += sum(1 for e in rig.update(t, hand(t, b)) if e[2] == "down"); t += 1 / FPS
        for _ in range(8):
            rig.update(t, hand(t, 0.0)); t += 1 / FPS
    assert det.hi < hi0 and notes >= 4


def test_unpatched_rig_still_works_without_learner():
    from pianolib.detect import TapDetector, ZoneMap
    from pianolib.adapt import zone_bounds
    z = ZoneMap(0.5, 4, 0.1)
    assert zone_bounds(z, 0) == (z.left, z.left + 0.1)
    assert isinstance(adapt.AdaptiveTapDetector(0.12), TapDetector)


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("ok  ", fn.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL", fn.__name__, "-", e)
    print("%d passed, %d failed" % (len(fns) - failed, failed))
    sys.exit(1 if failed else 0)
