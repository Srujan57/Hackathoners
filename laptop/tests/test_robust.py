"""Robustness: high/held lifts always count, a foot that blurs out mid-lift still counts, and the other foot
twitching while one foot slides is not a note.     cd laptop && python -m pytest tests/test_robust.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import sim                                                     # noqa: E402
from pianolib.profile import PRESETS                          # noqa: E402
from pianolib.rig import Rig                                   # noqa: E402

ASPECT = 16 / 9.0


class HeldFoot(sim.FootScript):
    """Taps may hold at the top ('hold' seconds) and may blur out of tracking ('blur': (t0, t1))."""

    def __init__(self, *a, blur=(), **kw):
        super().__init__(*a, **kw)
        self.blur = list(blur)

    def _lift(self, t):
        toe = step = 0.0
        for tp in self.taps:
            hold = tp.get("hold", 0.0)
            u = t - tp["t"]
            if not 0 <= u <= self.rise + hold + self.fall:
                continue
            if u < self.rise:
                prof = sim.raised_cos(0.5 * u / self.rise)
            elif u < self.rise + hold:
                prof = 1.0
            else:
                prof = sim.raised_cos(0.5 + 0.5 * (u - self.rise - hold) / self.fall)
            if tp.get("kind", "toe") == "toe":
                toe = max(toe, tp["amp"] * prof)
            else:
                step = max(step, tp["amp"] * prof)
        return toe, step

    def sample(self, t):
        pts = super().sample(t)
        if any(a <= t < b for a, b in self.blur):
            return {k: (x, y, 0.1) for k, (x, y, s) in pts.items()}       # model lost the foot
        return pts


def run(left, right, dur):
    rig = Rig(PRESETS["two_feet"], aspect=ASPECT, user=None)
    downs = []
    for t in sim.frames(dur):
        obs = {"L_FOOT": left.sample(t), "R_FOOT": right.sample(t)}
        downs += [(t, e, z) for e, z, k in rig.update(t, obs) if k == "down"]
    assert not rig.calibrating
    return rig, downs


def notes(downs, eid, t0=0.0, t1=99.0):
    return [d for d in downs if d[1] == eid and t0 <= d[0] <= t1]


def test_high_lift_held_at_the_top_still_plays():
    L = HeldFoot(x0=0.6, seed=1, taps=[{"t": 3.0 + 2.0 * i, "amp": 0.09, "hold": h}
                                       for i, h in enumerate((0.0, 0.4, 0.8))])
    R = HeldFoot(x0=1.2, seed=2)
    _, d = run(L, R, 12)
    assert len(notes(d, "L_FOOT")) == 3, notes(d, "L_FOOT")


def test_whole_foot_step_held_up_still_plays():
    L = HeldFoot(x0=0.6, seed=3, taps=[{"t": 3.0, "amp": 0.06, "hold": 0.9, "kind": "step"}])
    R = HeldFoot(x0=1.2, seed=4)
    _, d = run(L, R, 6)
    assert len(notes(d, "L_FOOT")) == 1


def test_foot_that_blurs_out_at_the_top_of_a_lift_still_plays():
    L = HeldFoot(x0=0.6, seed=5, taps=[{"t": 3.0, "amp": 0.09, "hold": 0.4}], blur=[(3.15, 3.55)])
    R = HeldFoot(x0=1.2, seed=6)
    _, d = run(L, R, 6)
    assert len(notes(d, "L_FOOT")) == 1


def test_resting_a_foot_up_for_a_long_time_does_not_get_stuck():
    # foot raised and left there for 3 s (resting on something): the baseline must re-plant, then normal
    # taps from the new height work again
    L = HeldFoot(x0=0.6, seed=7, taps=[{"t": 3.0, "amp": 0.06, "hold": 3.0, "kind": "step"},
                                       {"t": 8.0, "amp": 0.06}, {"t": 9.0, "amp": 0.06}])
    R = HeldFoot(x0=1.2, seed=8)
    _, d = run(L, R, 11)
    assert len(notes(d, "L_FOOT", 7.5, 11)) == 2


def test_other_foot_twitching_while_one_foot_slides_is_not_a_note():
    for seed in range(5):
        L = HeldFoot(x0=0.6, seed=10 + seed, slides=[(4.0, 4.5, -0.36), (7.0, 7.4, 0.36)])
        R = HeldFoot(x0=1.2, seed=20 + seed, taps=[{"t": 4.15, "amp": 0.02}, {"t": 7.1, "amp": 0.022}])
        _, d = run(L, R, 9)
        assert notes(d, "R_FOOT") == [], (seed, notes(d, "R_FOOT"))


def test_real_taps_with_the_other_foot_still_play_during_a_slide():
    L = HeldFoot(x0=0.6, seed=31, slides=[(4.0, 4.6, -0.36)])
    R = HeldFoot(x0=1.2, seed=32, taps=[{"t": 4.2, "amp": 0.05}, {"t": 6.0, "amp": 0.05}])
    _, d = run(L, R, 8)
    assert len(notes(d, "R_FOOT")) == 2


def test_small_taps_still_play_when_the_other_foot_is_still():
    L = HeldFoot(x0=0.6, seed=41)
    R = HeldFoot(x0=1.2, seed=42, taps=[{"t": 3.0 + 0.8 * i, "amp": 0.03} for i in range(5)])
    _, d = run(L, R, 8)
    assert len(notes(d, "R_FOOT")) == 5


def test_sliding_foot_itself_still_plays_where_it_lands():
    L = HeldFoot(x0=0.6, seed=51, slides=[(4.0, 4.5, -0.36)], taps=[{"t": 5.0, "amp": 0.05}])
    R = HeldFoot(x0=1.2, seed=52)
    _, d = run(L, R, 7)
    assert len(notes(d, "L_FOOT")) == 1
