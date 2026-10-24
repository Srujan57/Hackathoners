"""Tests for pianolib/adapt.py. Run: python -m pytest test_adapt.py -q  (or python test_adapt.py)"""
import os
import random
import tempfile

from adapt import AdaptiveModel, Warmup, NUM_KEYS, LIFT_MIN, LIFT_MAX, MIN_ZONE_WIDTH


def taps(model, limb, n, peak, xs, t0=0.0, dur=0.3, gap=1.0):
    t = t0
    for i in range(n):
        model.on_tap(limb, t, xs[i % len(xs)], peak, dur)
        t += gap
    return t


def test_defaults_before_learning():
    m = AdaptiveModel("u")
    th = m.thresholds("L")
    assert th["lift"] == 0.045 and th["land"] == 0.02
    assert m.zone_of("L", 0.0) == 0 and m.zone_of("L", 0.99) == 7


def test_weak_lifter_gets_lower_threshold():
    m = AdaptiveModel("u")
    taps(m, "L", 15, 0.03, [0.5])          # small lifts, under the 0.045 default
    assert m.thresholds("L")["lift"] < 0.03, "a normal lift for this person must count"


def test_strong_lifter_capped():
    m = AdaptiveModel("u")
    taps(m, "L", 15, 0.5, [0.5])
    assert m.thresholds("L")["lift"] == LIFT_MAX


def test_tremor_raises_threshold():
    m = AdaptiveModel("u")
    rng = random.Random(1)
    for i in range(400):                    # resting with a shake
        m.observe("L", i / 30, 0.5, abs(rng.gauss(0, 0.012)))
    taps(m, "L", 15, 0.02, [0.5])           # small taps
    th = m.thresholds("L")
    assert th["lift"] >= 3 * m.limb("L").noise
    assert th["smooth_alpha"] < 0.6          # smoothing got stronger


def test_reach_stretches_zones():
    m = AdaptiveModel("u")
    xs = [0.40 + 0.20 * i / 19 for i in range(20)]   # only moves within 0.40-0.60
    taps(m, "L", 20, 0.05, xs)
    lo, hi = m.limb("L").reach()
    assert 0.35 < lo < 0.41 and 0.59 < hi < 0.65
    assert m.zone_of("L", 0.41) == 0 and m.zone_of("L", 0.59) == 7
    zones = {m.zone_of("L", x) for x in xs}
    assert zones == set(range(NUM_KEYS)), "all 8 keys reachable inside their small range"


def test_warmup_fits_boundaries_to_aim():
    m = AdaptiveModel("u")
    w = Warmup(m, "R")
    # this person consistently lands a bit right of center, and crowded on the left
    centers = [0.10, 0.17, 0.26, 0.37, 0.50, 0.62, 0.74, 0.86]
    rng = random.Random(2)
    while not w.done():
        k = w.target()
        res = w.hit(centers[k] + rng.gauss(0, 0.01))
    assert res is True
    for k, c in enumerate(centers):
        assert m.zone_of("R", c) == k
    e = m.limb("R").edges()
    assert all(e[i + 1] - e[i] >= MIN_ZONE_WIDTH - 1e-9 for i in range(NUM_KEYS))


def test_warmup_rejects_inconsistent_aim():
    m = AdaptiveModel("u")
    w = Warmup(m, "R", rounds=1)
    while not w.done():
        res = w.hit(0.5)                     # lands in the same spot every time
    assert res is False and m.limb("R").boundaries is None


def test_correction_moves_boundary():
    m = AdaptiveModel("u")
    lm = m.limb("L")
    before = lm.edges()[4]                    # edge between zone 3 and 4 (0.5)
    # meant G (zone 4) but clipped F (zone 3) at x=0.49, short note, fixed fast
    t = 0.0
    for _ in range(4):
        m.on_tap("L", t, 0.49, 0.05, 0.08)
        m.on_tap("L", t + 0.25, 0.56, 0.05, 0.4)
        t += 2
    after = lm.edges()[4]
    assert after < before and m.zone_of("L", 0.49) == 4
    assert lm.corrections >= 1


def test_normal_melody_not_treated_as_correction():
    m = AdaptiveModel("u")
    t = 0.0
    for x in [0.06, 0.19, 0.31, 0.44, 0.56, 0.69, 0.81, 0.94]:  # scale, normal note lengths
        m.on_tap("L", t, x, 0.05, 0.35)
        t += 0.4
    assert m.limb("L").corrections == 0


def test_fatigue_detected():
    m = AdaptiveModel("u")
    t = taps(m, "L", 10, 0.06, [0.5])
    taps(m, "L", 15, 0.03, [0.5], t0=t)
    assert m.limb("L").fatigue()


def test_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "users", "maria.json")
        m = AdaptiveModel("maria", p)
        taps(m, "L", 12, 0.03, [0.3, 0.6])
        m.save()
        m2 = AdaptiveModel("maria", p)
        assert m2.load()
        assert m2.thresholds("L") == m.thresholds("L")
        assert m2.limb("L").edges() == m.limb("L").edges()


def test_lift_never_below_floor():
    m = AdaptiveModel("u")
    taps(m, "L", 20, 0.001, [0.5])
    assert m.thresholds("L")["lift"] == LIFT_MIN


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok ", fn.__name__)
    print(f"{len(fns)} passed")
