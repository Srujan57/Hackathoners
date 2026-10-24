"""Old tap logic vs new, on the same synthetic foot motion.  Run:  python tests/bench_vs_legacy.py

This is a benchmark of LOGIC on simulated keypoints. It says nothing about how well the pose model
tracks real feet - that needs real video (see README, "Measuring accuracy").
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from pianolib.profile import EffectorSpec, InputProfile      # noqa: E402
from pianolib.rig import Rig                                  # noqa: E402
import sim                                                    # noqa: E402


def run_new(script, duration, sens=1.0, spec=None):
    prof = InputProfile("t", [spec or EffectorSpec("R_FOOT", zones=4)], sensitivity=sens)
    rig = Rig(prof)
    fires = []
    for t in sim.frames(duration):
        evs = rig.update(t, {"R_FOOT": script.sample(t)})
        fires += [t for (_, _, k) in evs if k == "down"]
    return fires


def run_old(script, duration):
    lg = sim.LegacyFoot()
    for t in sim.frames(duration):
        lg.update(t, script.sample(t)["toe"][1])
    return lg.fires


def taps_every(first, gap, n, amp, kind="toe"):
    return [{"t": first + i * gap, "amp": amp, "kind": kind} for i in range(n)]


SCENARIOS = {
    "toe taps, still foot": dict(taps=taps_every(3.0, 1.0, 12, 0.05)),
    "step taps (whole foot)": dict(taps=taps_every(3.0, 1.0, 12, 0.08, "step")),
    "small toe taps (3 cm-ish)": dict(taps=taps_every(3.0, 1.0, 12, 0.035)),
    "fast toe taps (4 per s)": dict(taps=taps_every(3.0, 0.25, 24, 0.05)),
    "toe taps + depth drift +-0.06": dict(taps=taps_every(3.0, 1.0, 12, 0.05),
                                           drift=lambda t: 0.06 * math.sin(2 * math.pi * t / 9.0)),
    "NO taps, only depth drift +-0.06": dict(taps=[], drift=lambda t: 0.06 * math.sin(2 * math.pi * t / 9.0)),
    "NO taps, slide foot sideways": dict(taps=[], slides=[(4.0, 5.0, 0.15), (8.0, 9.0, -0.15)]),
    "NO taps, noisy keypoints (2x)": dict(taps=[], noise=0.008),
}


def main():
    print("%-36s %-5s %6s %6s %6s %12s" % ("scenario", "logic", "hits", "miss", "false", "latency ms"))
    for name, kw in SCENARIOS.items():
        script = sim.FootScript(**kw)
        dur = 16.0
        contacts = script.contacts()
        for label, fn in (("old", run_old), ("new", run_new)):
            script = sim.FootScript(**kw)          # fresh RNG so both see identical noise
            fires = fn(script, dur)
            # the first 1.5 s are calibration; ignore anything before taps begin
            h, m, f, lat = sim.match_events(contacts, [x for x in fires if x > 2.0])
            med = ("%+.0f" % (1000 * sorted(lat)[len(lat) // 2])) if lat else "-"
            print("%-36s %-5s %6d %6d %6d %12s" % (name if label == "old" else "", label, h, m, f, med))


if __name__ == "__main__":
    main()
