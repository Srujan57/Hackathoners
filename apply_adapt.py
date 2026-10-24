"""
Wires pianolib/adapt.py into pianolib/rig.py. Run from the repo root:

    python apply_adapt.py

Makes small, targeted edits (about 25 lines) and refuses to touch the file if any spot it expects
has changed. Safe to run twice. A backup is written to laptop/pianolib/rig.py.bak first.
"""
import re
import shutil
import sys

PATH = "laptop/pianolib/rig.py"
if len(sys.argv) > 1:
    PATH = sys.argv[1]

EDITS = [
    ("import adapt",
     "from .filters import OneEuro",
     "from .filters import OneEuro\n"
     "from .adapt import (AdaptiveFlexDetector, AdaptiveTapDetector, AdaptiveZoneMap, Learner,\n"
     "                    zone_bounds)"),

    ("effectors know the learner",
     "self.pts = {}\n        self.warnings = []\n\n    # ---- helpers",
     "self.pts = {}\n        self.warnings = []\n        self.learner = None\n\n    # ---- helpers"),

    ("adaptive tap detector",
     "self.det = TapDetector(lift_hi=hi)",
     "self.det = (AdaptiveTapDetector(lift_hi=hi, noise_sigma=sigma, stats=self.learner.strength(self.id))\n"
     "                    if self.learner else TapDetector(lift_hi=hi))"),

    ("zones learn from each tap",
     "z = self.zones.zone_for(x, self.prev_zone)",
     "z = self.zones.zone_for(x, self.prev_zone)\n"
     "                if hasattr(self.zones, \"observe_tap\"):\n"
     "                    self.zones.observe_tap(t, x, z)"),

    ("adaptive finger detector",
     "dets[g] = FlexDetector(hi=hi, lo=0.6 * hi)",
     "dets[g] = (AdaptiveFlexDetector(hi=hi, lo=0.6 * hi, noise_sigma=sigma,\n"
     "                                            stats=self.learner.strength(self.id + \"/\" + g))\n"
     "                       if self.learner else FlexDetector(hi=hi, lo=0.6 * hi))"),

    ("rig takes a user",
     "def __init__(self, profile, aspect=16 / 9.0, min_score=MIN_SCORE):",
     "def __init__(self, profile, aspect=16 / 9.0, min_score=MIN_SCORE, user=None):"),

    ("rig creates the learner",
     "self.effs = {e.id: make_effector(e, profile.sensitivity, min_score) for e in profile.effectors}",
     "self.effs = {e.id: make_effector(e, profile.sensitivity, min_score) for e in profile.effectors}\n"
     "        self.learner = Learner(user or profile.meta.get(\"user\"))\n"
     "        for e in self.effs.values():\n"
     "            e.learner = self.learner"),

    ("learner sees every event",
     "events += [(eid, slot, kind) for slot, kind in evs]\n        return events",
     "events += [(eid, slot, kind) for slot, kind in evs]\n"
     "        self.learner.on_events(t, events, self.warnings)\n"
     "        return events"),

    ("adaptive zones",
     "e.zones = ZoneMap(left + e.n * w / 2.0, e.n, w)",
     "e.zones = AdaptiveZoneMap(left + e.n * w / 2.0, e.n, w, rest_x=e.rest_x, home=e.home,\n"
     "                                      state=self.learner.zones(e.id))"),

    ("UI draws learned zone sizes",
     "return [(e.id, i, e.zones.left + i * e.zones.width, e.zones.left + (i + 1) * e.zones.width)",
     "return [(e.id, i) + tuple(zone_bounds(e.zones, i))"),

    ("status lines for the UI",
     "return \"Calibrating - rest feet flat / hands open.  \" + (\"; \".join(msgs) if msgs else \"hold still...\")",
     "return \"Calibrating - rest feet flat / hands open.  \" + (\"; \".join(msgs) if msgs else \"hold still...\")\n"
     "\n"
     "    def learn_status(self):\n"
     "        \"\"\"One line per effector describing what has been learned (for the UI).\"\"\"\n"
     "        return self.learner.status()\n"
     "\n"
     "    def save_learning(self):\n"
     "        self.learner.save()"),
]


def loose(s):
    """Regex for s that tolerates any amount of whitespace between tokens."""
    parts = re.split(r"(\s+)", s)
    return "".join(r"\s*" if p.isspace() else re.escape(p) for p in parts if p)


def main():
    src = open(PATH, encoding="utf-8").read()
    if "from .adapt import" in src:
        print("rig.py already wired for adaptive learning - nothing to do.")
        return
    missing = [name for name, old, _ in EDITS if len(re.findall(loose(old), src)) != 1]
    if missing:
        print("Stopped, rig.py was NOT changed. These spots were not found exactly once:")
        for m in missing:
            print("  -", m)
        sys.exit(1)
    shutil.copy(PATH, PATH + ".bak")
    for name, old, new in EDITS:
        src = re.sub(loose(old), lambda _m, n=new: n, src, count=1)
        print("ok ", name)
    open(PATH, "w", encoding="utf-8").write(src)
    print("rig.py updated (backup: %s.bak)" % PATH)


if __name__ == "__main__":
    main()
