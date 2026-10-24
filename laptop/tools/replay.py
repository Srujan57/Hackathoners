"""Replay a recording made with  foot_piano.py --record FILE  through the tap logic, offline.

    python tools/replay.py rec.jsonl                       # what the app did, plus a summary
    python tools/replay.py rec.jsonl --sens 0.7            # same footage, different sensitivity
    python tools/replay.py rec.jsonl --zone-width 0.14
    python tools/replay.py rec.jsonl --csv lift.csv        # per-frame lift signal, to plot

The recording holds keypoint coordinates only - no video or images.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pianolib.profile import InputProfile   # noqa: E402
from pianolib.rig import Rig                # noqa: E402


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        head = json.loads(f.readline())
        frames = [json.loads(line) for line in f if line.strip()]
    return head, frames


def replay(head, frames, sens=None, zone_width=None, csv_path=None):
    d = dict(head["profile"])
    if sens is not None:
        d["sensitivity"] = sens
    if zone_width is not None:
        d["zone_width"] = zone_width
    prof = InputProfile.from_dict(d)
    rig = Rig(prof, head.get("aspect", 16 / 9.0))
    events, cal_t, rows = [], None, []
    for fr in frames:
        t = fr["t"]
        obs = {eid: {n: tuple(v) for n, v in pts.items()} for eid, pts in fr["obs"].items()}
        evs = rig.update(t, obs)
        if cal_t is None and not rig.calibrating:
            cal_t = t
        for eid, slot, kind in evs:
            events.append((t, eid, slot, kind))
        if csv_path:
            row = {"t": t, "calibrating": int(rig.calibrating)}
            for eid, e in rig.effs.items():
                row[eid + "_lift"] = round(getattr(e, "last_lift", 0.0), 4)
                a = e.anchor
                row[eid + "_x"] = round(a[0], 4) if a else ""
            rows.append(row)
    if csv_path and rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rig, events, cal_t


def summarise(head, frames, rig, events, cal_t):
    t0, t1 = frames[0]["t"], frames[-1]["t"]
    dur = t1 - t0
    print("frames %d, %.1f s, average %.1f fps" % (len(frames), dur, len(frames) / dur if dur else 0))
    print("calibrated at %s" % ("never" if cal_t is None else "%.1f s" % (cal_t - t0)))
    for eid, e in rig.effs.items():
        if hasattr(e, "det"):
            print("  %s: lift threshold %.2f foot-lengths, noise sigma %.3f, foot length %.3f frame-heights"
                  % (eid, e.det.hi, getattr(e, "sigma", 0.0), e.ref))
    for w in rig.warnings:
        print("  warning:", w)
    downs = [e for e in events if e[3] == "down"]
    print("\nnotes: %d" % len(downs))
    for (eid, slot), n in sorted(Counter((e[1], e[2]) for e in downs).items(), key=str):
        print("  %s zone/finger %s: %d" % (eid, slot, n))
    bursts = 0
    last = {}
    for t, eid, slot, kind in downs:
        if eid in last and t - last[eid] < 0.3:
            bursts += 1
        last[eid] = t
    print("notes that came <0.3 s after another note from the same foot/hand: %d" % bursts)
    if downs and dur:
        print("notes per minute after calibration: %.0f" % (60.0 * len(downs) / max(t1 - (cal_t or t0), 1e-6)))
    live = sum(1 for fr in frames for e in fr.get("ev", []) if e[2] == "down")
    print("(the live run played %d notes)" % live)
    print("\nfirst 25 notes:  time(s)  who  zone")
    for t, eid, slot, kind in downs[:25]:
        print("  %7.2f  %-7s %s" % (t - t0, eid, slot))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--sens", type=float)
    ap.add_argument("--zone-width", type=float)
    ap.add_argument("--csv")
    a = ap.parse_args()
    head, frames = load(a.file)
    if not frames:
        sys.exit("empty recording")
    rig, events, cal_t = replay(head, frames, a.sens, a.zone_width, a.csv)
    summarise(head, frames, rig, events, cal_t)
