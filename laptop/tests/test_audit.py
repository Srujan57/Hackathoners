"""Regression tests for problems found in the code audit."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import sim                                                            # noqa: E402
from helpers import H, W, foot_kpts, hand_points                      # noqa: E402
from pianolib import layout                                           # noqa: E402
from pianolib.player import PianoLink                                 # noqa: E402
from pianolib.pose import extract_observations, select_person         # noqa: E402
from pianolib.profile import (PRESETS, EffectorSpec, InputProfile, ProfileFile,   # noqa: E402
                              load_profile)
from pianolib.rig import Rig                                          # noqa: E402


# ---------------------------------------------------------------- profile JSON from a frontend
GOOD = {"name": "x", "effectors": [{"id": "L_FOOT", "zones": 4}]}


@pytest.mark.parametrize("patch", [
    {"sensitivity": "high"}, {"sensitivity": None}, {"sensitivity": True}, {"sensitivity": float("nan")},
    {"sensitivity": float("inf")}, {"zone_width": "0.1"}, {"tap_len": [1]}, {"sustain": "false"},
    {"sustain": 1}, {"mode": ["FREE"]}, {"mode": 5}, {"chords": ["L_FOOT"]}, {"meta": [1]},
    {"effectors": {"id": "L_FOOT"}}, {"effectors": "L_FOOT"}, {"effectors": [5]},
    {"effectors": [{"id": ["L_FOOT"]}]}, {"effectors": [{"id": "L_FOOT", "zones": "4"}]},
    {"effectors": [{"id": "L_FOOT", "zones": 3.5}]}, {"effectors": [{"id": "L_FOOT", "zones": True}]},
    {"effectors": [{"id": "L_FOOT", "zones": [4]}]},
    {"effectors": [{"id": "R_HAND", "fingers": "index"}]},
    {"effectors": [{"id": "R_HAND", "fingers": [["index"]]}]},
    {"effectors": [{"id": "R_HAND", "fingers": [1]}]},
])
def test_wrong_types_raise_valueerror_never_anything_else(patch):
    d = dict(GOOD)
    d.update(patch)
    with pytest.raises(ValueError):
        InputProfile.from_dict(d)


def test_name_is_sanitised():
    p = InputProfile.from_dict(dict(GOOD, name="a\x00b\n" + "z" * 200))
    assert p.name.isprintable() and len(p.name) <= 60
    assert InputProfile.from_dict(dict(GOOD, name=12345)).name == "12345"
    assert InputProfile.from_dict(dict(GOOD, name="")).name == "custom"


def test_oversized_profile_file_is_refused(tmp_path):
    f = tmp_path / "p.json"
    f.write_text('{"name": "%s", "effectors": [{"id": "L_FOOT"}]}' % ("a" * 200000))
    with pytest.raises(ValueError):
        load_profile(str(f))


def test_profilefile_survives_garbage_and_picks_up_a_fix(tmp_path, capsys):
    f = tmp_path / "p.json"
    f.write_text('{"name": "a", "effectors": [{"id": "L_FOOT"}]}')
    w = ProfileFile(str(f))
    assert w.poll() is None
    f.write_text('{"sensitivity": "bad", "effectors": [{"id": "L_FOOT"}]}')    # same size class, new content
    os.utime(f, ns=(1, 1))
    assert w.poll() is None                                                    # ignored, no exception
    f.write_text('{"name": "b", "effectors": [{"id": "R_FOOT"}]}')
    os.utime(f, ns=(2, 2))
    assert w.poll().name == "b"


def test_profilefile_notices_a_rewrite_with_the_same_mtime(tmp_path):
    f = tmp_path / "p.json"
    f.write_text('{"name": "a", "effectors": [{"id": "L_FOOT"}]}')
    os.utime(f, ns=(5, 5))
    w = ProfileFile(str(f))
    f.write_text('{"name": "abc", "effectors": [{"id": "L_FOOT"}]}')
    os.utime(f, ns=(5, 5))                                                     # timestamp did not move
    assert w.poll().name == "abc"


def test_profilefile_deleted_file_is_not_an_error(tmp_path):
    f = tmp_path / "p.json"
    f.write_text('{"effectors": [{"id": "L_FOOT"}]}')
    w = ProfileFile(str(f))
    f.unlink()
    assert w.poll() is None


# ---------------------------------------------------------------- serial link
class FakeSerial:
    def __init__(self, fail=False, backlog=0):
        self.fail, self.out, self.in_waiting, self.read_n = fail, [], backlog, 0

    def write(self, b):
        if self.fail:
            raise OSError("device disconnected")
        self.out.append(b)

    def read(self, n):
        self.read_n += n
        self.in_waiting = 0
        return b"x" * n


def test_a_dead_serial_port_does_not_crash_the_app():
    link = PianoLink(None)
    link.ser = FakeSerial(fail=True)
    link.press(0)
    link.release(0)
    link.release_all()
    link.close()
    assert link.error


def test_link_recovers_and_clears_the_error():
    link = PianoLink(None)
    link.ser = FakeSerial(fail=True)
    link.press(0)
    assert link.error
    link.ser.fail = False
    link.release(0)
    assert link.error == "" and link.ser.out == [b"R0\n"]


def test_pico_output_is_drained():
    link = PianoLink(None)
    link.ser = FakeSerial(backlog=300)
    link.press(1)
    assert link.ser.read_n == 300 and link.ser.out == [b"P1\n"]


def test_missing_serial_port_gives_a_clear_exit(monkeypatch):
    serial = pytest.importorskip("serial")

    def boom(*a, **k):
        raise serial.SerialException("could not open port 'COM99'")
    monkeypatch.setattr(serial, "Serial", boom)
    with pytest.raises(SystemExit) as e:
        PianoLink("COM99")
    assert "COM99" in str(e.value)


# ---------------------------------------------------------------- NaN keypoints
def test_nan_keypoints_cannot_wedge_a_foot():
    rig = Rig(PRESETS["one_foot"], aspect=W / float(H))
    sc = sim.FootScript(x0=0.8, taps=[{"t": t0, "amp": 0.05} for t0 in (4.0, 5.0, 8.0, 9.0)], seed=3)
    ev = []
    for t in sim.frames(11):
        pts = sc.sample(t)
        if 6.0 <= t < 6.2:                                    # the model emits garbage for a few frames
            pts = {k: (float("nan"), float("nan"), 0.9) for k in pts}
        ev += [(t,) + e for e in rig.update(t, {"R_FOOT": pts})]
    downs = [e[0] for e in ev if e[3] == "down"]
    assert any(abs(d - 8.22) < 0.3 for d in downs) and any(abs(d - 9.22) < 0.3 for d in downs)


# ---------------------------------------------------------------- finger scale (hand moves in depth)
def scaled_hand(k, flex=None):
    pts = hand_points((1.0, 0.8), flex=flex)
    wx, wy = pts["wrist"][:2]
    return {n: (wx + (x - wx) * k, wy + (y - wy) * k, s) for n, (x, y, s) in pts.items()}


def finger_run(scale_at, press_at=()):
    rig = Rig(InputProfile("t", [EffectorSpec("R_HAND", fingers=("index", "middle"))]))
    ev = []
    for t in sim.frames(10):
        flex = {"index": 1.0} if any(a <= t < b for a, b in press_at) else {}
        ev += [(t,) + e for e in rig.update(t, {"R_HAND": scaled_hand(scale_at(t), flex)})]
    return ev


def test_hand_moving_away_does_not_press_fingers():
    # relaxed hand drifts to 60% size after calibration; unnormalised that reads as flexion ~0.4 > threshold
    ev = finger_run(lambda t: 1.0 if t < 3 else max(0.6, 1.0 - 0.2 * (t - 3)))
    assert [e for e in ev if e[3] == "down"] == []


def test_hand_moving_closer_does_not_press_or_block_real_presses():
    ev = finger_run(lambda t: 1.0 if t < 3 else min(1.4, 1.0 + 0.2 * (t - 3)), press_at=[(8.0, 8.6)])
    assert [(e[2], e[3]) for e in ev] == [("index", "down"), ("index", "up")]


def test_real_press_still_works_when_the_hand_is_small():
    ev = finger_run(lambda t: 1.0 if t < 3 else 0.7, press_at=[(7.0, 7.5)])
    assert [(e[2], e[3]) for e in ev] == [("index", "down"), ("index", "up")]


# ---------------------------------------------------------------- choosing the player among several people
def two_people(x_player, x_other, s_player=0.6, s_other=0.95):
    """Two people with only ankle keypoints; returns (kpts, scores) in the un-mirrored pixel layout."""
    k = np.zeros((2, 26, 2))
    s = np.zeros((2, 26))
    for p, (x, sc) in enumerate(((x_player, s_player), (x_other, s_other))):
        for i in (15, 16, 20, 21, 24, 25):
            k[p, i] = (W - x * H, 0.8 * H)          # un-mirrored pixels for a mirrored x in frame-height units
            s[p, i] = sc
    return k, s


def test_bystander_with_better_keypoints_does_not_steal_tracking():
    k, s = two_people(x_player=0.7, x_other=1.6)
    idx = [15, 16, 20, 21, 24, 25]
    assert select_person(s, idx) == 1                                           # no history: most confident
    assert select_person(s, idx, k, {"R_FOOT": 0.72}, W, H) == 0                # history: stay with the player


def test_lock_follows_the_player_when_they_move_a_little():
    k, s = two_people(x_player=0.9, x_other=1.6)
    assert select_person(s, [15, 16, 20, 21, 24, 25], k, {"R_FOOT": 0.7}, W, H) == 0


def test_lock_falls_back_if_nobody_is_near_and_player_confidence_is_low():
    k, s = two_people(0.7, 1.6, s_player=0.05, s_other=0.9)
    # player's keypoints are all below the minimum mean, so only the confident person is a candidate
    assert select_person(s, [15, 16, 20, 21, 24, 25], k, {"R_FOOT": 0.7}, W, H) == 1


def test_extract_observations_uses_the_lock_after_calibration():
    prof = PRESETS["one_foot"]
    k, s = two_people(0.7, 1.6)
    # keypoints for the player are present but weaker; with history the player's foot is reported
    obs = extract_observations(k, s, "halpe26", W, H, prof, {"R_FOOT": 0.7}, False)
    x = obs["R_FOOT"]["ankle"][0]
    assert abs(x - 0.7) < 0.05


# ---------------------------------------------------------------- small things
def test_chord_labels_are_ascii_for_the_overlay_font():
    assert all(n.isascii() for n in layout.CHORD_NAMES + layout.NOTE_NAMES)


def test_app_keeps_the_old_profile_if_the_new_model_cannot_load():
    import foot_piano

    class Fake:
        def __init__(self, hands):
            self.need_hands, self.layout, self.refiner = hands, "coco133" if hands else "halpe26", None

    def factory(hands):
        if hands:
            raise RuntimeError("download failed")
        return Fake(hands)

    link = PianoLink(None)
    app = foot_piano.App(PRESETS["two_feet"], link, factory)
    with pytest.raises(RuntimeError):
        app.set_profile(PRESETS["hand_foot"])
    assert app.profile is PRESETS["two_feet"] and app.backend.need_hands is False
    app.press_button("PROFILE")                    # next preset needs hands in some cases: must not raise
    app.press_button("PROFILE")
    app.press_button("PROFILE")


def test_foot_kpts_helper_roundtrip_still_ok():
    k, s = foot_kpts({"left": {"ankle": (0.5, 0.8, 0.9)}})
    assert k.shape[2] == 2 and math.isfinite(float(k.sum()))


# ---------------------------------------------------------------- calibration stillness
def _still_foot_run(glitch_every=0, jitter=0.0, dur=3.0):
    rig = Rig(PRESETS["one_foot"], aspect=W / float(H))
    sc = sim.FootScript(x0=0.8, noise=0.002, seed=2)
    for i, t in enumerate(sim.frames(dur)):
        pts = sc.sample(t)
        if glitch_every and i % glitch_every == 0 and i:
            pts = {k: (x + 0.08, y - 0.08, s) for k, (x, y, s) in pts.items()}     # one wild frame
        rig.update(t, {"R_FOOT": pts})
    return rig


def test_one_glitch_frame_does_not_block_calibration():
    assert not _still_foot_run(glitch_every=45).calibrating


def test_a_clearly_moving_player_still_blocks_calibration():
    rig = Rig(PRESETS["one_foot"], aspect=W / float(H))
    sc = sim.FootScript(x0=0.8, slides=[(0.0, 3.0, 0.5)], noise=0.002, seed=2)
    for t in sim.frames(3):
        rig.update(t, {"R_FOOT": sc.sample(t)})
    assert rig.calibrating and "hold still" in rig.status() and "moved" in rig.status()


def test_status_names_the_point_that_is_moving():
    rig = Rig(PRESETS["one_foot"], aspect=W / float(H))
    sc = sim.FootScript(x0=0.8, slides=[(0.0, 3.0, 0.5)], noise=0.002, seed=2)
    for t in sim.frames(2.0):
        rig.update(t, {"R_FOOT": sc.sample(t)})
    assert any(n in rig.status() for n in ("toe", "heel", "ankle"))


# ---------------------------------------------------------------- recorder / replay tool
def test_replay_reproduces_the_live_events(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import json as _json

    import replay

    prof = PRESETS["one_foot"]
    sc = sim.FootScript(x0=0.8, taps=[{"t": 3.0 + i, "amp": 0.05} for i in range(5)], seed=4)
    rig = Rig(prof, aspect=W / float(H))
    f = tmp_path / "rec.jsonl"
    with open(f, "w") as fh:
        fh.write(_json.dumps({"profile": prof.to_dict(), "aspect": W / float(H)}) + "\n")
        for t in sim.frames(10):
            obs = {"R_FOOT": sc.sample(t)}
            evs = rig.update(t, obs)
            fh.write(_json.dumps({"t": t, "ev": [list(e) for e in evs],
                                  "obs": {k: {n: list(v) for n, v in p.items()} for k, p in obs.items()}}) + "\n")
    head, frames = replay.load(str(f))
    rig2, events, cal_t = replay.replay(head, frames, csv_path=str(tmp_path / "lift.csv"))
    live = [(round(fr["t"], 3), e[0], e[1], e[2]) for fr in frames for e in fr["ev"]]
    again = [(round(e[0], 3), e[1], e[2], e[3]) for e in events]
    assert again == live
    assert len([e for e in events if e[3] == "down"]) == 5
    assert (tmp_path / "lift.csv").read_text().startswith("t,calibrating")
    replay.summarise(head, frames, rig2, events, cal_t)          # must not raise
    # a different sensitivity is accepted
    replay.replay(head, frames, sens=0.7)


# ---------------------------------------------------------------- camera set-up advice
from pianolib.rig import distance_cm, setup_advice   # noqa: E402

ASPECT = W / float(H)


def feet_obs(length, y=0.85, lx=0.6, rx=1.1, hide=()):
    def foot(x):
        return {"toe": (x, y, 0.9), "heel": (x, y + length, 0.9), "ankle": (x, y + length * 0.5, 0.9)}
    obs = {"L_FOOT": foot(lx), "R_FOOT": foot(rx)}
    for h in hide:
        obs.pop(h)
    return obs


def test_good_distance_gives_no_advice():
    assert setup_advice(feet_obs(0.2, y=0.7), PRESETS["two_feet"], ASPECT) == ""


def test_too_far_says_move_closer_with_a_number():
    msg = setup_advice(feet_obs(0.08, y=0.7), PRESETS["two_feet"], ASPECT)
    assert "Too far" in msg and "closer" in msg and "cm" in msg


def test_too_close_says_step_back_with_a_number():
    msg = setup_advice(feet_obs(0.45, y=0.45), PRESETS["two_feet"], ASPECT)
    assert "Too close" in msg and "step back" in msg and "cm" in msg


def test_the_numbers_move_in_the_right_direction():
    import re
    far = int(re.search(r"(\d+) cm", setup_advice(feet_obs(0.05, y=0.7), PRESETS["two_feet"], ASPECT)).group(1))
    less_far = int(re.search(r"(\d+) cm", setup_advice(feet_obs(0.10, y=0.7), PRESETS["two_feet"], ASPECT)).group(1))
    assert far > less_far
    assert distance_cm(0.1) > distance_cm(0.2) > distance_cm(0.3)


def test_cropped_feet_and_missing_foot_and_edges():
    assert "cut off" in setup_advice(feet_obs(0.2, y=0.85), PRESETS["two_feet"], ASPECT)
    assert "Can't see R_FOOT" in setup_advice(feet_obs(0.2, y=0.7, hide=("R_FOOT",)), PRESETS["two_feet"], ASPECT)
    assert "edge" in setup_advice(feet_obs(0.2, y=0.7, lx=0.02), PRESETS["two_feet"], ASPECT)


def test_hand_only_profile_gets_no_foot_advice():
    assert setup_advice({}, PRESETS["one_hand_fingers"], ASPECT) == ""


def test_rig_shows_the_advice_only_while_calibrating():
    rig = Rig(PRESETS["two_feet"], aspect=ASPECT)
    rig.update(0.0, feet_obs(0.45, y=0.45))
    assert "step back" in rig.setup_hint
