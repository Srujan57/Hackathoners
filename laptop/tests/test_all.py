"""Run:  cd laptop && python -m pytest tests -q"""

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np                                                    # noqa: E402

import sim                                                            # noqa: E402
from helpers import H, W, foot_kpts, hand_kpts, hand_points          # noqa: E402
from pianolib import layout                                           # noqa: E402
from pianolib.detect import FlexDetector, TapDetector, ZoneMap       # noqa: E402
from pianolib.filters import OneEuro                                  # noqa: E402
from pianolib.player import PianoLink, Player                         # noqa: E402
from pianolib.pose import extract_observations, select_person   # noqa: E402
from pianolib.profile import (PRESETS, EffectorSpec, InputProfile, ProfileFile,   # noqa: E402
                              load_profile)
from pianolib.rig import Rig                                          # noqa: E402
from pianolib.ui import DwellMenu                                     # noqa: E402


# ---------------------------------------------------------------- profile
class TestProfile:
    def test_presets_are_valid_and_round_trip(self):
        for name, p in PRESETS.items():
            again = InputProfile.from_dict(json.loads(json.dumps(p.to_dict())))
            assert again.to_dict() == p.to_dict(), name

    @pytest.mark.parametrize("bad", [
        {"effectors": []},
        {"effectors": [{"id": "HEAD"}]},
        {"effectors": [{"id": "R_HAND", "fingers": ["toe"]}]},
        {"effectors": [{"id": "R_FOOT", "fingers": ["index"]}]},
        {"effectors": [{"id": "R_FOOT"}, {"id": "R_FOOT"}]},
        {"effectors": [{"id": "R_FOOT"}], "mode": "DISCO"},
        {"effectors": [{"id": "R_FOOT"}], "chords": "L_FOOT"},
        {"effectors": [{"id": "R_FOOT"}], "sensitivty": 1.0},
        {"effectors": [{"id": "R_FOOT", "zones": 20}]},
        {"effectors": [{"id": "R_FOOT"}], "zone_width": 0.9},
    ])
    def test_rejects_bad_profiles(self, bad):
        with pytest.raises(ValueError):
            InputProfile.from_dict(bad)

    def test_frontend_extras_are_allowed_in_meta(self):
        p = InputProfile.from_dict({"effectors": [{"id": "R_FOOT"}], "meta": {"user": "x"}, "version": 1})
        assert p.meta == {"user": "x"}

    def test_hot_reload_ignores_half_written_files(self, tmp_path):
        f = tmp_path / "p.json"
        f.write_text(json.dumps(PRESETS["one_foot"].to_dict()))
        w = ProfileFile(str(f))
        assert w.poll() is None
        f.write_text("{ not json")
        os.utime(f, (os.path.getmtime(f) + 5, os.path.getmtime(f) + 5))
        assert w.poll() is None                                  # bad content: keep playing
        f.write_text(json.dumps(PRESETS["two_feet"].to_dict()))
        os.utime(f, (os.path.getmtime(f) + 10, os.path.getmtime(f) + 10))
        assert w.poll().name == "two feet"

    def test_load_by_name_and_unknown(self):
        assert load_profile("one_foot").name == "one foot"
        with pytest.raises(ValueError):
            load_profile("nope")

    def test_hand_needs_hands_model(self):
        assert PRESETS["hand_foot"].needs_hands and not PRESETS["two_feet"].needs_hands


# ---------------------------------------------------------------- layout
class TestLayout:
    @pytest.mark.parametrize("preset", list(PRESETS))
    @pytest.mark.parametrize("mode", layout.available_modes(PRESETS["two_feet"]) + ["SPLIT"])
    def test_every_slot_gets_playable_keys(self, preset, mode):
        p = PRESETS[preset]
        m = layout.build_slotmap(p, mode)
        assert set(m) == set(layout.all_slots(p))
        for a in m.values():
            assert a.is_next or (a.keys and all(0 <= k < layout.NUM_KEYS for k in a.keys))

    def test_key_sets_by_slot_count(self):
        assert [a.keys[0] for a in layout.note_actions(4)] == [0, 2, 4, 7]
        assert [a.keys[0] for a in layout.note_actions(3)] == [0, 2, 4]
        assert [a.keys[0] for a in layout.note_actions(8)] == list(range(8))
        assert [a.keys[0] for a in layout.note_actions(5)] == [0, 1, 2, 4, 5]   # pentatonic

    def test_more_slots_than_keys_still_valid_and_monotonic(self):
        ks = [a.keys[0] for a in layout.note_actions(10)]
        assert len(ks) == 10 and ks == sorted(ks) and set(ks) <= set(range(8))

    def test_two_feet_free_split_between_feet(self):
        m = layout.build_slotmap(PRESETS["two_feet"], "FREE")
        assert [m[("L_FOOT", i)].keys[0] for i in range(4)] == [0, 1, 2, 3]
        assert [m[("R_FOOT", i)].keys[0] for i in range(4)] == [4, 5, 6, 7]

    def test_hand_foot_split_foot_chords_hand_melody(self):
        m = layout.build_slotmap(PRESETS["hand_foot"], "SPLIT")
        assert [m[("L_FOOT", i)].label for i in range(4)] == ["C", "F", "G", "Am"]
        fingers = [m[("R_HAND", g)] for g in ("index", "middle", "ring")]
        assert [f.keys[0] for f in fingers] == [0, 2, 4]          # C E G, low->high

    def test_left_hand_fingers_go_up_in_pitch_pinky_to_thumb(self):
        spec = EffectorSpec("L_HAND", fingers=("thumb", "index", "pinky"))
        assert spec.slot_ids() == ["pinky", "index", "thumb"]
        assert EffectorSpec("R_HAND", fingers=("thumb", "index", "pinky")).slot_ids() == ["thumb", "index", "pinky"]

    def test_left_before_right_foot_before_hand(self):
        p = InputProfile("x", [EffectorSpec("R_HAND", fingers=("index",)), EffectorSpec("R_FOOT"),
                               EffectorSpec("L_HAND", fingers=("index",)), EffectorSpec("L_FOOT")])
        assert [e.id for e in p.effectors] == ["L_FOOT", "L_HAND", "R_FOOT", "R_HAND"]

    def test_single_effector_has_no_split_and_falls_back(self):
        p = PRESETS["one_foot"]
        assert "SPLIT" not in layout.available_modes(p)
        assert len(layout.build_slotmap(p, "SPLIT")) == 4

    def test_guided_makes_every_slot_play_next(self):
        assert all(a.is_next for a in layout.build_slotmap(PRESETS["hand_foot"], "GUIDED").values())


# ---------------------------------------------------------------- detectors
class TestDetectors:
    def test_one_euro_smooths_still_noise_but_follows_steps(self):
        import random
        rng, f = random.Random(0), OneEuro()
        raw = [0.5 + rng.gauss(0, 0.004) for i in range(90)]
        out = [f(x, i / 30.0) for i, x in enumerate(raw)]
        assert max(out[30:]) - min(out[30:]) < 0.5 * (max(raw[30:]) - min(raw[30:]))
        f2 = OneEuro()
        ys = [f2(0.0 if i < 10 else 0.1, i / 30.0) for i in range(20)]
        assert ys[16] > 0.07                                     # follows a fast step within ~6 frames

    def test_zone_hysteresis(self):
        z = ZoneMap(1.0, 4, 0.2)                                 # zones: [0.6,0.8) [0.8,1.0) [1.0,1.2) [1.2,1.4)
        assert z.zone_for(0.7) == 0 and z.zone_for(1.35) == 3
        assert z.zone_for(1.0 + 0.01, prev=1) == 1               # just over the border: stay
        assert z.zone_for(1.0 + 0.06, prev=1) == 2               # clearly over: switch
        assert z.zone_for(-5) == 0 and z.zone_for(9) == 3        # clamps

    def test_flex_detector_hysteresis(self):
        d = FlexDetector(hi=0.4, lo=0.2)
        seq = [(0.0, 0.1), (0.1, 0.45), (0.2, 0.3), (0.3, 0.15), (0.4, 0.5)]
        out = [d.update(t, v) for t, v in seq]
        assert out == [[], ["down"], [], ["up"], ["down"]]

    def test_tap_detector_rejects_jitter_and_fires_once(self):
        d = TapDetector(lift_hi=0.12)
        ev = []
        for i in range(60):
            ev += d.update(i / 30.0, 0.04 * math.sin(i))         # jitter below threshold
        assert ev == []
        t = 2.0
        lifts = [0.0, 0.1, 0.25, 0.3, 0.2, 0.1, 0.02, 0.0, 0.0]
        out = []
        for i, l in enumerate(lifts):
            out += d.update(t + i / 30.0, l)
        assert out.count("down") == 1


# ---------------------------------------------------------------- tap logic on simulated motion
def run_feet(script_kw, duration, sens=1.0, seeds=(1, 2, 3)):
    hits = miss = false = 0
    for sd in seeds:
        kw = dict(script_kw, seed=sd)
        script = sim.FootScript(**kw)
        rig = Rig(InputProfile("t", [EffectorSpec("R_FOOT", zones=4)], sensitivity=sens))
        fires = []
        for t in sim.frames(duration):
            fires += [t for (_, _, k) in rig.update(t, {"R_FOOT": script.sample(t)}) if k == "down"]
        h, m, f, _ = sim.match_events(sim.FootScript(**kw).contacts(), [x for x in fires if x > 2.0])
        hits, miss, false = hits + h, miss + m, false + f
    return hits, miss, false


def taps(first, gap, n, amp, kind="toe"):
    return [{"t": first + i * gap, "amp": amp, "kind": kind} for i in range(n)]


DRIFT = lambda t: 0.06 * math.sin(2 * math.pi * t / 9.0)                 # noqa: E731


class TestTapLogic:
    @pytest.mark.parametrize("name,kw,dur,min_recall", [
        ("toe taps", dict(taps=taps(3.0, 1.0, 12, 0.05)), 16, 1.0),
        ("step taps", dict(taps=taps(3.0, 1.0, 12, 0.08, "step")), 16, 1.0),
        ("small toe taps", dict(taps=taps(3.0, 1.0, 12, 0.035)), 16, 1.0),
        ("2 taps/s", dict(taps=taps(3.0, 0.5, 20, 0.05)), 15, 0.97),
        ("3 taps/s", dict(taps=taps(3.0, 0.33, 30, 0.05)), 15, 0.95),
        ("taps + depth drift", dict(taps=taps(3.0, 1.0, 12, 0.05), drift=DRIFT), 16, 0.95),
        ("taps + 2x noise", dict(taps=taps(3.0, 1.0, 12, 0.05), noise=0.008), 16, 0.9),
    ])
    def test_detects_taps(self, name, kw, dur, min_recall):
        h, m, f = run_feet(kw, dur)
        assert h / float(h + m) >= min_recall, (name, h, m, f)
        assert f <= 1, (name, h, m, f)

    @pytest.mark.parametrize("name,kw", [
        ("depth drift only", dict(taps=[], drift=DRIFT)),
        ("sideways slides", dict(taps=[], slides=[(4.0, 5.0, 0.15), (8.0, 9.0, -0.15)])),
        ("3x noise only", dict(taps=[], noise=0.012)),
        ("fast drift", dict(taps=[], drift=lambda t: 0.1 * math.sin(2 * math.pi * t / 6.0))),
    ])
    def test_no_ghost_notes(self, name, kw):
        h, m, f = run_feet(kw, 20)
        assert f == 0, (name, f)

    def test_known_limit_heavy_noise_plus_heavy_drift(self):
        # Documented limit: 2x keypoint jitter AND a foot drifting +-5 cm toward/away from the camera.
        # It may occasionally invent a note (~3 per 100 s in simulation); it must not be worse than that.
        h, m, f = run_feet(dict(taps=[], noise=0.008, drift=DRIFT), 20)
        assert f <= 4, f

    def test_known_limit_worst_case_fails_safe_by_missing_not_inventing(self):
        h, m, f = run_feet(dict(taps=taps(3.0, 1.0, 12, 0.05), noise=0.008, drift=DRIFT), 16)
        assert f <= 1, (h, m, f)             # recall is poor here (see README), but no ghost notes

    def test_higher_sensitivity_catches_smaller_taps(self):
        kw = dict(taps=taps(3.0, 1.0, 12, 0.014))
        lo = run_feet(kw, 16, sens=1.0)[0]
        hi = run_feet(kw, 16, sens=2.5)[0]
        assert hi > lo

    def test_foot_resting_raised_replants_silently(self):
        # the player lifts a foot and simply holds it up: must not play when put back down
        script = sim.FootScript(taps=[], noise=0.003)
        rig = Rig(InputProfile("t", [EffectorSpec("R_FOOT", zones=4)]))
        fires = []
        for t in sim.frames(14):
            pts = script.sample(t)
            if 4.0 <= t < 9.0:                                # raised by 0.06 for 5 s
                pts = {k: (x, y - 0.06, s) for k, (x, y, s) in pts.items()}
            fires += [t for (_, _, k) in rig.update(t, {"R_FOOT": pts}) if k == "down"]
        assert [x for x in fires if x > 4.0 and x < 9.0] == []

    def test_slide_then_tap_picks_the_new_zone(self):
        script = sim.FootScript(x0=0.9, taps=taps(8.0, 1.0, 3, 0.05),
                                slides=[(4.0, 5.0, 0.0), (5.0, 6.0, 0.0)])
        # slide the foot to the right by two zone widths (default 0.1*aspect each) before tapping
        script = sim.FootScript(x0=0.9, taps=taps(8.0, 1.0, 3, 0.05), slides=[(4.0, 5.0, 2 * 0.1 * W / H)])
        rig = Rig(InputProfile("t", [EffectorSpec("R_FOOT", zones=4)]), aspect=W / float(H))
        slots = []
        for t in sim.frames(12):
            slots += [s for (_, s, k) in rig.update(t, {"R_FOOT": script.sample(t)}) if k == "down" and t > 7]
        assert slots and all(s >= 3 for s in slots[:3]), slots


# ---------------------------------------------------------------- rig with two feet
def two_feet_frame(t, scripts):
    return {"L_FOOT": scripts[0].sample(t), "R_FOOT": scripts[1].sample(t)}


class TestRig:
    def make(self, lx=0.55, rx=1.25, ltaps=(), rtaps=(), **kw):
        scripts = (sim.FootScript(x0=lx, taps=list(ltaps), seed=5, **kw),
                   sim.FootScript(x0=rx, taps=list(rtaps), seed=6, **kw))
        rig = Rig(PRESETS["two_feet"], aspect=W / float(H))
        return rig, scripts

    def run(self, rig, scripts, dur):
        ev = []
        for t in sim.frames(dur):
            ev += [(t,) + e for e in rig.update(t, two_feet_frame(t, scripts))]
        return ev

    def test_auto_calibrates_then_ready(self):
        rig, sc = self.make()
        assert rig.calibrating
        self.run(rig, sc, 3)
        assert not rig.calibrating and rig.status() == ""

    def test_close_feet_get_one_row_keyboard_with_full_width_zones(self):
        rig, sc = self.make(lx=0.80, rx=1.00)                  # feet 0.2 apart: lanes would overlap
        self.run(rig, sc, 3)
        w = PRESETS["two_feet"].zone_width * rig.aspect
        assert {round(e.zones.width, 6) for e in rig.effs.values()} == {round(w, 6)}
        assert rig.mat and not rig.stacked
        assert len(set(rig.lane_rows().values())) == 1        # drawn as one row
        L, R = rig.effs["L_FOOT"].zones, rig.effs["R_FOOT"].zones
        assert abs((L.left + L.n * L.width) - R.left) < 1e-9   # side by side, no overlap, no gap
        assert L.left <= 0.80 <= L.left + L.n * L.width        # each foot stands inside its own lane
        assert R.left <= 1.00 <= R.left + R.n * R.width

    def test_stacked_layout_still_available(self):
        from pianolib.profile import InputProfile
        p = InputProfile.from_dict(dict(PRESETS["two_feet"].to_dict(), meta={"lanes": "stacked"}))
        _, sc = self.make(lx=0.80, rx=1.00)
        rig = Rig(p, aspect=W / float(H))
        self.run(rig, sc, 3)
        assert rig.stacked and not rig.mat
        rows = rig.lane_rows()
        assert rows["L_FOOT"][1] <= rows["R_FOOT"][0] + 1e-9

    def test_close_feet_tapping_in_place_plays_the_zone_under_each_foot(self):
        rig, sc = self.make(lx=0.80, rx=1.00, ltaps=taps(3.0, 0.7, 8, 0.05), rtaps=taps(3.3, 0.7, 8, 0.05))
        ev = self.run(rig, sc, 9.5)
        assert rig.mat
        for eid, x in (("L_FOOT", 0.80), ("R_FOOT", 1.00)):
            zs = [z for _, e, z, k in ev if e == eid and k == "down"]
            assert len(zs) >= 7, (eid, zs)
            assert set(zs) == {rig.effs[eid].zones.zone_for(x)}, (eid, zs)

    def test_far_apart_feet_share_one_row(self):
        rig, sc = self.make(lx=0.35, rx=1.45)
        self.run(rig, sc, 3)
        assert not rig.stacked
        assert len(set(rig.lane_rows().values())) == 1

    def test_tapping_in_place_always_plays_the_same_zone(self):
        rig, sc = self.make(ltaps=taps(3.0, 0.7, 10, 0.05), rtaps=taps(3.3, 0.7, 10, 0.05))
        ev = [e for e in self.run(rig, sc, 12) if e[3] == "down"]
        assert {e[2] for e in ev if e[1] == "L_FOOT"} == {2}      # left foot: home zone is 2 of 0-3
        assert {e[2] for e in ev if e[1] == "R_FOOT"} == {1}      # right foot: home zone is 1 of 0-3

    def test_resting_position_is_the_centre_of_a_zone(self):
        rig, sc = self.make()
        self.run(rig, sc, 3)
        for eid, home in (("L_FOOT", 2), ("R_FOOT", 1)):
            e = rig.effs[eid]
            assert e.zones.zone_center(home) == pytest.approx(e.rest_x, abs=1e-6)

    def test_each_foot_plays_only_its_own_lane(self):
        rig, sc = self.make(ltaps=taps(3.0, 1.0, 3, 0.05), rtaps=taps(3.5, 1.0, 3, 0.05))
        ev = [e for e in self.run(rig, sc, 8) if e[3] == "down"]
        assert {e[1] for e in ev} == {"L_FOOT", "R_FOOT"}
        assert sum(e[1] == "L_FOOT" for e in ev) == 3 and sum(e[1] == "R_FOOT" for e in ev) == 3

    def test_foot_lost_releases_held_note_and_reappearance_is_silent(self):
        prof = InputProfile("t", [EffectorSpec("R_FOOT", zones=4)], sustain=True)
        rig = Rig(prof)
        script = sim.FootScript(taps=taps(3.0, 5.0, 1, 0.05))
        ev = []
        for t in sim.frames(10):
            raw = {} if 3.5 <= t < 6.0 else script.sample(t)
            ev += [(t,) + e for e in rig.update(t, {"R_FOOT": raw})]
        kinds = [(round(e[0], 1), e[3]) for e in ev]
        assert ("down" in [k for _, k in kinds])
        downs = [e for e in ev if e[3] == "down"]
        assert len(downs) == 1                                   # nothing played when it came back
        assert any(e[3] == "up" and 3.0 < e[0] < 4.5 for e in ev)

    def test_notes_blocked_inside_menu_strip(self):
        rig = Rig(InputProfile("t", [EffectorSpec("R_FOOT", zones=4)]))
        script = sim.FootScript(y0=0.05, taps=taps(3.0, 1.0, 3, 0.05))
        ev = []
        for t in sim.frames(8):
            ev += rig.update(t, {"R_FOOT": script.sample(t)}, menu_y=0.09)
        assert [e for e in ev if e[2] == "down"] == []


# ---------------------------------------------------------------- fingers
class FingerRun:
    def __init__(self, fingers=("index", "middle", "ring"), short=(), side="R", sens=1.0):
        spec = EffectorSpec(side + "_HAND", fingers=fingers)
        self.id = spec.id
        self.rig = Rig(InputProfile("t", [spec], sensitivity=sens))
        self.fingers, self.short = fingers, short
        self.events = []

    def step(self, t, flex):
        pts = hand_points((1.0, 0.8), flex=flex, short=self.short)
        self.events += [(t,) + e for e in self.rig.update(t, {self.id: pts})]

    def play(self, schedule, dur):
        for t in sim.frames(dur):
            flex = {g: (1.0 if a <= t < b else 0.0) for g, a, b in schedule}
            self.step(t, flex)


class TestFingers:
    def test_flexing_a_finger_presses_only_that_key(self):
        r = FingerRun()
        r.play([("middle", 3.0, 3.6), ("ring", 4.5, 5.0)], 7)
        got = [(e[2], e[3]) for e in r.events]
        assert got == [("middle", "down"), ("middle", "up"), ("ring", "down"), ("ring", "up")]

    def test_two_fingers_at_once(self):
        r = FingerRun()
        r.play([("index", 3.0, 4.0), ("ring", 3.2, 4.0)], 6)
        downs = [e[2] for e in r.events if e[3] == "down"]
        assert sorted(downs) == ["index", "ring"]

    def test_unused_fingers_are_ignored(self):
        r = FingerRun(fingers=("index",))
        r.play([("pinky", 3.0, 4.0)], 6)
        assert r.events == []

    def test_too_short_finger_is_disabled_with_a_warning(self):
        r = FingerRun(fingers=("index", "middle"), short=("middle",))
        r.play([("index", 3.0, 3.5), ("middle", 4.0, 4.5)], 6)
        assert [(e[2], e[3]) for e in r.events] == [("index", "down"), ("index", "up")]
        assert any("too short" in w for w in r.rig.warnings)

    def test_hand_swaying_does_not_press(self):
        r = FingerRun()
        for t in sim.frames(8):
            pts = hand_points((1.0 + 0.05 * math.sin(t * 2), 0.8 + 0.03 * math.sin(t * 3)))
            r.events += r.rig.update(t, {r.id: pts})
        assert r.events == []


# ---------------------------------------------------------------- pose keypoints -> observations
def feet_dict(lx, rx, y=0.78):
    f = lambda x: sim.FootScript(x0=x, y0=y, noise=0.0).sample(0.0)       # noqa: E731
    return {"left": f(lx), "right": f(rx)}


class TestPose:
    def test_halpe_mapping_and_mirroring(self):
        k, s = foot_kpts(feet_dict(0.55, 1.25))
        obs = extract_observations(k, s, "halpe26", W, H, PRESETS["two_feet"], {}, True)
        assert set(obs) == {"L_FOOT", "R_FOOT"}
        assert obs["L_FOOT"]["toe"][0] == pytest.approx(0.55 + 0.06, abs=1e-6)
        assert obs["R_FOOT"]["heel"][0] == pytest.approx(1.25 - 0.06, abs=1e-6)
        assert obs["L_FOOT"]["toe"][1] == pytest.approx(0.78, abs=1e-6)

    def test_coco133_feet_use_the_wholebody_indices(self):
        k, s = foot_kpts(feet_dict(0.55, 1.25), layout="coco133")
        obs = extract_observations(k, s, "coco133", W, H, PRESETS["two_feet"], {}, True)
        assert obs["L_FOOT"]["toe"][0] == pytest.approx(0.61, abs=1e-6)

    def test_swapped_model_labels_are_corrected_by_position(self):
        k, s = foot_kpts(feet_dict(0.55, 1.25), swap_labels=True)
        obs = extract_observations(k, s, "halpe26", W, H, PRESETS["two_feet"], {}, True)
        assert obs["L_FOOT"]["toe"][0] < obs["R_FOOT"]["toe"][0]
        # and once running, identity follows position rather than the (flipping) labels
        prev = {"L_FOOT": 0.55, "R_FOOT": 1.25}
        k2, s2 = foot_kpts(feet_dict(0.56, 1.24), swap_labels=True)
        obs2 = extract_observations(k2, s2, "halpe26", W, H, PRESETS["two_feet"], prev, False)
        assert obs2["L_FOOT"]["toe"][0] == pytest.approx(0.62, abs=1e-6)

    def test_one_foot_profile_follows_its_foot_not_the_other(self):
        prof = PRESETS["one_foot"]                                  # R_FOOT only
        k, s = foot_kpts(feet_dict(0.55, 1.25))
        obs = extract_observations(k, s, "halpe26", W, H, prof, {}, True)
        assert obs["R_FOOT"]["toe"][0] == pytest.approx(1.31, abs=1e-6)     # the right one at the start
        obs = extract_observations(*foot_kpts(feet_dict(0.58, 1.20), swap_labels=True), "halpe26", W, H,
                                   prof, {"R_FOOT": 1.25}, False)
        assert obs["R_FOOT"]["toe"][0] == pytest.approx(1.26, abs=1e-6)     # tracked by position

    def test_low_confidence_points_are_dropped_by_the_rig(self):
        feet = feet_dict(0.55, 1.25)
        feet["left"]["toe"] = (feet["left"]["toe"][0], feet["left"]["toe"][1], 0.05)
        obs = extract_observations(*foot_kpts(feet), "halpe26", W, H, PRESETS["two_feet"], {}, True)
        rig = Rig(PRESETS["two_feet"])
        rig.update(0.0, obs)
        assert "toe" not in rig.effs["L_FOOT"].pts and "heel" in rig.effs["L_FOOT"].pts

    def test_person_selection_prefers_the_confident_person(self):
        s = np.zeros((2, 26))
        s[1, [15, 16, 20, 21, 24, 25]] = 0.9
        s[0, [15, 16, 20, 21, 24, 25]] = 0.2
        assert select_person(s, [15, 16, 20, 21, 24, 25]) == 1

    def test_hand_keypoints_map_to_named_fingers(self):
        pts = hand_points((1.0, 0.8))
        k, s = hand_kpts({"right": pts})
        prof = PRESETS["one_hand_fingers"]                          # R_HAND, all fingers
        obs = extract_observations(k, s, "coco133", W, H, prof, {}, True)
        o = obs["R_HAND"]
        assert o["wrist"][:2] == pytest.approx((1.0, 0.8), abs=1e-6)
        assert o["tip_index"][1] < o["mcp_index"][1]                # fingertip is above its knuckle
        assert o["tip_pinky"][0] == pytest.approx(1.045, abs=1e-6)

    def test_no_detection_gives_no_observations(self):
        assert extract_observations(np.zeros((0, 26, 2)), np.zeros((0, 26)), "halpe26", W, H,
                                    PRESETS["two_feet"], {}, True) == {}


# ---------------------------------------------------------------- player / link
class FakeLink(PianoLink):
    def __init__(self):
        super().__init__(None)
        self.sent = []

    def _send(self, msg):
        self.sent.append(msg)


class TestPlayer:
    def test_tap_note_ends_after_tap_len(self):
        link = FakeLink()
        p = Player(link, PRESETS["one_foot"])
        link.sent.clear()
        p.on_event("R_FOOT", 0, "down", 10.0)
        assert link.sent == ["P0"]
        p.tick(10.1)
        assert link.sent == ["P0"]
        p.tick(10.3)
        assert link.sent == ["P0", "R0"]

    def test_up_event_is_ignored_unless_sustain(self):
        link = FakeLink()
        p = Player(link, PRESETS["one_foot"])
        link.sent.clear()
        p.on_event("R_FOOT", 0, "down", 0.0)
        p.on_event("R_FOOT", 0, "up", 0.05)
        assert link.sent == ["P0"]
        link2 = FakeLink()
        prof = InputProfile("s", [EffectorSpec("R_FOOT", zones=4)], sustain=True)
        p2 = Player(link2, prof)
        link2.sent.clear()
        p2.on_event("R_FOOT", 0, "down", 0.0)
        p2.on_event("R_FOOT", 0, "up", 0.05)
        assert link2.sent == ["P0", "R0"]

    def test_chord_overlap_holds_shared_key(self):
        link = FakeLink()
        prof = InputProfile("c", [EffectorSpec("R_FOOT", zones=4)], mode="CHORD", sustain=True)
        p = Player(link, prof)
        p.on_event("R_FOOT", 0, "down", 0.0)         # C  = keys 0,2,4
        p.on_event("R_FOOT", 2, "down", 0.1)         # G  = keys 4,6,1
        p.on_event("R_FOOT", 0, "up", 0.2)
        assert "R4" not in link.sent and link.held(4)    # key 4 is still held by G
        p.on_event("R_FOOT", 2, "up", 0.3)
        assert not link.held(4)

    def test_guided_advances_through_the_song(self):
        link = FakeLink()
        prof = InputProfile("g", [EffectorSpec("R_FOOT", zones=2)], mode="GUIDED")
        p = Player(link, prof)
        for i in range(3):
            p.on_event("R_FOOT", i % 2, "down", float(i))
            p.tick(float(i) + 0.5)
        assert [m for m in link.sent if m.startswith("P")] == ["P2", "P2", "P3"]   # Ode to Joy: E E F

    def test_mode_cycle_depends_on_profile(self):
        p = Player(FakeLink(), PRESETS["one_foot"])
        seen = []
        for _ in range(5):
            seen.append(p.mode)
            p.next_mode()
        assert "SPLIT" not in seen
        p2 = Player(FakeLink(), PRESETS["hand_foot"])
        modes = []
        for _ in range(4):
            modes.append(p2.mode)
            p2.next_mode()
        assert "SPLIT" in modes

    def test_changing_profile_releases_everything(self):
        link = FakeLink()
        p = Player(link, PRESETS["one_foot"])
        p.on_event("R_FOOT", 0, "down", 0.0)
        p.set_profile(PRESETS["two_feet"])
        assert link.sent[-1] == "A" and not any(link.holders)


# ---------------------------------------------------------------- dwell menu
class TestDwell:
    def test_activates_after_dwell_not_before_and_has_cooldown(self):
        m = DwellMenu(dwell_s=1.0, cooldown_s=2.0)
        r = m.rects(W, H)["MODE"]
        p = [((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)]
        assert m.update(0.0, p, W, H) is None
        assert m.update(0.5, p, W, H) is None and 0.4 < m.progress["MODE"] < 0.6
        assert m.update(1.05, p, W, H) == "MODE"
        assert m.update(1.5, p, W, H) is None                   # cooldown
        assert m.update(3.2, p, W, H) is None and m.update(4.3, p, W, H) == "MODE"

    def test_leaving_resets_the_timer(self):
        m = DwellMenu(dwell_s=1.0)
        r = m.rects(W, H)["SONG"]
        inside = [((r[0] + r[2]) / 2, 10)]
        m.update(0.0, inside, W, H)
        m.update(0.6, [(50, 600)], W, H)
        assert m.update(1.1, inside, W, H) is None


# ---------------------------------------------------------------- whole pipeline
class FakeBackend:
    def __init__(self, need_hands, frames):
        self.need_hands = need_hands
        self.layout = "coco133" if need_hands else "halpe26"
        self.frames = frames
        self.i = 0

    def infer(self, frame):
        out = self.frames[self.i]
        self.i += 1
        return out


class TestPipeline:
    def test_two_feet_end_to_end_prints_the_right_pico_commands(self):
        import foot_piano
        sc_l = sim.FootScript(x0=0.55, taps=taps(3.0, 2.0, 2, 0.05), seed=1)
        sc_r = sim.FootScript(x0=1.25, taps=taps(3.5, 2.0, 2, 0.05), seed=2)
        times = sim.frames(9)
        frames = [foot_kpts({"left": sc_l.sample(t), "right": sc_r.sample(t)}) for t in times]
        link = FakeLink()
        app = foot_piano.App(PRESETS["two_feet"], link, lambda need: FakeBackend(need, frames), W / float(H))
        blank = np.zeros((H, W, 3), dtype=np.uint8)
        import time as _t
        base = _t.monotonic()
        for t in times:
            app.process(blank, base + t)
            app.player.tick(base + t)
        presses = [m for m in link.sent if m.startswith("P")]
        # left foot lane = keys 0-3, right foot lane = keys 4-7; all four taps landed in the middle zones
        assert len(presses) == 4
        assert {int(m[1:]) for m in presses[::2]} <= {0, 1, 2, 3}
        assert {int(m[1:]) for m in presses[1::2]} <= {4, 5, 6, 7}

    def test_switching_to_a_hand_profile_swaps_the_model_and_recalibrates(self):
        import foot_piano
        made = []
        app = foot_piano.App(PRESETS["two_feet"], FakeLink(),
                             lambda need: made.append(need) or FakeBackend(need, []), W / float(H))
        app.set_profile(PRESETS["hand_foot"])
        assert made == [False, True] and app.rig.calibrating
        app.set_profile(PRESETS["one_foot"])
        assert made == [False, True, False]


class TestChordLayout8:
    def test_eight_slot_chords_have_no_accidental_duplicate_at_the_start(self):
        labels = [a.label for a in layout.chord_actions(8)]
        assert labels == ["C", "Dm", "Em", "F", "G", "Am", "Bdim", "C"]
