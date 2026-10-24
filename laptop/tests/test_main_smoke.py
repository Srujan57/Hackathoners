"""Runs foot_piano.main() end to end with a fake camera and fake pose model (real OpenCV drawing, no window)."""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

cv2 = pytest.importorskip("cv2")

import foot_piano                                   # noqa: E402
import sim                                          # noqa: E402
from helpers import foot_kpts                       # noqa: E402
from pianolib import pose                           # noqa: E402

FRAMES = 300                                        # 10 s at 30 fps


class FakeCamera:
    def __init__(self, *a, **k):
        self.i = 0
        self.released = False

    def read(self, timeout=1.0):
        if self.i >= FRAMES:
            return None, 0.0
        t = self.i / 30.0
        self.i += 1
        return np.zeros((720, 1280, 3), dtype=np.uint8), 1000.0 + t

    def release(self):
        self.released = True


class FakeBackend:
    layout = "halpe26"
    refiner = None

    def __init__(self, need_hands, **kw):
        self.need_hands = need_hands
        self.n = 0
        self.closed = False
        self.left = sim.FootScript(x0=0.55, taps=[{"t": 3.0 + i, "amp": 0.05} for i in range(5)], seed=1)
        self.right = sim.FootScript(x0=1.25, seed=2)

    def infer(self, frame):
        t = self.n / 30.0
        self.n += 1
        k, s = foot_kpts({"left": self.left.sample(t), "right": self.right.sample(t)})
        return k, s

    def close(self):
        self.closed = True


@pytest.fixture()
def patched(monkeypatch):
    made = {}

    def cam(*a, **k):
        made["cam"] = FakeCamera()
        return made["cam"]

    def backend(*a, **k):
        made["be"] = FakeBackend(*a, **k)
        return made["be"]
    monkeypatch.setattr(pose, "LatestFrameCamera", cam)
    monkeypatch.setattr(pose, "PoseBackend", backend)
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *a: 0xFF)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: 1.0)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    return made


def run_main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["foot_piano.py", "--no-pico", "--mute"] + list(args))
    foot_piano.main()


def test_whole_app_plays_the_taps_and_cleans_up(patched, monkeypatch, capsys, tmp_path):
    rec = tmp_path / "s.jsonl"
    run_main(monkeypatch, "--record", str(rec))
    out = capsys.readouterr().out
    presses = [ln for ln in out.splitlines() if ln.startswith("[pico] P")]
    assert len(presses) == 5, out[-600:]                  # five left-foot taps, nothing else
    assert all(ln.endswith("P2") for ln in presses)       # left foot rests on zone 2 = E = key 2
    assert patched["cam"].released and patched["be"].closed
    lines = rec.read_text().splitlines()
    head = json.loads(lines[0])
    assert head["profile"]["name"] == "two feet" and len(lines) == FRAMES + 1


def test_record_refuses_to_overwrite_an_existing_file(patched, monkeypatch, tmp_path):
    f = tmp_path / "keep.jsonl"
    f.write_text("precious")
    with pytest.raises(SystemExit) as e:
        run_main(monkeypatch, "--record", str(f))
    assert "already exists" in str(e.value) and f.read_text() == "precious"
    assert patched["cam"].released                        # cleaned up even though it stopped early


def test_a_model_error_on_one_frame_does_not_end_the_session(patched, monkeypatch, capsys):
    orig = FakeBackend.infer
    calls = {"n": 0}

    def flaky(self, frame):
        calls["n"] += 1
        if calls["n"] == 100:
            raise RuntimeError("onnx hiccup")
        return orig(self, frame)
    monkeypatch.setattr(FakeBackend, "infer", flaky)
    run_main(monkeypatch)
    out = capsys.readouterr().out
    assert "[error] frame skipped" in out and calls["n"] == FRAMES


def test_the_window_close_button_ends_the_app(patched, monkeypatch):
    seen = {"n": 0}

    def prop(*a):
        seen["n"] += 1
        return 0.0 if seen["n"] > 20 else 1.0
    monkeypatch.setattr(cv2, "getWindowProperty", prop)
    run_main(monkeypatch)
    assert patched["cam"].i < 40                          # stopped soon after the window was closed


def test_q_key_quits(patched, monkeypatch):
    keys = iter([0xFF] * 10 + [ord("q")])
    monkeypatch.setattr(cv2, "waitKey", lambda *a: next(keys, 0xFF))
    run_main(monkeypatch)
    assert patched["cam"].i == 11


def test_sens_out_of_range_is_rejected(patched, monkeypatch):
    with pytest.raises(SystemExit) as e:
        run_main(monkeypatch, "--sens", "9")
    assert "between 0.2 and 5" in str(e.value)


def test_sens_is_applied_and_the_hud_readout_does_not_crash(patched, monkeypatch, capsys):
    run_main(monkeypatch, "--sens", "2")
    assert len([l for l in capsys.readouterr().out.splitlines() if l.startswith("[pico] P")]) >= 1
