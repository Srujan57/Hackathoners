"""Software synth, sound hook-up, and the detector-skipping wrapper (fake models, no sound card needed)."""

import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from pianolib.audio import KEY_FREQS, SAMPLE_RATE, AudioOut, Synth   # noqa: E402
from pianolib.player import PianoLink                                # noqa: E402
from pianolib.pose import FastPose                                   # noqa: E402


def peak_hz(x, sr=SAMPLE_RATE):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1.0 / sr)[int(np.argmax(spec))]


# ---------------------------------------------------------------- synth
@pytest.mark.parametrize("key", range(8))
def test_each_key_sounds_at_its_pitch(key):
    s = Synth()
    s.note_on(key)
    x = s.render(SAMPLE_RATE // 2)[2000:]
    assert abs(peak_hz(x) - KEY_FREQS[key]) < 3.0


def test_silence_when_nothing_plays_and_no_nan_or_clipping():
    s = Synth()
    assert not np.any(s.render(1024))
    for k in range(8):
        s.note_on(k)
    x = s.render(SAMPLE_RATE)
    assert np.all(np.isfinite(x)) and np.max(np.abs(x)) <= 1.0 and np.max(np.abs(x)) > 0.05


def test_release_fades_to_silence_and_frees_the_voice():
    s = Synth()
    s.note_on(0)
    s.render(2000)
    s.note_off(0)
    s.render(SAMPLE_RATE // 3)
    assert s.active == [] and not np.any(s.render(256))


def test_held_note_keeps_sounding_but_decays():
    s = Synth()
    s.note_on(2)
    a = np.max(np.abs(s.render(4000)))
    s.render(SAMPLE_RATE)
    b = np.max(np.abs(s.render(4000)))
    assert 0 < b < a


def test_chord_has_all_three_pitches():
    s = Synth()
    for k in (0, 2, 4):
        s.note_on(k)
    x = s.render(SAMPLE_RATE)[3000:]
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / SAMPLE_RATE)
    for k in (0, 2, 4):
        band = spec[(freqs > KEY_FREQS[k] - 3) & (freqs < KEY_FREQS[k] + 3)].max()
        assert band > 0.1 * spec.max()


def test_retrigger_and_bad_keys_are_safe():
    s = Synth()
    s.note_on(1)
    s.render(500)
    s.note_on(1)
    s.note_on(99)
    s.note_off(5)
    s.all_off()
    assert np.all(np.isfinite(s.render(512))) and s.active == [1]


def test_no_discontinuity_between_blocks():
    s = Synth()
    s.note_on(0)
    s.render(3000)
    a = s.render(256)
    b = s.render(256)
    assert abs(float(b[0]) - float(a[-1])) < 0.1


# ---------------------------------------------------------------- hook-up
class Sink:
    def __init__(self):
        self.calls = []

    def note_on(self, k):
        self.calls.append(("on", k))

    def note_off(self, k):
        self.calls.append(("off", k))

    def all_off(self):
        self.calls.append(("all",))


def test_link_drives_the_sound_with_chord_overlap_counting():
    link = PianoLink(None)
    sink = Sink()
    link.sinks.append(sink)
    link.press(0)
    link.press(0)                         # second holder: no new note
    link.release(0)                       # still held by the other: no note-off yet
    assert sink.calls == [("on", 0)]
    link.release(0)
    assert sink.calls == [("on", 0), ("off", 0)]
    link.release_all()
    assert sink.calls[-1] == ("all",)


def test_a_failing_sound_sink_cannot_break_playing(capsys):
    class Bad:
        def note_on(self, k):
            raise RuntimeError("no audio device")

    link = PianoLink(None)
    link.sinks.append(Bad())
    link.press(3)
    assert link.holders[3] == 1


def test_audioout_callback_fills_the_buffer(monkeypatch):
    started = {}

    class FakeStream:
        def __init__(self, **kw):
            started.update(kw)

        def start(self):
            started["on"] = True

        def stop(self):
            started["on"] = False

        def close(self):
            pass

    fake = types.ModuleType("sounddevice")
    fake.OutputStream = FakeStream
    fake.query_devices = lambda *a, **k: {"name": "Fake Speakers", "default_samplerate": 48000.0}
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    out = AudioOut()
    assert started["on"] and started["channels"] == 1 and started["callback"] == out._callback
    out.note_on(0)
    buf = np.zeros((256, 1), dtype=np.float32)
    out._callback(buf, 256, None, None)
    assert np.max(np.abs(buf)) > 0
    out.close()
    assert started["on"] is False


# ---------------------------------------------------------------- detector skipping
FRAME = np.zeros((8, 8, 3), dtype=np.uint8)


class FakeSol:
    """Counts detector/pose calls; the 'person' is at a fixed place with confident keypoints."""

    def __init__(self, people=1, score=0.9):
        self.det_calls = self.pose_calls = 0
        self.last_boxes = None
        self.people, self.score = people, score
        outer = self

        class Det:
            def __call__(self, img):
                outer.det_calls += 1
                return np.array([[100, 100, 300, 500]] * outer.people, dtype=float)

        class Pose:
            def __call__(self, img, bboxes):
                outer.pose_calls += 1
                outer.last_boxes = [list(b) for b in bboxes]
                n = len(bboxes)
                k = np.tile(np.array([[150.0, 400.0], [200.0, 420.0], [180.0, 450.0], [160.0, 480.0]]), (n, 1, 1))
                return k, np.full((n, 4), outer.score)
        self.det_model, self.pose_model = Det(), Pose()


def test_detector_runs_every_n_frames_but_pose_every_frame():
    sol = FakeSol()
    fp = FastPose(sol, det_every=5, async_det=False)
    for _ in range(20):
        k, s = fp(FRAME)
        assert len(k) == 1
    assert sol.det_calls == 4 and sol.pose_calls == 20


def test_boxes_in_between_come_from_the_previous_keypoints():
    sol = FakeSol()
    fp = FastPose(sol, det_every=10, async_det=False)
    fp(FRAME)
    fp(FRAME)
    x1, y1, x2, y2 = sol.last_boxes[0]
    assert x1 < 150 and x2 > 200 and y1 < 400 and y2 > 480


def test_losing_the_player_triggers_detection_on_the_next_frame():
    sol = FakeSol(score=0.05)               # keypoints all unconfident
    fp = FastPose(sol, det_every=10, async_det=False)
    for _ in range(5):
        fp(FRAME)
    assert sol.det_calls == 5


def test_nobody_detected_falls_back_to_whole_frame_and_lower_box():
    sol = FakeSol(people=0)
    fp = FastPose(sol, det_every=3, async_det=False)
    k, s_ = fp(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert sol.pose_calls == 1 and len(sol.last_boxes) == 2
    full, low = sol.last_boxes
    assert full == [0.0, 0.0, 1280.0, 720.0]
    assert 0 <= low[0] < low[2] <= 1280 and 0 < low[1] < 720 and low[3] == 720.0


def test_fallback_boxes_stay_inside_small_and_odd_frames():
    from pianolib.pose import fallback_boxes
    for W, H in ((1280, 720), (640, 480), (320, 1000), (8, 8)):
        for b in fallback_boxes(W, H):
            assert 0 <= b[0] < b[2] <= W and 0 <= b[1] < b[3] <= H


def test_tracked_player_is_kept_when_the_detector_only_finds_someone_else():
    sol = FakeSol()
    fp = FastPose(sol, det_every=1, async_det=False)
    fp(FRAME)                                    # locks on the player (box from keypoints near x 150-200)
    tracked = [list(b) for b in fp.boxes]
    merged = fp._merge(tracked, [[900.0, 100.0, 1100.0, 500.0]])       # a bystander far away
    assert tracked[0] in merged and len(merged) == 2
    same = fp._merge(tracked, [[t + 3 for t in tracked[0]]])           # same person again: not duplicated
    assert len(same) == 1


# ---------------------------------------------------------------- background detector
import threading   # noqa: E402
import time        # noqa: E402


class SlowDetSol(FakeSol):
    """Detector that blocks until released, to prove the frame loop does not wait for it."""

    def __init__(self):
        super().__init__()
        self.gate = threading.Event()
        inner = self.det_model

        class Det:
            def __call__(s, img):
                self.in_det = True
                self.gate.wait(5.0)
                return inner(img)
        self.det_model = Det()
        self.in_det = False


def test_background_detector_never_blocks_a_frame():
    sol = SlowDetSol()                               # the detector hangs from the very first call
    fp = FastPose(sol, det_every=3)
    t0 = time.perf_counter()
    for _ in range(12):
        k, s = fp(FRAME)
        assert len(k) >= 1                           # pose still runs (on the fall-back boxes)
    assert time.perf_counter() - t0 < 1.0
    assert sol.pose_calls == 12
    sol.gate.set()
    fp.close()


def test_background_result_is_picked_up_when_ready():
    sol = FakeSol()
    fp = FastPose(sol, det_every=2)
    for _ in range(10):
        fp(FRAME)
        time.sleep(0.01)
    assert sol.det_calls >= 2                        # asked for in the background, more than once
    fp.close()


def test_while_lost_it_keeps_searching_in_the_background_without_blocking():
    sol = FakeSol(score=0.05)                        # keypoints never confident: stays lost
    fp = FastPose(sol, det_every=100)
    for _ in range(6):
        fp(FRAME)
        time.sleep(0.01)
    assert sol.pose_calls == 6 and sol.det_calls >= 1
    fp.close()
