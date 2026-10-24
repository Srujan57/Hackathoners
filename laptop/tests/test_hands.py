"""Hand refinement tests (fake hand model; says nothing about the real model's accuracy)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from helpers import H, W, hand_kpts, hand_points                       # noqa: E402
from pianolib.pose import (COCO133_HAND_BASE, HandRefiner, extract_observations,   # noqa: E402
                           hand_box)
from pianolib.profile import EffectorSpec, InputProfile               # noqa: E402

FRAME = np.zeros((H, W, 3), dtype=np.uint8)


def hand21(cx, cy, size=60.0, score=0.9):
    """A 21-point hand in pixels: wrist at the bottom, fingers fanning up."""
    k = np.zeros((21, 2))
    s = np.full(21, score)
    k[0] = (cx, cy + size / 2)
    for f in range(5):
        for j in range(4):
            k[1 + 4 * f + j] = (cx + (f - 2) * size * 0.2, cy + size / 2 - (j + 1) * size * 0.22)
    return k, s


class FakeHand:
    def __init__(self, kpts, scores):
        self.kpts, self.scores, self.boxes = kpts, scores, []

    def __call__(self, frame, box):
        self.boxes.append(list(box))
        return self.kpts[None], self.scores[None]


def body_with(side, hk, hs):
    k = np.zeros((133, 2))
    s = np.zeros(133)
    b = COCO133_HAND_BASE[side]
    k[b:b + 21], s[b:b + 21] = hk, hs
    return k, s


def test_box_is_square_and_covers_points():
    hk, hs = hand21(500, 400)
    b = hand_box(hk, hs)
    assert abs((b[2] - b[0]) - (b[3] - b[1])) < 1e-6
    assert b[0] < hk[:, 0].min() and b[2] > hk[:, 0].max() and b[1] < hk[:, 1].min() and b[3] > hk[:, 1].max()


def test_box_none_when_too_few_confident_points():
    hk, hs = hand21(500, 400)
    hs[:] = 0.1
    hs[:3] = 0.9
    assert hand_box(hk, hs) is None


def test_box_ignores_scattered_outliers():
    hk, hs = hand21(500, 400)
    hk[20] = (1200, 50)                       # one wild point, still "confident"
    b = hand_box(hk, hs)
    assert b[2] - b[0] < 200


def test_tiny_hand_box_has_minimum_size():
    hk, hs = hand21(500, 400, size=6)
    b = hand_box(hk, hs)
    assert b[2] - b[0] >= 36


def test_refinement_replaces_the_hand_slice_and_leaves_the_rest():
    good, gs = hand21(500, 400)
    noisy = good + np.random.RandomState(0).normal(0, 12, good.shape)
    k, s = body_with("right", noisy, np.full(21, 0.5))
    k[15], s[15] = (123.0, 456.0), 0.8        # an ankle that must stay untouched
    r = HandRefiner(FakeHand(good, gs))
    k2, s2 = r.refine(FRAME, k, s, 0.0)
    b = COCO133_HAND_BASE["right"]
    assert np.allclose(k2[b:b + 21], good) and np.allclose(s2[b:b + 21], gs)
    assert tuple(k2[15]) == (123.0, 456.0)
    assert r.last["right"] == "refined"
    assert np.allclose(k[b:b + 21], noisy)    # input not modified in place


def test_low_confidence_refinement_is_rejected():
    good, _ = hand21(500, 400)
    noisy = good + 3.0
    k, s = body_with("left", noisy, np.full(21, 0.6))
    r = HandRefiner(FakeHand(good, np.full(21, 0.1)))
    k2, s2 = r.refine(FRAME, k, s, 0.0)
    b = COCO133_HAND_BASE["left"]
    assert np.allclose(k2[b:b + 21], noisy) and r.last["left"] == "fallback"


def test_model_exception_does_not_propagate():
    class Boom:
        def __call__(self, *a):
            raise RuntimeError("onnx failed")
    good, gs = hand21(500, 400)
    k, s = body_with("right", good, gs)
    k2, s2 = HandRefiner(Boom()).refine(FRAME, k, s, 0.0)
    assert np.allclose(k2, k)


def test_box_is_held_briefly_when_whole_body_loses_the_hand():
    good, gs = hand21(500, 400)
    m = FakeHand(good, gs)
    r = HandRefiner(m)
    k, s = body_with("right", good, gs)
    r.refine(FRAME, k, s, 0.0)
    empty_k, empty_s = np.zeros((133, 2)), np.zeros(133)
    r.refine(FRAME, empty_k, empty_s, 0.1)                 # lost: reuse the last box
    assert len(m.boxes) == 2
    r.refine(FRAME, empty_k, empty_s, 1.0)                 # lost for too long: no crop
    assert len(m.boxes) == 2 and r.last["right"] == "none"


def test_box_size_changes_are_rate_limited():
    good, gs = hand21(500, 400, size=60)
    m = FakeHand(good, gs)
    r = HandRefiner(m)
    k, s = body_with("right", good, gs)
    r.refine(FRAME, k, s, 0.0)
    big, bs = hand21(500, 400, size=200)
    k, s = body_with("right", big, bs)
    m.kpts = big                                          # the hand model agrees, so tracking follows it
    r.refine(FRAME, k, s, 0.03)
    w0 = m.boxes[0][2] - m.boxes[0][0]
    w1 = m.boxes[1][2] - m.boxes[1][0]
    assert w1 <= w0 * 1.31


def test_box_is_clipped_to_the_frame():
    good, gs = hand21(10, 10)
    m = FakeHand(good, gs)
    k, s = body_with("left", good, gs)
    HandRefiner(m).refine(FRAME, k, s, 0.0)
    x1, y1, x2, y2 = m.boxes[0]
    assert x1 >= 0 and y1 >= 0 and x2 <= W and y2 <= H


def test_extract_observations_uses_refined_points_for_fingers():
    prof = InputProfile(name="t", effectors=[EffectorSpec("R_HAND", fingers=("index", "middle"))])
    pts = hand_points((0.9, 0.6), flex={"index": 0.0, "middle": 0.0})
    k, s = hand_kpts({"right": pts})
    calls = []

    def refine(kk, ss):
        calls.append(1)
        return kk, ss
    obs = extract_observations(k, s, "coco133", W, H, prof, {}, True, refine=refine)
    assert calls == [1] and "R_HAND" in obs and "tip_index" in obs["R_HAND"]


def test_refine_not_called_for_feet_only_profiles():
    prof = InputProfile(name="t", effectors=[EffectorSpec("L_FOOT"), EffectorSpec("R_FOOT")])
    k = np.zeros((1, 26, 2))
    s = np.zeros((1, 26))
    called = []
    extract_observations(k, s, "halpe26", W, H, prof, {}, True, refine=lambda a, b: called.append(1) or (a, b))
    assert not called
