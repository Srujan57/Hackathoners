"""Builders for fake pose-model output and fake hands, used by the tests."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from pianolib.pose import (COCO133_FEET, COCO133_HAND_BASE, FINGER_MCP, FINGER_TIP,  # noqa: E402
                           HALPE26_FEET)

W, H = 1280, 720


def to_px(pt):
    """Mirrored frame-height units -> un-mirrored pixel coordinates the model would return."""
    x, y, s = pt
    return (W - x * H, y * H, s)


def foot_kpts(feet, layout="halpe26", swap_labels=False, n_kp=None):
    """feet = {'left': {'toe','heel','ankle'} , 'right': {...}} in mirrored frame-height units.
    'left' is the player's LEFT (appears on screen-left after mirroring). swap_labels mimics a model that
    swaps its left/right outputs."""
    table = HALPE26_FEET if layout == "halpe26" else COCO133_FEET
    K = n_kp or (26 if layout == "halpe26" else 133)
    k = np.zeros((1, K, 2), dtype=float)
    s = np.zeros((1, K), dtype=float)
    for side, pts in feet.items():
        label = side
        if swap_labels:
            label = "right" if side == "left" else "left"
        for name, idx in table[label].items():
            if name in pts:
                x, y, sc = to_px(pts[name])
                k[0, idx] = (x, y)
                s[0, idx] = sc
    return k, s


def hand_points(wrist_xy, flex=None, fingers=("thumb", "index", "middle", "ring", "pinky"),
                lengths=None, score=0.9, short=()):
    """Point dict for a hand in mirrored frame-height units. flex[finger] in 0..1 (0 = open)."""
    flex = flex or {}
    lengths = lengths or {}
    wx, wy = wrist_xy
    pts = {"wrist": (wx, wy, score), "palm": (wx, wy - 0.09, score)}
    offs = {"thumb": -0.045, "index": -0.025, "middle": 0.0, "ring": 0.025, "pinky": 0.045}
    for g in fingers:
        base = (wx + offs[g], wy - 0.09)
        length = lengths.get(g, 0.07) * (0.15 if g in short else 1.0)
        tip_y = base[1] - length * (1.0 - 0.7 * flex.get(g, 0.0))
        pts["mcp_" + g] = (base[0], base[1], score)
        pts["tip_" + g] = (base[0], tip_y, score)
    return pts


def hand_kpts(hands, fingers_all=("thumb", "index", "middle", "ring", "pinky")):
    """hands = {'left': pts-dict, 'right': pts-dict} -> COCO-WholeBody arrays (feet left empty)."""
    k = np.zeros((1, 133, 2), dtype=float)
    s = np.zeros((1, 133), dtype=float)
    for side, pts in hands.items():
        base = COCO133_HAND_BASE[side]
        if "wrist" in pts:
            x, y, sc = to_px(pts["wrist"])
            k[0, base], s[0, base] = (x, y), sc
        for g in fingers_all:
            for prefix, table in (("mcp_", FINGER_MCP), ("tip_", FINGER_TIP)):
                if prefix + g in pts:
                    x, y, sc = to_px(pts[prefix + g])
                    k[0, base + table[g]], s[0, base + table[g]] = (x, y), sc
        # the palm point is the middle finger's MCP
        if "palm" in pts:
            x, y, sc = to_px(pts["palm"])
            k[0, base + FINGER_MCP["middle"]], s[0, base + FINGER_MCP["middle"]] = (x, y), sc
    return k, s
