"""Pose model (RTMPose via rtmlib), keypoint mapping and a low-latency camera.

Model choice: rtmlib's BodyWithFeet (RTMPose, 26-keypoint Halpe layout: ankle + big toe + small toe + heel
per foot) when the profile only uses feet, and Wholebody (RTMW, 133 keypoints: feet + both hands with 21
points each) when it uses a hand or fingers. Both are top-down: a person detector, then keypoints.

The pure functions here (extract_observations) take numpy arrays, so they are tested without the model.
"""

import sys
import threading
import time

import numpy as np

# ---- keypoint layouts (indices verified against rtmlib's own skeleton tables) ----
# Halpe-26: 15/16 ankles, 20/21 big toes, 24/25 heels (left/right)
HALPE26_FEET = {
    "left": {"ankle": 15, "toe": 20, "heel": 24},
    "right": {"ankle": 16, "toe": 21, "heel": 25},
}
# COCO-WholeBody-133: 15/16 ankles, 17/20 big toes, 19/22 heels; hands 91-111 (left) and 112-132 (right)
COCO133_FEET = {
    "left": {"ankle": 15, "toe": 17, "heel": 19},
    "right": {"ankle": 16, "toe": 20, "heel": 22},
}
COCO133_HAND_BASE = {"left": 91, "right": 112}
# 21-point hand: 0 wrist, then 4 per finger (thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20)
FINGER_MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
FINGER_TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}


# Dedicated 21-point hand model (RTMPose-m, trained on the "hand5" dataset mix), same point order as the
# whole-body hand slice. It takes a 256x256 crop of ONE hand, so a hand that is only ~25 px wide in the camera
# frame is seen at roughly 10x the resolution the whole-body model gives it (that one squeezes the entire
# person into 192x256 pixels).
HAND5_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
             "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip")


def _expand_box(pts, k=1.3):
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = max(x2 - x1, 8.0) * k / 2.0, max(y2 - y1, 8.0) * k / 2.0
    return [cx - hw, cy - hh, cx + hw, cy + hh]


def _contained(a, b):
    """Fraction of the SMALLER box that lies inside the other. A legs-only keypoint box sits inside the
    person detector's whole-body box, so plain IoU would call them different people."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / small if small > 0 else 0.0


def fallback_boxes(W, H):
    """Where to look for the player when no person detector has found them: the whole frame, and a wide box
    over the lower part of the picture (a floor-level camera that only sees legs is not what the person
    detector was trained on, so it may find nobody)."""
    low = [max(0.0, W / 2.0 - 0.6 * H), 0.35 * H, min(float(W), W / 2.0 + 0.6 * H), float(H)]
    return [[0.0, 0.0, float(W), float(H)], low]


class _AsyncDetector:
    """Runs the person detector on a background thread so it never stalls a frame (onnxruntime releases the
    GIL while it runs). One job at a time; the newest finished result is picked up with take()."""

    def __init__(self, det_model, lock):
        self.det = det_model
        self._lock = lock
        self._cv = threading.Condition()
        self._req = None
        self._res = None
        self._busy = False
        self._run = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def submit(self, frame):
        with self._cv:
            if self._busy or self._req is not None:
                return False
            self._req = frame.copy()                 # the camera thread may reuse the original
            self._cv.notify()
            return True

    def take(self):
        with self._cv:
            r, self._res = self._res, None
            return r

    def _loop(self):
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._req is not None or not self._run)
                if not self._run:
                    return
                frame, self._req, self._busy = self._req, None, True
            try:
                with self._lock:
                    boxes = self.det(frame)
            except Exception as e:
                print("[detector] error:", e)
                boxes = None
            with self._cv:
                self._res, self._busy = boxes, False

    def close(self):
        with self._cv:
            self._run = False
            self._cv.notify()


class FastPose:
    """Keeps the (slow) person detector off the critical path and never blocks a frame on it.

    The pose model runs on every frame, on boxes taken from the previous frame's confident keypoints.
    The detector runs on a background thread (every `det_every` frames while tracking, every free moment while
    the player is lost) and its boxes are merged in: tracked boxes are kept, newly seen people are added.
    While the player is lost and the detector has found nobody, the pose model is tried on fall-back boxes
    (whole frame + lower part), so a legs-only view that the person detector misses can still lock on.
    async_det=False runs the detector in line instead (used by the tests)."""

    def __init__(self, solution, det_every=30, conf=0.3, min_pts=4, async_det=True):
        self.sol, self.det_every, self.conf, self.min_pts = solution, max(1, int(det_every)), conf, min_pts
        self.boxes, self.n = None, 0
        self._lock = threading.Lock()                 # one detector call at a time
        self.bg = _AsyncDetector(solution.det_model, self._lock) if async_det else None

    @staticmethod
    def _merge(tracked, fresh):
        out = list(tracked)
        for f in fresh:
            if not any(_contained(t, f) > 0.5 for t in tracked):
                out.append(list(f))
        return out

    def __call__(self, frame):
        H, W = frame.shape[:2]
        lost = self.boxes is None
        if self.bg is not None:
            fresh = self.bg.take()                    # a background result that finished since last frame
            self.n += 1
            if lost or self.n % self.det_every == 0:
                self.bg.submit(frame)
        else:
            redetect = lost or self.n % self.det_every == 0
            self.n += 1
            with self._lock:
                fresh = self.sol.det_model(frame) if redetect else None
        if fresh is not None and len(fresh) == 0:
            fresh = None
        if lost:
            boxes = [list(b) for b in fresh] if fresh is not None else fallback_boxes(W, H)
        elif fresh is not None:
            boxes = self._merge(self.boxes, fresh)
        else:
            boxes = self.boxes
        k, s = self.sol.pose_model(frame, bboxes=boxes)
        nb = []
        for kp, sc in zip(k, s):
            ok = np.asarray(sc) >= self.conf
            if int(ok.sum()) >= self.min_pts:
                nb.append(_expand_box(np.asarray(kp)[ok]))
        self.boxes = nb or None
        return k, s

    def close(self):
        if self.bg is not None:
            self.bg.close()


class PoseBackend:
    def __init__(self, need_hands, mode="lightweight", device="cpu", refine_hands=True, det_every=30):
        from rtmlib import BodyWithFeet, Wholebody
        self.need_hands = need_hands
        self.refiner = None
        self.det_every = det_every
        if need_hands:
            self.model = Wholebody(mode=mode, backend="onnxruntime", device=device)
            self.layout = "coco133"
            if refine_hands:
                from rtmlib import RTMPose
                hand = RTMPose(HAND5_URL, model_input_size=(256, 256), backend="onnxruntime", device=device)
                self.refiner = HandRefiner(lambda img, box: hand(img, bboxes=[box]))
        else:
            self.model = BodyWithFeet(mode=mode, backend="onnxruntime", device=device)
            self.layout = "halpe26"

        self.fast = FastPose(self.model, det_every) if det_every > 1 else None

    def close(self):
        if self.fast is not None:
            self.fast.close()

    def infer(self, frame_bgr):
        if self.fast is not None:
            return self.fast(frame_bgr)
        return self.model(frame_bgr)


# --------------------------------------------------------------------------
# Hand refinement: crop each hand out of the FULL-resolution frame and run the hand-only model on it
# --------------------------------------------------------------------------
def hand_box(hk, hs, thr=0.3, min_pts=6, margin=0.15, min_side=36.0):
    """Square xyxy pixel box around the believable points of a 21-point hand, or None.

    Low-confidence whole-body hand points can be scattered far from the hand, so points much farther from the
    median than the typical point are dropped before the box is taken.
    """
    hk = np.asarray(hk, dtype=float)
    ok = np.asarray(hs) >= thr
    if int(ok.sum()) < min_pts:
        return None
    p = hk[ok]
    med = np.median(p, axis=0)
    d = np.hypot(p[:, 0] - med[0], p[:, 1] - med[1])
    p = p[d <= 2.5 * float(np.median(d)) + 2.0]
    if len(p) < min_pts:
        return None
    x1, y1 = p.min(axis=0)
    x2, y2 = p.max(axis=0)
    side = max(x2 - x1, y2 - y1, min_side) * (1.0 + 2.0 * margin)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]


class HandRefiner:
    """Replaces the whole-body model's hand keypoints with a hand-only model's, run on a crop.

    model(frame, xyxy_box) -> (kpts (1,21,2) or (21,2) in FRAME pixels, scores (1,21) or (21,)).
    The crop box comes from the whole-body hand points; it is smoothed over time, and when the whole-body
    model loses the hand the box from the last accepted refinement is reused for a short while.
    A refined result is only used if the hand model itself is reasonably confident, so a crop on empty air
    cannot invent a hand.
    """

    def __init__(self, model, accept=0.35, box_alpha=0.6, max_grow=0.3, hold_s=0.3):
        self.model = model
        self.accept, self.alpha, self.max_grow, self.hold_s = accept, box_alpha, max_grow, hold_s
        self._state = {}                                  # side -> (cx, cy, side, t)
        self.last = {}                                    # side -> "refined" | "fallback" | "none" (for the UI)

    def reset(self):
        self._state.clear()
        self.last.clear()

    def _smooth(self, side, box, t):
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        sz = box[2] - box[0]
        old = self._state.get(side)
        if old is not None and t - old[3] <= self.hold_s * 3:
            ocx, ocy, osz, _ = old
            sz = min(max(sz, osz * (1 - self.max_grow)), osz * (1 + self.max_grow))
            cx, cy = ocx + self.alpha * (cx - ocx), ocy + self.alpha * (cy - ocy)
            sz = osz + self.alpha * (sz - osz)
        return cx, cy, sz

    def refine(self, frame, kpts, scores, t):
        """kpts (K,2), scores (K,) of ONE person (133-point layout). Returns new copies."""
        H, W = frame.shape[:2]
        k, s = np.array(kpts, dtype=float), np.array(scores, dtype=float)
        for side, base in COCO133_HAND_BASE.items():
            box = hand_box(k[base:base + 21], s[base:base + 21])
            old = self._state.get(side)
            if box is not None:
                cx, cy, sz = self._smooth(side, box, t)
            elif old is not None and t - old[3] <= self.hold_s:
                cx, cy, sz = old[:3]
            else:
                self.last[side] = "none"
                continue
            h = sz / 2.0
            x1, y1 = max(0.0, cx - h), max(0.0, cy - h)
            x2, y2 = min(float(W), cx + h), min(float(H), cy + h)
            if x2 - x1 < 8 or y2 - y1 < 8:
                self.last[side] = "none"
                continue
            try:
                hk, hs = self.model(frame, [x1, y1, x2, y2])
            except Exception as e:                        # a model failure must not stop the instrument
                print("[hand-refine] model error:", e)
                self.last[side] = "none"
                continue
            hk = np.asarray(hk, dtype=float).reshape(-1, 2)
            hs = np.asarray(hs, dtype=float).reshape(-1)
            if len(hk) != 21 or float(hs.mean()) < self.accept:
                self.last[side] = "fallback"
                if box is not None:
                    self._state[side] = (cx, cy, sz, t)
                continue
            k[base:base + 21], s[base:base + 21] = hk, hs
            nb = hand_box(hk, hs)                         # track from the better points next frame
            if nb is not None:
                cx, cy = (nb[0] + nb[2]) / 2.0, (nb[1] + nb[3]) / 2.0
                sz = nb[2] - nb[0]
            self._state[side] = (cx, cy, sz, t)
            self.last[side] = "refined"
        return k, s


# --------------------------------------------------------------------------
# keypoints -> observations
# --------------------------------------------------------------------------
def _pt(kpts, scores, i, W, H, mirror):
    x = (W - kpts[i][0]) if mirror else kpts[i][0]
    return (float(x) / H, float(kpts[i][1]) / H, float(scores[i]))


def _foot_candidates(kpts, scores, layout, W, H, mirror):
    table = HALPE26_FEET if layout == "halpe26" else COCO133_FEET
    out = []
    for side, idx in table.items():
        pts = {name: _pt(kpts, scores, i, W, H, mirror) for name, i in idx.items()}
        out.append((side, pts))
    return out


def _hand_candidates(kpts, scores, W, H, mirror, fingers_needed):
    out = []
    for side, base in COCO133_HAND_BASE.items():
        pts = {"wrist": _pt(kpts, scores, base, W, H, mirror),
               "palm": _pt(kpts, scores, base + FINGER_MCP["middle"] + 0, W, H, mirror)}
        for g in fingers_needed:
            pts["tip_" + g] = _pt(kpts, scores, base + FINGER_TIP[g], W, H, mirror)
            pts["mcp_" + g] = _pt(kpts, scores, base + FINGER_MCP[g], W, H, mirror)
        out.append((side, pts))
    return out


def _cand_x(pts, names):
    for n in names:
        if n in pts:
            return pts[n][0]
    return 0.5


def associate(cands, eids, prev_x, force_order, anchor_names, min_score):
    """Decide which candidate belongs to which effector.

    The model's own left/right labels can flip when it only sees legs, so after calibration we track
    identity by position: pick the assignment closest to where each effector was last seen. During
    calibration (no history yet) the left-most candidate on screen is the left effector.
    """
    cands = [c for c in cands if any(p[2] >= min_score for p in c[1].values())]
    if not cands or not eids:
        return {}
    xs = [_cand_x(c[1], anchor_names) for c in cands]
    sides = {e: e[0] for e in eids}                  # "L_FOOT" -> "L"

    if len(eids) >= 2 and len(cands) >= 2:
        order = sorted(range(len(cands)), key=lambda i: xs[i])
        left_c, right_c = order[0], order[-1]
        el = next((e for e in eids if sides[e] == "L"), eids[0])
        er = next((e for e in eids if sides[e] == "R"), eids[-1])
        if not force_order and el in prev_x and er in prev_x:
            straight = abs(xs[left_c] - prev_x[el]) + abs(xs[right_c] - prev_x[er])
            swapped = abs(xs[right_c] - prev_x[el]) + abs(xs[left_c] - prev_x[er])
            if swapped < straight:
                left_c, right_c = right_c, left_c
        return {el: cands[left_c][1], er: cands[right_c][1]}

    if len(eids) >= 2:                                # two effectors, one candidate
        x = xs[0]
        best = min(eids, key=lambda e: abs(x - prev_x[e]) if e in prev_x else
                   (0 if (cands[0][0] == "left") == (sides[e] == "L") else 1))
        return {best: cands[0][1]}

    e = eids[0]                                       # one effector, one or two candidates
    if e in prev_x and not force_order:
        i = min(range(len(cands)), key=lambda j: abs(xs[j] - prev_x[e]))
    else:
        want = "left" if sides[e] == "L" else "right"
        i = next((j for j, c in enumerate(cands) if c[0] == want), 0)
    return {e: cands[i][1]}


def select_person(scores, idxs, kpts=None, ref_x=None, W=1, H=1, mirror=True, min_mean=0.25):
    """Index of the player among the detected people.

    With no history (calibration) it is the person whose relevant keypoints are most confident. Once the
    player's position is known (ref_x = their effectors' last x positions, frame-height units), it is the
    person whose keypoints are nearest to that - so a helper or bystander walking in cannot take over.
    """
    def mean_score(p):
        return float(sum(scores[p][i] for i in idxs)) / max(len(idxs), 1)

    ranked = sorted(range(len(scores)), key=mean_score, reverse=True)
    if not ranked:
        return 0
    if kpts is None or not ref_x or not idxs:
        return ranked[0]
    cands = [p for p in ranked if mean_score(p) >= min_mean] or ranked[:1]
    best, best_d = cands[0], float("inf")
    for p in cands:
        xs = [(W - kpts[p][i][0]) if mirror else kpts[p][i][0] for i in idxs if scores[p][i] >= 0.3]
        if not xs:
            continue
        cx = float(sum(xs)) / len(xs) / H
        d = min(abs(cx - rx) for rx in ref_x.values())
        if d < best_d:
            best, best_d = p, d
    return best


def extract_observations(kpts, scores, layout, W, H, profile, prev_x, calibrating,
                         min_score=0.35, mirror=True, refine=None):
    """kpts: (N, K, 2) pixels, scores: (N, K). Returns {effector_id: {point_name: (x, y, score)}}.

    Coordinates come back in frame-height units, mirrored so that the player's left is on screen-left.
    refine: optional callable (k, s) -> (k, s) applied to the chosen player's keypoints (hand refinement).
    """
    if kpts is None or len(kpts) == 0:
        return {}
    feet_ids = [e.id for e in profile.effectors if e.kind == "foot"]
    hand_ids = [e.id for e in profile.effectors if e.kind == "hand"]
    table = HALPE26_FEET if layout == "halpe26" else COCO133_FEET
    idxs = [i for side in table.values() for i in side.values()] if feet_ids else []
    if hand_ids and layout == "coco133":
        idxs += [b + FINGER_MCP["middle"] for b in COCO133_HAND_BASE.values()]
    p = select_person(scores, idxs or [0], kpts, None if calibrating else prev_x, W, H, mirror)
    k, s = kpts[p], scores[p]
    if refine is not None and hand_ids and layout == "coco133":
        k, s = refine(k, s)                          # hand-only model on full-resolution crops

    obs = {}
    if feet_ids:
        c = _foot_candidates(k, s, layout, W, H, mirror)
        obs.update(associate(c, feet_ids, prev_x, calibrating, ("ankle", "toe"), min_score))
    if hand_ids and layout == "coco133":
        needed = sorted({g for e in profile.effectors if e.kind == "hand" for g in e.fingers})
        c = _hand_candidates(k, s, W, H, mirror, needed)
        obs.update(associate(c, hand_ids, prev_x, calibrating, ("wrist", "palm"), min_score))
    return obs


# --------------------------------------------------------------------------
# Camera: always hand back the NEWEST frame (no queue of stale frames)
# --------------------------------------------------------------------------
class LatestFrameCamera:
    def __init__(self, index=0, width=1280, height=720):
        import cv2
        self.cv2 = cv2
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # much higher fps than raw YUY2
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cond = threading.Condition()
        self._frame, self._t, self._seq, self._seen = None, 0.0, 0, 0
        self._run = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _loop(self):
        while self._run:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cond:
                self._frame, self._t = frame, time.monotonic()
                self._seq += 1
                self._cond.notify_all()

    def read(self, timeout=1.0):
        """(frame, capture_time) of the newest unseen frame, or (None, 0) on timeout."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq != self._seen, timeout):
                return None, 0.0
            self._seen = self._seq
            return self._frame, self._t

    def release(self):
        self._run = False
        self._th.join(timeout=1.0)
        self.cap.release()
