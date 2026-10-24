"""Input profiles: which body parts a player uses.

A profile is what the frontend writes (JSON) and what the app reads. It answers "how does this person
want to play?" - e.g. one hand with three fingers plus one foot. Everything downstream (key layout,
calibration, which model runs) is derived from it, so a new combination needs no code change.

JSON shape (every field except "effectors" is optional):

{
  "name": "right hand (3 fingers) + left foot",
  "effectors": [
    {"id": "L_FOOT", "zones": 4},
    {"id": "R_HAND", "fingers": ["index", "middle", "ring"]}
  ],
  "mode": "FREE",              # FREE | GUIDED | CHORD | SPLIT  (starting mode)
  "chords": "L_FOOT",          # SPLIT mode: which effector plays chords (default: first foot)
  "sensitivity": 1.0,          # >1 = smaller movements count, <1 = needs bigger movements
  "zone_width": 0.10,          # width of one foot/hand zone, as a fraction of the frame width
  "sustain": false,            # false: a tap plays a fixed-length note; true: note lasts while held
  "tap_len": 0.25,             # seconds, used when sustain is false
  "meta": {}                   # free space for the frontend; ignored here
}

Effector ids: L_FOOT, R_FOOT, L_HAND, R_HAND.
  - foot: tap detection, "zones" = how many notes it can pick between (default 4).
  - hand with "fingers": each listed finger is one key (flexing the finger presses it).
  - hand without "fingers": the whole hand is one tap pad that can pick between "zones" (default 3).
"""

import json
import math
import os

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
ALL_IDS = ("L_FOOT", "R_FOOT", "L_HAND", "R_HAND")
MODES = ("FREE", "GUIDED", "CHORD", "SPLIT")
_TOP_KEYS = {"name", "effectors", "mode", "chords", "sensitivity", "zone_width",
             "sustain", "tap_len", "meta", "version"}
MAX_PROFILE_BYTES = 64 * 1024     # a profile is a few hundred bytes; refuse anything absurd
DEFAULT_FOOT_ZONES = 4
DEFAULT_HAND_ZONES = 3


def _number(name, v, lo, hi):
    """Strict numeric check: JSON gives us whatever a frontend sent (strings, null, true, NaN...)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError("%s must be a number" % name)
    if not lo <= v <= hi:
        raise ValueError("%s must be between %g and %g" % (name, lo, hi))
    return float(v)


class EffectorSpec:
    def __init__(self, id, fingers=(), zones=0):
        if not isinstance(id, str) or id not in ALL_IDS:
            raise ValueError("unknown effector id %r (use one of %s)" % (id, ", ".join(ALL_IDS)))
        if fingers is not None and not isinstance(fingers, (list, tuple)):
            raise ValueError("%s: fingers must be a list" % id)
        fingers = tuple(fingers or ())
        for f in fingers:
            if not isinstance(f, str) or f not in FINGERS:
                raise ValueError("%s: unknown finger %r (use %s)" % (id, f, ", ".join(FINGERS)))
        if len(set(fingers)) != len(fingers):
            raise ValueError("%s: duplicate finger" % id)
        if fingers and id.endswith("FOOT"):
            raise ValueError("%s: feet cannot have fingers" % id)
        if zones is None:
            zones = 0
        if isinstance(zones, bool) or not isinstance(zones, int) or not 0 <= zones <= 8:
            raise ValueError("%s: zones must be a whole number 0-8" % id)
        self.id, self.fingers, self.zones = id, fingers, zones

    @property
    def kind(self):
        return "foot" if self.id.endswith("FOOT") else "hand"

    @property
    def side(self):
        return self.id[0]

    @property
    def uses_fingers(self):
        return self.kind == "hand" and len(self.fingers) > 0

    @property
    def n_zones(self):
        return self.zones or (DEFAULT_FOOT_ZONES if self.kind == "foot" else DEFAULT_HAND_ZONES)

    def slot_ids(self):
        """Slots in ascending-pitch order. Fingers follow piano convention:
        right hand thumb->pinky goes up in pitch, left hand pinky->thumb goes up in pitch."""
        if self.uses_fingers:
            order = FINGERS if self.side == "R" else tuple(reversed(FINGERS))
            return [f for f in order if f in self.fingers]
        return list(range(self.n_zones))

    def to_dict(self):
        d = {"id": self.id}
        if self.fingers:
            d["fingers"] = list(self.fingers)
        if self.zones:
            d["zones"] = self.zones
        return d


def order_effectors(effectors):
    """Left-to-right body order, lowest notes first: left side before right, foot before hand."""
    return sorted(effectors, key=lambda e: (e.side == "R", e.kind == "hand"))


class InputProfile:
    def __init__(self, name, effectors, mode="FREE", chords=None, sensitivity=1.0,
                 zone_width=0.10, sustain=False, tap_len=0.25, meta=None):
        if not effectors:
            raise ValueError("a profile needs at least one effector")
        ids = [e.id for e in effectors]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate effector in profile")
        if not isinstance(mode, str) or mode not in MODES:
            raise ValueError("mode must be one of %s" % ", ".join(MODES))
        if chords is not None and (not isinstance(chords, str) or chords not in ids):
            raise ValueError("chords effector %r is not in this profile" % (chords,))
        sensitivity = _number("sensitivity", sensitivity, 0.2, 5.0)
        zone_width = _number("zone_width (fraction of frame width)", zone_width, 0.03, 0.3)
        tap_len = _number("tap_len (seconds)", tap_len, 0.05, 3.0)
        if not isinstance(sustain, bool):
            raise ValueError("sustain must be true or false")
        if meta is not None and not isinstance(meta, dict):
            raise ValueError("meta must be an object")
        name = "".join(c for c in str(name) if c.isprintable())[:60].strip() or "custom"
        self.name = name
        self.effectors = order_effectors(effectors)
        self.mode, self.chords = mode, chords
        self.sensitivity, self.zone_width = float(sensitivity), float(zone_width)
        self.sustain, self.tap_len = bool(sustain), float(tap_len)
        self.meta = meta or {}

    @property
    def needs_hands(self):
        return any(e.kind == "hand" for e in self.effectors)

    def effector(self, eid):
        for e in self.effectors:
            if e.id == eid:
                return e
        return None

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict):
            raise ValueError("profile must be a JSON object")
        unknown = set(d) - _TOP_KEYS
        if unknown:
            raise ValueError("unknown profile field(s): %s" % ", ".join(sorted(unknown)))
        raw_effs = d.get("effectors", [])
        if not isinstance(raw_effs, list):
            raise ValueError("effectors must be a list")
        effs = []
        for e in raw_effs:
            if not isinstance(e, dict) or "id" not in e:
                raise ValueError("each effector needs an 'id'")
            extra = set(e) - {"id", "fingers", "zones"}
            if extra:
                raise ValueError("unknown effector field(s): %s" % ", ".join(sorted(extra)))
            effs.append(EffectorSpec(e["id"], e.get("fingers", ()), e.get("zones", 0)))
        return cls(d.get("name", "custom"), effs, d.get("mode", "FREE"), d.get("chords"),
                   d.get("sensitivity", 1.0), d.get("zone_width", 0.10), d.get("sustain", False),
                   d.get("tap_len", 0.25), d.get("meta"))

    def to_dict(self):
        d = {"name": self.name, "effectors": [e.to_dict() for e in self.effectors],
             "mode": self.mode, "sensitivity": self.sensitivity, "zone_width": self.zone_width,
             "sustain": self.sustain, "tap_len": self.tap_len}
        if self.chords:
            d["chords"] = self.chords
        if self.meta:
            d["meta"] = self.meta
        return d


# --------------------------------------------------------------------------
# Presets (also written to profiles/*.json so the frontend can start from them)
# --------------------------------------------------------------------------
def _p(name, effs, **kw):
    return InputProfile(name, effs, **kw)


PRESETS = {
    "two_feet": _p("two feet", [EffectorSpec("L_FOOT", zones=4), EffectorSpec("R_FOOT", zones=4)]),
    "one_foot": _p("one foot", [EffectorSpec("R_FOOT", zones=4)]),
    "hand_foot": _p("right hand (3 fingers) + left foot",
                    [EffectorSpec("L_FOOT", zones=4),
                     EffectorSpec("R_HAND", fingers=("index", "middle", "ring"))],
                    chords="L_FOOT"),
    "hand_foot_mirror": _p("left hand (3 fingers) + right foot",
                           [EffectorSpec("R_FOOT", zones=4),
                            EffectorSpec("L_HAND", fingers=("index", "middle", "ring"))],
                           chords="R_FOOT"),
    "one_hand_fingers": _p("right hand, all fingers",
                           [EffectorSpec("R_HAND", fingers=FINGERS)]),
    "two_hands": _p("two hands, all fingers",
                    [EffectorSpec("L_HAND", fingers=FINGERS), EffectorSpec("R_HAND", fingers=FINGERS)]),
}
PRESET_ORDER = list(PRESETS)


def load_profile(spec):
    """spec = preset name or path to a JSON file."""
    if spec in PRESETS:
        return PRESETS[spec]
    if not os.path.exists(spec):
        raise ValueError("no preset or file called %r (presets: %s)" % (spec, ", ".join(PRESET_ORDER)))
    if os.path.getsize(spec) > MAX_PROFILE_BYTES:
        raise ValueError("profile file is too large (limit %d bytes)" % MAX_PROFILE_BYTES)
    with open(spec, "r", encoding="utf-8") as f:
        return InputProfile.from_dict(json.load(f))


class ProfileFile:
    """Watches a profile JSON so a frontend can change the profile while the app is running."""

    def __init__(self, path):
        self.path = path
        self._sig = self._signature()

    def poll(self):
        """Returns a new InputProfile if the file changed and is valid, else None."""
        sig = self._signature()
        if sig is None or sig == self._sig:
            return None
        self._sig = sig
        try:
            return load_profile(self.path)
        except Exception as e:               # half-written file or bad content: keep playing the old profile
            print("[profile] ignored change:", e)
            return None

    def _signature(self):
        """(mtime, size): size catches rewrites that land inside the filesystem's timestamp resolution."""
        try:
            st = os.stat(self.path) if self.path else None
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size) if st else None
