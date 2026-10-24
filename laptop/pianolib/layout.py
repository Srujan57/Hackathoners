"""Adaptive layout: turns (profile, mode) into "this slot plays that key / chord".

A *slot* is one thing a player can trigger: a foot zone, or a finger. Slots are ordered low->high
across the body (left before right, foot before hand), then keys are handed out so the layout is
always musical for however many slots the player has - 8 slots get 8 keys, 4 slots get a C-E-G-C
arpeggio, 3 get C-E-G, and so on. Nothing here depends on which body parts are used.

Keys are indices into the one octave: 0=C 1=D 2=E 3=F 4=G 5=A 6=B 7=high C.
"""

from .profile import order_effectors

NUM_KEYS = 8
NOTE_NAMES = ["C", "D", "E", "F", "G", "A", "B", "C"]
CHORD_NAMES = ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]   # triads on each scale degree of C major

# Which keys a player with N slots gets. 4 slots = chord tones only, 5 = pentatonic (can't sound
# wrong), 6+ add notes. More than 8 slots share keys (see _spread).
KEYSETS = {
    1: [0],
    2: [0, 4],
    3: [0, 2, 4],
    4: [0, 2, 4, 7],
    5: [0, 1, 2, 4, 5],
    6: [0, 1, 2, 4, 5, 7],
    7: [0, 1, 2, 3, 4, 5, 6],
    8: [0, 1, 2, 3, 4, 5, 6, 7],
}
# Which chords a player with N slots gets (scale degrees): I, V, IV, vi cover most pop songs.
CHORD_DEGREES = {
    1: [0],
    2: [0, 4],
    3: [0, 3, 4],
    4: [0, 3, 4, 5],
    5: [0, 1, 3, 4, 5],
    6: [0, 1, 2, 3, 4, 5],
    7: [0, 1, 2, 3, 4, 5, 6],
}


class Action:
    """What a slot does when triggered."""

    def __init__(self, keys=(), label="", is_next=False):
        self.keys = tuple(keys)
        self.label = label
        self.is_next = is_next      # GUIDED: plays whatever song note is next

    def __repr__(self):
        return "Action(%s %s)" % (self.label, list(self.keys))


def chord_keys(degree):
    """Triad on a scale degree, kept inside the one octave (same voicing the project always used)."""
    return [degree, (degree + 2) % 7, (degree + 4) % 7]


def _spread(n, table, top):
    """Pick n entries from table (dict by count). For n > top, spread top entries over n slots."""
    if n <= top:
        return list(table[n])
    base = table[top]
    return [base[min(len(base) - 1, int(i * len(base) / n))] for i in range(n)]


def note_actions(n):
    return [Action([k], NOTE_NAMES[k]) for k in _spread(n, KEYSETS, 8)]


def chord_actions(n):
    degrees = _spread(n, CHORD_DEGREES, 7)
    return [Action(chord_keys(d), CHORD_NAMES[d]) for d in degrees]


def all_slots(profile):
    return [(e.id, s) for e in order_effectors(profile.effectors) for s in e.slot_ids()]


def available_modes(profile):
    modes = ["FREE", "GUIDED", "CHORD"]
    if len(profile.effectors) >= 2:
        modes.append("SPLIT")
    return modes


def chord_effector_id(profile):
    """SPLIT mode: who plays chords. Explicit in the profile, otherwise the first foot, else the first."""
    if profile.chords:
        return profile.chords
    effs = order_effectors(profile.effectors)
    for e in effs:
        if e.kind == "foot":
            return e.id
    return effs[0].id


def build_slotmap(profile, mode):
    """Returns {(effector_id, slot_id): Action}."""
    slots = all_slots(profile)
    if mode == "GUIDED":
        return {s: Action((), "next", is_next=True) for s in slots}
    if mode == "CHORD":
        return dict(zip(slots, chord_actions(len(slots))))
    if mode == "SPLIT":
        cid = chord_effector_id(profile)
        chord_slots = [s for s in slots if s[0] == cid]
        melody_slots = [s for s in slots if s[0] != cid]
        if not melody_slots:           # only one effector: SPLIT makes no sense, fall back
            return dict(zip(slots, note_actions(len(slots))))
        m = dict(zip(chord_slots, chord_actions(len(chord_slots))))
        m.update(zip(melody_slots, note_actions(len(melody_slots))))
        return m
    return dict(zip(slots, note_actions(len(slots))))   # FREE
