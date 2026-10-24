"""On-screen display and the hands-free dwell menu.

Every control that used to need a keyboard key (calibrate, mode, song, release, switch input) is now a
button in a strip along the top of the picture. Hold any tracked body part over a button for ~1.2 s to
press it. Notes are never played from inside the strip. The keyboard still works for a helper.
"""

MENU_H = 0.09                                    # menu strip height, as a fraction of frame height
BUTTONS = ["MODE", "SONG", "PROFILE", "RECAL", "STOP"]


class DwellMenu:
    def __init__(self, dwell_s=1.2, cooldown_s=1.5):
        self.dwell_s, self.cooldown_s = dwell_s, cooldown_s
        self.entered = {}            # button -> time the pointer arrived
        self.cool_until = 0.0
        self.progress = {}           # button -> 0..1 for drawing

    @staticmethod
    def rects(W, H):
        w = W / float(len(BUTTONS))
        return {b: (int(i * w), 0, int((i + 1) * w), int(MENU_H * H)) for i, b in enumerate(BUTTONS)}

    def update(self, t, pointers_px, W, H):
        """pointers_px: list of (x, y) in pixels. Returns a button name when one is activated."""
        self.progress = {}
        if t < self.cool_until:
            self.entered = {}
            return None
        hit = {}
        for b, (x0, y0, x1, y1) in self.rects(W, H).items():
            if any(x0 <= px < x1 and y0 <= py < y1 for px, py in pointers_px):
                hit[b] = True
        for b in list(self.entered):
            if b not in hit:
                del self.entered[b]
        for b in hit:
            self.entered.setdefault(b, t)
            p = (t - self.entered[b]) / self.dwell_s
            self.progress[b] = min(p, 1.0)
            if p >= 1.0:
                self.cool_until = t + self.cooldown_s
                self.entered = {}
                return b
        return None


def draw(frame, cv2, rig, player, link, menu, fps, lat_ms, hud_extra=()):
    """Draws onto an already-mirrored frame. Coordinates from the rig are frame-height units."""
    H, W = frame.shape[:2]

    # zone bands for foot / hand-pad lanes (placeholders until calibration has placed them, so hide them)
    _draw_zones(frame, cv2, rig, player, H, W)

    # finger chips (one per finger, bottom-right)
    y = int(H * 0.70)
    for eid, e in rig.effs.items():
        if not e.spec.uses_fingers:
            continue
        x = W - 12
        for slot in reversed(e.spec.slot_ids()):
            held = (eid, slot) in player.active
            ok = (not rig.calibrating) and e.ok.get(slot, False)
            color = (60, 200, 90) if held else ((90, 90, 90) if ok or rig.calibrating else (40, 40, 160))
            x0 = x - 74
            cv2.rectangle(frame, (x0, y), (x, y + 52), color, -1)
            cv2.rectangle(frame, (x0, y), (x, y + 52), (220, 220, 220), 1)
            cv2.putText(frame, "%s %s" % (eid[0], slot[:3]), (x0 + 4, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, player.label_for((eid, slot)), (x0 + 8, y + 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            x = x0 - 6
        y -= 62

    # tracked points
    for eid, e in rig.effs.items():
        a = e.anchor
        if a is None:
            continue
        held = any(k[0] == eid for k in player.active)
        c = (60, 220, 90) if held else (255, 255, 255)
        p = (int(a[0] * H), int(a[1] * H))
        cv2.circle(frame, p, 12, c, -1)
        cv2.putText(frame, eid[0], (p[0] - 6, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # menu strip
    for b, (x0, y0, x1, y1) in menu.rects(W, H).items():
        cv2.rectangle(frame, (x0, y0), (x1, y1), (50, 50, 50), -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (180, 180, 180), 1)
        prog = menu.progress.get(b, 0.0)
        if prog > 0:
            cv2.rectangle(frame, (x0, y1 - 8), (x0 + int((x1 - x0) * prog), y1), (0, 200, 255), -1)
        cv2.putText(frame, b, (x0 + 10, y1 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # status text
    pname = player.profile.name.encode("ascii", "replace").decode()      # OpenCV's font is ASCII-only
    lines = ["%s   Mode: %s   %4.1f fps   %3.0f ms" % (pname, player.mode, fps, lat_ms)]
    if link.error:
        lines.append("PICO LINK PROBLEM: " + link.error)
    if player.mode == "GUIDED":
        i, n = player.progress()
        from .player import SONGS
        lines.append("Song: %s (%d/%d)   Next: %s" % (SONGS[player.song][0], i, n, player.next_note_name()))
    if rig.calibrating:
        lines.append(rig.status())
        if rig.setup_hint:
            lines.append("SET-UP: " + rig.setup_hint)
    lines += list(hud_extra)
    lines += rig.warnings[:2]
    lines.append("hold a body part over a top button  |  keys: C M N P R Q")
    for i, t in enumerate(lines):
        y0 = int(H * MENU_H) + 28 + 26 * i
        cv2.putText(frame, t, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, t, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


LANE_COLORS = {"L": (200, 140, 40), "R": (40, 150, 230)}      # BGR: left = blue, right = orange
LANE_NAMES = {"L_FOOT": "LEFT FOOT", "R_FOOT": "RIGHT FOOT", "L_HAND": "LEFT HAND", "R_HAND": "RIGHT HAND"}


def _text(frame, cv2, s, org, scale, color=(255, 255, 255), thick=2, center=False):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(s, font, scale, thick)
    x, y = org
    if center:
        x -= tw // 2
        y += th // 2
    cv2.putText(frame, s, (x, y), font, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(frame, s, (x, y), font, scale, color, thick, cv2.LINE_AA)


def _draw_zones(frame, cv2, rig, player, H, W):
    """One clear piano-mat row along the floor: every zone shows its note in big letters, tinted by which
    foot/hand plays it. The zone a foot is over right now gets a white frame; a zone being played lights up
    green. Overlapping lanes (only with profile.meta["lanes"] = "stacked") are drawn in separate rows."""
    if rig.calibrating:
        return
    zones = rig.zone_rects()
    if not zones:
        return
    rows = rig.lane_rows()
    k = H / 720.0
    over = {}                                            # effector -> zone its anchor is above now
    for eid, e in rig.effs.items():
        a = e.anchor
        if a is not None and hasattr(e, "zones") and getattr(e, "calibrated", False):
            over[eid] = e.zones.zone_for(a[0])
    px = lambda v: int(round(v * H))

    overlay = frame.copy()
    for eid, slot, x0, x1 in zones:
        ry0, ry1 = rows[eid]
        held = (eid, slot) in player.active
        base = LANE_COLORS.get(eid[0], (120, 120, 120))
        shade = 1.0 if slot % 2 == 0 else 0.72
        color = (60, 210, 90) if held else tuple(int(c * shade) for c in base)
        cv2.rectangle(overlay, (px(x0), px(ry0)), (px(x1), px(ry1)), color, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    lane_span = {}
    for eid, slot, x0, x1 in zones:
        ry0, ry1 = rows[eid]
        held = (eid, slot) in player.active
        p0, p1 = (px(x0), px(ry0)), (px(x1), px(ry1))
        cv2.rectangle(frame, p0, p1, (235, 235, 235), max(1, int(1.5 * k)))
        if over.get(eid) == slot and not held:
            cv2.rectangle(frame, (p0[0] + int(3 * k), p0[1] + int(3 * k)), (p1[0] - int(3 * k), p1[1] - int(3 * k)),
                          (255, 255, 255), max(2, int(5 * k)))
        label = str(player.label_for((eid, slot)))
        big = 1.9 if len(label) <= 2 else 1.25
        if len(rows) and len(set(rows.values())) > 1:
            big *= 0.7
        _text(frame, cv2, label, ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2 + int(8 * k)), big * k,
              (20, 20, 20) if held else (255, 255, 255), max(2, int(3 * k)), center=True)
        lo, hi = lane_span.get(eid, (p0[0], p1[0]))
        lane_span[eid] = (min(lo, p0[0]), max(hi, p1[0]))

    for eid, (lo, hi) in lane_span.items():
        ry0 = px(rows[eid][0])
        name = LANE_NAMES.get(eid, eid)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.7 * k, max(1, int(2 * k)))
        cx = (lo + hi) // 2
        y1 = ry0 - int(6 * k)
        cv2.rectangle(frame, (cx - tw // 2 - int(10 * k), y1 - th - int(14 * k)), (cx + tw // 2 + int(10 * k), y1),
                      LANE_COLORS.get(eid[0], (90, 90, 90)), -1)
        cv2.putText(frame, name, (cx - tw // 2, y1 - int(7 * k)), cv2.FONT_HERSHEY_DUPLEX, 0.7 * k,
                    (255, 255, 255), max(1, int(2 * k)), cv2.LINE_AA)
    if getattr(rig, "mat", False):                       # thick divider where one foot's keys end
        ids = [i for i in rig._lane_order if i in lane_span]
        for a, b in zip(ids, ids[1:]):
            x = (lane_span[a][1] + lane_span[b][0]) // 2
            cv2.line(frame, (x, px(rows[a][0]) - int(34 * k)), (x, H), (255, 255, 255), max(2, int(4 * k)))
