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
    overlay = frame.copy()

    # zone bands for foot / hand-pad lanes
    for eid, slot, x0, x1 in rig.zone_rects():
        held = (eid, slot) in player.active
        color = (60, 200, 90) if held else ((70, 70, 70) if slot % 2 else (110, 110, 110))
        cv2.rectangle(overlay, (int(x0 * H), int(H * 0.78)), (int(x1 * H), H), color, -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for eid, slot, x0, x1 in rig.zone_rects():
        label = player.label_for((eid, slot))
        cx = int((x0 + x1) / 2 * H)
        cv2.rectangle(frame, (int(x0 * H), int(H * 0.78)), (int(x1 * H), H), (200, 200, 200), 1)
        cv2.putText(frame, label, (cx - 14, H - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

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
    lines += list(hud_extra)
    lines += rig.warnings[:2]
    lines.append("hold a body part over a top button  |  keys: C M N P R Q")
    for i, t in enumerate(lines):
        y0 = int(H * MENU_H) + 28 + 26 * i
        cv2.putText(frame, t, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, t, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
