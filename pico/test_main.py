"""Host-side tests for main.py (NOT copied to the Pico).  Run:  cd pico && python -m pytest -q

main.py is loaded against a fake `machine`/`uselect`/clock, and every I2C write is recorded exactly as the
PCA9685 would receive it. This checks the motion maths and protocol - it does NOT prove anything about real
servo behaviour (stall current, noise, how the key feels); that needs the hardware.
"""

import importlib.util
import os
import sys
import time
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CLOCK = [0]


class FakeI2C:
    def __init__(self, *a, **k):
        self.writes = []               # (clock_ms, addr, reg, bytes)

    def writeto_mem(self, addr, reg, data):
        self.writes.append((CLOCK[0], addr, reg, bytes(data)))

    def readfrom_mem(self, addr, reg, n):
        return bytes([0] * n)


class FakePin:
    OUT = 1

    def __init__(self, *a, **k):
        self.is_on = False

    def on(self):
        self.is_on = True

    def off(self):
        self.is_on = False


@pytest.fixture()
def pico():
    fm = types.ModuleType("machine")
    fm.Pin, fm.I2C = FakePin, FakeI2C
    sys.modules["machine"] = fm
    sys.modules["uselect"] = types.ModuleType("uselect")
    CLOCK[0] = 1000
    time.ticks_ms = lambda: CLOCK[0]
    time.ticks_diff = lambda a, b: ((a - b + 2 ** 29) % 2 ** 30) - 2 ** 29
    time.ticks_add = lambda a, b: (a + b) % 2 ** 30
    time.sleep_ms = lambda n: None
    time.sleep_us = lambda n: None
    spec = importlib.util.spec_from_file_location("pico_main_under_test", os.path.join(HERE, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.REST_DEG[:] = [90] * 8
    mod.PRESS_DEG[:] = [120] * 8
    eng = mod.build_engine()
    for k in eng.keys:
        k.jump_to(k.rest)
    eng.i2c = eng.keys[0].pca.i2c
    eng.i2c.writes.clear()
    eng.mod = mod
    return eng


def to_deg(mod, pca, ticks):
    us = ticks * pca.period_us / 4096.0
    return (us - mod.SERVO_US_MIN) * 180.0 / (mod.SERVO_US_MAX - mod.SERVO_US_MIN)


def channel_trace(eng, ch):
    """[(time, angle)] of every pulse-width write to a channel, decoded from the raw register bytes."""
    out = []
    pca = eng.keys[0].pca
    for t, addr, reg, data in eng.i2c.writes:
        if reg == 0x06 + 4 * ch:
            on = data[0] | (data[1] << 8)
            off = data[2] | (data[3] << 8)
            out.append((t, to_deg(eng.mod, pca, (off - on) & 0xFFF), on))
    return out


def run(eng, ms):
    end = CLOCK[0] + ms
    while CLOCK[0] < end:
        CLOCK[0] += eng.mod.LOOP_MS
        eng.update(CLOCK[0])


def test_oscillator_and_period(pico):
    pca = pico.keys[0].pca
    assert 19900 < pca.period_us < 20100            # ~50 Hz
    assert abs(pca.ticks_for(1500) - 1500 * 4096 / pca.period_us) <= 1


def test_press_is_a_smooth_monotonic_ramp_with_no_overshoot(pico):
    pico.handle("P0", CLOCK[0])
    run(pico, 300)
    tr = channel_trace(pico, 0)
    angles = [a for _, a, _ in tr]
    assert len(angles) > 10
    assert all(b >= a - 1e-9 for a, b in zip(angles, angles[1:]))        # never goes backwards
    assert max(angles) <= 120.0 + 0.5 and angles[-1] == pytest.approx(120.0, abs=0.5)
    # duration of the move ~ 30 deg / ATTACK_DPS
    t_first, t_last = tr[0][0], tr[-1][0]
    assert abs((t_last - t_first) - 30.0 / pico.mod.ATTACK_DPS * 1000) < 25


def test_peak_speed_stays_inside_what_an_sg90_can_follow(pico):
    pico.handle("P0", CLOCK[0])
    run(pico, 200)
    tr = channel_trace(pico, 0)
    # speed over 24 ms windows (single 0.44-degree steps are too coarse to measure)
    worst = 0.0
    for i, (t, a, _) in enumerate(tr):
        for t2, a2, _ in tr[i + 1:]:
            if t2 - t >= 24:
                worst = max(worst, (a2 - a) / ((t2 - t) / 1000.0))
                break
    assert worst < 600                               # SG90 spec: 0.1 s / 60 deg = 600 deg/s, no load
    assert worst > 200                               # and it is genuinely fast, not sluggish


def test_release_is_gentler_than_attack(pico):
    pico.handle("P0", CLOCK[0])
    run(pico, 300)
    pico.i2c.writes.clear()
    pico.handle("R0", CLOCK[0])
    run(pico, 400)
    tr = channel_trace(pico, 0)
    dur = tr[-1][0] - tr[0][0]
    assert dur > 30.0 / pico.mod.ATTACK_DPS * 1000 + 20
    assert tr[-1][1] == pytest.approx(90.0, abs=0.5)


def test_reversing_mid_press_does_not_jump(pico):
    pico.handle("P0", CLOCK[0])
    run(pico, 30)
    pico.handle("R0", CLOCK[0])
    run(pico, 400)
    angles = [a for _, a, _ in channel_trace(pico, 0)]
    assert max(abs(b - a) for a, b in zip(angles, angles[1:])) < 4.0       # continuous, no snap
    assert angles[-1] == pytest.approx(90.0, abs=0.5)


def test_channels_start_their_pulses_at_different_counts(pico):
    for k in range(8):
        pico.handle("P%d" % k, CLOCK[0])
    run(pico, 200)
    ons = {ch: channel_trace(pico, ch)[0][2] for ch in range(8)}
    assert len(set(ons.values())) == 8
    assert ons[3] == 3 * 256


def test_pulse_that_wraps_past_the_period_is_written_correctly(pico):
    pca = pico.keys[0].pca
    pico.i2c.writes.clear()
    pca.set_ticks(15, 400)
    _, _, reg, data = pico.i2c.writes[-1]
    on, off = data[0] | (data[1] << 8), data[2] | (data[3] << 8)
    assert on == 3840 and off == (3840 + 400) & 0xFFF and off < on          # wraps; width = 400
    assert (off - on) & 0xFFF == 400


def test_a_chord_starts_its_servos_staggered_not_all_at_once(pico):
    for k in (0, 2, 4):
        pico.handle("P%d" % k, CLOCK[0])
    run(pico, 100)
    first = {ch: channel_trace(pico, ch)[0][0] for ch in (0, 2, 4)}
    starts = sorted(first.values())
    assert starts[1] - starts[0] >= pico.mod.STAGGER_MS - pico.mod.LOOP_MS
    assert starts[2] - starts[1] >= pico.mod.STAGGER_MS - pico.mod.LOOP_MS
    assert starts[2] - starts[0] <= 2 * pico.mod.STAGGER_MS + 2 * pico.mod.LOOP_MS    # but still prompt


def test_tap_does_not_block_and_releases_itself(pico):
    pico.handle("T3", CLOCK[0])
    run(pico, 20)
    pico.handle("P5", CLOCK[0])                      # a command 20 ms later is handled immediately
    run(pico, 100)
    assert channel_trace(pico, 5)[0][0] < CLOCK[0]
    run(pico, 500)
    assert channel_trace(pico, 3)[-1][1] == pytest.approx(90.0, abs=0.5)    # tap released on its own
    assert pico.keys[3].held_since == 0 and pico.keys[5].held_since != 0


def test_safety_release_after_max_hold(pico):
    pico.handle("P2", CLOCK[0])
    run(pico, pico.mod.MAX_HOLD_MS + 600)
    assert pico.keys[2].held_since == 0
    assert channel_trace(pico, 2)[-1][1] == pytest.approx(90.0, abs=0.5)


def test_release_all(pico):
    for k in range(8):
        pico.handle("P%d" % k, CLOCK[0])
    run(pico, 150)
    pico.handle("A", CLOCK[0])
    run(pico, 500)
    assert all(channel_trace(pico, ch)[-1][1] == pytest.approx(90.0, abs=0.5) for ch in range(8))


@pytest.mark.parametrize("line", ["", "   ", "P99", "P-1", "Pabc", "R", "C", "C1 2", "Cx y z", "Z7", "T", "\x00\x01"])
def test_garbage_commands_never_crash(pico, line):
    pico.handle(line, CLOCK[0])
    run(pico, 20)


def test_lowercase_commands_work(pico):
    pico.handle("p1", CLOCK[0])
    run(pico, 150)
    assert channel_trace(pico, 1)[-1][1] == pytest.approx(120.0, abs=0.5)


def test_live_calibration_changes_the_target(pico):
    pico.handle("C0 85 105", CLOCK[0])
    run(pico, 300)
    assert channel_trace(pico, 0)[-1][1] == pytest.approx(85.0, abs=0.5)
    pico.i2c.writes.clear()
    pico.handle("P0", CLOCK[0])
    run(pico, 300)
    assert channel_trace(pico, 0)[-1][1] == pytest.approx(105.0, abs=0.5)


def test_idle_makes_no_i2c_traffic(pico):
    run(pico, 500)
    assert pico.i2c.writes == []


def test_led_follows_held_keys(pico):
    pico.handle("P0", CLOCK[0])
    assert pico.led.is_on
    pico.handle("R0", CLOCK[0])
    assert not pico.led.is_on


def test_key_table_length_mismatch_is_reported(pico):
    mod = pico.mod
    saved = list(mod.KEY_MAP)
    mod.KEY_MAP.append((0, 8))
    try:
        with pytest.raises(ValueError):
            mod.build_engine()
    finally:
        mod.KEY_MAP[:] = saved


def test_overlong_line_is_dropped_entirely(pico):
    r = pico.mod.LineReader()
    out = [r.feed(c) for c in "X" * 60 + "P3\n"]                # a 62-char line ending in a valid command
    assert [o for o in out if o is not None] == []
    out = [r.feed(c) for c in "P1\n"]                           # and the next line works normally
    assert [o for o in out if o is not None] == ["P1"]


def test_line_reader_handles_crlf_and_blank_lines(pico):
    r = pico.mod.LineReader()
    got = [x for x in (r.feed(c) for c in "P1\r\nR1\r\n") if x is not None]
    assert got == ["P1", "", "R1", ""]


@pytest.mark.parametrize("line", ["C0 -5 100", "C0 90 181", "C0 400 90"])
def test_calibration_angles_outside_servo_range_are_refused(pico, line):
    before = (list(pico.mod.REST_DEG), list(pico.mod.PRESS_DEG))
    pico.handle(line, CLOCK[0])
    run(pico, 50)
    assert (pico.mod.REST_DEG, pico.mod.PRESS_DEG) == before
