"""Software piano so the instrument can be heard (and tested) without the servo hardware.

Synth.render() is plain numpy, so it is tested without a sound card. AudioOut streams it to the speakers
with the `sounddevice` package. It is a simple additive tone (a few decaying harmonics) - good enough to hear
which key fired and when, not a sampled grand piano.
"""

import math
import threading

import numpy as np

SAMPLE_RATE = 44100
# one octave, C4..C5, same key order as the servos: 0=C 1=D 2=E 3=F 4=G 5=A 6=B 7=high C
KEY_FREQS = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
HARMONICS = (1.0, 0.5, 0.28, 0.16, 0.09)


class Synth:
    ATTACK_S = 0.004          # fade-in, avoids a click
    DECAY_PER_S = 1.6         # how fast a held note fades (piano-like)
    RELEASE_PER_S = 28.0      # how fast it dies after the key is let go
    GAIN = 0.22
    SILENT = 1e-3          # -60 dB: inaudible, free the voice

    def __init__(self, sr=SAMPLE_RATE, freqs=None):
        self.sr = sr
        self.freqs = list(freqs or KEY_FREQS)
        self._v = {}                          # key -> [phase, age_samples, env, released]
        self._lock = threading.Lock()

    def note_on(self, key):
        if 0 <= key < len(self.freqs):
            with self._lock:
                self._v[key] = [0.0, 0, 1.0, False]

    def note_off(self, key):
        with self._lock:
            v = self._v.get(key)
            if v is not None:
                v[3] = True

    def all_off(self):
        with self._lock:
            for v in self._v.values():
                v[3] = True

    @property
    def active(self):
        with self._lock:
            return sorted(self._v)

    def render(self, n):
        """n samples of float32 mono audio in [-1, 1]."""
        out = np.zeros(n, dtype=np.float64)
        idx = np.arange(n)
        att_n = max(1, int(self.ATTACK_S * self.sr))
        with self._lock:
            for key in list(self._v):
                phase, age, env, released = self._v[key]
                f = self.freqs[key]
                rate = self.RELEASE_PER_S if released else self.DECAY_PER_S
                e = env * np.exp(-rate * (idx + 1) / self.sr)
                a = np.minimum(1.0, (age + idx + 1) / att_n)
                ph = phase + 2.0 * math.pi * f * idx / self.sr
                tone = sum(w * np.sin((h + 1) * ph) for h, w in enumerate(HARMONICS))
                out += tone * e * a
                env_end = float(env * math.exp(-rate * n / self.sr))
                self._v[key] = [(phase + 2.0 * math.pi * f * n / self.sr) % (2.0 * math.pi),
                                age + n, env_end, released]
                if released and env_end < self.SILENT:
                    del self._v[key]
        return np.tanh(out * self.GAIN).astype(np.float32)


class AudioOut:
    """Plays a Synth through the default output device. Raises on failure; the caller decides what to do.

    Uses the device's own sample rate (many Windows devices run at 48000 Hz) and falls back to safer stream
    settings if the low-latency ones are refused."""

    def __init__(self, synth=None, blocksize=256):
        import sounddevice as sd
        self.sd = sd
        try:
            info = sd.query_devices(None, "output")
            rate = int(info.get("default_samplerate") or SAMPLE_RATE)
            self.device_name = str(info.get("name", "default output"))
        except Exception:
            rate, self.device_name = SAMPLE_RATE, "default output"
        self.synth = synth or Synth(sr=rate)
        last = None
        for kw in ({"blocksize": blocksize, "latency": "low"}, {"blocksize": 0, "latency": "high"}):
            try:
                self.stream = sd.OutputStream(samplerate=self.synth.sr, channels=1, dtype="float32",
                                              callback=self._callback, **kw)
                self.stream.start()
                return
            except Exception as e:                 # try the next, safer settings
                last = e
        raise RuntimeError("could not open the audio output (%s): %s" % (self.device_name, last))

    def _callback(self, outdata, frames, time_info, status):
        try:
            outdata[:, 0] = self.synth.render(frames)
        except Exception:                          # an exception here would end the stream for good
            outdata[:, 0] = 0.0

    def chime(self):
        """Short C-E-G arpeggio so you can hear that sound works without any camera or tapping."""
        import threading
        import time

        def run():
            for k in (0, 2, 4):
                self.note_on(k)
                time.sleep(0.16)
                self.note_off(k)
            self.note_on(0)
            self.note_on(2)
            self.note_on(4)
            time.sleep(0.5)
            self.all_off()
        threading.Thread(target=run, daemon=True).start()

    # PianoLink calls these
    def note_on(self, key):
        self.synth.note_on(key)

    def note_off(self, key):
        self.synth.note_off(key)

    def all_off(self):
        self.synth.all_off()

    def close(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


def self_test():
    """python foot_piano.py --test-sound   -> returns an exit message; prints what it found."""
    import time
    try:
        import sounddevice as sd
    except ImportError:
        return ("The 'sounddevice' package is not installed in this Python environment.\n"
                "Fix:  pip install sounddevice      (use the same terminal/venv you run foot_piano.py in)")
    try:
        devices = sd.query_devices()
        out = [(i, d["name"]) for i, d in enumerate(devices) if d.get("max_output_channels", 0) > 0]
        print("Output devices found:")
        for i, name in out:
            print("  [%d] %s" % (i, name))
        if not out:
            return "No audio output device found. Check the speakers/headphones and the Windows sound settings."
        a = AudioOut()
    except Exception as e:
        return "Could not start audio: %s" % e
    print("Playing on: %s (%d Hz). You should hear C, E, G and then a C major chord." % (a.device_name, a.synth.sr))
    a.chime()
    time.sleep(1.5)
    a.close()
    return "Done. If you heard nothing, check the volume / that this is the output you expect (Windows sound settings)."
