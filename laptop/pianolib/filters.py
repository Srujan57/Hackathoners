"""One-Euro filter (Casiez et al., 2012): heavy smoothing when still, light smoothing when moving.

Pose-model keypoints jitter by a few pixels even when the body is still. A fixed low-pass filter
would remove the jitter but add lag to every tap; the One-Euro filter raises its cutoff with speed,
so slow drift is smoothed hard and fast taps pass through with little delay.

All coordinates in this project are in "frame-height units" (pixels / frame height), so the
default parameters do not depend on camera resolution.
"""

import math


DEFAULT_MIN_CUTOFF = 2.0   # Hz, smoothing when still
DEFAULT_BETA = 20.0        # how fast the cutoff opens up with speed (tuned on simulated taps)
DEFAULT_D_CUTOFF = 2.0


class OneEuro:
    def __init__(self, min_cutoff=None, beta=None, d_cutoff=None):
        self.min_cutoff = DEFAULT_MIN_CUTOFF if min_cutoff is None else min_cutoff
        self.beta = DEFAULT_BETA if beta is None else beta
        self.d_cutoff = DEFAULT_D_CUTOFF if d_cutoff is None else d_cutoff
        self.reset()

    def reset(self):
        self._t = None
        self._x = 0.0
        self._dx = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    @property
    def last_time(self):
        return self._t

    def __call__(self, x, t):
        if self._t is None:
            self._t, self._x, self._dx = t, x, 0.0
            return x
        dt = max(t - self._t, 1e-3)
        self._t = t
        dx = (x - self._x) / dt
        self._dx += self._alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        self._x += self._alpha(cutoff, dt) * (x - self._x)
        return self._x
