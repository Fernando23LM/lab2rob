"""Shared runtime pieces: the 20 Hz control step and the model hash."""

from __future__ import annotations

import hashlib

import numpy as np

CONTROL_HZ = 20
CONTROL_DT = 1.0 / CONTROL_HZ


class RoutedStep:
    """Mixin for a biolab env: every env.step goes through the session's ticks.

    Before the session is live (the env's own reset and settling) the env
    steps itself at the same 50 ms rate, unrecorded.
    """

    _session = None

    def step(self, seconds: float) -> None:
        import mujoco
        session = getattr(self, "_session", None)
        n = max(1, round(float(seconds) / CONTROL_DT))
        if session is None or not session._live:
            sub = round(CONTROL_DT / self.model.opt.timestep)
            for _ in range(n):
                mujoco.mj_step(self.model, self.data, sub)
                self._after_tick()
            return
        for _ in range(n):
            session._tick()


def model_sha(model) -> str:
    """Every array and option of the model except the install paths of its meshes."""
    h = hashlib.sha256()
    for name in sorted(dir(model)):
        if name.startswith("_") or name in ("paths", "npaths", "nbuffer") or name.endswith("_pathadr"):
            continue
        try:
            value = getattr(model, name)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(value, np.ndarray):
            h.update(name.encode() + np.ascontiguousarray(value).tobytes())
        elif isinstance(value, (int, float, bool)):
            h.update(name.encode() + repr(value).encode())
    return h.hexdigest()[:16]
