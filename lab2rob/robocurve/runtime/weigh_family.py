"""Weighing family: biolab's balance bench, a salt vial passed between the arms."""

from __future__ import annotations

import numpy as np

from biolab import judge
from biolab.env import WeighEnv
from biolab.scene import ARMS, BOAT_SIZE, PAN_TOP, Rack, WeighLayout

from rc_base import RoutedStep

CAMERA = "bench"


def layout_from(doc: dict) -> WeighLayout:
    def rack(r):
        return Rack(r["name"], tuple(r["xy"]), nx=r["nx"], ny=r["ny"], yaw=r["yaw"])
    return WeighLayout(rack(doc["stock"]), rack(doc["handover"]), tuple(doc["balance"]),
                       tuple(doc["cell"]), None)


class Env(RoutedStep, WeighEnv):
    """biolab's weighing bench, pinned to the instance and ticking at 20 Hz."""

    READING_WINDOW = 10  # the display averages half a second (10 steps at 20 Hz)

    def __init__(self, instance: dict, session):
        self._instance, self._session = instance, session
        super().__init__(seed=instance["seed"], randomize=False)

    def _new_episode(self, rng):
        super()._new_episode(rng)
        self.layout = layout_from(self._instance["layout"])
        self.task.target_g = float(self._instance["target_g"])
        self.task.tolerance_g = float(self._instance["scoring"]["full_g"])


def make_env(instance: dict, session):
    return Env(instance, session)


def state_extra(env) -> list:
    return [list(env._deflection), env._tare]


def api(env) -> dict:
    from biolab.api import WeighApi
    return WeighApi(env).functions()


def measure(env) -> dict:
    """What a person would measure: balance, salt left in the vial, spills, contacts."""
    m, d = env.model, env.data
    vial = d.body(env.VIAL)
    frame = vial.xmat.reshape(3, 3)
    bx, by = env.layout.balance
    in_vial = on_boat = elsewhere = 0.0
    grains = [b for b in range(m.nbody) if m.body(b).name.startswith(f"{env.VIAL}_grain_")]
    for b in grains:
        p, mass = d.xpos[b], float(m.body_mass[b])
        local = frame.T @ (p - vial.xpos)
        if np.hypot(local[0], local[1]) < 0.0145 and 0.058 < local[2] < 0.082:
            in_vial += mass
        elif (abs(p[0] - bx) < BOAT_SIZE / 2 and abs(p[1] - by) < BOAT_SIZE / 2
              and PAN_TOP - 0.004 < p[2] < PAN_TOP + 0.04):
            on_boat += mass
        else:
            elsewhere += mass
    pan = m.body("balance_pan").id
    others = {b for b in range(m.nbody) if m.body(b).name.split("/")[0] in ARMS} | {vial.id}
    touched = any({m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} & {pan}
                  and {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} & others
                  for c in d.contact[: d.ncon])
    tilt = float(np.degrees(np.arccos(np.clip(frame[2, 2], -1, 1))))
    rack = env.layout.handover
    in_handover = any(judge._in_cell(env, env.VIAL, rack, c) for c in rack.cells()) and tilt < 15
    return {"reading_g": float(env.balance_reading()),
            "salt_in_vial_g": round(in_vial * 1000, 3),
            "salt_on_boat_g": round(on_boat * 1000, 3),
            "salt_spilled_g": round(elsewhere * 1000, 3),
            "pan_touched": bool(touched), "vial_tilt_deg": round(tilt, 1),
            "vial_in_handover_rack": bool(in_handover)}


def score(got: dict, instance: dict) -> tuple:
    """(score, reason), exactly as task.json's scoring text states it."""
    s = instance["scoring"]
    target = float(instance["target_g"])
    err = abs(got["reading_g"] - target)
    if got["pan_touched"]:
        value, reason = 0.0, "something other than salt touches the pan or the boat"
    elif err <= s["full_g"]:
        value, reason = 1.0, f"within {s['full_g']} g of the target"
    elif err <= s["three_quarter_g"]:
        value, reason = 0.75, f"within {s['three_quarter_g']} g"
    elif err <= s["half_g"]:
        value, reason = 0.5, f"within {s['half_g']} g"
    elif got["reading_g"] >= s["some_salt_g"]:
        value, reason = 0.25, f"at least {s['some_salt_g']} g on the boat"
    elif got["vial_in_handover_rack"]:
        value, reason = 0.1, "vial upright in the hand-over rack"
    else:
        value, reason = 0.0, "no salt on the boat"
    if got["salt_spilled_g"] > s["spill_cap_g"] and value > 0.5:
        value, reason = 0.5, reason + f"; capped: {got['salt_spilled_g']} g spilled"
    return value, reason, {"target_g": target, "error_g": round(err, 3)}


#: biolab's weighing script, with a gentler finish: salt leaves the vial in small
#: avalanches, so it switches to creeping 0.45 g early, tilts 0.002 rad at a time,
#: waits 1.5 s for the display each time, and stops within 0.05 g of the target.
DEMO_POUR = (('get_balance()["reading_g"] < target - 0.3', 'get_balance()["reading_g"] < target - 0.45'),
             ("tilt += 0.004", "tilt += 0.002"),
             (">= target - 0.06", ">= target - 0.05"),
             ("wait(0.8)", "wait(1.5)"))


def demo_script() -> str:
    from biolab.api import WEIGH_REFERENCE_POLICY
    code = WEIGH_REFERENCE_POLICY
    for old, new in DEMO_POUR:
        if code.count(old) != 1:
            raise RuntimeError(f"the reference script changed: {old!r} not found once")
        code = code.replace(old, new)
    return code
