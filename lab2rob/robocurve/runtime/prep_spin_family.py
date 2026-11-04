"""Prep-and-spin family: a protocol's add-reagents-then-centrifuge run as one task.

The right arm holds a pipette (biolab's) and adds the protocol's reagents from
troughs into an open sample tube, then water into a balance tube to match. The
left arm loads both tubes into opposite rotor slots and shuts the lid. The
centrifuge is the centrifuge family's (LabUtopia's Centrifuge_02).

Liquids are bookkeeping, as in biolab: each tube carries a well that follows
it, and every reagent is tracked by name (a well's ``extra`` holds
``reagent:<name>`` microlitres), so the sample's contents can be checked
reagent by reagent. Shutting the lid on the rotor records a spin event with
what each tube held at that moment: the order of the protocol is checked
against it (a reagent added after the spin does not count).

Actions: the arms' 14 targets plus the pipette plunger (``pipette_plunger``,
microlitres drawn), as in lab2rob's pipetting sessions. A real gripper-held
pipette cannot be driven this way; see the fidelity note in task.json.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from biolab.liquid import Well
from biolab.protein import (BASE, TROUGH_DEPTH, TROUGH_OPENING, TROUGH_WALL, PipetteEnv,
                            ProteinLayout, _add_block, add_liquids_and_pipette)
from biolab.scene import Rack, _yaw_quat

import centrifuge_family as cf
from rc_base import RoutedStep

CAMERA = "bench"
EXTRA_ACTIONS = ("pipette_plunger",)
EXTRA_UNITS = ("uL",)

# Open 1.5 mL tube: hexagonal wall, a floor, no cap.
TUBE_IN_R = 0.0042
TUBE_WALL = 0.0008
TUBE_H = 0.037
TUBE_FLOOR = 0.001
TUBE_MASS = 0.002
TUBE_HALF = TUBE_IN_R * 0.85        # the well's square opening, inscribed in the bore
TUBE_DEPTH = TUBE_H - TUBE_FLOOR
SAMPLE, BALANCE, WATER = "sample", "balance", "water"
TUBE_COLORS = {SAMPLE: (0.95, 0.75, 0.1, 1), BALANCE: (0.2, 0.45, 0.85, 1)}
PIPETTE_CAPACITY = 200.0
PREFILL_MARGIN = 1.2                 # troughs hold this much more than the task draws
BALANCE_TOL = 0.10                   # balance tube within 10% of the sample's volume
SPILL_CAP_UL = 10.0

# Aiming (as lab2rob's certifier): approach above the rim, then correct the tip's height.
APPROACH_M, SUBMERGE_M, FLOOR_MARGIN_M = 0.03, 0.001, 0.0004
TRAVEL_SPEED, WELL_SPEED, TIP_TOL_M, TIP_CORRECTIONS = 0.15, 0.04, 0.0002, 3
PLUNGER_HOLD_S = 0.4


@dataclass
class PrepLayout(cf.CentrifugeLayout):
    """The centrifuge on the left arm's side (lid hinged towards the left arm), the tube rack in
    the strip both arms reach, and reagent troughs on the right arm's side."""

    centrifuge_xy: tuple = (0.05, 0.12)
    centrifuge_yaw: float = float(-np.pi / 2)
    rack: cf.TubeRack = field(default_factory=lambda: cf.TubeRack("tube_rack", (-0.08, -0.025)))
    tubes: dict = field(default_factory=lambda: {SAMPLE: 0, BALANCE: 2})
    reservoir_xy: tuple = (0.02, -0.36)
    troughs: tuple = ("reagent_1", WATER)

    def reservoir(self) -> Rack:
        return Rack("reservoir", tuple(self.reservoir_xy), nx=len(self.troughs), ny=1)

    def trough_wells(self) -> list:
        rack = self.reservoir()
        return [Well(name, ProteinLayout.cell(rack, i, 0, TROUGH_OPENING + TROUGH_WALL), BASE,
                     TROUGH_OPENING / 2, TROUGH_DEPTH, yaw=rack.yaw)
                for i, name in enumerate(self.troughs)]

    def tube_wells(self) -> list:
        return [Well(name, self.rack.cell_xy(cell), cf.RACK_BASE + 0.0005 + TUBE_FLOOR, TUBE_HALF,
                     TUBE_DEPTH, yaw=self.rack.yaw)
                for name, cell in self.tubes.items()]

    def wells(self) -> list:
        return self.trough_wells() + self.tube_wells()

    def add_to(self, spec: mujoco.MjSpec) -> None:
        cf.add_centrifuge(spec, self)
        cf.add_tube_rack(spec, self.rack)
        for name, cell in self.tubes.items():
            _add_open_tube(spec, name, self.rack.cell_xy(cell), TUBE_COLORS.get(name, (0.85, 0.2, 0.2, 1)))
        _add_block(spec, self.reservoir(), TROUGH_OPENING, TROUGH_WALL, TROUGH_DEPTH, [0.75, 0.8, 0.85, 1])
        # Liquid in the troughs, the pipette, and the "bench" camera over the whole bench.
        add_liquids_and_pipette(spec, self.trough_wells(), target=(0.0, -0.06, 0.03))


def _add_open_tube(spec, name: str, xy, rgba) -> None:
    """Open 1.5 mL tube (10 mm across, 37 mm tall): a hexagonal wall and a floor; origin at the bottom."""
    body = spec.worldbody.add_body(name=name, pos=[xy[0], xy[1], cf.RACK_BASE + 0.0005])
    body.add_freejoint(name=f"{name}_free")
    wall = dict(type=mujoco.mjtGeom.mjGEOM_BOX, rgba=[0.92, 0.92, 0.95, 0.55], friction=[1.5, 0.01, 0.0005],
                condim=4, mass=TUBE_MASS / 7)
    body.add_geom(name=f"{name}_floor", size=[TUBE_IN_R, TUBE_IN_R, TUBE_FLOOR / 2],
                  pos=[0, 0, TUBE_FLOOR / 2], **wall)
    side = (TUBE_IN_R + TUBE_WALL) * np.tan(np.pi / 6)
    for j in range(6):
        a = 2 * np.pi * j / 6
        body.add_geom(size=[TUBE_WALL / 2, side, TUBE_H / 2],
                      pos=[(TUBE_IN_R + TUBE_WALL / 2) * np.cos(a), (TUBE_IN_R + TUBE_WALL / 2) * np.sin(a), TUBE_H / 2],
                      quat=_yaw_quat(a), **wall)
    # A coloured band near the rim tells the sample (yellow) from the balance tube (blue).
    body.add_geom(name=f"{name}_band", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[TUBE_IN_R + TUBE_WALL + 0.0004, 0.002, 0], pos=[0, 0, TUBE_H - 0.004],
                  rgba=rgba, contype=0, conaffinity=0, density=0)
    body.add_geom(name=f"liquid_{name}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[TUBE_IN_R * 0.95, 1e-4, 0],
                  pos=[0, 0, TUBE_FLOOR], rgba=[0.6, 0.7, 0.9, 0.0], contype=0, conaffinity=0, density=0)


def layout_to_json(layout: PrepLayout) -> dict:
    out = cf.layout_to_json(layout)
    out.update(reservoir_xy=[float(v) for v in layout.reservoir_xy], troughs=list(layout.troughs))
    return out


def layout_from(doc: dict) -> PrepLayout:
    r = doc["rack"]
    return PrepLayout(tuple(doc["centrifuge_xy"]), float(doc["centrifuge_yaw"]),
                      cf.TubeRack(r["name"], tuple(r["xy"]), r["n"], r["yaw"]), dict(doc["tubes"]),
                      tuple(doc["reservoir_xy"]), tuple(doc["troughs"]))


# --------------------------------------------------------------------------- env
@dataclass
class PrepTask:
    reagents: tuple   # ({"name", "trough", "volume_ul", "step"}, ...)

    @property
    def instruction(self) -> str:
        return "Prepare the sample, balance it, and centrifuge it."


class Env(RoutedStep, PipetteEnv):
    PIPETTE_CAPACITY = PIPETTE_CAPACITY

    def __init__(self, instance: dict, session):
        self._instance, self._session = instance, session
        self._plunger_target, self._stroke, self._last_plunger = 0.0, 0.0, {}
        self.spin = None
        super().__init__(seed=instance["seed"], randomize=False)

    def _new_episode(self, rng):
        self.layout = layout_from(self._instance["layout"])
        self.task = PrepTask(tuple(self._instance["reagents"]))
        self._noise = np.random.default_rng(rng.integers(1 << 31))

    def _fill(self) -> None:
        drawn = {r["trough"]: 0.0 for r in self.task.reagents}
        for r in self.task.reagents:
            drawn[r["trough"]] += r["volume_ul"]
        drawn[WATER] = sum(r["volume_ul"] for r in self.task.reagents)
        for trough, volume in drawn.items():
            amount = round(volume * PREFILL_MARGIN + 20.0, 3)
            name = next((r["name"] for r in self.task.reagents if r["trough"] == trough), WATER)
            self.wells[trough].add(amount, 0.0, {f"reagent:{name}": amount})

    def reset(self, seed=None):
        self._plunger_target, self._stroke, self.spin = 0.0, 0.0, None
        out = super().reset(seed)
        adr = self.model.joint("lid_hinge").qposadr[0]
        self.data.qpos[adr] = cf.LID_OPEN
        mujoco.mj_forward(self.model, self.data)
        self._follow_tubes()
        self.step(0.5)
        return out

    # tubes carry their wells
    def _follow_tubes(self) -> None:
        for name in self.layout.tubes:
            well, body = self.wells[name], self.data.body(name)
            well.center = body.xpos[:2].copy()
            well.floor = float(body.xpos[2]) + TUBE_FLOOR
            m = body.xmat.reshape(3, 3)
            well.yaw = float(np.arctan2(m[1, 0], m[0, 0]))

    def _sync_liquids(self) -> None:
        m = self.model
        for well in self.wells.values():
            g = m.geom(f"liquid_{well.name}").id
            height = max(1e-4, (well.surface - well.floor) / 2)
            m.geom_size[g][1 if well.name in self.layout.tubes else 2] = height
            if well.name in self.layout.tubes:
                m.geom_pos[g][2] = TUBE_FLOOR + height
            else:
                m.geom_pos[g][2] = well.floor + height
            m.geom_rgba[g] = [0.85, 0.3, 0.35, 0.85] if well.volume > 0 else [0, 0, 0, 0]

    def _after_tick(self) -> None:
        jid = self.model.joint("lid_hinge").id
        q = self.data.qpos[self.model.jnt_qposadr[jid]]
        ramp = np.clip((q - cf.LATCH_FROM) / np.radians(5), 0.0, 1.0)
        self.data.qfrc_applied[self.model.jnt_dofadr[jid]] = cf.LATCH_TORQUE * ramp
        if getattr(self, "wells", None):
            self._follow_tubes()
            if self.spin is None and self.lid_angle() >= np.degrees(cf.LATCH_FROM) + 5:
                self.spin = {"time": float(self.data.time), **_load(self)}

    def lid_angle(self) -> float:
        return float(np.degrees(self.data.qpos[self.model.joint("lid_hinge").qposadr[0]]))

    def tube_pose(self, name: str):
        b = self.data.body(name)
        return b.xpos.copy(), b.xpos + b.xmat.reshape(3, 3)[:, 2] * TUBE_H


def make_env(instance: dict, session):
    return Env(instance, session)


# ----------------------------------------------------------------- the plunger
def extra_range(env):
    return [0.0], [env.pipette.capacity]


def set_extra(env, values) -> None:
    env._plunger_target = float(values[0])


def extra_command(env) -> list:
    return [env._plunger_target]


def before_step(env) -> None:
    """The plunger moves liquid at the start of a step, where the tip is now."""
    delta = env._plunger_target - env._stroke
    if delta == 0.0:
        return
    env._stroke = env._plunger_target
    if not env.holding_pipette():
        env._last_plunger = {"volume_ul": 0.0, "note": "the right gripper is not holding the pipette"}
        return
    wells = list(env.wells.values())
    if delta > 0:
        env._last_plunger = env.pipette.aspirate(wells, env.tip(), delta)
    else:
        env._last_plunger = env.pipette.dispense(wells, env.tip(), -delta)
    env._sync_liquids()


def _contents(well) -> dict:
    return {k.split(":", 1)[1]: round(v, 4) for k, v in sorted(well.extra.items())
            if k.startswith("reagent:") and v > 1e-6}


def state_extra(env) -> list:
    wells = {n: [w.volume, w.conc, sorted(w.extra.items())] for n, w in sorted(env.wells.items())}
    p = env.pipette
    return [wells, [p.volume, p.spilled, sorted(p.extra.items())], env._stroke, env._plunger_target,
            env.spin]


# --------------------------------------------------------------------- skills API
def _grip_offset(env) -> np.ndarray:
    return env.data.site(env._site["right"]).xpos - env.tip()


class PrepApi:
    def __init__(self, env: Env):
        self.sim = env

    def functions(self) -> dict:
        return {name: getattr(self, name) for name in (
            "get_task", "get_wells", "get_tube_pose", "get_slots", "get_lid", "get_rack", "move_to",
            "set_gripper", "go_home", "wait", "move_tip_into", "withdraw", "aspirate", "dispense")}

    def get_task(self) -> dict:
        return {"reagents": [dict(r) for r in self.sim.task.reagents], "water_trough": WATER,
                "tubes": {"sample": SAMPLE, "balance": BALANCE}}

    def get_wells(self) -> dict:
        return {n: {"center": np.array([*w.center, w.floor + w.depth]), "floor_z": w.floor, "depth": w.depth,
                    "volume_ul": w.volume, "surface_z": w.surface} for n, w in self.sim.wells.items()}

    def get_tube_pose(self, tube: str):
        return self.sim.tube_pose(tube)

    def get_slots(self) -> list:
        return [self.sim.layout.slot_world(k) for k in range(cf.SLOT_COUNT)]

    def get_lid(self) -> dict:
        lay = self.sim.layout
        hinge = np.array([*lay.to_world(cf.HINGE[:2]), cf.HINGE[2]])
        outward = np.append(cf._rot(lay.centrifuge_yaw) @ np.array([-1.0, 0.0]), 0.0)
        return {"deg": self.sim.lid_angle(), "hinge": hinge, "outward": outward, "thickness": 0.021}

    def get_rack(self) -> dict:
        lay = self.sim.layout
        return {"cells": [np.array([*lay.rack.cell_xy(i), cf.RACK_BASE]) for i in range(lay.rack.n)],
                "tubes": dict(lay.tubes)}

    def move_to(self, arm: str, position, yaw: float = 0.0, tilt: float = 0.0, speed: float = 0.15) -> bool:
        return bool(self.sim.command(arm, np.asarray(position, dtype=float), yaw, tilt, speed))

    def set_gripper(self, arm: str, opening: float) -> None:
        self.sim.set_gripper(arm, float(opening))

    def go_home(self, arm: str) -> None:
        self.sim.home(arm)

    def wait(self, seconds: float) -> None:
        self.sim.step(float(np.clip(seconds, 0.0, 5.0)))

    def move_tip_into(self, well: str) -> bool:
        """Tip just under the well's surface (or near its floor), aimed by the measured tip."""
        w = self.sim.wells[well]
        x, y = float(w.center[0]), float(w.center[1])
        top = w.floor + w.depth + APPROACH_M
        tip_z = max(w.floor + 0.0002, max(w.floor + FLOOR_MARGIN_M, w.surface - SUBMERGE_M))
        if not self.move_to("right", np.array([x, y, top]) + _grip_offset(self.sim), 0.0, 0.0, TRAVEL_SPEED):
            return False
        goal = np.array([x, y, tip_z])
        aim = goal + _grip_offset(self.sim)
        ok = self.move_to("right", aim, 0.0, 0.0, WELL_SPEED)
        for _ in range(TIP_CORRECTIONS):
            error = goal - self.sim.tip()
            if np.abs(error).max() <= TIP_TOL_M:
                break
            aim = aim + error
            ok = self.move_to("right", aim, 0.0, 0.0, WELL_SPEED)
        return ok

    def withdraw(self, well: str) -> bool:
        w = self.sim.wells[well]
        top = w.floor + w.depth + APPROACH_M
        return self.move_to("right", np.array([w.center[0], w.center[1], top]) + _grip_offset(self.sim),
                            0.0, 0.0, TRAVEL_SPEED)

    def _plunger(self, volume_ul: float, well, sign: float) -> dict:
        if well is not None and not self.move_tip_into(well):
            return {"volume_ul": 0.0, "note": f"could not reach {well}"}
        env = self.sim
        env._plunger_target = float(np.clip(env._plunger_target + sign * max(0.0, float(volume_ul)),
                                            0.0, env.pipette.capacity))
        env._last_plunger = {"volume_ul": 0.0, "note": "plunger did not move"}
        self.wait(PLUNGER_HOLD_S)
        result = dict(env._last_plunger)
        if well is not None:
            self.withdraw(well)
        return result

    def aspirate(self, volume_ul: float, well: str | None = None) -> dict:
        return self._plunger(volume_ul, well, 1.0)

    def dispense(self, volume_ul: float, well: str | None = None) -> dict:
        return self._plunger(volume_ul, well, -1.0)


def api(env) -> dict:
    return PrepApi(env).functions()


# --------------------------------------------------------------------- measuring
def _load(env) -> dict:
    """The rotor's load and what each tube holds, now."""
    rotor = cf.measure(env)
    tubes = {n: {"volume_ul": round(env.wells[n].volume, 3), "reagents": _contents(env.wells[n])}
             for n in env.layout.tubes}
    return {"slots": rotor["slots"], "where": rotor["where"], "opposite": rotor["balanced"], "tubes": tubes}


def measure(env) -> dict:
    now = _load(env)
    return {**now, "lid_deg": round(env.lid_angle(), 1),
            "lid_closed": bool(env.lid_angle() >= np.degrees(cf.LATCH_FROM) + 5),
            "spilled_ul": round(env.pipette.spilled, 3), "spin": env.spin}


def score(got: dict, instance: dict) -> tuple:
    """(score, reason), exactly as task.json's scoring text states it."""
    spin = got["spin"]
    state = spin or got
    sample, balance = state["tubes"][SAMPLE], state["tubes"][BALANCE]
    targets = instance["scoring"]["reagents"]
    sample_ul = sample["volume_ul"]
    matched = balance["volume_ul"] > 0 and abs(balance["volume_ul"] - sample_ul) <= BALANCE_TOL * max(sample_ul, 1e-9)
    if spin is not None and not (spin["opposite"] and matched):
        return 0.0, "lid shut on an unbalanced load", {}
    right = [name for name, t in targets.items()
             if abs(sample["reagents"].get(name, 0.0) - t["target_ul"]) <= t["tol_ul"]]
    extra = [name for name in sample["reagents"] if name not in targets]
    reagent_part = 0.4 * len(right) / len(targets) if not extra else 0.0
    value = reagent_part + 0.2 * matched + 0.2 * bool(state["opposite"]) + (0.2 if spin else 0.0)
    value = round(value, 3)
    reason = (f"{len(right)}/{len(targets)} reagents right"
              + (f" (also {', '.join(extra)}: no reagent credit)" if extra else "")
              + f"; balance tube {'matched' if matched else 'not matched'}"
              + f"; tubes {'opposite' if state['opposite'] else 'not opposite'}"
              + f"; {'spun' if spin else 'not spun'}" + (" (judged at the spin)" if spin else ""))
    if got["spilled_ul"] > SPILL_CAP_UL and value > 0.5:
        value, reason = 0.5, reason + f"; capped: {got['spilled_ul']} uL spilled"
    dropped = [t for t, w in got["where"].items() if w == "elsewhere"]
    if dropped and value > 0.3:
        value, reason = 0.3, reason + f"; capped: {', '.join(dropped)} outside the rack and centrifuge"
    return value, reason, {}


# --------------------------------------------------------------------------- demo
DEMO = '''
import numpy as np

TRAVEL = 0.15
GRIP = 0.008
FINGERTIPS = 0.017
left_base = np.array([-0.20, 0.28])

def transfer(src, dst, volume):
    left = volume
    while left > 1e-9:
        v = min(left, 200.0)
        aspirate(v, src)
        dispense(v, dst)
        left -= v

def reach(arm, target, speed=0.1, yaws=(0.0, 0.5, -0.5, 1.0, -1.0, 1.5), tilts=(0.0, 0.3, 0.6, 0.9, 1.2)):
    for tilt in tilts:
        for yaw in yaws:
            if move_to(arm, target, yaw, tilt, speed):
                return yaw, tilt
    raise RuntimeError(f"{arm} cannot reach {target}")

YAWS = (0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5)

def descend(arm, xy, low, speed):
    """Above xy, then straight down to height low, at the first wrist yaw that reaches both."""
    for yaw in YAWS:
        if move_to(arm, [xy[0], xy[1], TRAVEL], yaw, speed=0.1) and move_to(arm, [xy[0], xy[1], low], yaw, speed=speed):
            return yaw
    raise RuntimeError(f"{arm} cannot reach down to {xy} at {low}")

def pick(arm, tube):
    bottom, top = get_tube_pose(tube)
    set_gripper(arm, 0.03)
    yaw = descend(arm, top[:2], top[2] - GRIP, 0.04)
    set_gripper(arm, 0.0)
    move_to(arm, [top[0], top[1], TRAVEL], yaw, speed=0.06)

def place_in(arm, slot):
    s = slots[slot]
    yaw = descend(arm, s[:2], s[2] + 0.002 + FINGERTIPS, 0.03)
    set_gripper(arm, 0.03)
    wait(0.5)
    move_to(arm, [s[0], s[1], TRAVEL], yaw, speed=0.06)

task = get_task()
for r in task["reagents"]:
    transfer(r["trough"], "sample", r["volume_ul"])
transfer(task["water_trough"], "balance", sum(r["volume_ul"] for r in task["reagents"]))
go_home("right")

slots = get_slots()
pair = min(range(4), key=lambda k: max(np.linalg.norm(slots[k][:2] - left_base),
                                       np.linalg.norm(slots[k + 4][:2] - left_base)))
pick("left", "sample"); place_in("left", pair)
pick("left", "balance"); place_in("left", pair + 4)

lid = get_lid()
out, hinge = lid["outward"], lid["hinge"]
face = hinge + out * (lid["thickness"] + 0.004) + np.array([0, 0, 0.07])
set_gripper("left", 0.0)
# Approach behind the lid's outer face, as far out as the arm reaches, then push it past upright.
for back, up in ((0.04, 0.03), (0.025, 0.03), (0.015, 0.02), (0.01, 0.0)):
    try:
        yaw, tilt = reach("left", face + out * back + np.array([0, 0, up]))
        break
    except RuntimeError:
        continue
else:
    raise RuntimeError("the left arm cannot get behind the lid")
move_to("left", face + out * min(back, 0.01), yaw, tilt, 0.05)
move_to("left", face - out * 0.035, yaw, tilt, 0.03)
wait(1.5)
move_to("left", face + out * back + np.array([0, 0, up + 0.02]), yaw, tilt, 0.08)
go_home("left")
wait(1.0)
'''


def demo_script() -> str:
    return DEMO
