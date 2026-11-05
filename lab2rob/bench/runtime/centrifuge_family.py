"""Centrifuge family: load sample tubes into a microcentrifuge, balanced, and shut the lid.

The centrifuge is LabUtopia's Centrifuge_02 (CC BY-NC 4.0; meshes in
assets/centrifuge_02, converted by convert_labutopia_centrifuge.py). Its
meshes are for looks only: collision is built by hand from the measured
geometry (a base, 8 vertical slot pockets, the well wall, a box for the lid).
The lid turns on a hinge, rests open leaning back under gravity, and latches
shut past 80 degrees (a torque applied in the env, so the arms keep exactly
14 actuators). Nothing spins: loading and closing are the task.

Frames: the asset's own frame has the display at +x and the hinge at -x; the
layout places that frame on the bench at ``xy`` turned by ``yaw``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from biolab.env import TubeRackEnv
from biolab.scene import ARMS, _yaw_quat

from bench_base import RoutedStep

CAMERA = "bench"
ASSETS = Path(__file__).resolve().parent / "assets" / "centrifuge_02"

# Measured from the mesh (asset frame, metres).
ROTOR_C = (-0.0072, -0.0002)
SLOT_RADIUS = 0.0265      # slot centres around the rotor centre
SLOT_COUNT = 8
SLOT_FLOOR = 0.0446       # where a seated tube's bottom rests
SLOT_TOP = 0.069          # rim of the slots
POCKET_R = 0.0064         # inner radius of a slot pocket (1.2 mm clearance round a tube)
WELL_R = 0.047            # the rotor well's wall
RIM_Z = 0.087             # top of the body, where the shut lid lies
BODY = ((-0.0734, 0.0839), (-0.0641, 0.0641))
HINGE = (-0.063, 0.0, 0.087)  # lid hinge (the converter writes lid.stl relative to it)
LID_OPEN = -0.15          # rad: resting open, leaning back on its stop
LID_SHUT = np.pi / 2      # rad: lying on the rim
LATCH_FROM = np.radians(80)
LATCH_TORQUE = 0.15       # N m holding a shut lid down
LID_MASS = 0.15

# 1.5 mL microtube with its cap shut (simplified: a cylinder and a cap disc).
TUBE_R = 0.0052
TUBE_BODY = 0.036
CAP_R = 0.006
CAP_H = 0.004
TUBE_MASS = 0.003         # 2 g of plastic and 1 g of sample
TUBE_COLORS = ((0.95, 0.75, 0.1, 1), (0.2, 0.45, 0.85, 1), (0.85, 0.2, 0.2, 1), (0.25, 0.7, 0.35, 1))

# A low rack for the tubes: cells far enough apart for the fingers.
RACK_HOLE = 0.0116
RACK_WALL = 0.002
RACK_PITCH = 0.026
RACK_HEIGHT = 0.014
RACK_BASE = 0.003


def _rot(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


@dataclass
class TubeRack:
    name: str
    xy: tuple
    n: int = 4
    yaw: float = 0.0

    def cell_xy(self, i: int) -> np.ndarray:
        local = np.array([(i - (self.n - 1) / 2) * RACK_PITCH, 0.0])
        return np.array(self.xy) + _rot(self.yaw) @ local


@dataclass
class CentrifugeLayout:
    centrifuge_xy: tuple = (0.05, -0.10)
    centrifuge_yaw: float = -np.pi / 2   # hinge towards the left arm, display towards the right
    rack: TubeRack = field(default_factory=lambda: TubeRack("tube_rack", (0.10, -0.26)))
    tubes: dict = field(default_factory=lambda: {"tube_a": 0, "tube_b": 2})  # tube -> rack cell

    # ------------------------------------------------------------- geometry
    def to_world(self, local_xy) -> np.ndarray:
        return np.array(self.centrifuge_xy) + _rot(self.centrifuge_yaw) @ np.asarray(local_xy, float)

    def slot_local(self, k: int) -> np.ndarray:
        a = 2 * np.pi * k / SLOT_COUNT
        return np.array(ROTOR_C) + SLOT_RADIUS * np.array([np.cos(a), np.sin(a)])

    def slot_world(self, k: int) -> np.ndarray:
        return np.array([*self.to_world(self.slot_local(k)), SLOT_TOP])

    # ------------------------------------------------------------- building
    def add_to(self, spec: mujoco.MjSpec) -> None:
        add_centrifuge(spec, self)
        add_tube_rack(spec, self.rack)
        for i, (name, cell) in enumerate(self.tubes.items()):
            _add_microtube(spec, name, self.rack.cell_xy(cell), TUBE_COLORS[i % len(TUBE_COLORS)])
        add_bench_camera(spec, self)


def add_centrifuge(spec: mujoco.MjSpec, layout) -> None:
    """The centrifuge (meshes, collision, hinged lid) at layout.centrifuge_xy, turned by its yaw."""
    spec.compiler.meshdir = str(ASSETS)
    for part in ("body", "panel", "lid"):
        spec.add_mesh(name=f"centrifuge_{part}", file=f"{part}.stl")
    quat = _yaw_quat(layout.centrifuge_yaw)
    cf = spec.worldbody.add_body(name="centrifuge", pos=[*layout.centrifuge_xy, 0], quat=quat)
    # Visual only: no contacts and no mass (a mesh geom gets mass from its volume otherwise).
    vis = dict(type=mujoco.mjtGeom.mjGEOM_MESH, contype=0, conaffinity=0, group=1, density=0)
    cf.add_geom(name="centrifuge_body_visual", meshname="centrifuge_body", rgba=[0.93, 0.93, 0.92, 1], **vis)
    cf.add_geom(name="centrifuge_panel_visual", meshname="centrifuge_panel", rgba=[0.15, 0.6, 0.7, 1], **vis)
    box = dict(type=mujoco.mjtGeom.mjGEOM_BOX, rgba=[0.9, 0.9, 0.9, 0.0], group=3,
               friction=[0.8, 0.01, 0.0005])
    (x0, x1), (y0, y1) = BODY
    rx, ry = ROTOR_C
    # Solid below the slot floors.
    cf.add_geom(name="centrifuge_base", size=[(x1 - x0) / 2, (y1 - y0) / 2, SLOT_FLOOR / 2],
                pos=[(x0 + x1) / 2, (y0 + y1) / 2, SLOT_FLOOR / 2], **box)
    # The shell around the well, from the slot floor to the rim.
    hz, zc = (RIM_Z - SLOT_FLOOR) / 2, (RIM_Z + SLOT_FLOOR) / 2
    cf.add_geom(size=[(rx - WELL_R - x0) / 2, (y1 - y0) / 2, hz], pos=[(x0 + rx - WELL_R) / 2, 0, zc], **box)
    cf.add_geom(size=[(x1 - rx - WELL_R) / 2, (y1 - y0) / 2, hz], pos=[(x1 + rx + WELL_R) / 2, 0, zc], **box)
    cf.add_geom(size=[WELL_R, (y1 - ry - WELL_R) / 2, hz], pos=[rx, (y1 + ry + WELL_R) / 2, zc], **box)
    cf.add_geom(size=[WELL_R, (ry - WELL_R - y0) / 2, hz], pos=[rx, (y0 + ry - WELL_R) / 2, zc], **box)
    # The well's round wall, as an octagon (fills the corners of the square shell).
    t = 0.002
    for i in range(8):
        a = 2 * np.pi * i / 8 + np.pi / 8
        cf.add_geom(size=[t / 2, WELL_R * np.tan(np.pi / 8) + t, hz],
                    pos=[rx + (WELL_R + t / 2) * np.cos(a), ry + (WELL_R + t / 2) * np.sin(a), zc],
                    quat=_yaw_quat(a), **box)
    # Slot pockets: six walls round each slot, from the floor to the slot rim.
    ph, pz = (SLOT_TOP - SLOT_FLOOR) / 2, (SLOT_TOP + SLOT_FLOOR) / 2
    for k in range(SLOT_COUNT):
        cx, cy = layout.slot_local(k)
        for j in range(6):
            a = 2 * np.pi * j / 6
            cf.add_geom(size=[0.00075, POCKET_R * np.tan(np.pi / 6) + 0.0008, ph],
                        pos=[cx + (POCKET_R + 0.00075) * np.cos(a), cy + (POCKET_R + 0.00075) * np.sin(a), pz],
                        quat=_yaw_quat(a), **box)
        cf.add_site(name=f"slot_{k}", pos=[cx, cy, SLOT_FLOOR], size=[0.002, 0, 0], rgba=[1, 0, 0, 0],
                    group=4)
    # The lid, on its hinge; its joint range is its stop, open and shut.
    lid = cf.add_body(name="centrifuge_lid", pos=list(HINGE))
    # biolab's scene compiles angles in degrees (MuJoCo's default).
    lid.add_joint(name="lid_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0],
                  range=[float(np.degrees(LID_OPEN)), float(np.degrees(LID_SHUT))],
                  damping=0.02, armature=0.0005)
    lid.add_geom(name="lid_visual", meshname="centrifuge_lid", rgba=[0.93, 0.93, 0.92, 1], **vis)
    lid.add_geom(name="lid_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0105, 0.062, 0.063],
                 pos=[-0.0105, 0, 0.054], mass=LID_MASS, rgba=[0.9, 0.9, 0.9, 0.0], group=3,
                 friction=[0.8, 0.01, 0.0005])
    # MuJoCo skips parent-child contacts except with bodies fixed to the world, as the
    # centrifuge is: exclude the pair, so the joint range alone stops the lid.
    spec.add_exclude(bodyname1="centrifuge", bodyname2="centrifuge_lid")


def add_tube_rack(spec: mujoco.MjSpec, rack_def: "TubeRack") -> None:
    """A low rack, its cells along its own x axis."""
    rack = spec.worldbody.add_body(name=rack_def.name, pos=[*rack_def.xy, 0], quat=_yaw_quat(rack_def.yaw))
    lx, ly = rack_def.n * RACK_PITCH, RACK_PITCH
    rbox = dict(type=mujoco.mjtGeom.mjGEOM_BOX, rgba=[0.15, 0.55, 0.6, 1])
    rack.add_geom(size=[lx / 2, ly / 2, RACK_BASE / 2], pos=[0, 0, RACK_BASE / 2], **rbox)
    zc, hz = (RACK_BASE + RACK_HEIGHT) / 2, (RACK_HEIGHT - RACK_BASE) / 2
    for i in range(rack_def.n):
        cx = (i - (rack_def.n - 1) / 2) * RACK_PITCH
        for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rack.add_geom(size=[RACK_WALL / 2 if sx else RACK_HOLE / 2 + RACK_WALL,
                                RACK_WALL / 2 if sy else RACK_HOLE / 2 + RACK_WALL, hz],
                          pos=[cx + sx * (RACK_HOLE / 2 + RACK_WALL / 2), sy * (RACK_HOLE / 2 + RACK_WALL / 2), zc],
                          **rbox)


def add_bench_camera(spec: mujoco.MjSpec, layout) -> None:
    """A camera on the rack and the centrifuge, from the far side of the bench."""
    target = np.array([*layout.to_world(ROTOR_C), 0.06]) * [1, 1, 1] + [0.0, -0.05, 0.0]
    eye = target + [0.42, 0.05, 0.30]
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    spec.worldbody.add_camera(name="bench", pos=eye, xyaxes=[*right, *np.cross(right, forward)], fovy=50)


def _add_microtube(spec, name: str, xy, rgba) -> None:
    """Capped 1.5 mL tube (10.4 mm across, 40 mm tall). The body origin is the bottom centre."""
    body = spec.worldbody.add_body(name=name, pos=[xy[0], xy[1], RACK_BASE + 0.0005])
    body.add_freejoint(name=f"{name}_free")
    cyl = dict(type=mujoco.mjtGeom.mjGEOM_CYLINDER, friction=[1.5, 0.01, 0.0005], condim=4)
    foot, lift = 0.0035, 0.0005
    body.add_geom(name=f"{name}_foot", type=mujoco.mjtGeom.mjGEOM_BOX, size=[foot, foot, lift], pos=[0, 0, lift],
                  rgba=[0.92, 0.92, 0.95, 0.8], mass=0.0002, friction=[1.0, 0.01, 0.0005])
    body.add_geom(name=f"{name}_body", size=[TUBE_R, (TUBE_BODY - lift) / 2, 0], pos=[0, 0, (TUBE_BODY + lift) / 2],
                  rgba=[0.92, 0.92, 0.95, 0.8], mass=TUBE_MASS * 0.8, **cyl)
    body.add_geom(name=f"{name}_cap", size=[CAP_R, CAP_H / 2, 0], pos=[0, 0, TUBE_BODY + CAP_H / 2],
                  rgba=rgba, mass=TUBE_MASS * 0.2, **cyl)


def layout_to_json(layout: CentrifugeLayout) -> dict:
    return {"centrifuge_xy": [float(v) for v in layout.centrifuge_xy],
            "centrifuge_yaw": float(layout.centrifuge_yaw),
            "rack": {"name": layout.rack.name, "xy": [float(v) for v in layout.rack.xy], "n": layout.rack.n,
                     "yaw": float(layout.rack.yaw)},
            "tubes": dict(layout.tubes)}


def layout_from(doc: dict) -> CentrifugeLayout:
    r = doc["rack"]
    return CentrifugeLayout(tuple(doc["centrifuge_xy"]), float(doc["centrifuge_yaw"]),
                            TubeRack(r["name"], tuple(r["xy"]), r["n"], r["yaw"]), dict(doc["tubes"]))


# --------------------------------------------------------------------------- env
@dataclass
class CentrifugeTask:
    tubes: tuple

    @property
    def instruction(self) -> str:
        return ("Load the sample tubes into opposite slots of the centrifuge rotor so it is balanced, "
                "then close the lid.")


class Env(RoutedStep, TubeRackEnv):
    """The centrifuge bench, pinned to the instance and ticking at 20 Hz."""

    def __init__(self, instance: dict, session):
        self._instance, self._session = instance, session
        super().__init__(seed=instance["seed"], randomize=False)

    def _new_episode(self, rng):
        self.layout = layout_from(self._instance["layout"])
        self.task = CentrifugeTask(tuple(self.layout.tubes))

    def reset(self, seed=None):
        out = super().reset(seed)
        adr = self.model.joint("lid_hinge").qposadr[0]
        self.data.qpos[adr] = LID_OPEN  # open, resting on its stop
        mujoco.mj_forward(self.model, self.data)
        self.step(0.5)
        return out

    def _after_tick(self) -> None:
        jid = self.model.joint("lid_hinge").id
        dof = self.model.jnt_dofadr[jid]
        q = self.data.qpos[self.model.jnt_qposadr[jid]]
        # Ramped in over 5 degrees, so a lid lifted off the latch gets no kick.
        ramp = np.clip((q - LATCH_FROM) / np.radians(5), 0.0, 1.0)
        self.data.qfrc_applied[dof] = LATCH_TORQUE * ramp

    def lid_angle(self) -> float:
        """Degrees the lid has turned from upright towards shut (shut is 90)."""
        return float(np.degrees(self.data.qpos[self.model.joint("lid_hinge").qposadr[0]]))

    def tube_pose(self, name: str):
        b = self.data.body(name)
        axis = b.xmat.reshape(3, 3)[:, 2]
        return b.xpos.copy(), b.xpos + axis * (TUBE_BODY + CAP_H)

    def perceive(self) -> dict:
        d = self.data
        tubes = {}
        for name in self.layout.tubes:
            bottom, top = self.tube_pose(name)
            tubes[name] = {"bottom": bottom, "top": top}
        return {"tubes": tubes, "lid_deg": self.lid_angle(), "ee": {a: self.ee_pose(a) for a in ARMS},
                "time": float(d.time)}

    def evaluate(self) -> dict:
        got = measure(self)
        value, reason, _ = score(got, self._instance)
        return {"success": value == 1.0, "progress": value, "note": reason}


def make_env(instance: dict, session):
    return Env(instance, session)


def state_extra(env) -> list:
    return [float(env.data.qfrc_applied.sum())]


# --------------------------------------------------------------------- skills API
class CentrifugeApi:
    """What the reference script (and a code-writing agent) may call."""

    def __init__(self, env: Env):
        self.sim = env

    def functions(self) -> dict:
        return {name: getattr(self, name) for name in (
            "get_tube_pose", "get_rack", "get_slots", "get_lid", "move_to", "set_gripper", "go_home", "wait")}

    def get_tube_pose(self, tube: str):
        """(bottom, top) of a tube in metres, world frame."""
        bottom, top = self.sim.tube_pose(tube)
        return bottom, top

    def get_rack(self) -> dict:
        lay = self.sim.layout
        return {"cells": [np.array([*lay.rack.cell_xy(i), RACK_BASE]) for i in range(lay.rack.n)],
                "yaw": lay.rack.yaw, "tubes": dict(lay.tubes)}

    def get_slots(self) -> list:
        """Rim centre of each rotor slot, in order round the rotor: slot k is opposite k + 4."""
        return [self.sim.layout.slot_world(k) for k in range(SLOT_COUNT)]

    def get_lid(self) -> dict:
        """The lid: its angle, the hinge, and the outward direction of its outer face."""
        lay = self.sim.layout
        hinge = np.array([*lay.to_world(HINGE[:2]), HINGE[2]])
        outward = np.append(_rot(lay.centrifuge_yaw) @ np.array([-1.0, 0.0]), 0.0)  # away from the well
        return {"deg": self.sim.lid_angle(), "hinge": hinge, "outward": outward,
                "thickness": 0.021, "height": 0.117}

    def move_to(self, arm: str, position, yaw: float = 0.0, tilt: float = 0.0, speed: float = 0.15) -> bool:
        return bool(self.sim.command(arm, np.asarray(position, dtype=float), yaw, tilt, speed))

    def set_gripper(self, arm: str, opening: float) -> None:
        self.sim.set_gripper(arm, float(opening))

    def go_home(self, arm: str) -> None:
        self.sim.home(arm)

    def wait(self, seconds: float) -> None:
        self.sim.step(float(np.clip(seconds, 0.0, 5.0)))


def api(env) -> dict:
    return CentrifugeApi(env).functions()


# --------------------------------------------------------------------- measuring
def measure(env) -> dict:
    """What a photo shows: which slot each tube sits in, where the rest are, the lid."""
    lay = env.layout
    seated, where = {}, {}
    rack_c = np.array(lay.rack.xy)
    for name in lay.tubes:
        b = env.data.body(name)
        tilt = float(np.degrees(np.arccos(np.clip(b.xmat.reshape(3, 3)[2, 2], -1, 1))))
        slot = None
        for k in range(SLOT_COUNT):
            s = lay.slot_world(k)
            if (np.linalg.norm(b.xpos[:2] - s[:2]) < 0.003 and abs(b.xpos[2] - SLOT_FLOOR) < 0.004
                    and tilt < 10):
                slot = k
        seated[name] = slot
        if slot is not None:
            where[name] = "slot"
        elif np.linalg.norm(b.xpos[:2] - lay.to_world(ROTOR_C)) < WELL_R and b.xpos[2] > 0.03:
            where[name] = "centrifuge"
        elif np.linalg.norm(b.xpos[:2] - rack_c) < RACK_PITCH * lay.rack.n / 2 and b.xpos[2] < 0.03:
            where[name] = "rack"
        else:
            where[name] = "elsewhere"
    slots = sorted(k for k in seated.values() if k is not None)
    balanced = len(slots) == 2 and len(lay.tubes) == 2 and (slots[1] - slots[0]) == SLOT_COUNT // 2
    lid = env.lid_angle()
    return {"slots": {k: v for k, v in seated.items()}, "where": where, "seated": len(slots),
            "balanced": bool(balanced), "lid_deg": round(lid, 1),
            "lid_closed": bool(lid >= np.degrees(LATCH_FROM) + 5)}


def score(got: dict, instance: dict) -> tuple:
    """(score, reason), exactly as task.json's scoring text states it."""
    if got["lid_closed"] and not got["balanced"]:
        value, reason = 0.0, "lid closed on an unbalanced load"
    elif got["balanced"] and got["lid_closed"]:
        value, reason = 1.0, "both tubes in opposite slots, lid closed"
    elif got["balanced"]:
        value, reason = 0.6, "both tubes in opposite slots, lid open"
    elif got["seated"] >= 1:
        value, reason = 0.3, f"{got['seated']} tube(s) seated, not balanced, lid open"
    else:
        value, reason = 0.0, "no tube seated in a slot"
    dropped = [t for t, w in got["where"].items() if w == "elsewhere"]
    if dropped and value > 0.3:
        value, reason = 0.3, reason + f"; capped: {', '.join(dropped)} outside the rack and centrifuge"
    return value, reason, {}


# --------------------------------------------------------------------------- demo
DEMO = '''
import numpy as np

TRAVEL = 0.15           # pinch height when carrying a tube
GRIP = 0.008            # pads this far below the tube's top
FINGERTIPS = 0.017      # fingertips below the pinch point
right_base, left_base = np.array([-0.20, -0.28]), np.array([-0.20, 0.28])

def reach(arm, target, speed=0.1, yaws=(0.0, 0.5, -0.5, 1.0, -1.0, 1.5), tilts=(0.0, 0.3, 0.6, 0.9, 1.2)):
    for tilt in tilts:
        for yaw in yaws:
            if move_to(arm, target, yaw, tilt, speed):
                return yaw, tilt
    raise RuntimeError(f"{arm} cannot reach {target}")

def pick(arm, tube):
    bottom, top = get_tube_pose(tube)
    grasp = top.copy(); grasp[2] -= GRIP
    set_gripper(arm, 0.03)
    move_to(arm, [grasp[0], grasp[1], TRAVEL], 0.0)
    move_to(arm, grasp, 0.0, speed=0.04)
    set_gripper(arm, 0.0)
    move_to(arm, [grasp[0], grasp[1], TRAVEL], 0.0, speed=0.06)

def place_in(arm, slot):
    s = slots[slot]
    release = s[2] + 0.002 + FINGERTIPS     # fingertips just above the slot rim; the tube drops in
    move_to(arm, [s[0], s[1], TRAVEL], 0.0, speed=0.1)
    move_to(arm, [s[0], s[1], release], 0.0, speed=0.03)
    set_gripper(arm, 0.03)
    wait(0.5)
    move_to(arm, [s[0], s[1], TRAVEL], 0.0, speed=0.06)

slots = get_slots()
# The opposite pair nearest the right arm.
pair = min(range(4), key=lambda k: max(np.linalg.norm(slots[k][:2] - right_base),
                                       np.linalg.norm(slots[k + 4][:2] - right_base)))
tubes = list(get_rack()["tubes"])
pick("right", tubes[0]); place_in("right", pair)
pick("right", tubes[1]); place_in("right", pair + 4)
go_home("right")

# The left arm tips the open lid past upright from behind; it falls shut and latches.
lid = get_lid()
out, hinge = lid["outward"], lid["hinge"]
face = hinge + out * (lid["thickness"] + 0.004) + np.array([0, 0, 0.07])
set_gripper("left", 0.0)
yaw, tilt = reach("left", face + out * 0.04 + np.array([0, 0, 0.03]))
move_to("left", face + out * 0.03, yaw, tilt, 0.05)
move_to("left", face - out * 0.035, yaw, tilt, 0.03)
wait(1.5)
move_to("left", face + out * 0.05 + np.array([0, 0, 0.05]), yaw, tilt, 0.08)
go_home("left")
wait(1.0)
'''


def demo_script() -> str:
    return DEMO
