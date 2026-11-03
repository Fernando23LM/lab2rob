"""Session: one frozen biolab scene, driven at three levels, recorded as rows.

Every way of acting on the robot ends up as the same thing: rows of 15
targets, one per control step (CONTROL_DT, 20 ms), plus a short list of
instrument events. That is what a trajectory file holds and what the grader
replays, so a task is graded the same way whichever level the agent used.

Levels (``Session(instance, level=...)``):

- ``"joints"``: the agent sends rows itself with ``step(a, n)``: 6 joint
  targets + gripper per arm, and the pipette plunger.
- ``"skills"``: the agent calls motion skills (``move_to``, ``set_gripper``,
  ``move_joints``, ``aspirate``/``dispense`` with or without a well, ``wait``)
  and never writes a row; the skills write the rows for it.
- ``"mixed"``: both, in the same program: skills for the easy parts, raw rows
  and the kinematics helpers (``fk``/``ik``) for the rest.

Instruments (``read_plate``, ``report_concentration``, ``incubate``,
``inspect``, ``get_clock``) are available at every level. They are not robot
motions: each is recorded as an event at the row index where it happened,
followed by the rows of the arm holding still while the instrument works.

The plunger (entry 14) is the volume the pipette's plunger has drawn, in uL.
Whenever the target changes, the change is drawn (or pushed out) at the
start of that control step, through biolab's own liquid model: liquid comes
in only if the tip is inside a well and below its surface (else air), and
what is pushed out outside every well is spilled. Without the pipette held
in the right gripper the plunger moves but no liquid does.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from lab2rob.certify.run import _biolab
from lab2rob.registry.predicates import PREDICATES
from lab2rob.sim.instance import (ACTION_NAMES, N_ACTION, PLUNGER, InstanceMismatch,
                                  build_env, close_env, load_instance, model_sha,
                                  state_sha)

LEVELS = ("joints", "skills", "mixed")
#: Longest trajectory a session records or replays (1 h of robot time).
MAX_ROWS = 180_000
MAX_EVENTS = 10_000
#: Seconds the arm holds still while an instrument works (biolab's own times).
INSTRUMENT_HOLD_S = {"read_plate": 2.0, "incubate": 0.5, "inspect": 1.0,
                     "report_concentration": 0.0}
PLUNGER_HOLD_S = 0.4  # the plunger takes a moment (biolab's aspirate/dispense)
#: Well approach used by aspirate/dispense(slot=...), as in certification.
APPROACH_M = 0.03
SUBMERGE_M = 0.001
FLOOR_MARGIN_M = 0.0004
TRAVEL_SPEED = 0.15
WELL_SPEED = 0.04
TRAJECTORY_FORMAT = "lab2rob-trajectory/1"


class LevelError(Exception):
    """A call the session's level does not allow."""


def _allowed(*levels):
    def wrap(fn):
        def inner(self, *args, **kwargs):
            if self.level not in levels:
                raise LevelError(f"{fn.__name__}() is not available at level "
                                 f"{self.level!r} (allowed: {', '.join(levels)})")
            return fn(self, *args, **kwargs)
        inner.__name__, inner.__doc__ = fn.__name__, fn.__doc__
        return inner
    return wrap


class Session:
    """A frozen scene the agent can reset, drive, inspect and record."""

    def __init__(self, instance, level: str = "mixed"):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        self.instance = load_instance(instance)
        self.level = level
        self.env = None
        self.reset()

    # ----------------------------------------------------------------- episode
    def reset(self) -> dict:
        """Back to the frozen initial state (a fresh build: a couple of seconds)."""
        if self.env is not None:
            close_env(self.env)
        self.env = build_env(self.instance)
        env = self.env
        self._substeps = round(self.instance["control_dt"] / env.model.opt.timestep)
        self._lo = np.append(env.model.actuator_ctrlrange[:, 0], 0.0)
        self._hi = np.append(env.model.actuator_ctrlrange[:, 1],
                             env.pipette.capacity)
        self._stroke = 0.0
        self._plunger_target = 0.0
        self._last_plunger: dict = {}
        self._rows: list = []
        self._events: list = []
        self._quiet = False  # instrument calls: biolab's own waiting is ours
        self._sim_error = ""
        env.step = self._env_step  # every biolab motion now ticks through us
        if model_sha(env.model) != self.instance["model_sha"]:
            raise InstanceMismatch("model hash differs from the instance")
        if self.state_sha() != self.instance["init_sha"]:
            raise InstanceMismatch("initial state hash differs from the instance")
        return self.state()

    def close(self) -> None:
        if self.env is not None:
            close_env(self.env)

    # ----------------------------------------------------------------- ticking
    def _env_step(self, seconds: float) -> None:
        """biolab's env.step(seconds), routed through the recorder."""
        if self._quiet:
            return
        for _ in range(max(1, round(float(seconds) / self.instance["control_dt"]))):
            self._tick()

    def _apply_plunger(self) -> None:
        env = self.env
        delta = self._plunger_target - self._stroke
        if delta == 0.0:
            return
        self._stroke = self._plunger_target
        if not env.holding_pipette():
            self._last_plunger = {"volume_ul": 0.0,
                                  "note": "the right gripper is not holding the pipette"}
            return
        wells = list(env.wells.values())
        if delta > 0:
            result = env.pipette.aspirate(wells, env.tip(), delta)
        else:
            result = env.pipette.dispense(wells, env.tip(), -delta)
        env._sync_liquids()
        self._last_plunger = result

    def _tick(self) -> None:
        if self._sim_error:
            return
        if len(self._rows) >= MAX_ROWS:
            raise RuntimeError(f"trajectory cap reached ({MAX_ROWS} rows)")
        env = self.env
        self._apply_plunger()
        self._rows.append(np.append(env.data.ctrl, self._plunger_target))
        mujoco.mj_step(env.model, env.data, self._substeps)
        env._after_tick()
        if env.on_tick is not None:
            env.on_tick()
        warnings = env.data.warning
        for kind in (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQPOS,
                     mujoco.mjtWarning.mjWARN_BADQVEL):
            if warnings[kind].number > 0:
                self._sim_error = f"unstable simulation ({kind.name})"

    def _set_row(self, a) -> None:
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        if a.shape != (N_ACTION,):
            raise ValueError(f"an action row has {N_ACTION} entries, got {a.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError("action row has a non-finite entry")
        a = np.clip(a, self._lo, self._hi)
        self.env.data.ctrl[:] = a[:PLUNGER]
        self._plunger_target = float(a[PLUNGER])

    # --------------------------------------------------------------- low level
    @_allowed("joints", "mixed")
    def step(self, a, n: int = 1) -> bool:
        """Hold one row of 15 targets for n control steps. False after a sim error."""
        self._set_row(a)
        for _ in range(int(n)):
            self._tick()
        return not self._sim_error

    def actuators(self) -> list:
        """The 15 action entries in order: name, unit, range."""
        units = ["rad"] * 6 + ["m per finger"] + ["rad"] * 6 + ["m per finger"] + ["uL"]
        return [{"index": i, "name": name, "unit": units[i],
                 "range": [float(self._lo[i]), float(self._hi[i])]}
                for i, name in enumerate(ACTION_NAMES)]

    def command(self) -> np.ndarray:
        """The targets in effect now (a row that holds the current pose)."""
        return np.append(self.env.data.ctrl, self._plunger_target)

    # ------------------------------------------------------------------ skills
    @_allowed("skills", "mixed")
    def move_to(self, arm: str, position, yaw: float = 0.0, tilt: float = 0.0,
                speed: float = TRAVEL_SPEED) -> bool:
        """Move a gripper's pinch point in a straight line (biolab's move_to)."""
        return bool(self.env.move_to(arm, np.asarray(position, dtype=float),
                                     float(yaw), float(tilt), float(speed)))

    @_allowed("skills", "mixed")
    def move_joints(self, arm: str, q) -> None:
        """Move one arm's 6 joints to q at joint speed, then settle."""
        self.env.move_joints(arm, np.asarray(q, dtype=float))

    @_allowed("skills", "mixed")
    def home(self, arm: str) -> None:
        self.env.home(arm)

    @_allowed("skills", "mixed")
    def set_gripper(self, arm: str, opening: float, wait: float = 0.6) -> None:
        """Distance between the finger pads in m (0 closed .. 0.082 open)."""
        self.env.set_gripper(arm, float(opening), float(wait))

    @_allowed("skills", "mixed")
    def move_tip_into(self, slot: str) -> bool:
        """Bring the held pipette's tip just under the surface of a well."""
        well = self._well(slot)
        grip_above = self._grip_above()
        x, y = float(well.center[0]), float(well.center[1])
        top = well.floor + well.depth + APPROACH_M
        tip_z = max(well.floor + FLOOR_MARGIN_M, well.surface - SUBMERGE_M)
        tip_z = max(well.floor + 0.0002, tip_z)
        return (self.move_to("right", [x, y, top + grip_above], 0.0, 0.0, TRAVEL_SPEED)
                and self.move_to("right", [x, y, tip_z + grip_above], 0.0, 0.0,
                                 WELL_SPEED))

    @_allowed("skills", "mixed")
    def withdraw(self, slot: str) -> bool:
        """Lift the tip straight out of a well to the travel height."""
        well = self._well(slot)
        x, y = float(well.center[0]), float(well.center[1])
        top = well.floor + well.depth + APPROACH_M
        return self.move_to("right", [x, y, top + self._grip_above()], 0.0, 0.0,
                            TRAVEL_SPEED)

    def _well(self, slot: str):
        if slot not in self.env.wells:
            raise KeyError(f"no well {slot!r}; wells: {sorted(self.env.wells)}")
        return self.env.wells[slot]

    @staticmethod
    def _grip_above() -> float:
        protein, _ = _biolab()
        return protein.GRIP_ABOVE_TIP

    def _plunger_skill(self, volume_ul: float, slot: str | None, sign: float) -> dict:
        if slot is not None and not self.move_tip_into(slot):
            return {"volume_ul": 0.0, "note": f"could not reach well {slot!r}"}
        target = self._plunger_target + sign * max(0.0, float(volume_ul))
        self._plunger_target = float(np.clip(target, 0.0, self._hi[PLUNGER]))
        self._last_plunger = {"volume_ul": 0.0, "note": "plunger did not move"}
        self._env_step(PLUNGER_HOLD_S)
        result = dict(self._last_plunger)
        if slot is not None:
            self.withdraw(slot)
        return _plain(result)

    @_allowed("skills", "mixed")
    def aspirate(self, volume_ul: float, slot: str | None = None) -> dict:
        """Draw with the plunger; with slot, go into that well first and back out."""
        return self._plunger_skill(volume_ul, slot, 1.0)

    @_allowed("skills", "mixed")
    def dispense(self, volume_ul: float, slot: str | None = None) -> dict:
        """Push out with the plunger; with slot, go into that well first and back out."""
        return self._plunger_skill(volume_ul, slot, -1.0)

    def wait(self, seconds: float) -> None:
        """Hold the current targets for a while (every level)."""
        self._env_step(float(np.clip(seconds, 0.0, 60.0)))

    # ------------------------------------------------------------- instruments
    def _instrument(self, call: str, args: dict, record: bool = True):
        if len(self._events) >= MAX_EVENTS:
            raise RuntimeError(f"event cap reached ({MAX_EVENTS})")
        fn = getattr(self.env, call, None)
        if call == "report_concentration":
            self.env.reported = float(args["mg_per_ml"])
            result = None
        elif fn is None:
            raise LevelError(f"this scene ({self.instance['family']}) has no {call}()")
        else:
            self._quiet = True
            try:
                result = fn(**args)
            finally:
                self._quiet = False
        self._events.append({"at": len(self._rows), "call": call, "args": args})
        if record and INSTRUMENT_HOLD_S[call] > 0:
            self._env_step(INSTRUMENT_HOLD_S[call])
        return _plain(result)

    def read_plate(self) -> dict:
        """Absorbance of every plate well with at least 50 uL (protein scenes)."""
        return self._instrument("read_plate", {})

    def report_concentration(self, mg_per_ml: float) -> None:
        """Report the sample's concentration; the last report counts."""
        self._instrument("report_concentration", {"mg_per_ml": float(mg_per_ml)})

    def incubate(self, hours: float) -> None:
        """Incubate the dishes (culture scenes); at most 24 h per call."""
        self._instrument("incubate", {"hours": float(hours)})

    def inspect(self, dish: str) -> dict:
        """Look at a dish under the microscope (culture scenes)."""
        return self._instrument("inspect", {"dish": str(dish)})

    def get_clock(self) -> float:
        return float(getattr(self.env, "clock", 0.0))

    # ------------------------------------------------------- state and sensing
    def state(self) -> dict:
        """Privileged view: joints, targets, tool poses, pipette, wells."""
        env, d = self.env, self.env.data
        return {
            "qpos": {arm: d.qpos[env._qadr[arm]].copy() for arm in ("left", "right")},
            "ctrl": self.command(),
            "ee": {arm: _plain(env.ee_pose(arm)) for arm in ("left", "right")},
            "tip": env.tip(),
            "holding_pipette": bool(env.holding_pipette()),
            "pipette": {"volume_ul": env.pipette.volume, "stroke_ul": self._stroke,
                        "capacity_ul": env.pipette.capacity},
            "wells": self.wells(),
            "steps": len(self._rows),
            "time_s": float(d.time),
        }

    def wells(self) -> dict:
        """Every well: rim centre, floor, depth, opening, current volume."""
        out = {}
        for name, w in self.env.wells.items():
            out[name] = {"center": np.array([*w.center, w.floor + w.depth]),
                         "floor_z": w.floor, "depth": w.depth, "opening": 2 * w.half,
                         "volume_ul": w.volume, "surface_z": w.surface}
        return out

    def tip(self) -> np.ndarray:
        return self.env.tip()

    def ee_pose(self, arm: str) -> dict:
        return _plain(self.env.ee_pose(arm))

    def cameras(self) -> list:
        m = self.env.model
        return [m.camera(i).name for i in range(m.ncam)]

    def render_frame(self, camera: str = "overhead", width: int = 640,
                     height: int = 480) -> np.ndarray:
        return self.env.render(camera=camera, width=width, height=height)

    def fk(self, arm: str, q) -> dict:
        """Pinch-point position and rotation for joint angles q (no physics)."""
        env = self.env
        m, d = env.model, env._ik_data
        d.qpos[:] = env.data.qpos
        d.qpos[env._qadr[arm]] = np.asarray(q, dtype=float)
        mujoco.mj_kinematics(m, d)
        site = env._site[arm]
        return {"pos": d.site_xpos[site].copy(), "xmat": d.site_xmat[site].reshape(3, 3).copy()}

    def ik(self, arm: str, position, yaw: float = 0.0, tilt: float = 0.0, q0=None):
        """Joint angles that put the pinch point at position (no physics).

        Returns (q, error); error above about 1e-3 means unreachable.
        """
        env = self.env
        q0 = env.data.qpos[env._qadr[arm]].copy() if q0 is None else np.asarray(q0, float)
        return env._ik(arm, np.asarray(position, dtype=float), float(yaw), q0, float(tilt))

    # ----------------------------------------------------------------- grading
    def snapshot(self) -> dict:
        """Liquid state in the predicates' state protocol."""
        env = self.env
        return {"wells": {name: {"volume_ul": round(w.volume, 3), "conc": round(w.conc, 6)}
                          for name, w in env.wells.items()},
                "spilled_ul": round(env.pipette.spilled, 3),
                "pipette_ul": round(env.pipette.volume, 3),
                "reported": getattr(env, "reported", None),
                "clock_h": round(float(getattr(env, "clock", 0.0)), 6)}

    def check(self) -> list:
        """The instance's end-state checks on the current state."""
        snap = self.snapshot()
        rows = []
        for spec in self.instance.get("checks", []):
            pred = spec["pred"]
            if pred == "no_spill":
                result = PREDICATES["no_spill"](snap, spec.get("tol", 0.0))
            elif pred in ("volume", "concentration"):
                result = PREDICATES[pred](snap, spec["well"], spec["target"], spec["tol"])
            else:
                raise ValueError(f"unknown check {pred!r}")
            rows.append({**spec, "passed": bool(result.passed), "detail": result.detail})
        return rows

    def state_sha(self) -> str:
        return state_sha(self.env, {"stroke": self._stroke})

    def status(self) -> dict:
        checks = self.check()
        success = bool(checks) and all(c["passed"] for c in checks) and not self._sim_error
        return {"success": success, "sim_error": self._sim_error,
                "checks": checks, "steps": len(self._rows),
                "events": len(self._events), "state_sha": self.state_sha(),
                "instance_init_sha": self.instance["init_sha"]}

    # ------------------------------------------------------ record and replay
    def trajectory(self) -> dict:
        """Everything done since the last reset: rows + instrument events."""
        rows = np.array(self._rows, dtype=np.float64).reshape(-1, N_ACTION)
        return {"rows": rows, "events": [dict(e) for e in self._events]}

    def replay(self, trajectory) -> dict:
        """Reset, then run a trajectory open-loop; returns status()."""
        traj = load_trajectory(trajectory) if isinstance(trajectory, (str, Path)) else trajectory
        rows = np.asarray(traj["rows"], dtype=np.float64).reshape(-1, N_ACTION)
        events = sorted(traj.get("events", []), key=lambda e: e["at"])
        if len(rows) > MAX_ROWS or len(events) > MAX_EVENTS:
            raise ValueError("trajectory over the row or event cap")
        if events and not all(0 <= e["at"] <= len(rows) for e in events):
            raise ValueError("event index outside the trajectory")
        self.reset()
        pending = list(events)
        for i, row in enumerate(rows):
            while pending and pending[0]["at"] == i:
                ev = pending.pop(0)
                self._instrument(ev["call"], dict(ev.get("args", {})), record=False)
            self._set_row(row)
            self._tick()
            if self._sim_error:
                break
        for ev in pending:
            if not self._sim_error:
                self._instrument(ev["call"], dict(ev.get("args", {})), record=False)
        return self.status()

    def save_trajectory(self, trajectory=None, path="/app/output/trajectory.npz") -> Path:
        traj = self.trajectory() if trajectory is None else trajectory
        return save_trajectory(traj, path, self.instance["init_sha"])


def save_trajectory(traj: dict, path, init_sha: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(traj["rows"], dtype=np.float64).reshape(-1, N_ACTION)
    meta = {"format": TRAJECTORY_FORMAT, "action": list(ACTION_NAMES),
            "instance_init_sha": init_sha, "events": list(traj.get("events", []))}
    with open(path, "wb") as handle:
        np.savez(handle, rows=rows, meta=np.array(json.dumps(meta)))
    return path


def load_trajectory(path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        rows = np.asarray(data["rows"], dtype=np.float64)
        meta = json.loads(str(data["meta"]))
    if meta.get("format") != TRAJECTORY_FORMAT:
        raise ValueError(f"not a {TRAJECTORY_FORMAT} file")
    return {"rows": rows, "events": meta.get("events", []),
            "instance_init_sha": meta.get("instance_init_sha", "")}


def _plain(value):
    """numpy -> plain Python, recursively (results the agent reads)."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value
