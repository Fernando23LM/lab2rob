"""Frozen scene instances: everything needed to rebuild one exact scene.

An instance is a JSON document (``lab2rob-instance/1``) that pins the biolab
family, the layout (racks and placed labware, exact floats), the prefill, the
seed, the action layout, the end-state checks and two hashes:

- ``model_sha``: the compiled MuJoCo model right after the scene is built;
- ``init_sha``: the full simulation state (physics, liquids, pipette, clocks,
  noise generator) right after the scene is built.

A Session refuses to run when a rebuilt scene does not reproduce both hashes
(``InstanceMismatch``), so a trajectory is only ever graded against the
scene it was made in.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from lab2rob.build import labware as labware_mod
from lab2rob.certify.run import _biolab, _cert_env_class

SCHEMA = "lab2rob-instance/1"

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
#: One action row: the 14 actuator targets in MuJoCo order + the plunger.
ACTION_NAMES = tuple([f"left_{j}" for j in ARM_JOINTS] + ["left_gripper"]
                     + [f"right_{j}" for j in ARM_JOINTS] + ["right_gripper"]
                     + ["pipette_plunger"])
N_ACTION = len(ACTION_NAMES)
PLUNGER = N_ACTION - 1


class InstanceMismatch(Exception):
    """A rebuilt scene does not reproduce the instance's recorded hashes."""


def _scene_module():
    _biolab()
    import biolab.scene as scene
    return scene


def _layout_base(layout) -> type:
    if isinstance(layout, labware_mod._WithLabware):
        return type(layout).__mro__[2]
    return type(layout)


def layout_to_json(layout) -> dict:
    """Family layout (+ placed labware) -> plain JSON (floats kept exact)."""
    scene = _scene_module()
    base = _layout_base(layout)
    racks = {}
    for f in dataclasses.fields(base):
        value = getattr(layout, f.name)
        if not isinstance(value, scene.Rack):
            raise ValueError(f"layout field {f.name!r} is not a Rack")
        racks[f.name] = {"name": value.name, "xy": [float(v) for v in value.xy],
                         "nx": int(value.nx), "ny": int(value.ny),
                         "yaw": float(value.yaw)}
    placed = [{"uid": p.uid, "asset": p.asset, "xy": [float(v) for v in p.xy],
               "yaw": float(p.yaw)} for p in getattr(layout, "labware", [])]
    return {"class": base.__name__, "racks": racks, "labware": placed}


def layout_from_json(doc: dict):
    """Inverse of layout_to_json."""
    protein, culture = _biolab()
    scene = _scene_module()
    classes = {"ProteinLayout": protein.ProteinLayout,
               "CultureLayout": culture.CultureLayout}
    if doc.get("class") not in classes:
        raise ValueError(f"unknown layout class {doc.get('class')!r}")
    racks = {name: scene.Rack(r["name"], tuple(r["xy"]), nx=r["nx"], ny=r["ny"],
                              yaw=r["yaw"])
             for name, r in doc["racks"].items()}
    layout = classes[doc["class"]](**racks)
    for p in doc.get("labware", []):
        layout = labware_mod.with_labware(layout, labware_mod.PlacedLabware(
            uid=p["uid"], asset=p["asset"], xy=tuple(p["xy"]), yaw=p["yaw"]))
    return layout


def build_env(instance: dict):
    """A fresh biolab env for the instance (the scene's own reset included)."""
    layout = layout_from_json(instance["layout"])
    env_cls, _ = _cert_env_class(instance["family"])
    return env_cls(layout, instance.get("prefill", {}), seed=instance["seed"])


def model_sha(model) -> str:
    buf = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buf)
    return hashlib.sha256(buf.tobytes()).hexdigest()[:16]


def _sim_state(env, extra: dict | None = None) -> dict:
    """Everything that can change the outcome, as exact JSON-able values."""
    state = {
        "wells": {name: [w.volume, w.conc, sorted(w.extra.items())]
                  for name, w in sorted(env.wells.items())},
        "pipette": [env.pipette.volume, env.pipette.conc, env.pipette.spilled,
                    sorted(env.pipette.extra.items())],
        "clock": float(getattr(env, "clock", 0.0)),
        "reported": getattr(env, "reported", None),
        "noise": env._noise.bit_generator.state,
    }
    for name in ("attached", "starved", "crowded"):
        if hasattr(env, name):
            state[name] = sorted(getattr(env, name).items())
    if extra:
        state.update(extra)
    return state


def state_sha(env, extra: dict | None = None) -> str:
    d = env.data
    h = hashlib.sha256()
    for arr in (d.qpos, d.qvel, d.act, d.ctrl, d.qacc_warmstart):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    h.update(np.float64(d.time).tobytes())
    h.update(json.dumps(_sim_state(env, extra), sort_keys=True,
                        default=float).encode("utf-8"))
    return h.hexdigest()[:16]


def make_instance(built, prefill: dict, seed: int, checks: list,
                  source: dict | None = None, instruction: str | None = None) -> dict:
    """Freeze a built scene: rebuild it once and record its hashes."""
    instance = {
        "schema": SCHEMA,
        "family": built.family,
        "seed": int(seed),
        "layout": layout_to_json(built.layout),
        "prefill": {k: float(v) for k, v in sorted(prefill.items())},
        "action": list(ACTION_NAMES),
        "n_action": N_ACTION,
        "rule": "final",
        "checks": list(checks),
        "source": source or {},
        "instruction": instruction or "",
    }
    env = build_env(instance)
    try:
        instance["control_dt"] = float(scene_control_dt())
        instance["physics_dt"] = float(env.model.opt.timestep)
        instance["pipette_capacity_ul"] = float(env.pipette.capacity)
        instance["model_sha"] = model_sha(env.model)
        instance["init_sha"] = state_sha(env, {"stroke": 0.0})
    finally:
        close_env(env)
    return instance


def scene_control_dt() -> float:
    _biolab()
    import biolab.env as benv
    return benv.CONTROL_DT


def close_env(env) -> None:
    renderer = getattr(env, "_renderer", None)
    if renderer is not None:
        renderer.close()
        env._renderer = None


def save_instance(instance: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(instance, indent=1), encoding="utf-8")
    return path


def load_instance(source) -> dict:
    """An instance dict, or a path to its JSON file."""
    if isinstance(source, dict):
        instance = source
    else:
        instance = json.loads(Path(source).read_text(encoding="utf-8"))
    if instance.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} document: {instance.get('schema')!r}")
    return instance
