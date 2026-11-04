#!/usr/bin/env python3
"""A Robocurve task: one frozen bench scene, two YAM arms at 20 Hz, and its scorer.

This file is copied into every task folder next to ``instance.json`` (the frozen
scene and its scoring parameters), its family module (``weigh_family.py``,
``centrifuge_family.py``, ...) and ``sim/`` (the simulator it runs on). It
depends only on numpy, mujoco and, for videos, imageio with ffmpeg.

    python task.py score --traj trajectory.npz             # two fresh replays, one score
    python task.py demo --out .                            # scripted demo: demo.npz, demo.mp4
    python task.py video --traj trajectory.npz --out run.mp4
    python task.py setup --out setup.png                   # the setup screenshot

A trajectory is the rows a policy sent: (T, 14) joint and gripper targets, one
row per 50 ms control step, in ACTION_NAMES order. In Python, drive a trial with
``TaskSession``: ``reset()``, ``observe()``, ``step(action)``, ``trajectory()``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sim"))
sys.path.insert(0, str(HERE))

import mujoco  # noqa: E402

from biolab.scene import ARM_JOINTS, ARMS  # noqa: E402
from rc_base import CONTROL_DT, CONTROL_HZ, model_sha  # noqa: E402,F401

ACTION_NAMES = tuple([f"left_{j}" for j in ARM_JOINTS] + ["left_gripper"]
                     + [f"right_{j}" for j in ARM_JOINTS] + ["right_gripper"])
SETTLE_STEPS = 20  # the scorer holds the last targets this long (1 s) before reading
TRAJECTORY_FORMAT = "robocurve-bench-trajectory/1"


def family_module(name: str):
    return importlib.import_module(f"{name}_family")


# ------------------------------------------------------------------------- session
class TaskSession:
    """The frozen scene: reset, act at 20 Hz, observe, record, replay, score."""

    def __init__(self, instance=HERE / "instance.json", check: bool = True):
        self.instance = instance if isinstance(instance, dict) else json.loads(
            Path(instance).read_text(encoding="utf-8"))
        self.family = family_module(self.instance["family"])
        self.run_limit = int(self.instance["run_limit_steps"])
        self._check = check
        self.env = None
        self.reset()

    def reset(self) -> None:
        """Back to the frozen start (a fresh build, a few seconds)."""
        if self.env is not None and self.env._renderer is not None:
            self.env._renderer.close()
        self._live = False
        self.env = self.family.make_env(self.instance, self)
        m = self.env.model
        if m.nu != len(ACTION_NAMES):
            raise RuntimeError(f"the scene has {m.nu} actuators, not the arms' {len(ACTION_NAMES)}")
        self._substeps = round(CONTROL_DT / m.opt.timestep)
        self._lo, self._hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        self._rows: list = []
        self._live = True
        self._error = ""
        if self._check:
            for key, value in (("model_sha", model_sha(m)), ("init_sha", self.state_sha())):
                want = self.instance.get(key)
                if want and want != value:
                    raise RuntimeError(f"{key} {value} differs from the instance ({want}): "
                                       f"not the frozen scene")

    # one control step
    def _tick(self) -> None:
        if self._error:
            return
        if len(self._rows) >= self.run_limit:
            self._error = f"run limit reached ({self.run_limit} steps)"
            return
        env = self.env
        self._rows.append(env.data.ctrl.copy())
        mujoco.mj_step(env.model, env.data, self._substeps)
        env._after_tick()
        w = env.data.warning
        if any(w[k].number for k in (mujoco.mjtWarning.mjWARN_BADQACC,
                                     mujoco.mjtWarning.mjWARN_BADQPOS,
                                     mujoco.mjtWarning.mjWARN_BADQVEL)):
            self._error = "unstable simulation"

    def step(self, action, n: int = 1) -> bool:
        """Hold 14 joint and gripper targets for n control steps (50 ms each)."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (len(ACTION_NAMES),) or not np.all(np.isfinite(a)):
            raise ValueError(f"an action is {len(ACTION_NAMES)} finite numbers")
        self.env.data.ctrl[:] = np.clip(a, self._lo, self._hi)
        for _ in range(int(n)):
            self._tick()
        return not self._error

    def command(self) -> np.ndarray:
        """The targets in effect: a row that holds the current pose."""
        return self.env.data.ctrl.copy()

    def actuators(self) -> list:
        m = self.env.model
        return [{"index": i, "name": name, "unit": "m per finger" if "gripper" in name else "rad",
                 "range": [float(m.actuator_ctrlrange[i, 0]), float(m.actuator_ctrlrange[i, 1])]}
                for i, name in enumerate(ACTION_NAMES)]

    def cameras(self) -> list:
        m = self.env.model
        return [m.camera(i).name for i in range(m.ncam)]

    def render(self, camera: str = "bench", width: int = 640, height: int = 480) -> np.ndarray:
        return self.env.render(camera=camera, width=width, height=height)

    def observe(self, cameras=("bench", "left/wrist_cam", "right/wrist_cam"),
                width: int = 320, height: int = 240) -> dict:
        """What a policy sees: camera images and its own joints and grippers."""
        env = self.env
        q = []
        for arm in ARMS:
            q += [float(v) for v in env.data.qpos[env._qadr[arm]]]
            q.append(float(env.ee_pose(arm)["opening"]))
        return {"images": {c: self.render(c, width, height) for c in cameras},
                "joints": np.array(q), "step": len(self._rows)}

    # skills for scripted demos (the reference solution); they write rows like any policy
    def api(self) -> dict:
        return self.family.api(self.env)

    def trajectory(self) -> dict:
        return {"rows": np.array(self._rows, dtype=np.float64).reshape(-1, len(ACTION_NAMES))}

    def replay(self, traj) -> dict:
        """Reset, run the rows open-loop, settle, score."""
        rows = np.asarray(traj["rows"], dtype=np.float64).reshape(-1, len(ACTION_NAMES))
        self.reset()
        if len(rows) > self.run_limit:
            return self.score(over_limit=True)
        for row in rows:
            self.env.data.ctrl[:] = row
            self._tick()
            if self._error:
                break
        return self.score()

    def state_sha(self) -> str:
        d = self.env.data
        h = hashlib.sha256()
        for arr in (d.qpos, d.qvel, d.act, d.ctrl, d.qacc_warmstart):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        h.update(np.float64(d.time).tobytes())
        h.update(json.dumps(self.family.state_extra(self.env)).encode())
        return h.hexdigest()[:16]

    # ---------------------------------------------------------------------- scoring
    def measure(self) -> dict:
        """What a person would measure on the bench (the family says what)."""
        return self.family.measure(self.env)

    def score(self, over_limit: bool = False) -> dict:
        """The task's score, exactly as task.json's scoring text states it."""
        if over_limit:
            return {"score": 0.0, "reason": f"over the run limit ({self.run_limit} steps)"}
        hold = self.command()
        self._live = False  # the settle window is the scorer's, not the policy's
        for _ in range(SETTLE_STEPS):
            self.env.data.ctrl[:] = hold
            mujoco.mj_step(self.env.model, self.env.data, self._substeps)
            self.env._after_tick()
        self._live = True
        got = self.measure()
        if self._error and "unstable" in self._error:
            value, reason, extra = 0.0, self._error, {}
        else:
            value, reason, extra = self.family.score(got, self.instance)
        return {"score": value, "reason": reason, **extra, "steps": len(self._rows),
                "state_sha": self.state_sha(), **got}


# --------------------------------------------------------------------- trajectories
def save_trajectory(traj: dict, path) -> Path:
    path = Path(path)
    rows = np.asarray(traj["rows"], dtype=np.float64).reshape(-1, len(ACTION_NAMES))
    meta = {"format": TRAJECTORY_FORMAT, "control_hz": CONTROL_HZ, "action": list(ACTION_NAMES)}
    with open(path, "wb") as handle:
        np.savez(handle, rows=rows, meta=np.array(json.dumps(meta)))
    return path


def load_trajectory(path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("format") != TRAJECTORY_FORMAT:
            raise ValueError(f"{path}: not a {TRAJECTORY_FORMAT} file")
        return {"rows": np.asarray(data["rows"], dtype=np.float64)}


def write_video(session: TaskSession, rows, path, camera: str = "bench", every: int = 2,
                width: int = 640, height: int = 480) -> Path:
    """H.264 MP4 of a replay, real time (every 2nd step at 10 fps)."""
    import imageio.v2 as imageio
    session.reset()
    writer = imageio.get_writer(str(path), fps=CONTROL_HZ / every, codec="libx264",
                                pixelformat="yuv420p", ffmpeg_params=["-movflags", "+faststart"],
                                macro_block_size=8)
    try:
        writer.append_data(session.render(camera, width, height))
        for i, row in enumerate(rows):
            session.step(row)
            if i % every == every - 1:
                writer.append_data(session.render(camera, width, height))
        hold = session.command()
        for i in range(SETTLE_STEPS):
            if len(session._rows) < session.run_limit:
                session.step(hold)
            if i % every == every - 1:
                writer.append_data(session.render(camera, width, height))
    finally:
        writer.close()
    return Path(path)


# ---------------------------------------------------------------------------- demo
def demo_script(session: TaskSession) -> str:
    return session.family.demo_script()


def run_demo(session: TaskSession) -> dict:
    """The scripted reference solution, recorded as rows like any policy's."""
    session.reset()
    namespace = dict(session.api())
    namespace.update(np=np, print=lambda *a, **k: None)
    exec(demo_script(session), namespace)  # noqa: S102 - our own reference script
    return session.trajectory()


# ----------------------------------------------------------------------------- CLI
def _replay_main(args) -> int:
    session = TaskSession(args.instance)
    print(json.dumps(session.replay(load_trajectory(args.traj))))
    return 0


def _score_main(args) -> int:
    """Each replay in its own fresh process; the score counts only if they agree."""
    results = []
    for _ in range(args.repeat):
        out = subprocess.run([sys.executable, str(Path(__file__).resolve()), "replay",
                              "--instance", str(args.instance), "--traj", str(args.traj)],
                             capture_output=True, text=True, check=True)
        results.append(json.loads(out.stdout.strip().splitlines()[-1]))
    agree = len({r.get("state_sha") for r in results}) == 1
    final = dict(results[0], replays=len(results), deterministic=agree)
    if not agree:
        final.update(score=0.0, reason="replays disagree")
    print(json.dumps(final, indent=2))
    return 0


def _demo_main(args) -> int:
    session = TaskSession(args.instance)
    traj = run_demo(session)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_trajectory(traj, out / "demo.npz")
    result = session.replay(traj)
    if not args.no_video:
        write_video(session, traj["rows"], out / "demo.mp4", camera=session.family.CAMERA)
    print(json.dumps(result))
    return 0


def _video_main(args) -> int:
    session = TaskSession(args.instance)
    write_video(session, load_trajectory(args.traj)["rows"], args.out, camera=args.camera)
    return 0


def _setup_main(args) -> int:
    import imageio.v2 as imageio
    session = TaskSession(args.instance)
    imageio.imwrite(args.out, session.render(args.camera, 1280, 960))
    return 0


def _hashes_main(args) -> int:
    session = TaskSession(args.instance, check=False)
    print(json.dumps({"model_sha": model_sha(session.env.model), "init_sha": session.state_sha()}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("replay", _replay_main), ("score", _score_main), ("demo", _demo_main),
                     ("video", _video_main), ("setup", _setup_main), ("hashes", _hashes_main)):
        p = sub.add_parser(name)
        p.add_argument("--instance", default=str(HERE / "instance.json"))
        p.set_defaults(func=fn)
        if name in ("replay", "score", "video"):
            p.add_argument("--traj", required=True)
        if name == "score":
            p.add_argument("--repeat", type=int, default=2)
        if name in ("demo", "video", "setup"):
            p.add_argument("--out", required=True)
        if name in ("video", "setup"):
            p.add_argument("--camera", default="bench")
        if name == "demo":
            p.add_argument("--no-video", action="store_true")
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
