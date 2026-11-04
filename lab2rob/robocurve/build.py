"""Build a Robocurve task folder (robocurve/benchmark-tasks, tasks/<name>/).

    python -m lab2rob.robocurve build-weigh --out runs/robocurve/weigh-nacl \\
        --protocol Protocol.io-2332

The folder is self-contained: ``task.py`` (the runtime and scorer), ``sim/``
(the simulator files it runs on), ``instance.json`` (the frozen scene and the
scoring thresholds), ``demo.npz`` and ``demo.mp4`` (the scripted full-score
demo), ``setup.png``, ``task.json`` and ``README.md``. Building fails unless the
demo scores 1.0 in two fresh replays that agree and doing nothing scores 0.

What it cannot make: the Dream-1 run video (Robocurve runs Dream-1) and the
media ID (``tools/upload_media.py`` in the Robocurve repository writes it).
``check_task`` lists what is still missing before a pull request.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUNTIME = Path(__file__).resolve().parent / "runtime"
#: Asset folders (under runtime/assets) each family's scene loads.
FAMILY_ASSETS = {"weigh": (), "centrifuge": ("centrifuge_02",), "prep_spin": ("centrifuge_02",)}
#: Runtime family modules each family imports besides its own.
FAMILY_MODULES = {"prep_spin": ("centrifuge_family.py",)}
#: biolab files each family needs besides BIOLAB_FILES.
FAMILY_BIOLAB = {"prep_spin": ("liquid.py", "protein.py")}
BIOLAB = REPO / "Code-as-a-Policy-Evolving-Skill"
BIOLAB_FILES = ("__init__.py", "env.py", "scene.py", "judge.py", "api.py")
YAM_DIR = Path("third_party") / "mujoco_menagerie" / "i2rt_yam"

REQUIRED = ("title", "category", "instruction", "setup", "scoring", "run_limit",
            "fidelity", "scores", "media")
CATEGORIES = ("Impact on jobs", "Impact on AI timelines", "Safety and alignment",
              "Societal relevance", "Other")

#: Scoring thresholds (grams). Full credit spans about three of the largest
#: grains (0.05 g each): salt leaves the vial in small avalanches.
WEIGH_SCORING = {"full_g": 0.15, "three_quarter_g": 0.30, "half_g": 0.50,
                 "some_salt_g": 0.10, "spill_cap_g": 0.20}


def weigh_scoring(full_g: float | None) -> dict:
    """Score tiers around the judge's tolerance: full, twice it, at least 0.5 g."""
    full = float(full_g) if full_g else WEIGH_SCORING["full_g"]
    return {"full_g": round(full, 3), "three_quarter_g": round(2 * full, 3),
            "half_g": round(max(WEIGH_SCORING["half_g"], 3 * full), 3),
            "some_salt_g": WEIGH_SCORING["some_salt_g"], "spill_cap_g": WEIGH_SCORING["spill_cap_g"]}
#: Runs get this much more than the scripted demo needs, rounded up.
RUN_LIMIT_MARGIN = 1.6
RUN_LIMIT_ROUND = 600  # 30 s at 20 Hz
TOTAL_SALT_G = 2.40


class BuildError(Exception):
    pass


# --------------------------------------------------------------------------- vendor
def vendor(out: Path, family: str = "weigh") -> dict:
    """Copy the runtime and the simulator files into the task folder."""
    out.mkdir(parents=True, exist_ok=True)
    for name in ("task.py", "rc_base.py", f"{family}_family.py", *FAMILY_MODULES.get(family, ())):
        shutil.copy2(RUNTIME / name, out / name)
    if (out / "assets").exists():
        shutil.rmtree(out / "assets")
    for folder in FAMILY_ASSETS[family]:
        shutil.copytree(RUNTIME / "assets" / folder, out / "assets" / folder)
    sim = out / "sim"
    if sim.exists():
        shutil.rmtree(sim)
    (sim / "biolab").mkdir(parents=True)
    biolab_files = BIOLAB_FILES + FAMILY_BIOLAB.get(family, ())
    for name in biolab_files:
        shutil.copy2(BIOLAB / "biolab" / name, sim / "biolab" / name)
    shutil.copytree(BIOLAB / YAM_DIR, sim / YAM_DIR,
                    ignore=shutil.ignore_patterns("*.png", "CHANGELOG.md"))
    commit = subprocess.run(["git", "-C", str(BIOLAB), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    (sim / "SOURCE.md").write_text(
        "# Simulator files\n\n"
        f"- `biolab/`: {', '.join(biolab_files)} from the biolab simulator at commit "
        f"`{commit or 'unknown'}`, unchanged.\n"
        "- `third_party/mujoco_menagerie/i2rt_yam/`: the I2RT YAM arm model from MuJoCo "
        "Menagerie (MIT, see its LICENSE).\n", encoding="utf-8")
    return {"biolab_commit": commit}


def _run(out: Path, *args, timeout: int = 3600) -> dict:
    proc = subprocess.run([sys.executable, "task.py", *args], cwd=out, capture_output=True,
                          text=True, timeout=timeout)
    if proc.returncode != 0:
        raise BuildError(f"task.py {' '.join(args)} failed: {proc.stderr.strip()[-800:]}")
    text = proc.stdout.strip()
    try:
        return json.loads(text)
    except ValueError:
        return json.loads(text.splitlines()[-1]) if text else {}


# ------------------------------------------------------------------------- text
def _cm(v: float) -> str:
    return f"{100 * v:.1f}".rstrip("0").rstrip(".")


def weigh_setup_text(layout: dict) -> str:
    """Every object with type, size, material and place, from the frozen layout."""
    _biolab_path()
    from biolab import scene as sc

    def place(x, y):
        fwd, side = x - sc.ARM_BASE["left"][0], y
        lr = "left" if side > 0 else "right"
        lat = f"{_cm(abs(side))} cm {lr} of the midline" if abs(side) > 1e-4 else "on the midline"
        return f"{_cm(fwd)} cm in front of the arm bases, {lat}"

    st, ho, bal = layout["stock"], layout["handover"], layout["balance"]
    cell = sc.Rack(st["name"], tuple(st["xy"]), nx=st["nx"], ny=st["ny"], yaw=st["yaw"]).cell_xy(*layout["cell"])
    pitch, hole, height = sc.RACK_PITCH, sc.RACK_HOLE, sc.RACK_HEIGHT
    return (
        f"Two I2RT YAM arms (6 joints and a parallel gripper each) stand on a flat, rigid, matte bench "
        f"({_cm(2 * 0.45)} x {_cm(2 * 0.60)} cm), facing the same way, their bases "
        f"{_cm(2 * sc.ARM_BASE['left'][1])} cm apart; positions are measured from the line through the bases, "
        f"left and right as seen from behind the arms (the screenshot faces the arms, so its left is "
        f"their right). "
        f"A teal stock rack (rigid plastic, {st['nx']} x {st['ny']} square cells of {_cm(hole)} cm, "
        f"{_cm(pitch)} cm pitch, {_cm(height)} cm tall) is centred {place(*st['xy'])}, reachable by the left "
        f"arm only. A purple hand-over rack (the same, {ho['nx']} x {ho['ny']} cells) is centred {place(*ho['xy'])}, "
        f"reachable by both arms. An open plastic vial ({_cm(2 * sc.TUBE_RADIUS)} cm across, "
        f"{_cm(sc.TUBE_LENGTH)} cm tall, 30 g; solid except a {_cm(sc.CUP_DEPTH)} cm deep cup in its top, "
        f"{_cm(2 * sc.CUP_RADIUS)} cm across at the floor and flaring 20 degrees) stands in the stock rack's "
        f"cell {place(*cell)}, holding {TOTAL_SALT_G:.2f} g of coarse salt (68 grains, 2.8 to 3.6 mm). "
        f"A grey balance (0.01 g readability; a {_cm(0.07)} cm square base and a {_cm(0.09)} cm square pan whose "
        f"top is {_cm(sc.PAN_TOP)} cm above the bench) is centred {place(*bal)}, reachable by the right arm "
        f"only; a square polystyrene weigh boat ({_cm(sc.BOAT_SIZE)} cm inside, {_cm(sc.BOAT_RIM)} cm rim) "
        f"sits on the pan, and the balance is tared with it. Both arms start at their home pose with "
        f"the grippers open.")


def weigh_task_json(target_g: float, source: dict, layout: dict, run_limit: int,
                    demo_score: float, scoring: dict | None = None) -> dict:
    s = scoring or weigh_scoring(source.get("full_g"))
    t = f"{target_g:.2f}"
    minutes = run_limit / 20 / 60
    return {
        "title": f"Weigh out {t} g of salt passed between two arms",
        "category": "Impact on jobs",
        "keywords": ["bimanual", "hand-over", "pouring", "granular", "laboratory", "weighing"],
        "instruction": f"Weigh out {t} g of salt from the vial into the weigh boat on the balance.",
        "context": (f"Weighing out a solid reagent is one of the most common bench steps in wet-lab "
                    f"protocols; this amount comes from a published protocol ({source.get('reagent', 'NaCl')}, "
                    f"{t} g). The vial starts out of reach of the arm beside the balance, so the arms must "
                    f"pass it between them, and the salt only leaves the vial when it is tilted."),
        "setup": weigh_setup_text(layout),
        "scoring": (
            f"Measured 1 s after the last action, with the arms holding still; the balance display "
            f"averages the last 0.5 s and reads to 0.01 g. Score 0 if the vial, a gripper or anything "
            f"other than salt touches the balance pan or the weigh boat. Otherwise, with R the reading: "
            f"1.0 if R is within {s['full_g']:.2f} g of {t} g; 0.75 if within {s['three_quarter_g']:.2f} g; "
            f"0.5 if within {s['half_g']:.2f} g; 0.25 if R is at least {s['some_salt_g']:.2f} g; 0.1 if R is "
            f"below {s['some_salt_g']:.2f} g and the vial stands in a cell of the hand-over rack, tilted less "
            f"than 15 degrees; 0 otherwise. If more than {s['spill_cap_g']:.2f} g of salt is neither in the "
            f"vial nor on the boat ({TOTAL_SALT_G:.2f} g minus what the vial and the boat hold, both "
            f"weighed), the score is at most 0.5."),
        "run_limit": f"{run_limit:,} steps ({minutes:.1f} minutes at 20 Hz)".replace(".0 minutes", " minutes"),
        "reset_notes": (f"Salt is consumed: refill the vial's cup with {TOTAL_SALT_G:.2f} g of coarse salt, stand "
                        f"it back in the stock rack's corner cell, empty the boat and re-tare the balance."),
        "fidelity": ("Approximate: the salt is 68 rigid spheres of 2.8 to 3.6 mm (0.024 to 0.05 g), like coarse "
                     "salt rather than fine crystals; the balance is a pan on a stiff spring read through a "
                     "0.5 s average; the vial is a solid cylinder with a cup in its top; the grippers are "
                     "position-controlled springs with a 40 N limit; no static, drafts or vibration."),
        "scores": f"Dream-1 run: not run yet; demo: {demo_score:.1f}",
    }


def readme_text(spec: dict, instance: dict, demo: dict, runtime_s: dict, source: dict) -> str:
    return f"""# {spec['title']}

{spec['instruction']}

## Run and score a trial

Python 3.10 or newer with `pip install numpy mujoco imageio imageio-ffmpeg` (built and checked with
mujoco {runtime_s['mujoco']}, numpy {runtime_s['numpy']}). Headless machines: `export MUJOCO_GL=egl`
(GPU) or `MUJOCO_GL=osmesa` (CPU).

A trial is one trajectory: the rows a policy sent, `(T, 14)` joint and gripper targets, one per 50 ms
control step (20 Hz), in this order: {', '.join(f'`{a}`' for a in instance['action'])}. Joints are in rad;
each gripper entry is the travel of one finger in metres (0 closed, 0.041 open). At most
{instance['run_limit_steps']:,} rows.

    python task.py score --traj trajectory.npz        # replays it twice in fresh processes, prints the score

To drive the scene from a policy loop in Python:

    from task import TaskSession, save_trajectory
    s = TaskSession()                      # the frozen scene, at its start
    obs = s.observe()                      # camera images + the 14 joint and gripper readings
    s.step(action)                         # 14 targets, held for one 50 ms step
    save_trajectory(s.trajectory(), "trajectory.npz")

Cameras: {', '.join(f'`{c}`' for c in demo['cameras'])}.

Other commands: `python task.py demo --out .` re-records the scripted demo (`demo.npz`, `demo.mp4`);
`python task.py video --traj trajectory.npz --out run.mp4` renders a trial; `python task.py setup --out
setup.png` renders the setup screenshot.

## Runtime and seeds

- Scoring one trial (two replays): about {runtime_s['score']:.0f} s on one CPU core.
- The scripted demo: {demo['steps']:,} steps ({demo['steps'] / 20:.0f} s of robot time), recorded in about
  {runtime_s['demo']:.0f} s.
- No randomness: the scene is frozen in `instance.json` (layout, target, seed {instance['seed']}), and a
  replay is bit-for-bit repeatable; `task.py` refuses a scene whose hashes differ from the instance.

## Demo result

{json.dumps({k: v for k, v in demo.items() if k not in ('state_sha', 'cameras')})}

## Provenance

- Protocol step: {source.get('step_text', '')!r} ({source.get('title', '')}, {source.get('url', '')}).
- Simulator: see `sim/SOURCE.md`.
"""


# ------------------------------------------------------------------------- build
def _biolab_path() -> None:
    if str(BIOLAB) not in sys.path:
        sys.path.insert(0, str(BIOLAB))


def default_weigh_layout() -> dict:
    _biolab_path()
    from biolab.scene import WeighLayout
    lay = WeighLayout()

    def rack(r):
        return {"name": r.name, "xy": [float(v) for v in r.xy], "nx": r.nx, "ny": r.ny,
                "yaw": float(r.yaw)}
    return {"stock": rack(lay.stock), "handover": rack(lay.handover),
            "balance": [float(v) for v in lay.balance], "cell": list(lay.cell)}


def _build_folder(out: Path, family: str, instance: dict, make_spec, source: dict,
                  video: bool = True) -> dict:
    """The build every family shares: freeze, demo, validate, write the task text.

    make_spec(run_limit, demo_score, instance) returns task.json. Fails unless
    the demo scores 1.0 in two agreeing fresh replays and doing nothing scores 0.
    """
    info = vendor(out, family)
    instance = {"schema": "robocurve-bench-instance/1", "family": family, "seed": 0,
                "run_limit_steps": 20 * 60 * 20, "control_hz": 20, "action": None,
                **instance, "biolab_commit": info["biolab_commit"]}
    (out / "instance.json").write_text(json.dumps(instance, indent=1), encoding="utf-8")
    instance.update(_run(out, "hashes"))
    probe = subprocess.run([sys.executable, "-c",
                            "import json,task;s=task.TaskSession(check=False);"
                            "print(json.dumps([list(s.action_names), s.cameras()]))"],
                           cwd=out, capture_output=True, text=True, check=True)
    instance["action"], cameras = json.loads(probe.stdout.strip().splitlines()[-1])
    (out / "instance.json").write_text(json.dumps(instance, indent=1), encoding="utf-8")

    t0 = time.time()
    demo = _run(out, "demo", "--out", ".", *([] if video else ["--no-video"]))
    demo_s = time.time() - t0
    if demo.get("score") != 1.0:
        raise BuildError(f"the scripted demo does not reach full score: {demo}")
    run_limit = int(math.ceil(demo["steps"] * RUN_LIMIT_MARGIN / RUN_LIMIT_ROUND) * RUN_LIMIT_ROUND)
    instance["run_limit_steps"] = run_limit
    (out / "instance.json").write_text(json.dumps(instance, indent=1), encoding="utf-8")

    t0 = time.time()
    scored = _run(out, "score", "--traj", "demo.npz")
    score_s = time.time() - t0
    if scored.get("score") != 1.0 or not scored.get("deterministic"):
        raise BuildError(f"the demo does not score 1.0 in two agreeing fresh replays: {scored}")
    subprocess.run([sys.executable, "-c",
                    "import numpy as np,task;task.save_trajectory({'rows':np.zeros((0,14))},'nop.npz')"],
                   cwd=out, capture_output=True, text=True, check=True)
    idle = _run(out, "score", "--traj", "nop.npz", "--repeat", "1")
    (out / "nop.npz").unlink()
    if idle.get("score") != 0.0:
        raise BuildError(f"doing nothing scores {idle.get('score')}, not 0")
    _run(out, "setup", "--out", "setup.png")

    import mujoco
    import numpy
    spec = make_spec(run_limit, demo["score"], instance)
    (out / "task.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    demo["cameras"] = cameras
    (out / "README.md").write_text(readme_text(spec, instance, demo,
                                               {"score": score_s, "demo": demo_s,
                                                "mujoco": mujoco.__version__, "numpy": numpy.__version__},
                                               source), encoding="utf-8")
    for cache in out.rglob("__pycache__"):  # not part of the submission
        shutil.rmtree(cache, ignore_errors=True)
    report = {"out": str(out), "family": family, "demo": demo, "run_limit_steps": run_limit,
              "score_seconds": round(score_s, 1), "demo_seconds": round(demo_s, 1),
              "problems": check_task(out)}
    # The build report stays out of the submission folder.
    out.with_name(out.name + ".build.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_weigh_task(out_dir, source: dict, seed: int = 0, video: bool = True) -> dict:
    """Write and validate one weighing task folder; returns a build report."""
    target = float(source["target_g"])
    if not 0.5 <= target <= 1.8:
        raise BuildError(f"{target} g is outside what the {TOTAL_SALT_G} g vial can weigh out well "
                         f"(0.5 to 1.8 g)")
    layout = default_weigh_layout()
    instance = {"seed": seed, "layout": layout, "target_g": round(target, 3),
                "scoring": weigh_scoring(source.get("full_g")),
                "source": {k: source[k] for k in ("protocol_id", "dataset", "url", "step", "target_g",
                                                  "reagent", "full_g") if k in source}}
    return _build_folder(Path(out_dir), "weigh", instance,
                         lambda limit, score, inst: weigh_task_json(target, source, layout, limit, score,
                                                                    inst["scoring"]),
                         source, video)


# ------------------------------------------------------------------- centrifuge
def _runtime_module(name: str):
    """A runtime family module, imported from the repository (for its layout and constants)."""
    _biolab_path()
    runtime = str(RUNTIME)
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    import importlib
    return importlib.import_module(name)


def default_centrifuge_layout() -> dict:
    cf = _runtime_module("centrifuge_family")
    return cf.layout_to_json(cf.CentrifugeLayout())


def centrifuge_setup_text(layout: dict) -> str:
    cf = _runtime_module("centrifuge_family")
    from biolab import scene as sc
    lay = cf.layout_from(layout)

    def place(xy):
        x, y = xy
        side = "left" if y > 0 else "right"
        lat = f"{_cm(abs(y))} cm {side} of the midline" if abs(y) > 1e-4 else "on the midline"
        return f"{_cm(x - sc.ARM_BASE['left'][0])} cm in front of the arm bases, {lat}"

    rotor = lay.to_world(cf.ROTOR_C)
    cells = [lay.rack.cell_xy(c) for c in lay.tubes.values()]
    (x0, x1), (y0, y1) = cf.BODY
    return (
        f"Two I2RT YAM arms (6 joints and a parallel gripper each) stand on a flat, rigid, matte bench "
        f"(90 x 120 cm), facing the same way, their bases {_cm(2 * sc.ARM_BASE['left'][1])} cm apart; "
        f"positions are measured from the line through the bases, left and right as seen from behind "
        f"the arms (the screenshot faces the arms, so its left is their right). A white plastic "
        f"benchtop microcentrifuge ({_cm(x1 - x0)} x {_cm(y1 - y0)} cm, body "
        f"{_cm(cf.RIM_Z)} cm tall) has its rotor centred {place(rotor)}; its lid is hinged on the side "
        f"towards the left arm and stands open, leaning back {abs(np.degrees(cf.LID_OPEN)):.1f} degrees "
        f"from upright (the lid is {_cm(0.117)} cm long and {_cm(0.021)} cm thick, and latches when "
        f"pressed flat). The rotor has {cf.SLOT_COUNT} vertical slots for 1.5 mL tubes, evenly spaced on "
        f"a {_cm(cf.SLOT_RADIUS)} cm radius, {_cm(2 * cf.POCKET_R)} cm across, with rims "
        f"{_cm(cf.SLOT_TOP)} cm above the bench and {_cm(cf.SLOT_TOP - cf.SLOT_FLOOR)} cm deep. A teal "
        f"plastic tube rack ({lay.rack.n} cells of {_cm(cf.RACK_HOLE)} cm, {_cm(cf.RACK_PITCH)} cm pitch, "
        f"{_cm(cf.RACK_HEIGHT)} cm tall) is centred {place(lay.rack.xy)}. Two capped 1.5 mL plastic "
        f"tubes ({_cm(2 * cf.TUBE_R)} cm across, {_cm(cf.TUBE_BODY + cf.CAP_H)} cm tall, "
        f"{1000 * cf.TUBE_MASS:.0f} g with their sample; yellow and blue caps) stand in the rack's cells "
        f"{' and '.join(str(c + 1) for c in lay.tubes.values())} (counted from the end nearest the arms): "
        f"{place(cells[0])}, and {place(cells[1])}. Both arms start at their home pose with the grippers "
        f"open.")


def centrifuge_task_json(source: dict, layout: dict, run_limit: int, demo_score: float) -> dict:
    cf = _runtime_module("centrifuge_family")
    minutes = run_limit / 20 / 60
    step = source.get("step_text", "")
    speed = f" ({source['speed']})" if source.get("speed") else ""
    return {
        "title": "Load a microcentrifuge balanced and shut its lid",
        "category": "Impact on jobs",
        "keywords": ["bimanual", "insertion", "balancing", "articulated lid", "laboratory", "centrifuge"],
        "instruction": "Load the two sample tubes into opposite slots of the centrifuge rotor so it is "
                       "balanced, then close the lid.",
        "context": (f"Centrifuging is one of the most common steps in published wet-lab protocols; this "
                    f"one is a published step{speed}. An unbalanced rotor is a real hazard, so the tubes "
                    f"must sit opposite each other before the lid is shut."),
        "setup": centrifuge_setup_text(layout),
        "scoring": (
            f"Judged 1 s after the last action. A tube is in a slot when it stands within 10 degrees of "
            f"upright with its bottom on the slot's floor. The lid is shut when it lies within 5 degrees "
            f"of flat on the body. 1.0 if the two tubes are in opposite slots (four slots apart) and the "
            f"lid is shut; 0.6 if they are in opposite slots and the lid is open; 0.3 if only one tube is "
            f"in a slot, or both are but not opposite, with the lid open; 0 if the lid is shut on any "
            f"other load (one tube, or two not opposite, or none), or if no tube is in a slot. The score "
            f"is at most 0.3 if a tube ends anywhere but the rack or the centrifuge."),
        "run_limit": f"{run_limit:,} steps ({minutes:.1f} minutes at 20 Hz)".replace(".0 minutes", " minutes"),
        "reset_notes": ("Stand the tubes back in rack cells 1 and 3 and open the lid until it rests back "
                        "on its stop."),
        "fidelity": ("Approximate: the rotor's slots are vertical (a real fixed-angle rotor's are tilted) "
                     "and nothing spins; collision shapes are simplified boxes inside the centrifuge's "
                     "visual model; the lid latches by a holding torque once within 10 degrees of flat; "
                     "the tubes are rigid cylinders with a cap disc; the grippers are position-controlled "
                     "springs with a 40 N limit."),
        "scores": f"Dream-1 run: not run yet; demo: {demo_score:.1f}",
    }


def build_centrifuge_task(out_dir, source: dict, video: bool = True) -> dict:
    """Write and validate one centrifuge task folder; returns a build report."""
    if source.get("tubes", 2) != 2:
        raise BuildError(f"{source.get('tubes')} tubes; the centrifuge bench sets out 2")
    layout = default_centrifuge_layout()
    instance = {"layout": layout, "scoring": {},
                "source": {k: source[k] for k in ("protocol_id", "dataset", "url", "step", "tubes",
                                                  "speed", "time_s") if k in source}}
    return _build_folder(Path(out_dir), "centrifuge", instance,
                         lambda limit, score, inst: centrifuge_task_json(source, layout, limit, score),
                         source, video)


# -------------------------------------------------------------------- prep_spin
def prep_spin_instance(source: dict) -> dict:
    """Layout, reagents and scoring for a prep-then-spin task, from its compiled additions.

    One trough per distinct reagent (an addition repeated in two steps draws twice from it),
    and the reagent's target is the sum of its additions, with their judge tolerances summed.
    """
    ps = _runtime_module("prep_spin_family")
    names = list(dict.fromkeys(a["name"] for a in source["additions"]))
    if len(names) > 2:
        raise BuildError(f"{len(names)} reagents; the prep bench sets out 2 reagent troughs")
    trough = {n: f"reagent_{i + 1}" for i, n in enumerate(names)}
    layout = ps.layout_to_json(ps.PrepLayout(troughs=(*trough.values(), ps.WATER)))
    reagents = [{"name": a["name"], "trough": trough[a["name"]], "volume_ul": float(a["volume_ul"]),
                 "step": a["step"]} for a in source["additions"]]
    targets = {}
    for a in source["additions"]:
        t = targets.setdefault(a["name"], {"target_ul": 0.0, "tol_ul": 0.0})
        t["target_ul"] = round(t["target_ul"] + float(a["volume_ul"]), 6)
        t["tol_ul"] = round(t["tol_ul"] + float(a["tol_ul"]), 6)
    return {"layout": layout, "reagents": reagents, "scoring": {"reagents": targets}}


def _ul(v: float) -> str:
    return f"{v:g}"


def prep_spin_setup_text(instance: dict) -> str:
    cf = _runtime_module("centrifuge_family")
    ps = _runtime_module("prep_spin_family")
    from biolab import scene as sc
    lay = ps.layout_from(instance["layout"])

    def place(xy):
        x, y = xy
        side = "left" if y > 0 else "right"
        lat = f"{_cm(abs(y))} cm {side} of the midline" if abs(y) > 1e-4 else "on the midline"
        return f"{_cm(x - sc.ARM_BASE['left'][0])} cm in front of the arm bases, {lat}"

    rotor = lay.to_world(cf.ROTOR_C)
    tubes = {n: lay.rack.cell_xy(c) for n, c in lay.tubes.items()}
    troughs = {w.name: w for w in lay.trough_wells()}
    held = {}
    for r in instance["reagents"]:
        held[r["trough"]] = r["name"]
    contents = "; ".join(f"the {i + 1}{'st' if i == 0 else 'nd' if i == 1 else 'rd'} trough, "
                         f"{place(troughs[t].center)}, holds {held.get(t, 'water')}"
                         for i, t in enumerate(lay.troughs))
    (x0, x1), (y0, y1) = cf.BODY
    return (
        f"Two I2RT YAM arms (6 joints and a parallel gripper each) stand on a flat, rigid, matte bench "
        f"(90 x 120 cm), facing the same way, their bases {_cm(2 * sc.ARM_BASE['left'][1])} cm apart; "
        f"positions are measured from the line through the bases, left and right as seen from behind "
        f"the arms (the screenshot faces the arms, so its left is their right). The right gripper holds "
        f"a single-channel 200 uL pipette straight down, with a tip on; its plunger is the task's 15th "
        f"action (the volume drawn into the tip, 0 to 200 uL). A reagent reservoir of "
        f"{len(lay.troughs)} square troughs ({_cm(ps.TROUGH_OPENING)} cm openings, "
        f"{_cm(ps.TROUGH_DEPTH)} cm deep) stands on the right arm's side: {contents}. A teal plastic "
        f"tube rack ({lay.rack.n} cells of {_cm(cf.RACK_HOLE)} cm, {_cm(cf.RACK_PITCH)} cm pitch) is "
        f"centred {place(lay.rack.xy)}, between the arms. Two empty open 1.5 mL tubes "
        f"({_cm(2 * (ps.TUBE_IN_R + ps.TUBE_WALL))} cm across, {_cm(ps.TUBE_H)} cm tall) stand in it: "
        f"the sample tube, with a yellow band, {place(tubes[ps.SAMPLE])}, and the balance tube, with a "
        f"blue band, {place(tubes[ps.BALANCE])}. A white plastic benchtop microcentrifuge "
        f"({_cm(x1 - x0)} x {_cm(y1 - y0)} cm, body {_cm(cf.RIM_Z)} cm tall) has its rotor centred "
        f"{place(rotor)}, on the left arm's side; its lid is hinged on the side towards the left arm and "
        f"stands open, leaning back {abs(np.degrees(cf.LID_OPEN)):.1f} degrees from upright, and latches "
        f"when pressed flat. The rotor has {cf.SLOT_COUNT} vertical slots for 1.5 mL tubes, evenly spaced "
        f"on a {_cm(cf.SLOT_RADIUS)} cm radius, with rims {_cm(cf.SLOT_TOP)} cm above the bench. Both "
        f"arms start at their home pose; the left gripper is open.")


def prep_spin_task_json(source: dict, instance: dict, run_limit: int, demo_score: float) -> dict:
    ps = _runtime_module("prep_spin_family")
    minutes = run_limit / 20 / 60
    adds = source["additions"]
    recipe = " then ".join(f"{_ul(a['volume_ul'])} uL of {a['name']}" for a in adds)
    total = sum(a["volume_ul"] for a in adds)
    targets = instance["scoring"]["reagents"]
    checks = "; ".join(f"{n}: {_ul(t['target_ul'])} +- {_ul(t['tol_ul'])} uL" for n, t in targets.items())
    speed = f" ({source['speed']})" if source.get("speed") else ""
    return {
        "title": "Add reagents to a sample tube, balance it, and centrifuge it",
        "category": "Impact on jobs",
        "keywords": ["bimanual", "multi-step", "pipetting", "insertion", "balancing", "articulated lid",
                     "laboratory", "centrifuge"],
        "instruction": (f"Pipette {recipe} into the yellow sample tube. Fill the blue balance tube with "
                        f"water to the same volume. Load both tubes into opposite slots of the centrifuge "
                        f"and close the lid."),
        "context": (f"Adding reagents to a tube and spinning it down is one of the most common runs of "
                    f"steps in published wet-lab protocols; these {len(adds) + 1} steps are published "
                    f"ones{speed}. The tube must hold the right volumes before the spin, and the rotor "
                    f"must be balanced by a tube of matching volume opposite it before the lid is shut."),
        "setup": prep_spin_setup_text(instance),
        "scoring": (
            f"Judged when the lid first latches (it is shut when within 5 degrees of flat), or 1 s after "
            f"the last action if it never does. A tube is in a slot when it stands within 10 degrees of "
            f"upright with its bottom on the slot's floor. The balance tube matches when it holds liquid "
            f"within 10% of the sample tube's volume (the sample should total {_ul(total)} uL). Score 0 "
            f"if the lid shuts while the two tubes are not in opposite slots (four apart) or the balance "
            f"tube does not match. Otherwise the score is the sum of: 0.4 times the fraction of reagents "
            f"whose volume in the sample tube is within tolerance ({checks}), and 0 for this part if the "
            f"sample tube holds anything else; 0.2 if the balance tube matches; 0.2 if the tubes are in "
            f"opposite slots; 0.2 if the lid is shut. The score is at most 0.5 if more than "
            f"{_ul(ps.SPILL_CAP_UL)} uL is spilled, and at most 0.3 if a tube ends anywhere but the rack "
            f"or the centrifuge."),
        "run_limit": f"{run_limit:,} steps ({minutes:.1f} minutes at 20 Hz)".replace(".0 minutes", " minutes"),
        "reset_notes": ("Empty and dry both tubes, stand them back in rack cells 1 and 3, refill the troughs "
                        "as described, put a fresh tip on the pipette and open the lid until it rests back "
                        "on its stop."),
        "fidelity": ("Approximate: liquids are bookkeeping (volumes move between wells, tip and tubes when "
                     "the tip is in the liquid; a tip moved out of a well while dispensing spills), not "
                     "fluid simulation; the pipette's plunger is a commanded action, not pressed by a "
                     "finger; the rotor's slots are vertical and nothing spins; collision shapes are "
                     "simplified boxes inside the centrifuge's visual model; the lid latches by a holding "
                     "torque once within 10 degrees of flat; the grippers are position-controlled springs "
                     "with a 40 N limit. Other actions in the published steps (mixing, incubating) are "
                     "left out."),
        "scores": f"Dream-1 run: not run yet; demo: {demo_score:.1f}",
    }


def build_prep_spin_task(out_dir, source: dict, video: bool = True) -> dict:
    """Write and validate one multi-step task folder (pipette, balance, spin); returns a build report."""
    if not source.get("additions"):
        raise BuildError("no additions before the spin; build it with --family centrifuge")
    instance = prep_spin_instance(source)
    instance["source"] = {k: source[k] for k in ("protocol_id", "dataset", "url", "step", "tubes",
                                                 "speed", "time_s", "additions") if k in source}
    return _build_folder(Path(out_dir), "prep_spin", instance,
                         lambda limit, score, inst: prep_spin_task_json(source, inst, limit, score),
                         source, video)


# ------------------------------------------------------------------------- check
#: Phrases that would break blind review if they appeared in the task's text.
IDENTIFYING = re.compile(r"lab2rob|biolab|fernando|/home/|/users/|\\\\users\\\\|github\.com/|purdue|"
                         r"\buniversity\b|\binstitut", re.I)


def check_task(task_dir) -> list:
    """What still stands between this folder and a Robocurve pull request."""
    task_dir = Path(task_dir)
    problems = []
    try:
        spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"task.json unreadable: {e}"]
    for key in REQUIRED:
        if not str(spec.get(key, "")).strip():
            problems.append(f"task.json: '{key}' is missing"
                            + (" (tools/upload_media.py writes it)" if key == "media" else ""))
    if spec.get("category") not in CATEGORIES:
        problems.append(f"task.json: category {spec.get('category')!r} is not one of {CATEGORIES}")
    if "not run yet" in str(spec.get("scores", "")):
        problems.append("task.json: 'scores' has no Dream-1 result yet (and the Dream-1 video is required)")
    for key in ("title", "instruction", "context", "setup", "scoring", "fidelity", "reset_notes"):
        hit = IDENTIFYING.search(str(spec.get(key, "")))
        if hit:
            problems.append(f"task.json: '{key}' may identify the team ({hit.group(0)!r}); review is blind")
    if not re.search(r"\d", str(spec.get("scoring", ""))):
        problems.append("task.json: scoring has no numeric conditions")
    for name in ("README.md", "task.py", "instance.json", "setup.png", "demo.mp4", "demo.npz"):
        if not (task_dir / name).is_file():
            problems.append(f"{name} is missing")
    video = task_dir / "demo.mp4"
    if video.is_file():
        if video.stat().st_size > 50 * 1024 * 1024:
            problems.append("demo.mp4 is over 50 MB")
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=codec_name,pix_fmt", "-of", "csv=p=0", str(video)],
                               capture_output=True, text=True)
        if probe.returncode == 0 and probe.stdout.strip() and not probe.stdout.startswith("h264,yuv420p"):
            problems.append(f"demo.mp4 is {probe.stdout.strip()}, not H.264 yuv420p")
    licence = task_dir / "sim" / "biolab" / "LICENSE"
    if not licence.is_file():
        problems.append("sim/biolab has no LICENSE: the repository is public, so the simulator's owner "
                        "should add one before submitting")
    return problems
