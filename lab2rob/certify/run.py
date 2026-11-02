"""Certifier: run the reference program in the built scene, trace, grade.

Execution replays the planned ops through the biolab API exactly as a policy
would: move the held pipette over each assigned well, aspirate/dispense,
incubate/wait/read through the matching calls. Every op appends a trace row
with a full liquid snapshot; each planned step's judge predicates run
against the snapshot taken right after that step. Anything unexpected aborts
the run with a reason — a completed run is a clean run, and predicate
pass/fail is graded separately.

Source wells are prefilled with exactly the total each will give (computed
from the program): just-in-time labware prep, recorded in run.json.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lab2rob.registry.predicates import PREDICATES

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
_BIOLAB_ROOT = Path(__file__).resolve().parents[2] / "Code-as-a-Policy-Evolving-Skill"

PREFILL_CONC = 0.0
APPROACH_M = 0.03  # travel height above the well rim
SUBMERGE_M = 0.001  # tip depth below the surface for a draw
FLOOR_MARGIN_M = 0.0004  # shallow-well tip height (floor-draw covers it)
TRAVEL_SPEED = 0.15
WELL_SPEED = 0.04


class RunAborted(Exception):
    """Internal: abort the run with a reason (recorded, never raised out)."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass
class RunReport:
    run_id: str
    dir: Path
    status: str  # complete | aborted
    abort_reason: str = ""
    abort_detail: str = ""
    predicates: list = field(default_factory=list)
    n_ops: int = 0
    env: object = None  # live env, only when certify(return_env=True)


def _biolab():
    root = str(_BIOLAB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    import biolab.culture as culture
    import biolab.protein as protein
    return protein, culture


def _cert_env_class(family: str):
    """Env subclass pinned to our layout + prefill (task fields neutral)."""
    protein, culture = _biolab()
    if family == "protein":
        base, api_cls = protein.ProteinEnv, protein.ProteinApi

        def _new_episode(self, rng):
            self.layout = self._cert_layout
            self.task = protein.ProteinTask(2.0, 0.5, (0.45, 1.2, 0.9))
            self._noise = np.random.default_rng(rng.integers(1 << 31))
            self.reported = None
    elif family == "culture":
        base, api_cls = culture.CultureEnv, culture.CultureApi

        def _new_episode(self, rng):
            self.layout = self._cert_layout
            self.task = culture.CultureTask(1.8e6, 24.0, 0.0)
            self._noise = np.random.default_rng(rng.integers(1 << 31))
            self.clock = 0.0
            self.attached = {d: 0.0 for d in culture.DISHES}
            self.starved = {d: 0.0 for d in culture.DISHES}
            self.crowded = {d: 0.0 for d in culture.DISHES}
    else:
        raise ValueError(f"no cert env for family {family!r}")

    def _fill(self):
        for name, volume in self._cert_prefill.items():
            if name in self.wells and volume > 0:
                overflow = self.wells[name].add(volume, PREFILL_CONC)
                if overflow > 0:
                    raise RunAborted("over_capacity",
                                     f"prefill {volume:g} uL overflows {name}")

    def __init__(self, layout, prefill, seed=None):
        self._cert_layout = layout
        self._cert_prefill = dict(prefill)
        base.__init__(self, seed=seed, randomize=False)

    cls = type(f"Cert{family.title()}Env", (base,),
               {"__init__": __init__, "_new_episode": _new_episode, "_fill": _fill})
    return cls, api_cls


def compute_prefill(program, built) -> dict:
    """Per-slot aspirate totals (what each source well must start with)."""
    totals: dict[str, float] = {}
    for step in program.steps:
        if step.status != "planned":
            continue
        for op_idx, op in enumerate(step.ops):
            if op["skill"] != "aspirate":
                continue
            slot = built.assignments.get(f"{step.idx}:{op_idx}")
            volume = op["args"].get("volume_ul")
            if slot is not None and isinstance(volume, (int, float)):
                totals[slot] = totals.get(slot, 0.0) + float(volume)
    return totals


def prepare_env(built, prefill, seed: int = 0):
    """Env + API + grip height, shared by certify and the eval harness."""
    protein, _ = _biolab()
    env_cls, api_cls = _cert_env_class(built.family)
    env = env_cls(built.layout, prefill, seed=seed)
    return env, api_cls(env), protein.GRIP_ABOVE_TIP


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def snapshot(env) -> dict:
    """Liquid state in the predicates' state protocol + run metadata."""
    return {
        "wells": {name: {"volume_ul": round(w.volume, 3),
                         "conc": round(w.conc, 6)}
                  for name, w in env.wells.items()},
        "spilled_ul": round(env.pipette.spilled, 3),
        "pipette_ul": round(env.pipette.volume, 3),
        "clock_h": round(float(getattr(env, "clock", 0.0)), 6),
    }


def _tip_target(well, extra_mm: float = 0.0) -> float:
    """Tip z that draws/delivers: below the surface, or at the floor."""
    target = max(well.floor + FLOOR_MARGIN_M, well.surface - SUBMERGE_M)
    return max(well.floor + 0.0002, target - extra_mm / 1000.0)


def _move_tip(api, well, tip_z: float, grip_above: float, label: str,
              approach_m: float = APPROACH_M, well_speed: float = WELL_SPEED) -> None:
    x, y = float(well.center[0]), float(well.center[1])
    top = well.floor + well.depth + approach_m
    if not api.move_to("right", np.array([x, y, top + grip_above]), 0.0, 0.0, TRAVEL_SPEED):
        raise RunAborted("move_blocked", f"approach to {label} failed")
    if not api.move_to("right", np.array([x, y, tip_z + grip_above]), 0.0, 0.0, well_speed):
        raise RunAborted("move_blocked", f"descent into {label} failed")


def _withdraw(api, well, grip_above: float, approach_m: float = APPROACH_M) -> None:
    x, y = float(well.center[0]), float(well.center[1])
    top = well.floor + well.depth + approach_m
    api.move_to("right", np.array([x, y, top + grip_above]), 0.0, 0.0, TRAVEL_SPEED)


def execute_op(api, env, op: dict, slot: str | None, grip_above: float,
               approach_m: float = APPROACH_M, well_speed: float = WELL_SPEED,
               tip_extra_mm: float = 0.0):
    """Run one planned op through the API; returns the JSON-safe result."""
    skill = op["skill"]
    if skill in ("aspirate", "dispense"):
        if slot is None or slot not in env.wells:
            raise RunAborted("no_slot", f"{skill} has no assigned well")
        well = env.wells[slot]
        _move_tip(api, well, _tip_target(well, tip_extra_mm), grip_above, slot,
                  approach_m, well_speed)
        result = getattr(api, skill)(float(op["args"].get("volume_ul", 0.0)))
        _withdraw(api, well, grip_above, approach_m)
        return _jsonable(result)
    if skill == "incubate":
        return _jsonable(api.incubate(float(op["args"].get("hours", 0.0))))
    if skill == "wait":
        return _jsonable(api.wait(float(op["args"].get("seconds", 0.0))))
    if skill == "read_plate":
        return _jsonable(api.read_plate())
    if skill.startswith("get_"):
        try:
            return _jsonable(getattr(api, skill)())
        except TypeError as e:
            raise RunAborted("query_needs_args", f"{skill}: {e}") from None
    raise RunAborted("no_executor", f"certifier cannot run {skill!r}")


def _resolve_well(ref: str, step_idx: int, program, built) -> str | None:
    """Judge vessel ref -> concrete slot (None = unresolvable, skip)."""
    if ref != "unbound":
        for uid, vessel in built.vessels.items():
            if vessel["asset"] == ref:
                return vessel["slot"]
        for sub in built.substitutions:
            if sub["from"] == ref:
                for vessel in built.vessels.values():
                    if vessel["asset"] == sub["to"]:
                        return vessel["slot"]
        return None
    for step in program.steps:
        if step.idx != step_idx:
            continue
        dispenses = [built.assignments.get(f"{step_idx}:{i}")
                     for i, op in enumerate(step.ops) if op["skill"] == "dispense"]
        dispenses = [s for s in dispenses if s is not None]
        return dispenses[0] if dispenses else None
    return None


def _step_assignment_slots(step_idx: int, program, built) -> list:
    slots = []
    for step in program.steps:
        if step.idx == step_idx:
            for i in range(len(step.ops)):
                slot = built.assignments.get(f"{step_idx}:{i}")
                if slot is not None:
                    slots.append(slot)
    return slots


def evaluate_predicates(judge, program, built, snapshots: dict) -> list:
    """Grade each step's predicates on its post-step snapshot.

    Unknown refs, in-place volume checks (aspirate == dispense slot: volume
    is conserved, only spill is checkable), and unknown predicate names
    SKIP with a reason instead of failing correct runs.
    """
    rows = []
    for judge_step in judge.steps:
        state = snapshots.get(judge_step.idx, {"wells": {}, "spilled_ul": 0.0})
        for pred in judge_step.predicates:
            row = {"step": judge_step.idx, "pred": pred.pred, "args": pred.args,
                   "target": pred.target, "tol": pred.tol, "well": None,
                   "status": "skipped", "detail": ""}
            if pred.pred == "no_spill":
                result = PREDICATES["no_spill"](state, pred.tol or 0.0)
                row.update(status="passed" if result.passed else "failed",
                           detail=result.detail)
            elif pred.pred == "volume" and pred.args:
                slot = _resolve_well(pred.args[0], judge_step.idx, program, built)
                step_slots = _step_assignment_slots(judge_step.idx, program, built)
                if slot is None:
                    row["detail"] = f"cannot resolve well ref {pred.args[0]!r}"
                elif len(set(step_slots)) == 1 and len(step_slots) > 1:
                    row.update(well=slot, detail="in-place step: volume conserved, spill only")
                else:
                    result = PREDICATES["volume"](state, slot, pred.target, pred.tol)
                    row.update(well=slot,
                               status="passed" if result.passed else "failed",
                               detail=result.detail)
            else:
                row["detail"] = f"unknown predicate {pred.pred!r}" if pred.pred != "volume" else "volume pred has no well ref"
            rows.append(row)
    return rows


def certify(program, judge, built, seed: int = 0, runs_dir: Path = RUNS_DIR,
            approach_m: float = APPROACH_M, well_speed: float = WELL_SPEED,
            tip_extra_mm: float = 0.0, return_env: bool = False,
            run_suffix: str = "") -> RunReport:
    """Run the program in the built scene; trace + grades to runs/<run_id>/.

    approach_m/well_speed/tip_extra_mm are execution overrides for the
    repair pass (SC-05); return_env keeps the live env on the report for
    visual checks (never serialized).
    """
    run_id = f"{program.protocol_id}_s{seed}{run_suffix}"
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report = RunReport(run_id=run_id, dir=run_dir, status="complete")
    trace, snapshots = [], {}

    def record(kind: str, **fields):
        row = {"kind": kind, **_jsonable(fields)}
        trace.append(row)
        return row

    try:
        prefill = compute_prefill(program, built)
        env, api, grip_above = prepare_env(built, prefill, seed=seed)
        record("initial", state=snapshot(env), prefill=prefill)
        for step in program.steps:
            if step.status != "planned":
                continue
            for op_idx, op in enumerate(step.ops):
                slot = built.assignments.get(f"{step.idx}:{op_idx}")
                result = execute_op(api, env, op, slot, grip_above,
                                    approach_m, well_speed, tip_extra_mm)
                report.n_ops += 1
                record("op", step=step.idx, op=op_idx, skill=op["skill"],
                       args=op["args"], slot=slot, result=result,
                       state=snapshot(env))
            snapshots[step.idx] = snapshot(env)
        report.predicates = evaluate_predicates(judge, program, built, snapshots)
        if return_env:
            report.env = env
    except RunAborted as aborted:
        report.status = "aborted"
        report.abort_reason, report.abort_detail = aborted.reason, aborted.detail
        record("abort", reason=aborted.reason, detail=aborted.detail)
    except Exception as e:  # noqa: BLE001 - the trace must survive any crash
        report.status = "aborted"
        report.abort_reason, report.abort_detail = type(e).__name__, str(e)[:300]
        record("abort", reason=report.abort_reason, detail=report.abort_detail)

    counts = {"passed": 0, "failed": 0, "skipped": 0}
    for row in report.predicates:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    (run_dir / "run.json").write_text(json.dumps({
        "run_id": run_id, "protocol_id": program.protocol_id,
        "family": built.family, "seed": seed, "status": report.status,
        "exec": {"approach_m": approach_m, "well_speed": well_speed,
                 "tip_extra_mm": tip_extra_mm},
        "abort_reason": report.abort_reason, "abort_detail": report.abort_detail,
        "n_ops": report.n_ops, "predicate_counts": counts,
        "assignments": built.assignments, "aspirate_sources": built.aspirate_sources,
        "substitutions": built.substitutions,
        "labware": built.labware, "instruments": built.instruments,
        "redistributable": built.redistributable,
    }, indent=2), encoding="utf-8")
    with open(run_dir / "trace.jsonl", "w", encoding="utf-8") as handle:
        for row in trace:
            handle.write(json.dumps(row) + "\n")
    (run_dir / "predicates.json").write_text(
        json.dumps(report.predicates, indent=2), encoding="utf-8")
    return report
