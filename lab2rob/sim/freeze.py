"""Freeze a certified program into a graded instance + oracle trajectory.

freeze_task builds the frozen instance (scene, prefill, end-state checks,
hashes), records the reference program through a skills-level Session (the
recorded rows are the oracle), and proves the task is sound before writing
it out:

1. the oracle passes every end-state check in the recording process;
2. two replays of the saved oracle in fresh processes succeed and reach
   the same final state hash as the recording;
3. the empty trajectory (do nothing) fails.

Any failure raises FreezeError with a reason; nothing half-checked is saved
as frozen (a failed freeze still writes freeze.json with the reason).

End-state checks come from the independent judge, never from the program:
each judged well keeps the volume predicate of the last step that judges it
(the state after that step is the one that must survive to the end), plus
one no_spill check. A judged volume that the oracle's end state does not
meet fails the freeze: the judge then describes a per-step state that a
later step changes, which an end-state grader cannot check.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from lab2rob.certify.run import (RunAborted, _resolve_well, _step_assignment_slots,
                                 compute_prefill, execute_op)
from lab2rob.sim.instance import make_instance, save_instance
from lab2rob.sim.session import Session, load_trajectory, save_trajectory

REPO_ROOT = Path(__file__).resolve().parents[2]


class FreezeError(Exception):
    def __init__(self, reason: str, detail=""):
        self.reason, self.detail = reason, detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


def end_state_checks(judge, program, built) -> list:
    """Judge predicates -> end-state checks on concrete wells."""
    by_well: dict = {}
    spill_tol = None
    for judge_step in judge.steps:
        for pred in judge_step.predicates:
            if pred.pred == "no_spill":
                tol = float(pred.tol or 0.0)
                spill_tol = tol if spill_tol is None else max(spill_tol, tol)
            elif pred.pred == "volume" and pred.args:
                well = _resolve_well(pred.args[0], judge_step.idx, program, built)
                slots = _step_assignment_slots(judge_step.idx, program, built)
                if well is None or (len(set(slots)) == 1 and len(slots) > 1):
                    continue  # unresolvable, or in place (volume conserved)
                by_well[well] = {"pred": "volume", "well": well,
                                 "target": float(pred.target), "tol": float(pred.tol),
                                 "from_step": judge_step.idx}
    checks = [by_well[w] for w in sorted(by_well)]
    if spill_tol is not None:
        checks.append({"pred": "no_spill", "tol": spill_tol})
    return checks


def record_oracle(program, built, instance: dict) -> tuple:
    """Run the planned program through skills; returns (session, step_rows)."""
    session = Session(instance, level="skills")
    grip_above = session._grip_above()
    step_rows = []
    for step in program.steps:
        if step.status != "planned":
            continue
        for op_idx, op in enumerate(step.ops):
            slot = built.assignments.get(f"{step.idx}:{op_idx}")
            if not hasattr(session, op["skill"]) and not op["skill"].startswith("get_"):
                raise FreezeError("no_executor", f"session cannot run {op['skill']!r}")
            if op["skill"].startswith("get_"):
                continue  # queries change nothing in the scene
            try:
                execute_op(session, session.env, op, slot, grip_above)
            except RunAborted as aborted:
                raise FreezeError(aborted.reason, aborted.detail) from None
        step_rows.append({"step": step.idx, "end_row": len(session._rows)})
    return session, step_rows


def _fresh_replay(instance_path: Path, traj_path: Path, timeout_s: int = 1800) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    proc = subprocess.run(
        [sys.executable, "-m", "lab2rob.sim", "replay", "--instance", str(instance_path),
         "--traj", str(traj_path)],
        capture_output=True, text=True, timeout=timeout_s, env=env)
    if proc.returncode != 0:
        raise FreezeError("replay_crashed", proc.stderr.strip()[-500:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def freeze_task(program, judge, built, seed: int, out_dir, source: dict | None = None,
                instruction: str | None = None, fresh_replays: int = 2) -> dict:
    """Write instance.json + oracle.npz + freeze.json into out_dir; returns the report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"protocol_id": program.protocol_id, "seed": seed, "frozen": False}
    try:
        checks = end_state_checks(judge, program, built)
        if not any(c["pred"] in ("volume", "concentration") for c in checks):
            raise FreezeError("no_checks", "the judge gives no well to check at the end")
        instance = make_instance(built, compute_prefill(program, built), seed, checks,
                                 source=source, instruction=instruction)
        instance_path = save_instance(instance, out_dir / "instance.json")
        session, step_rows = record_oracle(program, built, instance)
        status = session.status()
        report.update(oracle=status, step_rows=step_rows)
        if not status["success"]:
            failed = [c for c in status["checks"] if not c["passed"]]
            raise FreezeError("oracle_fails_checks", failed or status["sim_error"])
        traj = session.trajectory()
        traj_path = save_trajectory(traj, out_dir / "oracle.npz", instance["init_sha"])
        session.close()

        replays = [_fresh_replay(instance_path, traj_path) for _ in range(fresh_replays)]
        report["replays"] = [{k: r[k] for k in ("success", "state_sha", "steps")}
                             for r in replays]
        if not all(r["success"] for r in replays):
            raise FreezeError("replay_fails", report["replays"])
        if {r["state_sha"] for r in replays} != {status["state_sha"]}:
            raise FreezeError("nondeterministic", report["replays"])

        nop = Session(instance, level="joints")
        nop_status = nop.replay({"rows": [], "events": []})
        nop.close()
        report["nop_success"] = nop_status["success"]
        if nop_status["success"]:
            raise FreezeError("trivial", "doing nothing already passes every check")
        report.update(frozen=True, n_rows=int(len(traj["rows"])),
                      n_events=len(traj["events"]))
        return report
    except FreezeError as e:
        report.update(reason=e.reason, detail=e.detail)
        raise
    finally:
        (out_dir / "freeze.json").write_text(json.dumps(report, indent=2, default=str),
                                             encoding="utf-8")


def check_oracle_file(instance_path, traj_path) -> dict:
    """In-process replay of a saved trajectory (tests and quick checks)."""
    session = Session(instance_path, level="joints")
    try:
        return session.replay(load_trajectory(traj_path))
    finally:
        session.close()
