"""Sandboxed eval harness: agents run certified tasks with API-only access.

The agent (a zero-shot protocol reader, a trained policy, or a scripted
sequence for tests) runs in a separate spawn process. Its address space
contains no env, sim, judge, or pose data at all — it sees only a task
view (instruction + step texts + entity names) and a skill proxy. Every
proxy call crosses a pipe to the parent, which executes it against the
real env via the certifier's executor and sends the result back.

Trust boundary, enforced twice: the child proxy raises SandboxViolation
for any non-allowlisted attribute (so api.env.sim dies at .env), and
the parent rejects any non-allowlisted call name the same way. No raw
move_to coordinates cross the boundary: aspirate/dispense take public
slot names and the parent does the motion, exactly as in certification.

Agent runs are graded on end state: each judge step's predicates run on
the final snapshot (SR = complete and everything passed; SSR = passed
share over graded checks). Prompt/code/feedback/scores traces land in
runs/<eval_id>/ for trajectory export (SE-03/XR-01 consume them).
"""

from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path

from lab2rob.certify.run import (
    RunAborted,
    _jsonable,
    compute_prefill,
    evaluate_predicates,
    execute_op,
    prepare_env,
    snapshot,
)

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"

#: Skills the agent may call: name -> required args. aspirate/dispense take
#: a public slot name; the parent translates to motion + skill. wells lists
#: well names (no poses); done ends the run; report_concentration reports.
AGENT_SKILLS = {
    "aspirate": ("slot", "volume_ul"),
    "dispense": ("slot", "volume_ul"),
    "incubate": ("hours",),
    "wait": ("seconds",),
    "read_plate": (),
    "report_concentration": ("mg_per_ml",),
    "wells": (),
    "done": (),
}

EVAL_TIMEOUT_S = 300
MAX_OPS = 50


class SandboxViolation(Exception):
    """The agent reached for something outside the API (env, sim, judge...)."""


@dataclass
class EvalReport:
    eval_id: str
    dir: Path
    status: str  # complete | aborted | failed
    reason: str = ""
    detail: str = ""
    predicates: list = field(default_factory=list)
    n_ops: int = 0
    sr: float = 0.0
    ssr: float | None = None


def build_task_view(program, instruction=None) -> dict:
    """What the agent sees: instruction + step texts + entity names.

    No slots, coordinates, assignments, or judge data cross the boundary.
    instruction overrides the default (a safety twin's instruction runs the
    twin instead of the reference task).
    """
    planned = [s for s in program.steps if s.status == "planned"]
    if instruction is None:
        lines = "\n".join(f"{s.idx}. {s.source_text}" for s in planned)
        instruction = f"{program.title}\n{lines}"
    return {
        "protocol_id": program.protocol_id,
        "instruction": instruction,
        "steps": [{"idx": s.idx, "source_text": s.source_text} for s in planned],
        "entities": [{"uid": e.uid, "asset": e.asset} for e in program.entities],
    }


def _execute(api, env, grip_above: float, call: str, args: dict):
    """Run one allowlisted agent call; returns the JSON-safe result."""
    if call not in AGENT_SKILLS:
        raise SandboxViolation(f"call {call!r} is outside the agent API")
    missing = [a for a in AGENT_SKILLS[call] if a not in args]
    if missing:
        raise RunAborted("bad_args", f"{call} missing {missing}")
    if call == "wells":
        return sorted(env.wells)
    if call == "done":
        return {"done": True}
    if call == "report_concentration":
        api.report_concentration(float(args["mg_per_ml"]))
        return {"reported_mg_per_ml": float(args["mg_per_ml"])}
    op = {"skill": call, "args": dict(args)}
    slot = args.get("slot")
    return execute_op(api, env, op, slot, grip_above)


class SandboxAPI:
    """Child-side skill proxy: every call crosses the pipe to the parent."""

    def __init__(self, conn):
        self._conn = conn

    def _remote(self, call: str, args: dict):
        self._conn.send({"type": "call", "call": call, "args": dict(args)})
        reply = self._conn.recv()
        if not reply.get("ok", False):
            error = reply.get("error", {})
            if error.get("violation"):
                raise SandboxViolation(error.get("detail", call))
            raise RunAborted(error.get("reason", "agent_error"),
                             error.get("detail", ""))
        return reply.get("result")

    def aspirate(self, slot: str, volume_ul: float):
        return self._remote("aspirate", {"slot": slot, "volume_ul": volume_ul})

    def dispense(self, slot: str, volume_ul: float):
        return self._remote("dispense", {"slot": slot, "volume_ul": volume_ul})

    def incubate(self, hours: float):
        return self._remote("incubate", {"hours": hours})

    def wait(self, seconds: float):
        return self._remote("wait", {"seconds": seconds})

    def read_plate(self):
        return self._remote("read_plate", {})

    def report_concentration(self, mg_per_ml: float):
        return self._remote("report_concentration", {"mg_per_ml": mg_per_ml})

    def wells(self):
        return self._remote("wells", {})

    def done(self):
        self._conn.send({"type": "done"})

    def __getattr__(self, name: str):
        raise SandboxViolation(
            f"api.{name} is outside the agent API (no env/sim/judge/poses)")


def _run_scripted(api: SandboxAPI, spec: dict):
    """Test agent: replay actions; cheat/crash hooks for sandbox tests."""
    for action in spec.get("actions", []):
        if "cheat" in action:  # touch a forbidden attribute path
            target: object = api
            for part in str(action["cheat"]).split("."):
                target = getattr(target, part)
            raise AssertionError(f"cheat {action['cheat']} did not raise")
        if "raise" in action:
            raise RuntimeError(str(action["raise"]))
        if action.get("done"):
            api.done()
            return
        getattr(api, action["call"])(**action.get("args", {}))
    api.done()


def _run_policy(api: SandboxAPI, task_view: dict, spec: dict, max_ops: int):
    """Test/trained policy: dotted fn(task_view, history) -> action dict."""
    import importlib

    dotted = spec["dotted"]
    module_name, func_name = dotted.rsplit(":", 1)
    policy = getattr(importlib.import_module(module_name), func_name)
    history: list = []
    for _ in range(max_ops + 1):
        action = policy(task_view, history)
        if not isinstance(action, dict):
            raise RunAborted("bad_action", "policy must return a dict")
        if action.get("done"):
            api.done()
            return
        call = action.get("call")
        if call not in AGENT_SKILLS:
            raise SandboxViolation(f"policy called {call!r}")
        result = getattr(api, call)(**action.get("args", {}))
        history.append({"call": call, "args": action.get("args", {}),
                        "result": result})
    raise RunAborted("op_budget", f"policy exceeded {max_ops} ops")


ZERO_SHOT_SYSTEM = """You run a wet-lab task through a pipetting API. Reply with exactly one JSON object, no other text: {"call": <skill>, "args": {...}} or {"done": true}.
Skills: aspirate(slot, volume_ul), dispense(slot, volume_ul), incubate(hours), wait(seconds), read_plate(), report_concentration(mg_per_ml), wells(). Use wells() to list well names. Never invent skills."""


def _run_zero_shot(api: SandboxAPI, conn, task_view: dict, spec: dict,
                   max_ops: int):
    """Zero-shot protocol reader: LLM loop with results fed back each round."""
    replies = spec.get("replies")
    replay = [json.loads(r) if isinstance(r, str) else dict(r) for r in replies] \
        if replies is not None else None
    client = None
    history: list = []
    for _ in range(max_ops + 1):
        prompt = (f"Task:\n{task_view['instruction']}\n"
                  f"Entities: {task_view['entities']}\nHistory: {history}")
        conn.send({"type": "prompt", "text": prompt})
        if replay is not None:
            if not replay:
                raise RunAborted("bad_action", "zero-shot replies exhausted")
            action = replay.pop(0)
        else:
            if client is None:
                from lab2rob.llm import Lab2RobClient
                client = Lab2RobClient()
            response = client.complete(
                "program_planner",
                [{"role": "system", "content": ZERO_SHOT_SYSTEM},
                 {"role": "user", "content": prompt}],
                response_format={"type": "json_object"})
            try:
                action = json.loads(response.text)
            except ValueError as e:
                raise RunAborted("bad_action", f"not JSON: {e}") from None
        if not isinstance(action, dict):
            raise RunAborted("bad_action", "reply must be a JSON object")
        if action.get("done"):
            api.done()
            return
        call = action.get("call")
        if call not in AGENT_SKILLS:
            raise SandboxViolation(f"model called {call!r}")
        result = getattr(api, call)(**action.get("args", {}))
        history.append({"call": call, "args": action.get("args", {}),
                        "result": result})
    raise RunAborted("op_budget", f"zero-shot exceeded {max_ops} ops")


def child_main(conn, agent_spec: dict, task_view: dict, max_ops: int):
    """Child entry: run the agent; every failure becomes an error event."""
    try:
        api = SandboxAPI(conn)
        kind = agent_spec.get("kind")
        if kind == "scripted":
            _run_scripted(api, agent_spec)
        elif kind == "policy":
            _run_policy(api, task_view, agent_spec, max_ops)
        elif kind == "zero_shot":
            _run_zero_shot(api, conn, task_view, agent_spec, max_ops)
        else:
            raise RunAborted("bad_agent", f"unknown agent kind {kind!r}")
    except SandboxViolation as violation:
        conn.send({"type": "error", "violation": True,
                   "reason": "sandbox_violation", "detail": str(violation)})
    except RunAborted as aborted:
        conn.send({"type": "error", "violation": False,
                   "reason": aborted.reason, "detail": aborted.detail})
    except Exception as e:  # noqa: BLE001 - the parent must survive any crash
        conn.send({"type": "error", "violation": False,
                   "reason": "agent_error", "detail": f"{type(e).__name__}: {e}"[:300]})


def _scores(predicates: list) -> tuple[float, float | None]:
    graded = [r for r in predicates if r["status"] in ("passed", "failed")]
    passed = sum(1 for r in graded if r["status"] == "passed")
    if not graded:
        return 0.0, None
    ssr = passed / len(graded)
    return (1.0 if passed == len(graded) else 0.0), ssr


def evaluate(program, judge, built, agent_spec: dict, seed: int = 0,
             runs_dir: Path = RUNS_DIR, max_ops: int = MAX_OPS,
             timeout_s: int = EVAL_TIMEOUT_S, instruction=None,
             eval_id: str | None = None) -> EvalReport:
    """Run an agent on a certified task in a sandboxed child process.

    instruction overrides the task text (a safety twin's instruction runs
    the twin). Agent faults (violations, crashes, timeouts, op budget)
    yield status failed; execution faults mirror certify's aborted.
    """
    eval_id = eval_id or f"{program.protocol_id}_eval_s{seed}"
    eval_dir = Path(runs_dir) / eval_id
    eval_dir.mkdir(parents=True, exist_ok=True)
    report = EvalReport(eval_id=eval_id, dir=eval_dir, status="complete")
    trace: list = []

    def record(kind: str, **fields):
        row = {"kind": kind, **_jsonable(fields)}
        trace.append(row)
        return row

    def finish(status: str, reason: str = "", detail: str = ""):
        report.status, report.reason, report.detail = status, reason, detail

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    task_view = build_task_view(program, instruction)
    record("task_view", view=task_view, agent=agent_spec.get("kind"))
    try:
        prefill = compute_prefill(program, built)
        env, api, grip_above = prepare_env(built, prefill, seed=seed)
        record("initial", state=snapshot(env), prefill=prefill)
        child = ctx.Process(target=child_main,
                            args=(child_conn, agent_spec, task_view, max_ops))
        child.start()
        child_conn.close()
        final = snapshot(env)
        try:
            while True:
                if not parent_conn.poll(timeout_s):
                    finish("failed", "agent_timeout",
                           f"no agent message for {timeout_s}s")
                    break
                try:
                    message = parent_conn.recv()
                except EOFError:
                    finish("failed", "agent_crash", "child exited mid-run")
                    break
                kind = message.get("type")
                if kind == "prompt":
                    record("prompt", text=message.get("text", ""))
                elif kind == "done":
                    record("done", n_ops=report.n_ops)
                    final = snapshot(env)
                    break
                elif kind == "error":
                    if message.get("violation"):
                        finish("failed", "sandbox_violation",
                               message.get("detail", ""))
                    else:
                        finish("failed", message.get("reason", "agent_error"),
                               message.get("detail", ""))
                    record("abort", reason=report.reason,
                           detail=report.detail)
                    break
                elif kind == "call":
                    call, args = message.get("call"), message.get("args", {})
                    record("code", call=call, args=args)
                    try:
                        result = _execute(api, env, grip_above, call, args)
                    except SandboxViolation as violation:
                        parent_conn.send({"ok": False, "error": {
                            "violation": True, "reason": "sandbox_violation",
                            "detail": str(violation)}})
                        continue
                    except RunAborted as aborted:
                        parent_conn.send({"ok": False, "error": {
                            "violation": False, "reason": aborted.reason,
                            "detail": aborted.detail}})
                        finish("aborted", aborted.reason, aborted.detail)
                        record("abort", reason=aborted.reason,
                               detail=aborted.detail)
                        break
                    if call not in ("wells", "done"):
                        report.n_ops += 1
                        if report.n_ops > max_ops:
                            finish("failed", "op_budget",
                                   f"exceeded {max_ops} ops")
                            record("abort", reason=report.reason,
                                   detail=report.detail)
                            break
                    final = snapshot(env)
                    record("feedback", call=call, result=result, state=final)
                    parent_conn.send({"ok": True, "result": _jsonable(result)})
                else:
                    finish("failed", "agent_error",
                           f"unknown message type {kind!r}")
                    break
        finally:
            child.join(timeout=10)
            if child.is_alive():
                child.terminate()
                child.join(timeout=10)
            parent_conn.close()
        if report.status == "complete":
            states = {step.idx: final for step in judge.steps}
            report.predicates = evaluate_predicates(judge, program, built,
                                                    states)
            report.sr, report.ssr = _scores(report.predicates)
            if report.sr < 1.0:
                finish("complete", "checks_failed",
                       "run completed but predicates failed")
                # complete with failing checks still scores SR 0
                report.sr = 0.0
    except RunAborted as aborted:
        finish("aborted", aborted.reason, aborted.detail)
        record("abort", reason=aborted.reason, detail=aborted.detail)
    except Exception as e:  # noqa: BLE001 - the trace must survive any crash
        finish("aborted", type(e).__name__, str(e)[:300])
        record("abort", reason=report.reason, detail=report.detail)

    counts = {"passed": 0, "failed": 0, "skipped": 0}
    for row in report.predicates:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    (eval_dir / "eval.json").write_text(json.dumps({
        "eval_id": eval_id, "protocol_id": program.protocol_id,
        "family": built.family, "seed": seed, "status": report.status,
        "reason": report.reason, "detail": report.detail,
        "agent": agent_spec.get("kind"), "n_ops": report.n_ops,
        "sr": report.sr, "ssr": report.ssr, "predicate_counts": counts,
    }, indent=2), encoding="utf-8")
    with open(eval_dir / "trace.jsonl", "w", encoding="utf-8") as handle:
        for row in trace:
            handle.write(json.dumps(row) + "\n")
    (eval_dir / "predicates.json").write_text(
        json.dumps(report.predicates, indent=2), encoding="utf-8")
    return report
