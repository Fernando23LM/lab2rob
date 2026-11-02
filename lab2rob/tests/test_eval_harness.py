"""SE-02: sandboxed eval harness runs agents with API-only access."""

import json
import time

import pytest

from lab2rob.build.scene import build_scene
from lab2rob.eval.harness import (
    SandboxViolation,
    _execute,
    build_task_view,
    evaluate,
)
from lab2rob.schema import StepPredicate
from lab2rob.specify.judge import JudgeStep, PlannedJudge
from lab2rob.specify.program import PlannedProgram, PlannedStep


def op(skill, args=None):
    return {"skill": skill, "args": args or {}, "at": None, "assumed": {}}


def prog(steps_ops):
    steps = [PlannedStep(idx=i + 1, source_text=f"s{i + 1}", status="planned",
                         ops=list(ops))
             for i, ops in enumerate(steps_ops)]
    return PlannedProgram(protocol_id="t", title="Move 25 uL", steps=steps)


def judge_for(target=25.0):
    preds = [StepPredicate(step=1, pred="volume", args=["unbound"],
                           target=target, tol=1.25),
             StepPredicate(step=1, pred="no_spill", args=[],
                           target=None, tol=0.0)]
    return PlannedJudge(protocol_id="t", title="t",
                        steps=[JudgeStep(idx=1, source_text="s1",
                                         predicates=preds)])


def honest_actions(built):
    src = built.assignments["1:0"]
    dst = built.assignments["1:1"]
    return [
        {"call": "wells", "args": {}},
        {"call": "aspirate", "args": {"slot": src, "volume_ul": 25.0}},
        {"call": "dispense", "args": {"slot": dst, "volume_ul": 25.0}},
        {"done": True},
    ]


def honest_policy(task_view, history):
    """Dotted-path policy for the spawn import test (mirrors honest script)."""
    calls = [h["call"] for h in history]
    if "wells" not in calls:
        return {"call": "wells", "args": {}}
    if "aspirate" not in calls:
        return {"call": "aspirate",
                "args": {"slot": "stock", "volume_ul": 25.0}}
    if "dispense" not in calls:
        return {"call": "dispense",
                "args": {"slot": "A1", "volume_ul": 25.0}}
    return {"done": True}


def sleeping_policy(task_view, history):
    time.sleep(60)
    return {"done": True}


def test_honest_scripted_agent_completes_and_scores(tmp_path):
    program = prog([[op("aspirate", {"volume_ul": 25.0}),
                     op("dispense", {"volume_ul": 25.0})]])
    built = build_scene(program, seed=0)
    report = evaluate(program, judge_for(), built,
                      {"kind": "scripted",
                       "actions": honest_actions(built)},
                      seed=0, runs_dir=tmp_path)
    assert report.status == "complete"
    assert report.n_ops == 2  # wells is a query, not an op
    assert report.sr == 1.0
    assert report.ssr == 1.0
    assert [(p["pred"], p["status"]) for p in report.predicates] == [
        ("volume", "passed"), ("no_spill", "passed")]
    saved = json.loads((report.dir / "eval.json").read_text(encoding="utf-8"))
    assert saved["sr"] == 1.0 and saved["agent"] == "scripted"
    kinds = [json.loads(line)["kind"] for line in
             (report.dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert kinds[0] == "task_view" and kinds[-1] == "done"
    assert "code" in kinds and "feedback" in kinds


def test_cheating_policy_touching_env_sim_fails(tmp_path):
    program = prog([[op("aspirate", {"volume_ul": 25.0})]])
    built = build_scene(program, seed=0)
    report = evaluate(program, judge_for(), built,
                      {"kind": "scripted",
                       "actions": [{"cheat": "env.sim"}]},
                      seed=0, runs_dir=tmp_path)
    assert report.status == "failed"
    assert report.reason == "sandbox_violation"
    assert report.sr == 0.0 and report.n_ops == 0


def test_parent_rejects_non_allowlisted_call_without_env():
    with pytest.raises(SandboxViolation):
        _execute(None, None, 0.0, "env", {})
    with pytest.raises(SandboxViolation):
        _execute(None, None, 0.0, "move_to", {})


def test_bad_slot_aborts_run(tmp_path):
    program = prog([[op("aspirate", {"volume_ul": 25.0})]])
    built = build_scene(program, seed=0)
    report = evaluate(program, judge_for(), built,
                      {"kind": "scripted", "actions": [
                          {"call": "aspirate",
                           "args": {"slot": "nope", "volume_ul": 5.0}}]},
                      seed=0, runs_dir=tmp_path)
    assert report.status == "aborted"
    assert report.reason == "no_slot"


def test_task_view_has_no_poses_slots_or_judge():
    program = prog([[op("aspirate", {"volume_ul": 25.0})]])
    view = build_task_view(program)
    blob = json.dumps(view).lower()
    for forbidden in ("pos", "slot", "assign", "judge", "pred", "layout",
                      "target", "tol"):
        assert forbidden not in blob
    assert view["steps"][0]["source_text"] == "s1"


def test_zero_shot_replays_scripted_replies(tmp_path):
    program = prog([[op("aspirate", {"volume_ul": 25.0}),
                     op("dispense", {"volume_ul": 25.0})]])
    built = build_scene(program, seed=0)
    src, dst = built.assignments["1:0"], built.assignments["1:1"]
    replies = [
        {"call": "aspirate", "args": {"slot": src, "volume_ul": 25.0}},
        {"call": "dispense", "args": {"slot": dst, "volume_ul": 25.0}},
        {"done": True},
    ]
    report = evaluate(program, judge_for(), built,
                      {"kind": "zero_shot", "replies": replies},
                      seed=0, runs_dir=tmp_path)
    assert report.status == "complete"
    assert report.sr == 1.0
    kinds = [json.loads(line)["kind"] for line in
             (report.dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert kinds.count("prompt") == 3  # one prompt per model round


def test_agent_crash_fails_but_parent_survives(tmp_path):
    program = prog([[op("aspirate", {"volume_ul": 25.0})]])
    built = build_scene(program, seed=0)
    report = evaluate(program, judge_for(), built,
                      {"kind": "scripted", "actions": [{"raise": "boom"}]},
                      seed=0, runs_dir=tmp_path)
    assert report.status == "failed"
    assert report.reason == "agent_error"
    assert (report.dir / "eval.json").exists()  # trace survived the crash


def test_op_budget_stops_runaway_agent(tmp_path):
    program = prog([[op("aspirate", {"volume_ul": 25.0}),
                     op("dispense", {"volume_ul": 25.0})]])
    built = build_scene(program, seed=0)
    report = evaluate(program, judge_for(), built,
                      {"kind": "scripted",
                       "actions": honest_actions(built)},
                      seed=0, runs_dir=tmp_path, max_ops=1)
    assert report.status == "failed"
    assert report.reason == "op_budget"


def test_wrong_volume_scores_zero_success(tmp_path):
    program = prog([[op("aspirate", {"volume_ul": 25.0}),
                     op("dispense", {"volume_ul": 25.0})]])
    built = build_scene(program, seed=0)
    src, dst = built.assignments["1:0"], built.assignments["1:1"]
    report = evaluate(program, judge_for(), built,
                      {"kind": "scripted", "actions": [
                          {"call": "aspirate",
                           "args": {"slot": src, "volume_ul": 25.0}},
                          {"call": "dispense",
                           "args": {"slot": dst, "volume_ul": 10.0}},
                          {"done": True}]},
                      seed=0, runs_dir=tmp_path)
    assert report.status == "complete"
    assert report.reason == "checks_failed"
    assert report.sr == 0.0
    assert report.ssr == 0.5  # volume failed, no_spill passed


def test_policy_agent_runs_in_child_process(tmp_path):
    program = prog([[op("aspirate", {"volume_ul": 25.0}),
                     op("dispense", {"volume_ul": 25.0})]])
    built = build_scene(program, seed=0)
    assert (built.assignments["1:0"], built.assignments["1:1"]) == (
        "stock", "A1")
    report = evaluate(
        program, judge_for(), built,
        {"kind": "policy",
         "dotted": "lab2rob.tests.test_eval_harness:honest_policy"},
        seed=0, runs_dir=tmp_path)
    assert report.status == "complete"
    assert report.sr == 1.0


def test_silent_agent_times_out(tmp_path):
    program = prog([[op("aspirate", {"volume_ul": 25.0})]])
    built = build_scene(program, seed=0)
    report = evaluate(
        program, judge_for(), built,
        {"kind": "policy",
         "dotted": "lab2rob.tests.test_eval_harness:sleeping_policy"},
        seed=0, runs_dir=tmp_path, timeout_s=2)
    assert report.status == "failed"
    assert report.reason == "agent_timeout"
