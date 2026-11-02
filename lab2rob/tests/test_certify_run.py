"""SC-04: certifier runs programs, traces, and grades predicates."""

import json

from lab2rob.build.scene import BuiltScene, build_scene
from lab2rob.certify.run import certify, evaluate_predicates
from lab2rob.ground.rules import ground_protocol
from lab2rob.ingest.bioprot import load_bioprot_id
from lab2rob.schema import Entity, StepPredicate
from lab2rob.specify.judge import JudgeStep, PlannedJudge, plan_judge
from lab2rob.specify.program import PlannedProgram, PlannedStep, plan_protocol


def op(skill, args=None, at=None):
    return {"skill": skill, "args": args or {}, "at": at, "assumed": {}}


def prog(steps_ops, entities=()):
    steps = [PlannedStep(idx=i + 1, source_text=f"s{i + 1}", status="planned",
                         ops=list(ops))
             for i, ops in enumerate(steps_ops)]
    return PlannedProgram(protocol_id="t", title="t", steps=steps,
                          entities=[Entity(uid=u, asset=a) for u, a in entities])


def judge_for(preds_by_step):
    steps = [JudgeStep(idx=idx, source_text=f"s{idx}", predicates=list(preds))
             for idx, preds in sorted(preds_by_step.items())]
    return PlannedJudge(protocol_id="t", title="t", steps=steps)


def test_1779_runs_end_to_end(tmp_path):
    grounding = ground_protocol(load_bioprot_id("1779"))
    program = plan_protocol(grounding)
    judge = plan_judge(grounding)
    report = certify(program, judge, build_scene(program, seed=0), seed=0,
                     runs_dir=tmp_path)
    assert report.status == "complete"
    assert report.n_ops == 2
    assert [(p["pred"], p["status"]) for p in report.predicates] == [
        ("no_spill", "passed"), ("volume", "passed")]
    run = json.loads((report.dir / "run.json").read_text(encoding="utf-8"))
    assert run["predicate_counts"] == {"passed": 2, "failed": 0, "skipped": 0}
    assert run["assignments"] == {"3:0": "stock", "3:1": "A1"}
    rows = [json.loads(line) for line in
            (report.dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["kind"] for r in rows] == ["initial", "op", "op"]
    assert rows[0]["state"]["wells"]["stock"]["volume_ul"] == 75.0
    assert rows[0]["prefill"] == {"stock": 75.0}
    assert rows[-1]["state"]["wells"]["A1"]["volume_ul"] == 75.0
    assert rows[-1]["state"]["spilled_ul"] == 0.0
    assert len(json.loads((report.dir / "predicates.json").read_text(encoding="utf-8"))) == 2


def test_evaluate_predicates_grades_and_skips():
    program = prog([[op("aspirate", {"volume_ul": 25.0}),
                     op("dispense", {"volume_ul": 25.0})]])
    built = BuiltScene(family="protein", seed=0, layout=None,
                       assignments={"1:0": "stock", "1:1": "A1"})
    judge = judge_for({1: [StepPredicate(step=1, pred="volume", args=["unbound"],
                                        target=25.0, tol=1.25),
                           StepPredicate(step=1, pred="no_spill", args=[],
                                         target=None, tol=0.0)]})
    good = {"wells": {"A1": {"volume_ul": 25.0, "conc": 0.0}}, "spilled_ul": 0.0}
    rows = evaluate_predicates(judge, program, built, {1: good})
    assert [(r["pred"], r["status"], r["well"]) for r in rows] == [
        ("volume", "passed", "A1"), ("no_spill", "passed", None)]
    bad = {"wells": {"A1": {"volume_ul": 10.0, "conc": 0.0}}, "spilled_ul": 0.0}
    assert evaluate_predicates(judge, program, built, {1: bad})[0]["status"] == "failed"

    aspirate_only = prog([[op("aspirate", {"volume_ul": 25.0})]])
    built_only = BuiltScene(family="protein", seed=0, layout=None,
                            assignments={"1:0": "stock"})
    skipped = evaluate_predicates(judge, aspirate_only, built_only, {1: good})
    assert skipped[0]["status"] == "skipped"  # no dispense target to check

    inplace = BuiltScene(family="protein", seed=0, layout=None,
                         assignments={"1:0": "A1", "1:1": "A1"})
    skipped = evaluate_predicates(judge, program, inplace, {1: good})
    assert skipped[0]["status"] == "skipped"
    assert "in-place" in skipped[0]["detail"]

    named = BuiltScene(family="protein", seed=0, layout=None,
                       assignments={"1:0": "stock", "1:1": "A1"},
                       vessels={"e1": {"asset": "plate_8", "slot": "A1"}},
                       substitutions=[{"from": "plate_96_well", "to": "plate_8"}])
    direct = judge_for({1: [StepPredicate(step=1, pred="volume", args=["plate_8"],
                                         target=25.0, tol=1.25)]})
    assert evaluate_predicates(direct, program, named, {1: good})[0]["well"] == "A1"
    via_sub = judge_for({1: [StepPredicate(step=1, pred="volume", args=["plate_96_well"],
                                          target=25.0, tol=1.25)]})
    assert evaluate_predicates(via_sub, program, named, {1: good})[0]["well"] == "A1"
    unknown = judge_for({1: [StepPredicate(step=1, pred="volume", args=["nope"],
                                          target=25.0, tol=1.25)]})
    assert evaluate_predicates(unknown, program, named, {1: good})[0]["status"] == "skipped"


def test_unknown_skill_aborts_run(tmp_path):
    program = prog([[op("teleport", {"where": "mars"})]])
    judge = judge_for({})
    report = certify(program, judge, build_scene(program, seed=0), seed=0,
                     runs_dir=tmp_path)
    assert report.status == "aborted"
    assert report.abort_reason == "no_executor"
    assert report.predicates == []
    assert (report.dir / "trace.jsonl").exists()


def test_prefill_over_capacity_aborts_fast(tmp_path):
    ops = [op("aspirate", {"volume_ul": 900.0}, at="e1") for _ in range(41)]
    program = prog([ops], entities=[("e1", "plate_8")])
    judge = judge_for({})
    report = certify(program, judge, build_scene(program, seed=0), seed=0,
                     runs_dir=tmp_path)
    assert report.status == "aborted"
    assert report.abort_reason == "over_capacity"
