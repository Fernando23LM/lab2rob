"""SC-02: independent judge planner and outcome checks."""

import inspect
import json

import pytest

from lab2rob.ground.rules import (
    Grounding,
    ProtocolGrounding,
    Quantity,
    ground_protocol,
)
from lab2rob.ingest.bioprot import load_bioprot_id
from lab2rob.registry.predicates import PREDICATES
from lab2rob.schema import OutcomeCheck
from lab2rob.specify.judge import plan_judge, save_judge
from lab2rob.specify.outcome import evaluate_outcome


def qty(value, unit):
    return Quantity(raw=f"{value} {unit}", value=float(value), unit=unit,
                    approximate=False, pint=f"{value} {unit}")


def make_step(idx=1, text="", action="transfer", status="grounded",
              quantities=(), entities=()):
    return Grounding(idx=idx, source_text=text, quantities=list(quantities),
                     reagents=[], entities=list(entities), action=action,
                     status=status, reason=None, detail="", by="rules")


def make_grounding(steps, title="", description=""):
    return ProtocolGrounding(protocol_id="t", title=title,
                             description=description, steps=list(steps))


def ent(name, scope="in_scope"):
    return {"name": name, "scope": scope}


def test_judge_takes_no_program_input():
    assert list(inspect.signature(plan_judge).parameters) == ["protocol_grounding"]


def test_liquid_step_gets_volume_and_no_spill():
    judge = plan_judge(make_grounding(
        [make_step(text="Add 25 uL dye", quantities=[qty(25, "ul")],
                   entities=[ent("plate_96_well")])]))
    preds = [(p.pred, p.args, p.target, p.tol) for p in judge.steps[0].predicates]
    assert preds == [("no_spill", [], None, 0.0),
                     ("volume", ["plate_96_well"], 25.0, 1.25)]


def test_predicates_execute_against_state():
    judge = plan_judge(make_grounding(
        [make_step(text="Add 25 uL dye", quantities=[qty(25, "ul")],
                   entities=[ent("plate_96_well")])]))
    volume = next(p for p in judge.steps[0].predicates if p.pred == "volume")
    run = PREDICATES["volume"]
    assert run({"wells": {"plate_96_well": {"volume_ul": 25.0}}},
               volume.args[0], volume.target, volume.tol).passed is True
    assert run({"wells": {"plate_96_well": {"volume_ul": 30.0}}},
               volume.args[0], volume.target, volume.tol).passed is False
    spill = PREDICATES["no_spill"]
    assert spill({"spilled_ul": 0.0}, 0.0).passed is True
    assert spill({"spilled_ul": 3.0}, 0.0).passed is False


def test_unbound_well_marked_when_text_names_no_vessel():
    judge = plan_judge(make_grounding(
        [make_step(text="Add 25 uL dye", quantities=[qty(25, "ul")])]))
    volume = next(p for p in judge.steps[0].predicates if p.pred == "volume")
    assert volume.args == ["unbound"]


def test_nonliquid_grounded_steps_get_no_checks():
    judge = plan_judge(make_grounding(
        [make_step(text="Incubate for 30 min", action="incubate",
                   quantities=[qty(30, "min")])]))
    assert judge.steps[0].predicates == []


def test_unplannable_steps_get_no_judge_step():
    grounded = make_step(idx=1, text="Add 25 uL dye", quantities=[qty(25, "ul")])
    oos = make_step(idx=2, text="Spin", action="centrifuge",
                    status="out_of_scope")
    judge = plan_judge(make_grounding([grounded, oos]))
    assert [s.idx for s in judge.steps] == [1]


def test_outcome_triggers_and_recipe_guard():
    bradford = plan_judge(make_grounding([], title="Bradford standard curve"))
    assert [(o.type, o.quantity) for o in bradford.outcomes] == [
        ("concentration", "mg_per_ml")]
    ic50 = plan_judge(make_grounding([], description="Report the IC50 after 48 h."))
    assert [o.type for o in ic50.outcomes] == ["ic50"]
    via = plan_judge(make_grounding([], title="Check cell viability"))
    assert [o.type for o in via.outcomes] == ["viable_cells"]
    # Recipe concentrations are targets to make, not outcomes to report.
    assert plan_judge(make_grounding([], description="1% (w/v) SDS buffer")).outcomes == []
    assert plan_judge(ground_protocol(load_bioprot_id("1778"))).outcomes == []


def test_evaluate_outcome_on_sample_episode():
    check = OutcomeCheck(type="concentration", quantity="mg_per_ml", tol_rel=0.15)
    assert evaluate_outcome(check, {"reported": 2.2, "truth": 2.0})["passed"] is True
    failed = evaluate_outcome(check, {"reported": 2.6, "truth": 2.0})
    assert failed["passed"] is False and failed["error_abs"] == pytest.approx(0.6)
    missing = evaluate_outcome(check, {"reported": 2.0})
    assert missing["passed"] is False and "truth" in missing["detail"]
    absolute = OutcomeCheck(type="concentration", tol_abs=0.05)
    assert evaluate_outcome(absolute, {"reported": 2.04, "truth": 2.0})["passed"] is True


def test_save_judge_roundtrip(tmp_path):
    judge = plan_judge(ground_protocol(load_bioprot_id("1779")))
    rows = [json.loads(line) for line in
            save_judge(judge, tmp_path / "judge.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {(r["step_idx"], r["pred"]) for r in rows} == {
        (3, "no_spill"), (3, "volume")}
    assert all(r["source_text"].startswith("Add 75") for r in rows)
    assert all(r["protocol_id"] == "1779" for r in rows)


def test_tier1_end_to_end():
    judge = plan_judge(ground_protocol(load_bioprot_id("1779")))
    assert [s.idx for s in judge.steps] == [3]
    volume = next(p for s in judge.steps for p in s.predicates if p.pred == "volume")
    assert (volume.target, volume.tol) == (75.0, 3.75)
    assert judge.outcomes == []


def test_judge_never_checks_volume_in_the_pipette():
    from lab2rob.ground.rules import Grounding, Quantity
    from types import SimpleNamespace
    step = Grounding(idx=1, source_text="Add 8 uL to the plate. Pipette to mix.",
                     quantities=[Quantity(raw="8 ul", value=8.0, unit="ul",
                                          approximate=False, pint="8 ul")],
                     entities=[{"name": "pipette", "scope": "in_scope"},
                               {"name": "plate_96_well", "scope": "in_scope"}],
                     action="transfer", status="grounded", by="rules")
    judge = plan_judge(SimpleNamespace(protocol_id="t", title="t", description="",
                                       steps=[step]))
    volume = [p for p in judge.steps[0].predicates if p.pred == "volume"]
    assert volume[0].args == ["plate_96_well"]


def test_judge_checks_the_balance_for_each_weighing():
    from lab2rob.ground.rules import Grounding, Quantity
    from types import SimpleNamespace

    def q(v, u):
        return Quantity(raw=f"{v} {u}", value=float(v), unit=u, approximate=False, pint=f"{v} {u}")
    step = Grounding(idx=3, source_text="Weigh out 0.9 g NaCl and 4 g PEG.",
                     quantities=[q(0.9, "g"), q(4, "g")], reagents=["NaCl", "PEG"],
                     action="weigh_sample", status="grounded", by="rules")
    judge = plan_judge(SimpleNamespace(protocol_id="t", title="t", description="", steps=[step]))
    preds = [(p.pred, p.args, p.target, p.tol) for p in judge.steps[0].predicates]
    assert preds == [("mass_on_balance", ["NaCl"], 0.9, 0.15),
                     ("mass_on_balance", ["PEG"], 4.0, 0.2)]
    ok = PREDICATES["mass_on_balance"]({"balance_g": 0.99, "pan_touched": False}, 0.9, 0.15)
    touched = PREDICATES["mass_on_balance"]({"balance_g": 0.9, "pan_touched": True}, 0.9, 0.15)
    assert ok.passed and not touched.passed
    assert not PREDICATES["mass_on_balance"]({}, 0.9, 0.15).passed
