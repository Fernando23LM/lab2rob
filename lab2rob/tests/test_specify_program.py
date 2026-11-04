"""SC-01: program planner maps grounded steps to registered skills only."""

import json

from lab2rob.ground.rules import Grounding, Quantity, ground_protocol
from lab2rob.ingest.bioprot import load_bioprot_id
from lab2rob.specify.program import plan_protocol, plan_step, save_pairs


def qty(value, unit):
    return Quantity(raw=f"{value} {unit}", value=float(value), unit=unit,
                    approximate=False, pint=f"{value} {unit}")


def make_step(idx=1, text="", action=None, status="grounded", reason=None,
              detail="", quantities=(), entities=()):
    return Grounding(idx=idx, source_text=text, quantities=list(quantities),
                     reagents=[], entities=list(entities), action=action,
                     status=status, reason=reason, detail=detail, by="rules")


def ent(name, scope="in_scope"):
    return {"name": name, "scope": scope}


def test_unknown_skill_becomes_missing_skill():
    out = plan_step(make_step(text="Teleport the tube", action="teleport"))
    assert (out.status, out.reason) == ("out_of_scope", "missing_skill")
    assert out.ops == []


def test_declared_skill_rejected_despite_grounded():
    out = plan_step(make_step(text="Vortex the tubes", action="vortex"))
    assert (out.status, out.reason) == ("out_of_scope", "missing_skill")


def test_transfer_expands_with_bound_volume_and_vessels():
    out = plan_step(make_step(text="Add 25 uL dye to the well", action="transfer",
                              quantities=[qty(25, "ul")],
                              entities=[ent("reagent_reservoir_50ml"),
                                        ent("plate_96_well")]))
    assert out.status == "planned"
    assert [(op["skill"], op["args"]) for op in out.ops] == [
        ("aspirate", {"volume_ul": 25.0}), ("dispense", {"volume_ul": 25.0})]
    assert [op["at"] for op in out.ops] == ["e1", "e2"]
    assert out.entities == ["e1", "e2"]


def test_missing_volume_is_ambiguous():
    out = plan_step(make_step(text="Aspirate media", action="aspirate"))
    assert (out.status, out.reason) == ("out_of_scope", "ambiguous")
    assert "volume" in out.detail


def test_mix_cycles_parsed_or_defaulted():
    parsed = plan_step(make_step(text="Mix 3 times with 100 uL", action="mix_by_pipetting",
                                 quantities=[qty(100, "ul")]))
    assert [op["skill"] for op in parsed.ops] == ["aspirate", "dispense"] * 3
    assert parsed.ops[0]["assumed"].get("cycles") is None
    defaulted = plan_step(make_step(text="Mix with 100 uL", action="mix_by_pipetting",
                                    quantities=[qty(100, "ul")]))
    assert len(defaulted.ops) == 6  # default 3 cycles
    assert "cycles" in defaulted.ops[0]["assumed"]


def test_incubate_hours_overnight_and_split():
    half_hour = plan_step(make_step(text="Incubate for 30 min", action="incubate",
                                    quantities=[qty(30, "min")]))
    assert [(op["skill"], op["args"]) for op in half_hour.ops] == [
        ("incubate", {"hours": 0.5})]
    night = plan_step(make_step(text="Incubate overnight", action="incubate"))
    assert night.ops[0]["args"] == {"hours": 16.0}
    assert "hours" in night.ops[0]["assumed"]
    long = plan_step(make_step(text="Incubate for 48 h", action="incubate",
                               quantities=[qty(48, "h")]))
    assert [op["args"] for op in long.ops] == [{"hours": 24.0}, {"hours": 24.0}]
    missing = plan_step(make_step(text="Incubate a while", action="incubate"))
    assert (missing.status, missing.reason) == ("out_of_scope", "ambiguous")


def test_wait_splits_at_sim_cap():
    out = plan_step(make_step(text="Wait 12 s", action="wait",
                              quantities=[qty(12, "s")]))
    assert [op["args"] for op in out.ops] == [
        {"seconds": 5.0}, {"seconds": 5.0}, {"seconds": 2.0}]


def test_reason_categories():
    cases = [(("out_of_scope", "not_implemented"), "missing_skill"),
             (("out_of_scope", "missing_skill"), "missing_skill"),
             (("out_of_scope", "missing_asset"), "missing_asset"),
             (("out_of_scope", "unmodeled_physics"), "missing_physics"),
             (("ambiguous", "ambiguous_text"), "ambiguous"),
             (("needs_llm", "unknown_skill"), "ambiguous")]
    for (status, reason), category in cases:
        out = plan_step(make_step(text="t", status=status, reason=reason,
                                  detail="why"))
        assert (out.status, out.reason) == ("out_of_scope", category), (status, reason)


def test_out_of_scope_entity_blocks_planning():
    out = plan_step(make_step(text="Add 1 uL to the Ifuge", action="transfer",
                              quantities=[qty(1, "ul")],
                              entities=[ent("plate_96_well"),
                                        ent("mystery_ifuge", "out_of_scope")]))
    assert (out.status, out.reason) == ("out_of_scope", "missing_asset")


def test_serial_dilution_points_and_fold():
    out = plan_step(make_step(text="8-point two-fold serial dilution with 100 uL",
                              action="serial_dilution",
                              quantities=[qty(100, "ul")]))
    assert len(out.ops) == 14  # 7 links x aspirate/dispense
    assert "8-point 2-fold" in out.ops[0]["assumed"]["series"]
    ranged = plan_step(make_step(text="8-12 point serial dilution with 100 uL",
                                 action="serial_dilution",
                                 quantities=[qty(100, "ul")]))
    assert (ranged.status, ranged.reason) == ("out_of_scope", "ambiguous")
    vague = plan_step(make_step(text="Serial dilution with 100 uL",
                                action="serial_dilution",
                                quantities=[qty(100, "ul")]))
    assert (vague.status, vague.reason) == ("out_of_scope", "ambiguous")


def test_tier1_end_to_end_and_pairs_roundtrip(tmp_path):
    program = plan_protocol(ground_protocol(load_bioprot_id("1779")))
    assert program.protocol_id == "1779"
    planned = [s for s in program.steps if s.status == "planned"]
    assert [(s.idx, [o["skill"] for o in s.ops]) for s in planned] == [
        (3, ["aspirate", "dispense"])]
    assert planned[0].ops[0]["args"] == {"volume_ul": 75.0}
    assert [(o.step, o.reason) for o in program.out_of_scope] == [
        (1, "missing_skill"), (2, "missing_skill"), (4, "missing_skill")]
    pairs_file = save_pairs(program, tmp_path / "pairs.jsonl")
    pairs = [json.loads(line) for line in pairs_file.read_text(encoding="utf-8").splitlines()]
    assert len(pairs) == 4
    assert pairs[2]["skills"] == ["aspirate", "dispense"]
    assert pairs[0]["reason"] == "missing_skill"


def _program(*groundings):
    from types import SimpleNamespace
    return plan_protocol(SimpleNamespace(protocol_id="t", title="t", steps=list(groundings)))


def test_tools_never_take_a_vessel_role():
    """'... pipette to mix' names the tool; binding a role to it left ops unassigned."""
    out = plan_step(make_step(text="Add 8 uL of glutaraldehyde to each sample. Pipette to mix.",
                              action="transfer", quantities=[qty(8, "ul")],
                              entities=[ent("pipette")]))
    assert out.status == "planned"
    assert [op["at"] for op in out.ops] == [None, None]
    assert out.entities  # still recorded for traceability


def test_one_named_vessel_is_the_destination():
    out = plan_step(make_step(text="Add 50 uL of stop solution to each well of the plate.",
                              action="transfer", quantities=[qty(50, "ul")],
                              entities=[ent("plate_96_well")]))
    source, target = out.ops
    assert source["at"] is None and target["at"] == out.entities[0]
    assert "source" in source["assumed"]


def test_dispense_with_an_empty_pipette_draws_first():
    program = _program(
        make_step(idx=1, text="Dispense 25 uL into each spot.", action="dispense",
                  quantities=[qty(25, "ul")]),
        make_step(idx=2, text="Aspirate 40 uL of buffer.", action="aspirate",
                  quantities=[qty(40, "ul")]),
        make_step(idx=3, text="Dispense 30 uL into the well.", action="dispense",
                  quantities=[qty(30, "ul")]),
        make_step(idx=4, text="Dispense 15 uL into the next well.", action="dispense",
                  quantities=[qty(15, "ul")]))
    ops = {s.idx: [(o["skill"], o["args"]["volume_ul"]) for o in s.ops] for s in program.steps}
    assert ops[1] == [("aspirate", 25.0), ("dispense", 25.0)]
    assert ops[2] == [("aspirate", 40.0)]
    assert ops[3] == [("dispense", 30.0)]  # covered by step 2's draw
    assert ops[4] == [("aspirate", 5.0), ("dispense", 15.0)]  # only the shortfall
    assert "empty pipette" in program.steps[0].ops[0]["assumed"]["source"]


def test_weighing_binds_mass_and_reagent_per_stated_mass():
    step = make_step(text="Weigh out 0.9 g sodium chloride (NaCl and 4.0 g PEG 8000.",
                     action="weigh_sample", quantities=[qty(0.9, "g"), qty(4000, "mg")])
    step.reagents = ["sodium chloride (NaCl", "PEG 8000"]
    out = plan_step(step)
    assert out.status == "planned"
    assert [o["args"] for o in out.ops] == [{"mass_g": 0.9, "reagent": "sodium chloride (NaCl)"},
                                            {"mass_g": 4.0, "reagent": "PEG 8000"}]
    unpaired = make_step(text="Weigh 1 g and 2 g.", action="weigh_sample",
                         quantities=[qty(1, "g"), qty(2, "g")])
    out = plan_step(unpaired)
    assert all(o["args"]["reagent"] is None and o["assumed"]["reagent"] for o in out.ops)
    no_mass = plan_step(make_step(text="Weigh the tube.", action="weigh_sample"))
    assert (no_mass.status, no_mass.reason) == ("out_of_scope", "ambiguous")


def test_each_scene_family_plans_only_its_own_steps():
    weigh = make_step(idx=1, text="Weigh 0.9 g NaCl.", action="weigh_sample",
                      quantities=[qty(0.9, "g")])
    weigh.reagents = ["NaCl"]
    pipette = make_step(idx=2, text="Add 25 uL to the well", action="transfer",
                        quantities=[qty(25, "ul")])
    from types import SimpleNamespace
    grounding = SimpleNamespace(protocol_id="t", title="t", steps=[weigh, pipette])
    default = plan_protocol(grounding)
    assert [(s.status, s.reason) for s in default.steps] == [("out_of_scope", "other_scene"),
                                                              ("planned", None)]
    weighing = plan_protocol(grounding, family="weigh")
    assert [(s.status, s.reason) for s in weighing.steps] == [("planned", None),
                                                               ("out_of_scope", "other_scene")]
    import pytest
    with pytest.raises(ValueError):
        plan_protocol(grounding, family="freezer")


def test_a_weighing_that_fails_to_bind_still_belongs_to_the_weighing_scene():
    """Its equipment must not leak into the pipetting scene (it chose the wrong family)."""
    from types import SimpleNamespace
    weigh = make_step(idx=1, text="Shave the tumours off the dish into a tared tube and weigh.",
                      action="weigh_sample", entities=[ent("culture_dish"), ent("tube_vial")])
    pipette = make_step(idx=2, text="Add 25 uL to the well", action="transfer",
                        quantities=[qty(25, "ul")])
    program = plan_protocol(SimpleNamespace(protocol_id="t", title="t", steps=[weigh, pipette]))
    assert (program.steps[0].status, program.steps[0].reason) == ("out_of_scope", "other_scene")
    assert [e.asset for e in program.entities] == []


def test_centrifuge_records_speed_and_time_and_counts_tubes():
    step = make_step(text="Centrifuge the two tubes at 12,000 x g for 5 min.", action="centrifuge",
                     quantities=[qty(5, "min")])
    out = plan_step(step)
    assert out.status == "planned"
    assert out.ops[0]["args"] == {"tubes": 2, "time_s": 300.0, "speed": "12,000 x g"}
    assert out.ops[0]["assumed"] == {}
    brief = plan_step(make_step(text="Spin down briefly.", action="centrifuge"))
    assert brief.ops[0]["args"] == {"tubes": 2, "time_s": None, "speed": None}
    assert "no tube count" in brief.ops[0]["assumed"]["tubes"]
    four = plan_step(make_step(text="Spin the 4 tubes at 1000 rpm.", action="centrifuge"))
    assert four.ops[0]["args"]["tubes"] == 4 and four.ops[0]["args"]["speed"] == "1000 rpm"
