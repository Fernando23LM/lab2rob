"""IG-05: rules ground the tier-1 steps; hints are expectations, not inputs."""

from dataclasses import replace

from lab2rob.ground.rules import (
    extract_quantities,
    extract_reagents,
    ground_protocol,
    match_verb,
    parse_steps,
)
from lab2rob.ingest.bioprot import load_bioprot_id, load_tier1_file

# (status, action-or-reason) per step, derived from expected_grounding_hints.
EXPECTED = {
    "1779": [("out_of_scope", "not_implemented"),   # dissolve -> mix (declared)
             ("out_of_scope", "missing_skill"),     # 12.5 mL > pipette, no pour
             ("grounded", "transfer"),              # 75 uL pipetting
             ("out_of_scope", "missing_skill")],    # make up 50 mL > pipette (dilute learned)
    "1780": [("out_of_scope", "not_implemented"),   # dissolve -> mix (declared)
             ("out_of_scope", "unmodeled_physics"),  # adjust pH: no titration
             ("out_of_scope", "missing_skill"),     # make up 50 mL > pipette (dilute learned)
             ("ambiguous", "ambiguous_text")],      # aliquot without a size
    "1778": [("ambiguous", "ambiguous_text"),       # prepare stock, no amounts
             ("ambiguous", "ambiguous_text"),       # prepare stock, no amounts
             ("out_of_scope", "not_implemented"),   # mix (declared)
             ("out_of_scope", "missing_skill"),     # make up 500 mL > pipette (dilute learned)
             ("out_of_scope", "not_implemented")],  # dissolve -> mix (declared)
}


def test_step_counts_and_noise_stripped():
    for raw in load_tier1_file():
        steps = parse_steps(raw.steps_raw)
        assert len(steps) == raw.number_of_steps
        for step in steps:
            assert "amount" not in step.text and "reagents" not in step.text
            assert step.text.strip().lower() != "none"


def test_notes_become_comments():
    steps = parse_steps(load_bioprot_id("1779").steps_raw)
    assert any("carrier nucleic acid" in c for c in steps[2].comments)
    assert "NOTES" not in steps[2].text


def test_continuation_lines_join():
    steps = parse_steps(load_bioprot_id("1778").steps_raw)
    assert "e.g., 5 g SDS in 500 mL" in steps[4].text


def test_quantities_with_pint():
    step3 = parse_steps(load_bioprot_id("1779").steps_raw)[2]
    quantities = extract_quantities(step3.text)
    ul = next(q for q in quantities if q.value == 75)
    assert (ul.value, ul.unit, ul.approximate) == (75.0, "ul", False)
    assert ul.pint is not None
    assert next(q for q in quantities if q.value == 20).unit == "mg/ml"
    approx = extract_quantities("Dissolve 0.75 g CTAB in ~ 20 mL water")
    assert [q.approximate for q in approx] == [False, True]
    assert extract_quantities("Adjust pH to 5.2 with acid") == []
    percent = extract_quantities("to a final concentration of 1% (w/v)")[0]
    assert (percent.value, percent.unit) == (1.0, "%")
    assert extract_quantities("4 M NaCl")[0].unit == "M"


def test_reagents_extracted():
    steps = parse_steps(load_bioprot_id("1779").steps_raw)
    assert "CTAB" in extract_reagents(steps[0].text)
    assert "NaCl" in extract_reagents(steps[1].text)
    assert "E. coli tRNA" in extract_reagents(steps[2].text)
    last = parse_steps(load_bioprot_id("1778").steps_raw)[4]
    assert "SDS" in extract_reagents(last.text)
    assert "DNA Buffer" in extract_reagents(last.text)


def test_verb_mapping():
    assert match_verb("Add 12.5 mL of 4 M NaCl") == ("add", "transfer")
    assert match_verb("Dissolve 0.75 g CTAB")[1] == "mix"
    assert match_verb("Make up to 50 mL with water") == ("make up", "dilute")
    assert match_verb("Mix 50 mL NaCl and 50 mL EDTA")[1] == "mix"


def test_per_step_verdicts_match_hints():
    for raw in load_tier1_file():
        grounding = ground_protocol(raw)
        assert len(grounding.steps) == raw.number_of_steps
        for step, (status, key) in zip(grounding.steps, EXPECTED[raw.protocol_id]):
            assert step.status == status, (raw.protocol_id, step.idx, step.source_text)
            assert (step.action if status == "grounded" else step.reason) == key


def test_pilot_needs_no_llm():
    for raw in load_tier1_file():
        for step in ground_protocol(raw).steps:
            assert step.status != "needs_llm", (raw.protocol_id, step.idx)


def test_report_counts():
    counts = {raw.protocol_id: ground_protocol(raw).counts for raw in load_tier1_file()}
    assert counts["1779"] == {"out_of_scope": 3, "rules": 1}
    assert counts["1780"] == {"out_of_scope": 3, "ambiguous": 1}
    assert counts["1778"] == {"out_of_scope": 3, "ambiguous": 2}


def test_description_flags():
    flags_1779 = ground_protocol(load_bioprot_id("1779")).description_flags
    assert any("Store at" in f["text"] for f in flags_1779)
    assert all(f["reason"] == "unmodeled_physics" for f in flags_1779)
    assert len(ground_protocol(load_bioprot_id("1778")).description_flags) >= 2
    assert ground_protocol(load_bioprot_id("1780")).description_flags == []


def test_hints_are_not_parser_inputs():
    for raw in load_tier1_file():
        assert ground_protocol(replace(raw, hints={})).steps == ground_protocol(raw).steps


def test_plural_time_units_and_nanoliter():
    quantities = extract_quantities("Incubate for 5 minutes, then hold for 30 seconds")
    assert [(q.value, q.unit) for q in quantities] == [(5.0, "min"), (30.0, "s")]
    assert all(q.pint is not None for q in quantities)
    assert extract_quantities("for 2 hours")[0].unit == "h"
    nano = extract_quantities("dispense 500nL per well")[0]
    assert (nano.value, nano.unit) == (500.0, "nl")
    assert nano.pint is not None
