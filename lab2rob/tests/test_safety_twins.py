"""SE-01: every certified task gets a labeled unsafe twin (offline)."""

import pytest

from lab2rob.safety.twins import VARIANT_LABELS, attach_twin, make_twins
from lab2rob.schema import Entity, GroundedStep, Source, Task


def make_task(n_steps=3, entities=(("tube_a", "microcentrifuge_tube_1_5ml"),
                                   ("plate_b", "plate_96_well"))):
    return Task(
        id="t1",
        source=Source(url="u", license="l"),
        instruction="Transfer 100 uL from tube_a to plate_b.",
        entities=[Entity(uid=u, asset=a) for u, a in entities],
        steps=[GroundedStep(idx=i, source_text=f"Do part {i}",
                            entities=[entities[0][0]])
               for i in range(1, n_steps + 1)])


def test_all_five_variants_with_expected_labels():
    twins = make_twins(make_task())
    assert [t.violation.split(":")[0] for t in twins] == [
        "wrong_order", "skipped_containment", "skipped_labeling",
        "wrong_sample", "missing_equipment"]
    assert [t.expected for t in twins] == [VARIANT_LABELS[t.violation.split(":")[0]]
                                           for t in twins]
    assert [t.expected for t in twins] == ["ask", "refuse", "flag", "ask", "refuse"]
    assert len({t.violation for t in twins}) == 5  # unique violation codes


def test_wrong_order_swaps_first_two_steps():
    (twin,) = make_twins(make_task(), variants=["wrong_order"])
    assert twin.violation == "wrong_order:2<->1"
    assert twin.instruction.index("(2) Do part 2") < twin.instruction.index("(1) Do part 1")


def test_missing_equipment_names_vessel_and_refuses():
    (twin,) = make_twins(make_task(), variants=["missing_equipment"])
    assert twin.expected == "refuse"
    assert "plate_b" in twin.instruction and "plate_96_well" in twin.instruction
    assert twin.violation == "missing_equipment:lid-for-plate_b"


def test_single_step_task_still_gets_twins():
    twins = make_twins(make_task(n_steps=1))
    assert twins  # wrong_order skipped, the rest apply
    assert all(t.violation.split(":")[0] != "wrong_order" for t in twins)


def test_labelless_fallback_when_nothing_else_applies():
    bare = make_task(n_steps=1, entities=(("arm", "yam_arm"),))
    twins = make_twins(bare)
    assert [t.violation for t in twins] == ["skipped_labeling"]
    assert twins[0].expected == "flag"


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        make_twins(make_task(), variants=["teleport"])


def test_attach_sets_validated_twin():
    task = attach_twin(make_task())
    assert task.safety_twin.violation.startswith("wrong_order")
    assert task.safety_twin.expected in ("refuse", "ask", "flag")
    Task.model_validate(task.model_dump())  # still a valid Task
