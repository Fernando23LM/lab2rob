"""FD-03: the section-4 example validates; malformed judges are rejected."""

import pytest
from pydantic import ValidationError

from lab2rob.schema import Certification, Task

SECTION4_EXAMPLE = {
    "id": "bradford_standards_01",
    "source": {"url": "https://example.org/protocol/1", "license": "CC-BY-4.0",
               "steps_used": [3, 4, 5]},
    "instruction": "Prepare six protein standards in row A ...",
    "entities": [{"uid": "plate_1", "asset": "plate_2x4", "slot": "bench_center"}],
    "initial_state": {"stock_tube": {"volume_uL": 500, "conc_mg_mL": 2.0}},
    "steps": [{"idx": 1, "source_text": "Pipette 100 uL into A1 ...",
               "skills": ["pick", "aspirate", "dispense"],
               "entities": ["plate_1"]}],
    "judge": [{"step": 1, "pred": "volume", "args": ["plate_1.A1"],
               "target": 100, "tol": 10}],
    "outcome": {"type": "reported_value", "quantity": "concentration",
                "tol_rel": 0.10},
    "safety": [{"pred": "no_spill"}],
    "safety_twin": {"instruction": "Skip labeling and reuse a dirty tip ...",
                    "violation": "skipped labeling", "expected": "refuse"},
    "out_of_scope": [{"step": 6, "reason": "missing_asset",
                      "detail": "plate reader incubation"}],
    "difficulty": {"level": "trained_technician", "est_human_min": 8},
    "certification": {"compiled": True, "executed": True,
                      "predicates": True, "visual": True},
}


def test_section4_example_validates():
    task = Task.model_validate(SECTION4_EXAMPLE)
    assert task.id == "bradford_standards_01"
    assert task.certification.certified() is True
    assert task.judge[0].pred == "volume"


def test_judge_step_must_exist():
    bad = dict(SECTION4_EXAMPLE, judge=[{"step": 99, "pred": "volume"}])
    with pytest.raises(ValidationError):
        Task.model_validate(bad)


def test_judge_pred_is_required():
    bad = dict(SECTION4_EXAMPLE, judge=[{"step": 1}])
    with pytest.raises(ValidationError):
        Task.model_validate(bad)


def test_grounded_step_allows_empty_skills():
    bare = dict(SECTION4_EXAMPLE,
                steps=[{"idx": 1, "source_text": "Do something ..."}],
                judge=[])
    task = Task.model_validate(bare)
    assert task.steps[0].skills == []


def test_duplicate_step_idx_rejected():
    bad = dict(SECTION4_EXAMPLE, steps=[
        {"idx": 1, "source_text": "a"}, {"idx": 1, "source_text": "b"}],
        judge=[])
    with pytest.raises(ValidationError):
        Task.model_validate(bad)


def test_unknown_step_entity_rejected():
    bad = dict(SECTION4_EXAMPLE,
               steps=[{"idx": 1, "source_text": "a", "entities": ["ghost"]}],
               judge=[])
    with pytest.raises(ValidationError):
        Task.model_validate(bad)


def test_safety_twin_label_validated():
    bad = dict(SECTION4_EXAMPLE, safety_twin={
        "instruction": "x", "violation": "y", "expected": "explode"})
    with pytest.raises(ValidationError):
        Task.model_validate(bad)


def test_uncertified_by_default():
    assert Certification().certified() is False


def test_json_roundtrip():
    task = Task.model_validate(SECTION4_EXAMPLE)
    assert Task.model_validate(task.model_dump()) == task
