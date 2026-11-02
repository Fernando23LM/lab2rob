"""IG-05: tier-1 BioProt protocols load with URL, license and raw text kept."""

import pytest

from lab2rob.ingest.bioprot import load_bioprot_id, load_tier1_file
from lab2rob.schema import GroundedStep


def test_tier1_loads_three_protocols():
    protocols = load_tier1_file()
    assert [p.protocol_id for p in protocols] == ["1779", "1780", "1778"]
    assert {p.dataset for p in protocols} == {"bioprot"}
    assert [p.number_of_steps for p in protocols] == [4, 4, 5]


def test_source_and_raw_text_preserved():
    protocol = load_bioprot_id("1779")
    assert protocol.source_url == "https://www.protocols.io/view/1779"
    assert protocol.license.strip()
    assert "1amount" in protocol.steps_raw  # verbatim noise, kept for the grounder
    assert "NOTES" in protocol.steps_raw
    assert protocol.title == "CTAB mix"


def test_to_task_builds_valid_schema_task():
    protocol = load_bioprot_id("1780")
    task = protocol.to_task([GroundedStep(idx=1, source_text="Dissolve ...")])
    assert task.id == "bioprot_1780"
    assert task.source.url == "https://www.protocols.io/view/1780"
    assert task.source.steps_used == [1]
    assert task.instruction.startswith("Sodium acetate solution:")


def test_load_unknown_id_raises():
    with pytest.raises(KeyError):
        load_bioprot_id("nope")


def test_repo_loading_is_an_explicit_stub():
    with pytest.raises(NotImplementedError):
        load_bioprot_id("1779", repo_root="/tmp/bioprot")
