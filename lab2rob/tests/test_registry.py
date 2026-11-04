"""FD-04 + FD-05: registry content, status taxonomy and grounder contract."""

import pytest

from lab2rob.registry import (
    asset_names,
    gen_skills,
    lexicon_index,
    load_assets,
    load_lexicon,
    load_skills,
    resolve_asset,
    resolve_skill,
    skill_names,
)

EXPECTED_IMPLEMENTED = [
    "aspirate",
    "dispense",
    "get_balance",
    "get_cap",
    "get_clock",
    "get_gripper_pose",
    "get_labware",
    "get_pipette",
    "get_rack",
    "get_task",
    "get_tube_pose",
    "go_home",
    "incubate",
    "inspect",
    "move_to",
    "read_plate",
    "report_concentration",
    "report_ic50",
    "set_gripper",
    "wait",
]

EXPECTED_MACROS = ["centrifuge", "dilute", "mix_by_pipetting", "serial_dilution", "transfer",
                   "weigh_sample"]

EXPECTED_DECLARED_SKILLS = [
    "aliquot",
    "attach_tip",
    "autoclave",
    "blot_dry",
    "change_tip",
    "concentrate_sample",
    "consolidate",
    "cover_plate",
    "distribute",
    "eject_tip",
    "filter_sterilize",
    "flash_freeze",
    "homogenize",
    "image_plate",
    "measure_fluorescence",
    "measure_luminescence",
    "measure_ph",
    "mix",
    "pool_samples",
    "seal_plate",
    "sonicate",
    "stamp_transfer",
    "thaw",
    "thermocycle",
    "uncover_plate",
    "unseal_plate",
    "uv_sterilize",
    "vortex",
    "wash_plate",
]

EXPECTED_ASSETS = [
    "yam_arm",
    "rack",
    "tube_vial",
    "snap_tube",
    "salt_vial",
    "salt_grain",
    "balance",
    "table",
    "floor",
    "pipette",
    "plate_8",
    "plate_assay_12",
    "plate_dilution_8",
    "reservoir_block",
    "reservoir_culture",
    "culture_dish",
    "microcentrifuge_tube_1_5ml",
    "pipette_tip_200ul",
    "pipette_tip_1000ul",
    "tip_rack",
    "reagent_reservoir_50ml",
    "plate_96_well",
    "plate_384_well",
    "petri_dish_90mm",
    "centrifuge_tube_15ml",
    "centrifuge_tube_50ml",
    "centrifuge_mini",
    "incubator_co2",
    "plate_reader",
    "vortexer",
    "ph_meter",
]

# Helpers defined inside reference-policy *strings*: real defs to a grep, but
# not API functions. transfer is excluded: it is a genuine macro now.
POLICY_STRING_HELPERS = {"lid_middle", "pick", "place", "pour_pose", "tip_to",
                         "go_into", "leave", "volume", "empty", "feed", "split"}


def test_skills_yaml_is_current():
    assert gen_skills.generate() == gen_skills.SKILLS_PATH.read_text(encoding="utf-8")


def test_implemented_skills_match_api():
    assert skill_names(status="implemented") == EXPECTED_IMPLEMENTED


def test_macro_names():
    assert skill_names(status="macro") == EXPECTED_MACROS


def test_declared_skill_names():
    assert skill_names(status="declared") == EXPECTED_DECLARED_SKILLS


def test_reference_policy_helpers_not_skills():
    assert set(skill_names()).isdisjoint(POLICY_STRING_HELPERS)


def test_macro_steps_bottom_out_in_implemented():
    by_name = {s["name"]: s for s in load_skills()}
    implemented = set(skill_names(status="implemented"))

    def closure(steps):
        out = set()
        for step in steps:
            if step in implemented:
                out.add(step)
            else:
                assert by_name[step]["status"] == "macro", step
                out |= closure(by_name[step]["steps"])
        return out

    for macro in (s for s in load_skills() if s["status"] == "macro"):
        assert closure(macro["steps"]) <= implemented
        assert macro["name"] not in closure(macro["steps"])


def test_macro_validation_rejects_dangling_and_cycles():
    implemented = {"aspirate", "dispense"}
    with pytest.raises(ValueError, match="unknown/non-implemented"):
        gen_skills.validate_macros(
            [{"name": "bad", "steps": ["centrifuge"]}], implemented, {})
    with pytest.raises(ValueError, match="recursive"):
        gen_skills.validate_macros(
            [{"name": "loop", "steps": ["loop"]}], implemented, {})


def test_resolve_skill_statuses():
    assert resolve_skill("aspirate")["scope"] == "in_scope"
    assert resolve_skill("transfer")["status"] == "macro"
    assert resolve_skill("transfer")["scope"] == "in_scope"
    decided = resolve_skill("vortex")  # a declared skill (centrifuge is a macro now)
    assert (decided["status"], decided["scope"], decided["reason"]) == (
        "declared", "out_of_scope", "not_implemented")
    missing = resolve_skill("teleport")
    assert (missing["status"], missing["scope"], missing["reason"]) == (
        None, "out_of_scope", "unknown_skill")


def test_resolve_asset_statuses():
    assert resolve_asset("pipette")["scope"] == "in_scope"
    assert resolve_asset("plate_96_well")["scope"] == "in_scope"
    decided = resolve_asset("petri_dish_90mm")
    assert (decided["status"], decided["scope"], decided["reason"]) == (
        "declared", "out_of_scope", "not_implemented")
    missing = resolve_asset("tricorder")
    assert (missing["status"], missing["scope"], missing["reason"]) == (
        None, "out_of_scope", "unknown_asset")


def test_assets_yaml_schema():
    assert asset_names() == EXPECTED_ASSETS
    required = {"name", "status", "description", "dimensions", "affordances",
                "license", "source"}
    for asset in load_assets():
        assert required <= set(asset), asset["name"]
        assert asset["status"] in ("implemented", "declared")
        assert isinstance(asset["affordances"], list)
        assert asset["license"].strip()


def test_lexicon_spot_checks():
    index = lexicon_index()
    assert index["eppendorf"] == "microcentrifuge_tube_1_5ml"
    assert index["spin"] == "centrifuge"
    assert index["Eppendorf".lower()] == "microcentrifuge_tube_1_5ml"
    assert index["pipette"] == "pipette"
    assert index["96-well plate"] == "plate_96_well"


def test_lexicon_canonicals_resolve():
    skills, assets = set(skill_names()), set(asset_names())
    for entry in load_lexicon():
        assert entry["canonical"] in skills | assets, entry["canonical"]
