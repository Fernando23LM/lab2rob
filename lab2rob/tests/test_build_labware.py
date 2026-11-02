"""ST-05: labware from Opentrons definitions, placed by the scene builder."""

import mujoco
import numpy as np
import pytest

from lab2rob.build import labware
from lab2rob.build.scene import SceneRejected, _biolab, build_scene, reach_envelope, to_mujoco
from lab2rob.certify.run import certify
from lab2rob.registry import load_assets
from lab2rob.schema import Entity, StepPredicate
from lab2rob.specify.judge import JudgeStep, PlannedJudge
from lab2rob.specify.program import PlannedProgram, PlannedStep


def op(skill, args=None, at=None):
    return {"skill": skill, "args": args or {}, "at": at, "assumed": {}}


def prog(steps_ops, entities=()):
    steps = [PlannedStep(idx=i + 1, source_text=f"s{i + 1}", status="planned",
                         ops=list(ops))
             for i, ops in enumerate(steps_ops)]
    return PlannedProgram(protocol_id="t", title="t", steps=steps,
                          entities=[Entity(uid=u, asset=a) for u, a in entities])


def transfer_into(uid, asset, volume=50.0):
    return prog([[op("aspirate", {"volume_ul": volume}),
                  op("dispense", {"volume_ul": volume}, at=uid)]],
                entities=[(uid, asset)])


@pytest.mark.parametrize("asset", sorted(a for a, k in labware.KINDS.items() if k.liquid))
def test_well_capacity_matches_definition(asset):
    kind = labware.KINDS[asset]
    rated = labware.load_definition(kind.definition)["wells"]["A1"]["totalLiquidVolume"]
    well = labware.geometry(asset).wells[0]
    capacity = (2 * well.half * 1000) ** 2 * (well.rim - well.floor) * 1000
    assert capacity <= rated * 1.0001
    assert capacity >= rated * 0.85  # flattened bottoms lose a little, never gain


def test_registry_entries_carry_opentrons_license():
    entries = {a["name"]: a for a in load_assets()}
    for asset, kind in labware.KINDS.items():
        assert entries[asset]["status"] == "implemented"
        assert "Apache-2.0" in entries[asset]["license"]
        assert kind.definition in entries[asset]["source"]


def test_96_well_plate_placed_and_compiles():
    protein, _, _ = _biolab()
    built = build_scene(transfer_into("plate_1", "plate_96_well"), seed=0)
    assert built.substitutions == []
    assert built.vessels["plate_1"] == {"asset": "plate_96_well", "slot": "plate_1:A1"}
    assert built.assignments == {"1:0": "stock", "1:1": "plate_1:A1"}
    (record,) = built.labware
    assert record["asset"] == "plate_96_well" and "Apache-2.0" in record["license"]
    assert isinstance(built.layout, protein.ProteinLayout)
    assert len(built.layout.wells()) == 12 + 96
    model = to_mujoco(built)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate_1") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "liquid_plate_1:H12") >= 0


@pytest.mark.parametrize("seed", range(5))
def test_placement_in_reach_and_clear_of_family_labware(seed):
    program = prog([[op("aspirate", {"volume_ul": 50.0}, at="tube_1"),
                     op("dispense", {"volume_ul": 50.0}, at="plate_1")]],
                   entities=[("tube_1", "microcentrifuge_tube_1_5ml"),
                             ("plate_1", "plate_96_well"), ("tips_1", "tip_rack")])
    built = build_scene(program, seed=seed)
    assert {r["asset"] for r in built.labware} == {
        "microcentrifuge_tube_1_5ml", "plate_96_well", "tip_rack"}
    envelope = reach_envelope()
    base = np.array(envelope["base"])
    family_wells = [w for w in built.layout.wells() if ":" not in w.name]
    for placed in built.layout.labware:
        corners = placed.corners()
        dists = np.linalg.norm(corners - base, axis=1)
        assert envelope["min_m"] <= dists.min() and dists.max() <= envelope["max_m"]
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        for w in family_wells:
            assert not (lo[0] < w.center[0] < hi[0] and lo[1] < w.center[1] < hi[1]), w.name


def test_deterministic_per_seed():
    program = transfer_into("plate_1", "plate_96_well")
    assert build_scene(program, seed=3).labware == build_scene(program, seed=3).labware
    assert build_scene(program, seed=3).slots == build_scene(program, seed=3).slots


def test_tall_tube_rejected_as_unreachable():
    for asset in ("centrifuge_tube_15ml", "centrifuge_tube_50ml"):
        with pytest.raises(SceneRejected) as exc:
            build_scene(transfer_into("falcon", asset))
        assert exc.value.reason == "unreachable"
        assert "needs tilt" in exc.value.detail


def test_tip_rack_is_a_prop_not_a_vessel():
    built = build_scene(prog([[op("aspirate", {"volume_ul": 25.0}),
                               op("dispense", {"volume_ul": 25.0})]],
                             entities=[("tips_1", "tip_rack")]))
    assert [r["asset"] for r in built.labware] == ["tip_rack"]
    assert "tips_1" not in built.vessels
    with pytest.raises(SceneRejected) as exc:
        build_scene(transfer_into("tips_1", "tip_rack"))
    assert exc.value.reason == "missing_asset"


def test_full_bench_noted_not_rejected():
    plates = [(f"plate_{i}", "plate_96_well") for i in range(8)]
    built = build_scene(prog([[op("aspirate", {"volume_ul": 25.0}),
                               op("dispense", {"volume_ul": 25.0})]], entities=plates))
    assert built.labware  # at least one fits
    assert any("no free bench space" in note for note in built.unplaced)


@pytest.mark.parametrize("uid,asset", [("plate_1", "plate_96_well"),
                                       ("tube_1", "microcentrifuge_tube_1_5ml")])
def test_certify_pipettes_into_new_labware(tmp_path, uid, asset):
    program = transfer_into(uid, asset)
    judge = PlannedJudge(protocol_id="t", title="t", steps=[JudgeStep(
        idx=1, source_text="s1",
        predicates=[StepPredicate(step=1, pred="volume", args=[asset], target=50.0, tol=2.5)])])
    report = certify(program, judge, build_scene(program, seed=0), seed=0, runs_dir=tmp_path)
    assert report.status == "complete", (report.abort_reason, report.abort_detail)
    assert [p["status"] for p in report.predicates] == ["passed"]


# -- AutoBio demo props --------------------------------------------------------

def with_instruments():
    return prog([[op("aspirate", {"volume_ul": 50.0}),
                  op("dispense", {"volume_ul": 50.0}, at="plate_1")]],
                entities=[("plate_1", "plate_96_well"), ("spin_1", "centrifuge_mini"),
                          ("vortex_1", "vortexer")])


def test_demo_props_off_by_default():
    built = build_scene(with_instruments(), seed=0)
    assert built.instruments == [] and built.redistributable
    model = to_mujoco(built)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spin_1/body") == -1


def test_demo_props_placed_and_marked_unlicensed():
    plain = to_mujoco(build_scene(with_instruments(), seed=0))
    built = build_scene(with_instruments(), seed=0, demo_props=True)
    assert [r["asset"] for r in built.instruments] == ["centrifuge_mini", "vortexer"]
    assert not built.redistributable
    assert all("UNLICENSED" in r["license"] for r in built.instruments)
    model = to_mujoco(built)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spin_1/body") >= 0
    assert (model.nu, model.njnt) == (plain.nu, plain.njnt)  # rigid scenery only


def test_certify_runs_with_demo_props(tmp_path):
    program = with_instruments()
    judge = PlannedJudge(protocol_id="t", title="t", steps=[JudgeStep(
        idx=1, source_text="s1",
        predicates=[StepPredicate(step=1, pred="volume", args=["plate_96_well"], target=50.0, tol=2.5)])])
    built = build_scene(program, seed=0, demo_props=True)
    report = certify(program, judge, built, seed=0, runs_dir=tmp_path)
    assert report.status == "complete", (report.abort_reason, report.abort_detail)
    assert [p["status"] for p in report.predicates] == ["passed"]
