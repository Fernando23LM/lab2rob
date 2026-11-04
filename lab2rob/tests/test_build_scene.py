"""SC-03: scene builder places vessels and rejects unreachable layouts."""

import mujoco
import pytest

from lab2rob.build.scene import (
    SceneRejected,
    _biolab,
    build_scene,
    check_reach,
    reach_envelope,
    to_mujoco,
)
from lab2rob.ground.rules import ground_protocol
from lab2rob.ingest.bioprot import load_bioprot_id
from lab2rob.schema import Entity
from lab2rob.specify.program import PlannedProgram, PlannedStep, plan_protocol


def op(skill, args=None, at=None):
    return {"skill": skill, "args": args or {}, "at": at, "assumed": {}}


def prog(steps_ops, entities=()):
    steps = [PlannedStep(idx=i + 1, source_text=f"s{i + 1}", status="planned",
                         ops=list(ops))
             for i, ops in enumerate(steps_ops)]
    return PlannedProgram(protocol_id="t", title="t", steps=steps,
                          entities=[Entity(uid=u, asset=a) for u, a in entities])


def test_tier1_builds_protein_scene():
    program = plan_protocol(ground_protocol(load_bioprot_id("1779")))
    built = build_scene(program, seed=0)
    assert built.family == "protein"
    assert built.assignments == {"3:0": "stock", "3:1": "A1"}
    assert built.aspirate_sources == ["stock"]
    assert built.substitutions == []


def test_deterministic_per_seed():
    program = plan_protocol(ground_protocol(load_bioprot_id("1779")))
    first, second = build_scene(program, seed=7), build_scene(program, seed=7)
    assert first.assignments == second.assignments
    assert first.slots == second.slots


def test_envelope_from_registry_and_boundaries():
    envelope = reach_envelope()
    assert (envelope["min_m"], envelope["max_m"], envelope["height_max_m"]) == (0.10, 0.35, 0.16)
    base_x, base_y = envelope["base"]
    assert check_reach({"ok": (base_x + 0.2, base_y, 0.02)}) == []
    (far,) = check_reach({"far": (base_x + 0.5, base_y, 0.02)})
    assert "envelope 0.1-0.35 m" in far
    (high,) = check_reach({"high": (base_x + 0.2, base_y, 0.20)})
    assert "tilt" in high


def test_unreachable_layout_rejected_with_reason():
    protein, _, _ = _biolab()
    layout = protein.ProteinLayout()
    layout.plate.xy = (0.5, 0.5)  # far outside the arm envelope
    with pytest.raises(SceneRejected) as exc:
        build_scene(prog([[op("dispense", {"volume_ul": 25.0})]]), layout=layout)
    assert exc.value.reason == "unreachable"
    assert "A1" in exc.value.detail and "envelope" in exc.value.detail


def test_missing_asset_rejected():
    with pytest.raises(SceneRejected) as exc:
        build_scene(prog([[op("aspirate", {"volume_ul": 25.0}, at="e1")]],
                         entities=[("e1", "tip_rack")]))
    assert exc.value.reason == "missing_asset"


def test_named_tube_rejected_needs_rack():
    with pytest.raises(SceneRejected) as exc:
        build_scene(prog([[op("aspirate", {"volume_ul": 25.0}, at="e1")]],
                         entities=[("e1", "tube_vial")]))
    assert exc.value.reason == "missing_asset"
    assert "rack" in exc.value.detail


def test_substitution_recorded():
    built = build_scene(
        prog([[op("aspirate", {"volume_ul": 25.0}, at="e1"),
               op("dispense", {"volume_ul": 25.0}, at="e1")]],
             entities=[("e1", "reagent_reservoir_50ml")]))
    assert built.substitutions == [{"from": "reagent_reservoir_50ml", "to": "reservoir_block"}]
    assert built.vessels["e1"]["asset"] == "reservoir_block"


def test_equipment_needs_no_placement():
    built = build_scene(
        prog([[op("aspirate", {"volume_ul": 25.0}, at="e1")]],
             entities=[("e1", "incubator_co2")]))
    assert built.unplaced == ["incubator_co2: no physical model needed"]
    # Equipment is not a vessel: the op falls back to a scene stock instead of
    # being left without a well (which certification then aborted on).
    assert built.assignments == {"1:0": "stock"}


def test_over_capacity_rejected():
    with pytest.raises(SceneRejected) as exc:
        build_scene(prog([[op("aspirate", {"volume_ul": 50000.0})]]))
    assert exc.value.reason == "over_capacity"
    twelve = [[op("dispense", {"volume_ul": 25.0})] for _ in range(12)]
    assert len(build_scene(prog(twelve)).assignments) == 12
    with pytest.raises(SceneRejected) as exc:
        build_scene(prog(twelve + [[op("dispense", {"volume_ul": 25.0})]]))
    assert exc.value.reason == "over_capacity"


def test_family_routing():
    ml = prog([[op("aspirate", {"volume_ul": 1000.0}),
                op("dispense", {"volume_ul": 1000.0})]])
    assert build_scene(ml).family == "culture"
    assert build_scene(prog([[op("incubate", {"hours": 16.0})]])).family == "culture"
    reader = prog([[op("aspirate", {"volume_ul": 25.0}), op("read_plate")]])
    assert build_scene(reader).family == "protein"
    with pytest.raises(SceneRejected) as exc:
        build_scene(prog([[op("inspect", {"dish": "A"}), op("read_plate")]]))
    assert exc.value.reason == "family_conflict"


def test_to_mujoco_compiles():
    program = plan_protocol(ground_protocol(load_bioprot_id("1779")))
    model = to_mujoco(build_scene(program, seed=0))
    assert model.nbody > 5
    liquid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "liquid_A1")
    assert liquid >= 0


def test_every_pipetting_op_gets_a_well_or_the_scene_is_rejected():
    """A tool or equipment ref degrades to an unbound role, never a skipped op."""
    built = build_scene(prog([[op("aspirate", {"volume_ul": 8.0}, at="e1"),
                               op("dispense", {"volume_ul": 8.0}, at="e1")]],
                             entities=[("e1", "pipette")]))
    assert set(built.assignments) == {"1:0", "1:1"}


def test_weighing_scene_accepts_one_salt_weighing_and_rejects_the_rest():
    from lab2rob.build.scene import build_weigh_scene, weigh_programs

    def weigh(grams, reagent):
        return op("weigh_sample", {"mass_g": grams, "reagent": reagent})
    built = build_scene(prog([[weigh(0.9, "NaCl")]]))
    assert built.family == "weigh" and built.assignments == {"1:0": "balance"}
    for steps, reason in (([[weigh(0.9, "NaCl")], [weigh(1.0, "KCl")]], "over_capacity"),
                          ([[weigh(3.0, "NaCl")]], "over_capacity"),
                          ([[weigh(1.0, "agarose")]], "missing_asset"),
                          ([[weigh(1.0, None)]], "missing_asset"),
                          ([[weigh(1.0, "NaCl")], [op("dispense", {"volume_ul": 5.0})]],
                           "family_conflict")):
        with pytest.raises(SceneRejected) as exc:
            build_weigh_scene(prog(steps))
        assert exc.value.reason == reason
    singles = weigh_programs(prog([[weigh(0.9, "NaCl"), weigh(1.0, "KCl")], [weigh(1.2, "salt")]]))
    assert [[(s.status, [o["args"]["mass_g"] for o in s.ops]) for s in p.steps] for p in singles] == [
        [("planned", [0.9]), ("out_of_scope", [])],
        [("planned", [1.0]), ("out_of_scope", [])],
        [("out_of_scope", []), ("planned", [1.2])]]


def test_centrifuge_scene_takes_two_microtubes_and_nothing_else():
    from lab2rob.build.scene import build_centrifuge_scene

    def spin(tubes=2, text="Centrifuge at 12,000 x g for 1 min."):
        p = prog([[op("centrifuge", {"tubes": tubes, "time_s": 60.0, "speed": "12,000 x g"})]])
        p.steps[0].source_text = text
        return p
    built = build_scene(spin())
    assert built.family == "centrifuge" and built.assignments == {"1:0": "rotor"}
    for program, reason in ((spin(4), "over_capacity"),
                            (spin(text="Spin the 50 mL conical tubes at 300 g."), "missing_asset"),
                            (spin(text="Centrifuge the plate at 1000 rpm."), "missing_asset")):
        with pytest.raises(SceneRejected) as exc:
            build_centrifuge_scene(program)
        assert exc.value.reason == reason
    mixed = prog([[op("centrifuge", {"tubes": 2})], [op("dispense", {"volume_ul": 5.0})]])
    with pytest.raises(SceneRejected) as exc:
        build_centrifuge_scene(mixed)
    assert exc.value.reason == "family_conflict"
