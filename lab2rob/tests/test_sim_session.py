"""Frozen-scene sessions: joint, skill and mixed levels record the same rows."""

import json
import subprocess
import sys

import numpy as np
import pytest

from lab2rob.build import labware
from lab2rob.build.scene import BuiltScene, build_scene
from lab2rob.certify.run import _biolab
from lab2rob.ground.rules import ground_protocol
from lab2rob.ingest.bioprot import load_bioprot_id
from lab2rob.sim import (ACTION_NAMES, InstanceMismatch, LevelError, Session,
                         load_trajectory, make_instance, save_trajectory)
from lab2rob.sim.freeze import REPO_ROOT, end_state_checks, freeze_task
from lab2rob.sim.instance import layout_from_json, layout_to_json
from lab2rob.specify.judge import plan_judge
from lab2rob.specify.program import plan_protocol

GRIP_ABOVE = 0.080
A1_75 = [{"pred": "volume", "well": "A1", "target": 75.0, "tol": 3.75},
         {"pred": "no_spill", "tol": 0.0}]


@pytest.fixture(scope="module")
def protein_instance():
    protein, _ = _biolab()
    built = BuiltScene(family="protein", seed=0, layout=protein.ProteinLayout())
    return make_instance(built, {"stock": 150.0}, 0, A1_75)


def _goto(s, pos, n=80, hold=40, plunger=None):
    """Joint-level move of the right arm: IK once, interpolate rows."""
    q, err = s.ik("right", pos)
    assert err < 1e-3
    row = s.command()
    start = row[7:13].copy()
    for k in range(1, n + 1):
        row[7:13] = start + (q - start) * k / n
        s.step(row)
    if plunger is not None:
        row[14] = plunger
    s.step(row, n=hold)


def _above(well):
    return [well["center"][0], well["center"][1], well["center"][2] + 0.03 + GRIP_ABOVE]


def _inside(well):
    return [well["center"][0], well["center"][1], well["floor_z"] + 0.0015 + GRIP_ABOVE]


def _joint_transfer(s, src, dst, volume):
    w = s.wells()
    _goto(s, _above(w[src]))
    _goto(s, _inside(w[src]), n=60)
    _goto(s, _inside(w[src]), n=1, hold=20, plunger=s.command()[14] + volume)
    _goto(s, _above(w[src]), n=60)
    _goto(s, _above(w[dst]))
    _goto(s, _inside(w[dst]), n=60)
    _goto(s, _inside(w[dst]), n=1, hold=20, plunger=s.command()[14] - volume)
    _goto(s, _above(w[dst]), n=60)


def test_action_layout_is_14_actuators_plus_plunger(protein_instance):
    s = Session(protein_instance, level="joints")
    acts = s.actuators()
    assert [a["name"] for a in acts] == list(ACTION_NAMES)
    assert len(acts) == 15 and acts[14]["unit"] == "uL"
    assert acts[14]["range"] == [0.0, 200.0]
    assert np.array_equal(s.command()[:14], s.env.data.ctrl)


def test_layout_round_trip_keeps_labware_exact():
    protein, _ = _biolab()
    layout = labware.with_labware(protein.ProteinLayout(), labware.PlacedLabware(
        uid="plate_1", asset="plate_96_well", xy=(0.123456789, -0.3), yaw=0.1))
    doc = layout_to_json(layout)
    again = layout_from_json(json.loads(json.dumps(doc)))
    assert layout_to_json(again) == doc
    names = [w.name for w in layout.wells()]
    assert [w.name for w in again.wells()] == names and "plate_1:A1" in names


def test_rebuilt_scene_must_match_instance_hashes(protein_instance):
    tampered = dict(protein_instance, init_sha="0" * 16)
    with pytest.raises(InstanceMismatch):
        Session(tampered)
    tampered = dict(protein_instance, prefill={"stock": 151.0})
    with pytest.raises(InstanceMismatch):
        Session(tampered)


def test_levels_restrict_calls(protein_instance):
    joints = Session(protein_instance, level="joints")
    with pytest.raises(LevelError):
        joints.move_to("right", [0.0, -0.3, 0.2])
    with pytest.raises(LevelError):
        joints.aspirate(10.0, slot="stock")
    joints.wait(0.1)  # holding still and instruments work at every level
    joints.read_plate()
    skills = Session(protein_instance, level="skills")
    with pytest.raises(LevelError):
        skills.step(skills.command())
    with pytest.raises(ValueError):
        Session(protein_instance, level="free")


def test_joint_level_agent_moves_liquid_and_replays_bit_identical(protein_instance):
    s = Session(protein_instance, level="joints")
    _joint_transfer(s, "stock", "A1", 75.0)
    status = s.status()
    assert status["success"], status["checks"]
    assert s.snapshot()["wells"]["stock"]["volume_ul"] == 75.0
    again = Session(protein_instance, level="joints").replay(s.trajectory())
    assert again["success"] and again["state_sha"] == status["state_sha"]
    assert again["steps"] == status["steps"]


def test_skill_level_rows_replay_at_joint_level(protein_instance):
    s = Session(protein_instance, level="skills")
    got = s.aspirate(75.0, slot="stock")
    assert got["volume_ul"] == 75.0 and got["from"] == "stock"
    put = s.dispense(75.0, slot="A1")
    assert put["volume_ul"] == 75.0 and put["into"] == "A1"
    status = s.status()
    assert status["success"]
    replayed = Session(protein_instance, level="joints").replay(s.trajectory())
    assert replayed["state_sha"] == status["state_sha"] and replayed["success"]


def test_mixed_level_combines_skills_and_raw_rows(protein_instance):
    s = Session(protein_instance, level="mixed")
    assert s.move_tip_into("stock")
    row = s.command()
    row[14] = 40.0  # raw plunger row, no aspirate() call
    s.step(row, n=20)
    assert s.state()["pipette"]["volume_ul"] == pytest.approx(40.0)
    s.withdraw("stock")
    _goto(s, _above(s.wells()["A2"]))  # raw joint motion to the next well
    out = s.dispense(40.0, slot="A2")
    assert out["volume_ul"] == pytest.approx(40.0)
    assert s.snapshot()["wells"]["A2"]["volume_ul"] == 40.0
    replayed = Session(protein_instance, level="joints").replay(s.trajectory())
    assert replayed["state_sha"] == s.state_sha()


def test_plunger_draws_air_outside_wells_and_spills_outside(protein_instance):
    s = Session(protein_instance, level="skills")
    air = s.aspirate(30.0)  # tip at the start pose, over no well
    assert air["volume_ul"] == 0.0 and "air" in air["note"]
    s.dispense(30.0)
    s.aspirate(30.0, slot="stock")
    spill = s.dispense(30.0)  # back at travel height, over the stock trough rim
    assert spill["volume_ul"] == 0.0 and s.snapshot()["spilled_ul"] == 30.0
    assert not s.status()["success"]


def test_instrument_events_are_recorded_and_replayed(protein_instance):
    s = Session(protein_instance, level="skills")
    s.aspirate(75.0, slot="stock")
    s.dispense(75.0, slot="A1")
    first = s.read_plate()
    s.report_concentration(0.5)
    traj = s.trajectory()
    assert [e["call"] for e in traj["events"]] == ["read_plate", "report_concentration"]
    assert traj["events"][0]["at"] < len(traj["rows"])
    other = Session(protein_instance, level="joints")
    other.replay(traj)
    assert other.env.reported == 0.5
    assert other.state_sha() == s.state_sha()  # noise generator advanced the same
    assert other.read_plate() == s.read_plate() and first["A1"] is not None


def test_trajectory_file_round_trip_and_cli_replay(protein_instance, tmp_path):
    s = Session(protein_instance, level="skills")
    s.aspirate(75.0, slot="stock")
    s.dispense(75.0, slot="A1")
    path = s.save_trajectory(path=tmp_path / "trajectory.npz")
    loaded = load_trajectory(path)
    assert np.array_equal(loaded["rows"], s.trajectory()["rows"])
    assert loaded["instance_init_sha"] == protein_instance["init_sha"]
    (tmp_path / "instance.json").write_text(json.dumps(protein_instance))
    out = subprocess.run([sys.executable, "-m", "lab2rob.sim", "replay", "--instance",
                          str(tmp_path / "instance.json"), "--traj", str(path)],
                         capture_output=True, text=True, cwd=REPO_ROOT, check=True)
    status = json.loads(out.stdout.splitlines()[-1])
    assert status["success"] and status["state_sha"] == s.state_sha()
    save_trajectory(loaded, tmp_path / "foreign.npz", init_sha="f" * 16)
    out = subprocess.run([sys.executable, "-m", "lab2rob.sim", "replay", "--instance",
                          str(tmp_path / "instance.json"), "--traj",
                          str(tmp_path / "foreign.npz")],
                         capture_output=True, text=True, cwd=REPO_ROOT, check=True)
    assert "another instance" in out.stdout


def test_bad_rows_are_rejected_or_clipped(protein_instance):
    s = Session(protein_instance, level="joints")
    with pytest.raises(ValueError):
        s.step(np.zeros(14))
    row = s.command()
    row[14] = np.nan
    with pytest.raises(ValueError):
        s.step(row)
    row = s.command()
    row[14] = 1e6
    s.step(row)
    assert s.command()[14] == 200.0 and s.trajectory()["rows"][-1][14] == 200.0


def test_culture_incubation_replays_with_the_clock():
    _, culture = _biolab()
    built = BuiltScene(family="culture", seed=0, layout=culture.CultureLayout())
    instance = make_instance(built, {}, 0, [{"pred": "no_spill", "tol": 0.0}])
    s = Session(instance, level="joints")
    s.incubate(2.0)
    s.inspect("A")
    assert s.get_clock() == pytest.approx(2.0)
    other = Session(instance, level="joints")
    other.replay(s.trajectory())
    assert other.get_clock() == pytest.approx(2.0)
    assert other.state_sha() == s.state_sha()


def test_end_state_checks_come_from_the_judge():
    grounding = ground_protocol(load_bioprot_id("1779"))
    program, judge = plan_protocol(grounding), plan_judge(grounding)
    built = build_scene(program, seed=0)
    assert end_state_checks(judge, program, built) == [
        {"pred": "volume", "well": "A1", "target": 75.0, "tol": 3.75, "from_step": 3},
        {"pred": "no_spill", "tol": 0.0}]


def test_freeze_writes_a_validated_task_and_the_grader_scores_it(tmp_path):
    grounding = ground_protocol(load_bioprot_id("1779"))
    program, judge = plan_protocol(grounding), plan_judge(grounding)
    built = build_scene(program, seed=0)
    report = freeze_task(program, judge, built, 0, tmp_path / "task")
    assert report["frozen"] and not report["nop_success"]
    assert [r["success"] for r in report["replays"]] == [True, True]
    assert len({r["state_sha"] for r in report["replays"]}) == 1
    instance, oracle = tmp_path / "task" / "instance.json", tmp_path / "task" / "oracle.npz"

    def grade(traj, out):
        proc = subprocess.run([sys.executable, "-m", "lab2rob.sim", "grade", "--instance",
                               str(instance), "--traj", str(traj), "--out", str(out)],
                              capture_output=True, text=True, cwd=REPO_ROOT, check=True)
        return json.loads(proc.stdout.splitlines()[-1]), (out / "reward.txt").read_text()

    result, reward = grade(oracle, tmp_path / "v1")
    assert reward == "1\n" and result["deterministic"] == 1
    empty = save_trajectory({"rows": np.zeros((0, 15)), "events": []}, tmp_path / "nop.npz")
    result, reward = grade(empty, tmp_path / "v2")
    assert reward == "0\n" and result["grader_error"] == 0
