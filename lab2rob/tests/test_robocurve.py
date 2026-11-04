"""Robocurve task builder: vendored runtime, scoring rules, task.json and checks."""

import json
import subprocess
import sys

import pytest

from lab2rob.robocurve import build


@pytest.fixture(scope="module")
def task_dir(tmp_path_factory):
    """A vendored task folder with a frozen 0.9 g instance (no demo run)."""
    out = tmp_path_factory.mktemp("task")
    build.vendor(out)
    instance = {"schema": "robocurve-bench-instance/1", "family": "weigh", "seed": 0,
                "layout": build.default_weigh_layout(), "target_g": 0.9,
                "scoring": dict(build.WEIGH_SCORING), "run_limit_steps": 400, "control_hz": 20}
    (out / "instance.json").write_text(json.dumps(instance))
    instance.update(build._run(out, "hashes"))
    (out / "instance.json").write_text(json.dumps(instance))
    return out


def _py(task_dir, code):
    proc = subprocess.run([sys.executable, "-c", code], cwd=task_dir, capture_output=True,
                          text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_vendored_folder_is_self_contained(task_dir):
    assert (task_dir / "task.py").is_file()
    for name in build.BIOLAB_FILES:
        assert (task_dir / "sim" / "biolab" / name).is_file()
    assert (task_dir / "sim" / build.YAM_DIR / "LICENSE").is_file()
    out = _py(task_dir, "import json, task, biolab; print(json.dumps([biolab.__file__, "
                        "list(task.ACTION_NAMES), task.CONTROL_HZ]))")
    assert out[0].startswith(str(task_dir))  # the vendored copy, not the repo's
    assert len(out[1]) == 14 and out[2] == 20


def test_doing_nothing_scores_zero_and_replays_agree(task_dir):
    out = _py(task_dir, """
import json, numpy as np, task
s = task.TaskSession()
hold = s.command()
s.step(hold, n=30)
a = s.replay(s.trajectory()); b = task.TaskSession().replay(s.trajectory())
print(json.dumps([a, b]))""")
    a, b = out
    assert a["score"] == 0.0 and a["reason"] == "no salt on the boat"
    assert a["state_sha"] == b["state_sha"] and a["steps"] == 30
    assert a["salt_in_vial_g"] == pytest.approx(build.TOTAL_SALT_G, abs=0.05)


def test_run_limit_and_frozen_scene_are_enforced(task_dir):
    out = _py(task_dir, """
import json, numpy as np, task
s = task.TaskSession()
over = s.replay({"rows": np.tile(s.command(), (401, 1))})
inst = json.load(open("instance.json")); inst["target_g"] = 1.0; inst["layout"]["balance"][0] += 0.01
try:
    task.TaskSession(inst); moved = "accepted"
except RuntimeError as e:
    moved = str(e)
print(json.dumps([over, moved]))""")
    over, moved = out
    assert over["score"] == 0.0 and "run limit" in over["reason"]
    assert "differs from the instance" in moved


@pytest.mark.parametrize("reading,touched,spilled,in_rack,want", [
    (0.90, False, 0.0, False, 1.0), (1.04, False, 0.0, False, 1.0),
    (1.10, False, 0.0, False, 0.75), (0.45, False, 0.0, False, 0.5),
    (1.60, False, 0.0, False, 0.25), (0.05, False, 0.0, True, 0.1),
    (0.05, False, 0.0, False, 0.0), (0.90, True, 0.0, False, 0.0),
    (0.90, False, 0.30, False, 0.5), (0.45, False, 0.30, False, 0.5),
])
def test_scoring_rules_match_the_task_text(task_dir, reading, touched, spilled, in_rack, want):
    got = _py(task_dir, f"""
import json, task
s = task.TaskSession()
s.measure = lambda: {{"reading_g": {reading}, "pan_touched": {touched}, "salt_spilled_g": {spilled},
                      "vial_in_handover_rack": {in_rack}, "salt_in_vial_g": 0, "salt_on_boat_g": 0,
                      "vial_tilt_deg": 0}}
print(json.dumps(s.score()))""")
    assert got["score"] == want, got["reason"]


def test_task_json_has_every_required_field_and_no_identity():
    source = {"target_g": 0.9, "reagent": "NaCl"}
    spec = build.weigh_task_json(0.9, source, build.default_weigh_layout(), 5400, 1.0)
    missing = [k for k in build.REQUIRED if k != "media" and not spec.get(k)]
    assert not missing and spec["category"] in build.CATEGORIES
    assert spec["run_limit"] == "5,400 steps (4.5 minutes at 20 Hz)"
    for word in ("stock rack", "hand-over rack", "vial", "balance", "weigh boat", "YAM"):
        assert word in spec["setup"]
    assert "0.15 g of 0.90 g" in spec["scoring"] and "at most 0.5" in spec["scoring"]
    assert not any(build.IDENTIFYING.search(str(v)) for v in spec.values())


def test_check_task_lists_what_is_missing(tmp_path):
    (tmp_path / "task.json").write_text(json.dumps({
        "title": "A task from /home/someone/lab2rob", "category": "Science",
        "instruction": "Do it.", "setup": "x", "scoring": "well done", "run_limit": "10",
        "fidelity": "x", "scores": "Dream-1 run: not run yet; demo: 1.0"}))
    problems = "\n".join(build.check_task(tmp_path))
    for expected in ("'media' is missing", "category 'Science'", "no Dream-1 result",
                     "'title' may identify", "no numeric conditions", "setup.png is missing",
                     "sim/biolab has no LICENSE"):
        assert expected in problems


def _raw(*steps):
    from lab2rob.ingest.bioprot import RawProtocol
    return RawProtocol(protocol_id="p1", title="T", description="",
                       steps_raw="\n".join(f"{i}. {t}" for i, t in enumerate(steps, 1)),
                       source_url="https://www.protocols.io/view/x", license="CC-BY",
                       number_of_steps=len(steps), dataset="protocolsio")


def test_compile_weighings_goes_through_the_compiler():
    from lab2rob.robocurve.compile import compile_weighings
    records = compile_weighings(_raw(
        "Using an analytical balance, weigh out 0.9 g sodium chloride (NaCl) and 4.0 g PEG 8000.",
        "Weigh 1 g agarose into a bottle.",
        "Add 100 uL of buffer to the tube."))
    got = [(r["source"]["step"], r["source"]["target_g"], r["source"]["reagent"], r["accepted"],
            r["reason"].split(":")[0]) for r in records]
    assert got == [(1, 0.9, "sodium chloride", True, ""),
                   (1, 4.0, "PEG 8000", False, "over_capacity"),
                   (2, 1.0, "agarose", False, "missing_asset")]
    accepted = records[0]
    assert accepted["source"]["full_g"] == 0.15  # from the judge's balance check
    assert accepted["built"].family == "weigh"
    assert accepted["built"].substitutions == [{"from": "sodium chloride", "to": "coarse salt"}]


def test_scoring_tiers_follow_the_judges_tolerance():
    assert build.weigh_scoring(0.15) == {"full_g": 0.15, "three_quarter_g": 0.3, "half_g": 0.5,
                                         "some_salt_g": 0.1, "spill_cap_g": 0.2}
    assert build.weigh_scoring(0.25)["half_g"] == 0.75
    spec = build.weigh_task_json(1.6, {"full_g": 0.25}, build.default_weigh_layout(), 5400, 1.0)
    assert "within 0.25 g of 1.60 g" in spec["scoring"]


def test_build_refuses_masses_outside_the_vial(tmp_path):
    with pytest.raises(build.BuildError):
        build.build_weigh_task(tmp_path / "out", {"target_g": 3.0})


@pytest.fixture(scope="module")
def centrifuge_dir(tmp_path_factory):
    """A vendored centrifuge task folder with a frozen instance (no demo run)."""
    out = tmp_path_factory.mktemp("centrifuge")
    build.vendor(out, "centrifuge")
    instance = {"schema": "robocurve-bench-instance/1", "family": "centrifuge", "seed": 0,
                "layout": build.default_centrifuge_layout(), "scoring": {}, "run_limit_steps": 400,
                "control_hz": 20}
    (out / "instance.json").write_text(json.dumps(instance))
    instance.update(build._run(out, "hashes"))
    (out / "instance.json").write_text(json.dumps(instance))
    return out


def test_centrifuge_scene_keeps_the_arms_action_layout(centrifuge_dir):
    assert (centrifuge_dir / "assets" / "centrifuge_02" / "ATTRIBUTION.md").is_file()
    out = _py(centrifuge_dir, """
import json, task
s = task.TaskSession()
print(json.dumps([s.env.model.nu, round(s.env.lid_angle(), 1), s.measure()]))""")
    nu, lid, got = out
    assert nu == 14 and lid == -8.6
    assert got["where"] == {"tube_a": "rack", "tube_b": "rack"} and got["seated"] == 0


def test_centrifuge_idle_scores_zero_and_replays_agree(centrifuge_dir):
    out = _py(centrifuge_dir, """
import json, task
s = task.TaskSession()
s.step(s.command(), n=30)
print(json.dumps([s.replay(s.trajectory()), task.TaskSession().replay(s.trajectory())]))""")
    a, b = out
    assert a["score"] == 0.0 and a["reason"] == "no tube seated in a slot"
    assert a["state_sha"] == b["state_sha"]


@pytest.mark.parametrize("slots,lid,where,want", [
    ({"tube_a": 1, "tube_b": 5}, True, {}, 1.0),
    ({"tube_a": 1, "tube_b": 5}, False, {}, 0.6),
    ({"tube_a": 1, "tube_b": None}, False, {}, 0.3),
    ({"tube_a": 1, "tube_b": 2}, False, {}, 0.3),
    ({"tube_a": 1, "tube_b": 2}, True, {}, 0.0),
    ({"tube_a": 1, "tube_b": None}, True, {}, 0.0),
    ({"tube_a": None, "tube_b": None}, False, {}, 0.0),
    ({"tube_a": None, "tube_b": None}, True, {}, 0.0),
    ({"tube_a": 1, "tube_b": 5}, False, {"tube_b": "elsewhere"}, 0.3),
])
def test_centrifuge_scoring_rules_match_the_task_text(centrifuge_dir, slots, lid, where, want):
    got = _py(centrifuge_dir, f"""
import json, task
slots = {slots!r}
seated = sorted(k for k in slots.values() if k is not None)
where = {{t: ("slot" if k is not None else "rack") for t, k in slots.items()}}
where.update({where!r})
fake = {{"slots": slots, "where": where, "seated": len(seated), "lid_deg": 90 if {lid} else -8.6,
         "balanced": len(seated) == 2 and seated[1] - seated[0] == 4, "lid_closed": {lid}}}
s = task.TaskSession()
s.measure = lambda: fake
print(json.dumps(s.score()))""")
    assert got["score"] == want, got["reason"]


def test_centrifuge_task_json_and_compile():
    from lab2rob.robocurve.compile import compile_candidates
    spec = build.centrifuge_task_json({"speed": "12,000 x g"}, build.default_centrifuge_layout(), 4200, 1.0)
    assert not [k for k in build.REQUIRED if k != "media" and not spec.get(k)]
    for word in ("microcentrifuge", "rotor", "lid", "tube rack", "1.5 mL", "YAM"):
        assert word in spec["setup"]
    assert "opposite slots" in spec["scoring"] and "at most 0.3" in spec["scoring"]
    assert not any(build.IDENTIFYING.search(str(v)) for v in spec.values())
    records = compile_candidates(_raw("Add 100 uL of buffer to the tube.",
                                      "Centrifuge the tubes at 12,000 x g for 5 min.",
                                      "Spin the conical tubes at 300 g for 5 min."), "centrifuge")
    assert [(r["source"]["step"], r["accepted"], r["reason"].split(":")[0]) for r in records] == [
        (2, True, ""), (3, False, "missing_asset")]
    assert records[0]["source"]["speed"] == "12,000 x g" and records[0]["source"]["time_s"] == 300.0


# --------------------------------------------------------------------- prep_spin
PREP_SOURCE = {"protocol_id": "p1", "step": 3, "speed": "11,000 x g", "additions": [
    {"name": "lysis solution", "volume_ul": 60.0, "tol_ul": 3.0, "step": 1},
    {"name": "precipitation solution", "volume_ul": 20.0, "tol_ul": 1.0, "step": 2}]}


def test_compile_prep_spin_takes_the_additions_before_a_spin():
    from lab2rob.robocurve.compile import compile_prep_spin
    records = compile_prep_spin(_raw(
        "Add 60 uL lysis solution, vortex 5 seconds.",
        "Add 20 uL ethanol, vortex 5 seconds.",
        "Centrifuge the tubes at 11,000 x g for 3 min.",
        "Add 50 uL ethanol and 100 uL isopropanol, mix well.",
        "Centrifuge the tubes at 11,000 x g for 3 min.",
        "Centrifuge the tubes at 11,000 x g for 1 min."))
    got = [(r["source"]["step"], r["additions"], r["accepted"], r["reason"].split(":")[0]) for r in records]
    # A spin straight after another spin has no additions: the centrifuge family's task, not listed.
    assert got == [(3, [1, 2], True, ""), (5, [4], False, "ambiguous")]
    adds = records[0]["source"]["additions"]
    assert [(a["name"], a["volume_ul"], a["tol_ul"]) for a in adds] == [
        ("lysis solution", 60.0, 3.0), ("ethanol", 20.0, 1.0)]
    assert records[0]["built"].family == "centrifuge"


def test_prep_spin_instance_and_task_json():
    inst = build.prep_spin_instance(PREP_SOURCE)
    assert inst["layout"]["troughs"] == ["reagent_1", "reagent_2", "water"]
    assert [r["trough"] for r in inst["reagents"]] == ["reagent_1", "reagent_2"]
    assert inst["scoring"]["reagents"]["lysis solution"] == {"target_ul": 60.0, "tol_ul": 3.0}
    twice = build.prep_spin_instance({"additions": [dict(PREP_SOURCE["additions"][0], step=1),
                                                    dict(PREP_SOURCE["additions"][0], step=2)]})
    assert twice["layout"]["troughs"] == ["reagent_1", "water"]
    assert twice["scoring"]["reagents"]["lysis solution"] == {"target_ul": 120.0, "tol_ul": 6.0}
    spec = build.prep_spin_task_json(PREP_SOURCE, inst, 13200, 1.0)
    assert not [k for k in build.REQUIRED if k != "media" and not spec.get(k)]
    assert "60 uL of lysis solution then 20 uL of precipitation solution" in spec["instruction"]
    for word in ("pipette", "15th action", "microcentrifuge", "troughs", "balance tube"):
        assert word in spec["setup"] + spec["instruction"]
    assert "lysis solution: 60 +- 3 uL" in spec["scoring"] and "Score 0" in spec["scoring"]
    assert not any(build.IDENTIFYING.search(str(v)) for v in spec.values())
    with pytest.raises(build.BuildError):
        build.build_prep_spin_task("unused", {"additions": []})


@pytest.fixture(scope="module")
def prep_dir(tmp_path_factory):
    """A vendored prep_spin task folder with a frozen instance (no demo run)."""
    out = tmp_path_factory.mktemp("prep_spin")
    build.vendor(out, "prep_spin")
    instance = {"schema": "robocurve-bench-instance/1", "family": "prep_spin", "seed": 0,
                **build.prep_spin_instance(PREP_SOURCE), "run_limit_steps": 400, "control_hz": 20}
    (out / "instance.json").write_text(json.dumps(instance))
    instance.update(build._run(out, "hashes"))
    (out / "instance.json").write_text(json.dumps(instance))
    return out


def test_prep_spin_scene_starts_ready_and_idle_scores_zero(prep_dir):
    out = _py(prep_dir, """
import json, task
s = task.TaskSession()
got = s.measure()
s.step(s.command(), n=30)
print(json.dumps([len(s.action_names), s.env.model.nu, got, s.replay(s.trajectory()),
                  task.TaskSession().replay(s.trajectory())]))""")
    n_actions, nu, got, a, b = out
    assert n_actions == 15 and nu == 14
    assert got["where"] == {"sample": "rack", "balance": "rack"} and got["spin"] is None
    assert got["tubes"]["sample"]["volume_ul"] == 0.0
    assert a["score"] == 0.0 and a["state_sha"] == b["state_sha"]


def _prep_state(sample, balance, opposite, where=None):
    tube = lambda reagents: {"volume_ul": sum(reagents.values()), "reagents": reagents}
    return {"slots": {}, "where": where or {"sample": "slot", "balance": "slot"}, "opposite": opposite,
            "tubes": {"sample": tube(sample), "balance": tube(balance)}}


GOOD = {"lysis solution": 60.0, "precipitation solution": 20.0}


@pytest.mark.parametrize("state,spun,spilled,want", [
    (_prep_state(GOOD, {"water": 80.0}, True), True, 0.0, 1.0),
    (_prep_state(GOOD, {"water": 80.0}, True), False, 0.0, 0.8),
    (_prep_state({"lysis solution": 60.0}, {"water": 60.0}, True), True, 0.0, 0.8),
    (_prep_state(dict(GOOD, ethanol=5.0), {"water": 85.0}, True), True, 0.0, 0.6),
    (_prep_state(GOOD, {"water": 40.0}, True), True, 0.0, 0.0),
    (_prep_state(GOOD, {"water": 80.0}, False), True, 0.0, 0.0),
    (_prep_state(GOOD, {"water": 80.0}, False), False, 0.0, 0.6),
    (_prep_state(GOOD, {"water": 80.0}, True), True, 12.0, 0.5),
    (_prep_state(GOOD, {}, False, {"sample": "rack", "balance": "elsewhere"}), False, 0.0, 0.3),
    (_prep_state({}, {}, False, {"sample": "rack", "balance": "rack"}), False, 0.0, 0.0),
])
def test_prep_spin_scoring_rules_match_the_task_text(prep_dir, state, spun, spilled, want):
    got = _py(prep_dir, f"""
import json, task
state = {state!r}
fake = dict(state, lid_deg=0.0, lid_closed={spun}, spilled_ul={spilled},
            spin=(dict(state, time=1.0) if {spun} else None))
s = task.TaskSession()
s.measure = lambda: fake
print(json.dumps(s.score()))""")
    assert got["score"] == want, got["reason"]


def test_prep_spin_reads_names_after_volumes_and_rejects_columns():
    from lab2rob.robocurve.compile import compile_prep_spin
    records = compile_prep_spin(_raw(
        "Add 1 ml of chilled acetone to the tube and keep it for 10 minutes at -20 C.",
        "Centrifuge the tubes at 12,000 x g for 10 min.",
        "Pipet 500 uL Buffer RPE onto the spin column.",
        "Centrifuge for 1 min at 8,000 x g.",
        "Transfer 500 uL to a new tube.",
        "Centrifuge at 8,000 x g for 1 min."))
    got = [(r["source"]["step"], r["accepted"], r["reason"].split(":")[0]) for r in records]
    assert got == [(2, True, ""), (4, False, "missing_asset"), (6, False, "ambiguous")]
    assert records[0]["source"]["additions"][0]["name"] == "chilled acetone"
