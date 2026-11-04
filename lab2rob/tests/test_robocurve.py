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
