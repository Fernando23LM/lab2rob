"""SC-05: SAM visual check and one repair (fake backend offline)."""

import json
from pathlib import Path

from lab2rob.build.scene import build_scene
from lab2rob.certify.repair import certify_with_repair, diagnose
from lab2rob.certify.run import RunReport, certify
from lab2rob.certify.visual import FakeBackend, LLMBackend
from lab2rob.ground.rules import ground_protocol
from lab2rob.ingest.bioprot import load_bioprot_id
from lab2rob.llm import Lab2RobClient
from lab2rob.llm.cache import LLMCache
from lab2rob.specify.judge import plan_judge
from lab2rob.specify.program import plan_protocol

PASS = {"passed": True, "detail": "scene looks right", "vessels_ok": True,
        "pipette_held": True, "liquid_visible": True}
FAIL = {"passed": False, "detail": "no liquid seen", "vessels_ok": True,
        "pipette_held": True, "liquid_visible": False}
EXEC = {"approach_m": 0.03, "well_speed": 0.04, "tip_extra_mm": 0.0}


def chain1779():
    grounding = ground_protocol(load_bioprot_id("1779"))
    program = plan_protocol(grounding)
    return program, plan_judge(grounding), build_scene(program, seed=0)


def test_visual_pass_no_repair_no_flag(tmp_path):
    program, judge, built = chain1779()
    backend = FakeBackend([dict(PASS)])
    out = certify_with_repair(program, judge, built, seed=0,
                              runs_dir=tmp_path, visual_backend=backend)
    assert out["human_flag"] is False
    assert out["repair"] is None and out["diagnosis"] is None
    assert [v["attempt"] for v in out["visual"]] == [1]
    assert backend.calls[0]["cameras"] == ["front", "overhead"]
    assert backend.calls[0]["expectations"] == {"liquid_expected": True}
    assert (out["initial"].dir / "visual_overhead_a1.png").exists()


def test_visual_fail_then_pass_repairs(tmp_path):
    program, judge, built = chain1779()
    out = certify_with_repair(program, judge, built, seed=0, runs_dir=tmp_path,
                              visual_backend=FakeBackend([dict(FAIL), dict(PASS)]))
    assert out["human_flag"] is False
    assert [v["attempt"] for v in out["visual"]] == [1, 2]
    assert out["repair"] is None  # the single repair went to the re-query


def test_visual_fail_twice_flags_human(tmp_path):
    program, judge, built = chain1779()
    out = certify_with_repair(program, judge, built, seed=0, runs_dir=tmp_path,
                              visual_backend=FakeBackend([dict(FAIL), dict(FAIL)]))
    assert out["human_flag"] is True
    assert len(out["visual"]) == 2  # exactly one re-query, then flag
    repair_json = json.loads((out["initial"].dir / "repair.json").read_text(encoding="utf-8"))
    assert repair_json["human_flag"] is True
    assert repair_json["repair_run"] is None


def test_diagnose_plans_and_declines(tmp_path):
    blocked = RunReport(run_id="r", dir=tmp_path, status="aborted",
                        abort_reason="move_blocked", abort_detail="descent into A1 failed")
    plan = diagnose(blocked, [], dict(EXEC))
    assert plan["exec"] == {"approach_m": 0.05, "well_speed": 0.02, "tip_extra_mm": 0.0}
    hopeless = RunReport(run_id="r", dir=tmp_path, status="aborted",
                         abort_reason="no_executor")
    assert diagnose(hopeless, [], dict(EXEC)) is None

    failed = [{"step": 1, "pred": "volume", "status": "failed"}]
    short = [{"kind": "op", "step": 1, "skill": "aspirate",
              "args": {"volume_ul": 25.0}, "result": {"volume_ul": 10.0}}]
    complete = RunReport(run_id="r", dir=tmp_path, status="complete", predicates=failed)
    plan = diagnose(complete, short, dict(EXEC))
    assert plan["exec"]["tip_extra_mm"] == 1.0
    assert "drew 10 of 25" in plan["why"]
    full = [{"kind": "op", "step": 1, "skill": "aspirate",
             "args": {"volume_ul": 25.0}, "result": {"volume_ul": 25.0}}]
    assert diagnose(complete, full, dict(EXEC)) is None
    clean = RunReport(run_id="r", dir=tmp_path, status="complete", predicates=[])
    assert diagnose(clean, full, dict(EXEC)) is None


def test_exec_overrides_run_clean(tmp_path):
    program, judge, built = chain1779()
    report = certify(program, judge, built, seed=0, runs_dir=tmp_path,
                     approach_m=0.05, tip_extra_mm=0.5)
    assert report.status == "complete"
    assert all(p["status"] == "passed" for p in report.predicates)


class CapTransport:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def complete(self, model, messages, **kwargs):
        self.calls.append((model, messages, kwargs))
        return {"text": self.texts.pop(0), "prompt_tokens": 10,
                "completion_tokens": 5, "reasoning_tokens": 0}


def make_visual_client(tmp_path, texts):
    from lab2rob.llm.cost_log import CostLog
    transport = CapTransport(texts)
    client = Lab2RobClient(cache=LLMCache(str(tmp_path / "cache")),
                           cost_log=CostLog(str(tmp_path / "costs.jsonl")),
                           transport=transport)
    return client, transport


def test_llm_backend_message_shape_and_retry(tmp_path):
    good = json.dumps({"vessels_ok": True, "pipette_held": True,
                       "liquid_visible": True, "detail": "ok"})
    client, transport = make_visual_client(tmp_path, ["not json", good])
    verdict = LLMBackend(client).check({"overhead": b"PNG1", "front": b"PNG2"},
                                       {"liquid_expected": True})
    assert verdict["passed"] is True
    assert len(transport.calls) == 2  # one retry on invalid JSON
    model, messages, _ = transport.calls[0]
    assert model == "muse-spark-1.3-contributor"  # ST-03 reroute of visual
    parts = messages[1]["content"]
    assert parts[0]["type"] == "text"
    assert [p["type"] for p in parts[1:]] == ["image_url", "image_url"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    exp = {"liquid_expected": False}
    client2, _ = make_visual_client(tmp_path / "c2", [good])
    assert LLMBackend(client2).check({"overhead": b"P"}, exp)["passed"] is False
