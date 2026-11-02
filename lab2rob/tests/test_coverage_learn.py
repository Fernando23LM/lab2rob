"""ST-02: skill learning from the gap queue (offline fakes + real sim smoke)."""

import json
import shutil

import pytest

from lab2rob.coverage import learn
from lab2rob.coverage.learn import (_parse_proposal, ranked_candidates,
                                    register_skill)


class FakeReply:
    def __init__(self, text):
        self.text = text


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.jobs = []

    def complete(self, job, messages, **kwargs):
        self.jobs.append(job)
        assert job == "evolve"
        assert kwargs.get("max_tokens", 0) >= 2048
        reply = self.replies.pop(0)
        return reply if isinstance(reply, FakeReply) else FakeReply(reply)


# Fictional skill for unit tests: the real "dilute" is registered by ST-02.
LEARNED = {"skill": "test_learn_dilute",
           "description": "Lower concentration by adding diluent.",
           "params": ["target", "volume_ul"], "steps": ["aspirate", "dispense"],
           "smoke_text": "Test learn dilute the sample with 25 uL water."}


def test_parse_proposal_accepts_valid():
    out = _parse_proposal(json.dumps(LEARNED))
    assert (out["skill"], out["steps"]) == ("test_learn_dilute",
                                            ["aspirate", "dispense"])


def test_parse_proposal_rejects():
    with pytest.raises(ValueError, match="not JSON"):
        _parse_proposal("nope")
    bad_step = dict(LEARNED, steps=["aspirate", "teleport"])
    with pytest.raises(ValueError, match="not implemented"):
        _parse_proposal(json.dumps(bad_step))
    taken = dict(LEARNED, skill="transfer")
    with pytest.raises(ValueError, match="already registered"):
        _parse_proposal(json.dumps(taken))
    assert _parse_proposal(json.dumps({"cannot_express": "needs a stirrer"})) == {
        "cannot_express": "needs a stirrer"}


def test_ranked_candidates_filters_unlearnable():
    queue = {"evolve_candidates": [
        {"capability": "centrifuge", "count": 20, "reasons": ["missing_skill"],
         "examples": [{"text": "spin it"}]},
        {"capability": "freeze", "count": 9, "reasons": ["unmodeled_physics"],
         "examples": [{"text": "freeze it"}]},
        {"capability": "dilute", "count": 11,
         "reasons": ["missing_skill", "not_implemented"],
         "examples": [{"text": "dilute it"}]},
    ]}
    ranked = ranked_candidates(queue, limit=5)
    assert [c["capability"] for c in ranked] == ["centrifuge", "dilute"]
    assert [c["capability"] for c in ranked_candidates(queue, limit=1)] == ["centrifuge"]


def test_register_skill_promotes_and_regenerates(tmp_path, monkeypatch):
    from lab2rob.registry import gen_skills

    registry = tmp_path / "registry"
    registry.mkdir()
    for name in ("macros.yaml", "declared.yaml"):
        shutil.copy(learn.REGISTRY_DIR / name, registry / name)
    monkeypatch.setattr(gen_skills, "MACROS_PATH", registry / "macros.yaml")
    monkeypatch.setattr(gen_skills, "DECLARED_PATH", registry / "declared.yaml")
    monkeypatch.setattr(gen_skills, "SKILLS_PATH", registry / "skills.yaml")
    declared_path = registry / "declared.yaml"  # seed a declared entry to promote
    declared_path.write_text(declared_path.read_text(encoding="utf-8")
                             + "  - name: test_learn_dilute\n    kind: action\n"
                             + "    description: Fixture skill.\n", encoding="utf-8")
    out = register_skill(dict(LEARNED), registry)
    assert out == {"skill": "test_learn_dilute", "steps": ["aspirate", "dispense"]}
    declared = declared_path.read_text(encoding="utf-8")
    assert "name: test_learn_dilute\n" not in declared  # promoted out
    assert "name: distribute\n" in declared  # neighbors untouched
    generated = (registry / "skills.yaml").read_text(encoding="utf-8")
    assert "name: test_learn_dilute" in generated and "status: macro" in generated
    assert gen_skills.main(["--check"]) == 0


def test_learn_candidate_orchestration(tmp_path, monkeypatch):
    candidate = {"capability": "dilute", "count": 2,
                 "examples": [{"text": "Dilute the sample with 25 uL water."}]}
    client = FakeClient(["not json", json.dumps(LEARNED)])
    calls = {"smokes": 0, "restores": 0}
    monkeypatch.setattr(learn, "smoke_test",
                        lambda *a, **k: (calls.update(smokes=calls["smokes"] + 1),
                                         {"passed": True, "detail": "2 ops"})[1])
    monkeypatch.setattr(learn, "register_skill", lambda *a, **k: {"skill": "dilute"})
    monkeypatch.setattr(learn, "_restore_registry",
                        lambda *a, **k: calls.update(restores=calls["restores"] + 1))
    result = learn.learn_candidate(candidate, client, tmp_path / "learn_dilute",
                                   rounds=3)
    assert result["learned"] is True and result["rounds"] == 2  # invalid, then pass
    assert calls == {"smokes": 1, "restores": 0}  # kept: no rollback
    assert (tmp_path / "learn_dilute" / "best.json").exists()
    assert (tmp_path / "learn_dilute" / "round_000" / "error.json").exists()

    gloomy = FakeClient([json.dumps({"cannot_express": "needs hardware"})])
    result = learn.learn_candidate({"capability": "spin", "count": 1,
                                    "examples": [{"text": "Spin it."}]},
                                   gloomy, tmp_path / "learn_spin")
    assert result["learned"] is False and calls["smokes"] == 1  # no smoke spent


def test_dilute_grounds_plans_and_certifies(tmp_path):
    """Committed proof for the really learned skill (needs registration)."""
    from lab2rob.ground.rules import ground_step, parse_steps
    from lab2rob.registry import resolve_skill
    from lab2rob.specify.program import plan_step

    assert resolve_skill("dilute")["status"] == "macro"
    grounding = ground_step(parse_steps("1. Dilute the sample with 25 uL water.")[0])
    assert (grounding.status, grounding.action) == ("grounded", "dilute")
    planned = plan_step(grounding)
    assert planned.status == "planned"
    assert [op["skill"] for op in planned.ops] == ["aspirate", "dispense"]
    smoke = learn.smoke_test("dilute", "Dilute the sample with 25 uL water.",
                             tmp_path / "smoke")
    assert smoke["passed"], smoke["detail"]


def test_dilute_bulk_stays_out_of_scope():
    """Pipette-scale dilute is learnable; bulk dilute still cannot compile."""
    from lab2rob.ground.rules import ground_step, parse_steps

    grounding = ground_step(parse_steps("1. Dilute the sample with 500 mL water.")[0])
    assert (grounding.status, grounding.reason) == ("out_of_scope", "missing_skill")


def test_dilute_gets_liquid_judge_checks():
    from lab2rob.ground.rules import Grounding, ProtocolGrounding, Quantity
    from lab2rob.specify.judge import plan_judge

    quantities = [Quantity(raw="25 uL", value=25.0, unit="ul",
                           approximate=False, pint=None)]
    grounding = Grounding(idx=1, source_text="Dilute the sample with 25 uL water.",
                          quantities=quantities, action="dilute", status="grounded")
    judge = plan_judge(ProtocolGrounding(protocol_id="t", title="t",
                                         steps=[grounding]))
    assert {p.pred for p in judge.steps[0].predicates} == {"no_spill", "volume"}
