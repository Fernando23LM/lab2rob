"""ST-04: sweep mines skill candidates from full-LLM verdicts (offline)."""

import json

import pytest

from lab2rob.ground.sweep import (
    _capability_key,
    _diff_usage,
    load_corpus,
    mine_new_skills,
    sweep_protocols,
    write_report,
)
from lab2rob.ingest.bioprot import RawProtocol


class FakeReply:
    def __init__(self, text):
        self.text = text


class FakeFullClient:
    def __init__(self, arrays):
        self.arrays = [json.dumps(a) for a in arrays]
        self.calls = 0

    def complete(self, job, messages, **kwargs):
        self.calls += 1
        assert job == "ground_fallback"
        return FakeReply(self.arrays.pop(0))


def verdict(idx, skill=None, reason=None, ambiguous=False, detail=""):
    return {"idx": idx, "skill": skill, "out_of_scope_reason": reason,
            "ambiguous": ambiguous, "detail": detail}


def raw(pid, steps):
    blob = "\n".join(f"{i}. {t}" for i, t in enumerate(steps, 1))
    return RawProtocol(protocol_id=pid, dataset="test", title=f"T{pid}",
                       description="", steps_raw=blob, source_url="u",
                       license="l", number_of_steps=len(steps))


def test_capability_keywords():
    assert _capability_key("Spin at 3000 g for 5 min") == "centrifuge"
    assert _capability_key("Resuspend the pellet") == "resuspend"  # beats pellet
    assert _capability_key("Pour off the supernatant") == "pour"
    assert _capability_key("Count cells on a hemocytometer") == "count_cells"
    assert _capability_key("Wash twice with PBS") == "wash"
    assert _capability_key("Flash freeze in LN2") == "cold_store"
    assert _capability_key("Do the thing for a while") == "unclustered"


def test_sweep_writes_records_and_mines_ranking(tmp_path):
    raws = [raw("p1", ["Add 1 uL dye", "Spin at 3000 g", "Pour off the waste"]),
            raw("p2", ["Do the thing", "Aspirate 5 uL"])]
    client = FakeFullClient([
        [verdict(1, skill="transfer", detail="ok"),
         verdict(2, reason="not_implemented", detail="needs centrifuge spin"),
         verdict(3, reason="missing_skill", detail="pouring motion unknown")],
        [verdict(1, ambiguous=True, detail="vague"),
         verdict(2, skill="aspirate", detail="ok")],
    ])
    records = sweep_protocols(raws, client, tmp_path)
    assert client.calls == 2
    assert {p.name for p in tmp_path.glob("test_*.json")} == {"test_p1.json", "test_p2.json"}
    assert [s["status"] for s in records[0]["steps"]] == [
        "grounded", "out_of_scope", "out_of_scope"]
    ranking = mine_new_skills(records)
    assert [(e["capability"], e["count"]) for e in ranking] == [
        ("centrifuge", 1), ("pour", 1), ("unclustered", 1)]
    assert ranking[0]["examples"][0]["protocol_id"] == "test:p1"
    assert ranking[0]["protocols"] == ["test:p1"]


def test_resume_skips_grounded(tmp_path):
    raws = [raw("p1", ["Add 1 uL dye"])]
    first = FakeFullClient([[verdict(1, skill="transfer", detail="ok")]])
    sweep_protocols(raws, first, tmp_path)
    assert first.calls == 1
    second = FakeFullClient([])
    records = sweep_protocols(raws, second, tmp_path, resume=True)
    assert second.calls == 0
    assert records[0]["steps"][0]["status"] == "grounded"


def test_failure_recorded_not_fatal(tmp_path):
    class Boom:
        def complete(self, *a, **k):
            raise RuntimeError("boom")

    records = sweep_protocols([raw("p9", ["Add 1 uL dye"])], Boom(), tmp_path)
    assert records[0]["error"].startswith("RuntimeError: boom")
    assert mine_new_skills(records) == []
    payload = write_report(tmp_path, records, [])
    assert payload["protocols_failed"] == 1
    assert payload["failed"] == ["test:p9"]


def test_all_ambiguous_protocol_warns(tmp_path, capsys):
    raws = [raw("pw", ["Do the thing"])]
    client = FakeFullClient([[verdict(1, ambiguous=True, detail="vague")]])
    sweep_protocols(raws, client, tmp_path)
    assert "WARNING" in capsys.readouterr().out


def test_write_report_files(tmp_path):
    ranking = mine_new_skills([{
        "protocol_id": "p1", "dataset": "test",
        "steps": [{"idx": 1, "text": "Spin fast", "status": "out_of_scope",
                   "reason": "not_implemented", "action": None, "detail": ""}]}])
    payload = write_report(tmp_path, [{"n_steps": 1}], ranking)
    assert payload["steps_mined"] == 1
    assert (tmp_path / "candidates.json").exists()
    text = (tmp_path / "candidates.md").read_text(encoding="utf-8")
    assert "## 1. centrifuge (1 steps" in text


def test_failed_steps_not_mined():
    records = [{"protocol_id": "pf", "dataset": "test",
                "steps": [{"idx": 1, "text": "Spin at 3000 g", "status": "failed",
                           "reason": "llm_truncated", "action": None,
                           "detail": "cut off"}]}]
    assert mine_new_skills(records) == []  # truncated steps carry no signal


def test_write_report_includes_llm_usage(tmp_path):
    usage = {"ground_fallback": {"calls": 3, "prompt_tokens": 30,
                                 "completion_tokens": 15,
                                 "reasoning_tokens": 12, "cost_usd": 0.06}}
    payload = write_report(tmp_path, [{"n_steps": 1}], [], usage=usage)
    assert payload["llm_usage"] == usage
    text = (tmp_path / "candidates.md").read_text(encoding="utf-8")
    assert "## LLM usage" in text
    assert "ground_fallback: 3 calls, 12 reasoning tokens, $0.0600" in text


def test_all_failed_protocol_warns(tmp_path, capsys):
    class TruncatedReply:
        def __init__(self, text="bad"):
            self.text = text
            self.completion_tokens = 1756
            self.reasoning_tokens = 1700
            self.finish_reason = "length"

    class TruncatedClient:
        def complete(self, job, messages, **kwargs):
            assert job == "ground_fallback"
            return TruncatedReply()

    records = sweep_protocols([raw("pf", ["Do the thing"])], TruncatedClient(),
                              tmp_path)
    assert records[0]["steps"][0]["status"] == "failed"
    assert "WARNING" in capsys.readouterr().out


def test_diff_usage_reports_only_this_run():
    before = {"ground_fallback": {"calls": 70, "prompt_tokens": 700,
                                  "completion_tokens": 7000,
                                  "reasoning_tokens": 6000, "cost_usd": 2.0}}
    after = {"ground_fallback": {"calls": 79, "prompt_tokens": 790,
                                 "completion_tokens": 7900,
                                 "reasoning_tokens": 6487,
                                 "cost_usd": 2.1214},
             "visual": {"calls": 1, "prompt_tokens": 10,
                        "completion_tokens": 5, "reasoning_tokens": 0,
                        "cost_usd": 0.001}}
    assert _diff_usage(before, after) == {
        "ground_fallback": {"calls": 9, "prompt_tokens": 90,
                            "completion_tokens": 900, "reasoning_tokens": 487,
                            "cost_usd": pytest.approx(0.1214)},
        "visual": {"calls": 1, "prompt_tokens": 10, "completion_tokens": 5,
                   "reasoning_tokens": 0, "cost_usd": pytest.approx(0.001)}}
    assert _diff_usage(after, after) == {}  # a fully resumed run reports zero


def test_load_corpus_tier1_and_limit():
    assert len(load_corpus(protocolsio=False)) == 3
    assert len(load_corpus(limit=2)) == 2
