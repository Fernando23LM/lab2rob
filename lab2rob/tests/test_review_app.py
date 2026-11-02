"""SC-06: review app routes serve runs and record decisions on disk."""

import json

from lab2rob.review.app import dispatch

PNG = b"\x89PNG\r\n\x1a" + b"\x00" * 64


def write_run(runs, run_id, status="complete", counts=None, human_flag=None,
              decision=None):
    run_dir = runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({
        "run_id": run_id, "protocol_id": "1779", "family": "protein", "seed": 0,
        "status": status, "predicate_counts": counts or {"passed": 2, "failed": 0}}),
        encoding="utf-8")
    (run_dir / "predicates.json").write_text(json.dumps([
        {"step": 3, "pred": "volume", "args": ["unbound"], "target": 75.0,
         "tol": 3.75, "well": "A1", "status": "passed", "detail": "ok"}]),
        encoding="utf-8")
    (run_dir / "trace.jsonl").write_text(
        json.dumps({"kind": "initial", "state": {"spilled_ul": 0.0}}) + "\n"
        + json.dumps({"kind": "op", "step": 3, "op": 1, "skill": "dispense",
                      "slot": "A1", "result": {"volume_ul": 75.0},
                      "state": {"spilled_ul": 0.0}}) + "\n", encoding="utf-8")
    if human_flag is not None:
        (run_dir / "repair.json").write_text(json.dumps({
            "human_flag": human_flag, "diagnosis": None, "visual": []}),
            encoding="utf-8")
    if decision is not None:
        (run_dir / "decision.json").write_text(json.dumps(decision), encoding="utf-8")
    (run_dir / "visual_overhead_a1.png").write_bytes(PNG)
    return run_dir


def make_runs(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    write_run(runs, "a_flagged", human_flag=True)
    write_run(runs, "b_decided", human_flag=True,
              decision={"decision": "accept", "note": "looks good",
                        "by": "human", "decided_at": "2026-01-01T00:00:00+00:00"})
    write_run(runs, "c_passed", human_flag=False)
    return runs


def get(runs, path, **kw):
    code, content_type, body = dispatch("GET", path, None, runs, **kw)
    if content_type.startswith("application/json"):
        return code, json.loads(body.decode("utf-8"))
    return code, body


def post(runs, path, obj, **kw):
    code, _, body = dispatch("POST", path, json.dumps(obj).encode("utf-8"), runs, **kw)
    return code, json.loads(body.decode("utf-8"))


class FakeGroundReply:
    def __init__(self, text):
        self.text = text


class FakeGroundClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, job, messages, **kwargs):
        self.calls += 1
        assert job == "ground_fallback"
        return FakeGroundReply(self.replies.pop(0))


def write_pio_cache(sources):
    sources.mkdir(parents=True, exist_ok=True)
    (sources / "protocols_io_4242.json").write_text(json.dumps({
        "protocol_id": "4242", "title": "Demo pipetting", "license": "CC-BY-4.0",
        "url": "https://example.invalid/4242", "query": "test",
        "raw": {"steps": [{"step": "Add 1 uL dye to the well.", "is_substep": False},
                          {"step": "Teleport the tube to Mars.", "is_substep": False}],
                "description": "A demo.", "before_start": "",
                "stats": {"number_of_steps": 2}}}), encoding="utf-8")


def write_tier1(path):
    path.write_text(json.dumps({"protocols": [{
        "protocol_id": "t1", "dataset": "bioprot", "title": "Tier one",
        "description": "",
        "steps_raw": "1. Add 1 uL dye to the well\n2. Teleport the tube\n",
        "source": {"url": "https://example.invalid/t1", "license": "CC-BY-4.0"},
        "number_of_steps": 2}]}), encoding="utf-8")


def make_corpora(tmp_path):
    sources = tmp_path / "sources"
    write_pio_cache(sources)
    tier1 = tmp_path / "tier1.json"
    write_tier1(tier1)
    runs = tmp_path / "runs"
    runs.mkdir()
    return {"sources_dir": sources, "tier1_path": tier1, "runs": runs}


def test_list_runs_states_and_order(tmp_path):
    code, runs = get(make_runs(tmp_path), "/api/runs")
    assert code == 200
    assert [(r["run_id"], r["state"]) for r in runs] == [
        ("a_flagged", "needs_review"), ("b_decided", "decided-accept"),
        ("c_passed", "passed")]


def test_run_detail_and_png(tmp_path):
    runs = make_runs(tmp_path)
    code, detail = get(runs, "/api/runs/a_flagged")
    assert code == 200
    assert detail["predicates"][0]["well"] == "A1"
    assert detail["trace"][-1]["skill"] == "dispense"
    assert detail["repair"]["human_flag"] is True
    assert detail["visuals"] == ["visual_overhead_a1.png"]
    assert detail["decision"] is None
    code, content_type, body = dispatch("GET", "/files/a_flagged/visual_overhead_a1.png",
                                       None, runs)
    assert (code, content_type, body) == (200, "image/png", PNG)


def test_accept_then_edit_decision_on_disk(tmp_path):
    runs = make_runs(tmp_path)
    code, _ = post(runs, "/api/runs/a_flagged/decision",
                   {"decision": "accept", "note": "verified"})
    assert code == 200
    record = json.loads((runs / "a_flagged" / "decision.json").read_text(encoding="utf-8"))
    assert record["decision"] == "accept" and record["note"] == "verified"
    assert record["by"] == "human" and record["decided_at"]
    assert get(runs, "/api/runs/a_flagged")[1]["state"] == "decided-accept"
    code, _ = post(runs, "/api/runs/a_flagged/decision",
                   {"decision": "reject", "note": "second look"})
    assert code == 200
    record = json.loads((runs / "a_flagged" / "decision.json").read_text(encoding="utf-8"))
    assert (record["decision"], record["note"]) == ("reject", "second look")


def test_rejects_bad_requests(tmp_path):
    runs = make_runs(tmp_path)
    assert post(runs, "/api/runs/a_flagged/decision", {"decision": "maybe"})[0] == 400
    assert post(runs, "/api/runs/a_flagged/decision",
                {"decision": "accept", "note": "x" * 2001})[0] == 400
    assert post(runs, "/api/runs/nope/decision", {"decision": "accept"})[0] == 404
    assert get(runs, "/api/runs/nope")[0] == 404
    assert get(runs, "/api/runs/../secret")[0] == 404
    assert get(runs, "/files/a_flagged/run.json")[0] == 404
    assert get(runs, "/nope")[0] == 404
    assert dispatch("PUT", "/api/runs", None, runs)[0] == 405


def test_serves_real_run_detail(tmp_path):
    from lab2rob.build.scene import build_scene
    from lab2rob.certify.run import certify
    from lab2rob.ground.rules import ground_protocol
    from lab2rob.ingest.bioprot import load_bioprot_id
    from lab2rob.specify.judge import plan_judge
    from lab2rob.specify.program import plan_protocol

    grounding = ground_protocol(load_bioprot_id("1779"))
    program = plan_protocol(grounding)
    certify(program, plan_judge(grounding), build_scene(program, seed=0),
            seed=0, runs_dir=tmp_path)
    code, detail = get(tmp_path, "/api/runs/1779_s0")
    assert code == 200
    assert all(p["status"] == "passed" for p in detail["predicates"])
    assert [r["skill"] for r in detail["trace"] if r["kind"] == "op"] == [
        "aspirate", "dispense"]
    assert detail["state"] == "needs_review"  # no repair pass ran yet
    code, content_type, body = dispatch("GET", "/", None, tmp_path)
    assert code == 200 and b"lab2rob review" in body


def test_protocols_list_both_corpora(tmp_path):
    fix = make_corpora(tmp_path)
    code, rows = get(fix["runs"], "/api/protocols",
                     sources_dir=fix["sources_dir"], tier1_path=fix["tier1_path"])
    assert code == 200
    assert [(r["dataset"], r["id"]) for r in rows] == [
        ("protocolsio", "4242"), ("bioprot", "t1")]
    assert rows[0]["steps"] == 2 and rows[0]["license"] == "CC-BY-4.0"


def test_ground_rules_mode_offline(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    fix = make_corpora(tmp_path)
    kw = {"sources_dir": fix["sources_dir"], "tier1_path": fix["tier1_path"]}
    code, session = post(fix["runs"], "/api/ground",
                         {"dataset": "bioprot", "protocol_id": "t1", "mode": "rules"},
                         **kw)
    assert code == 200
    assert session["grounding_id"] == "g_bioprot_t1_rules"
    assert session["llm_available"] is False
    first, second = session["steps"]
    assert first["verdict"]["by"] == "rules"  # grounded without spending a call
    assert first["quantities"][0]["raw"] == "1 uL"
    assert first["alternatives"] == []
    assert (second["verdict"]["status"], second["verdict"]["by"]) == (
        "needs_llm", "none")  # stays open: no key, no fallback
    saved = json.loads((fix["runs"] / "g_bioprot_t1_rules" / "grounding.json")
                       .read_text(encoding="utf-8"))
    assert [s["idx"] for s in saved["steps"]] == [1, 2]
    code, listed = get(fix["runs"], "/api/ground")
    assert code == 200
    assert [(r["grounding_id"], r["decided"], r["state"]) for r in listed] == [
        ("g_bioprot_t1_rules", 0, "needs_review")]
    code, detail = get(fix["runs"], "/api/ground/g_bioprot_t1_rules")
    assert code == 200 and detail["title"] == "Tier one"


def test_ground_rules_mode_fallback_becomes_alternative(tmp_path):
    fix = make_corpora(tmp_path)
    client = FakeGroundClient(['{"skill": null, "out_of_scope_reason": "missing_skill", '
                               '"ambiguous": false, "detail": "no teleporter"}'])
    kw = {"sources_dir": fix["sources_dir"], "tier1_path": fix["tier1_path"],
          "client_factory": lambda: client}
    code, session = post(fix["runs"], "/api/ground",
                         {"dataset": "bioprot", "protocol_id": "t1", "mode": "rules"},
                         **kw)
    assert code == 200 and client.calls == 1  # only the needs_llm step called
    first, second = session["steps"]
    assert first["alternatives"] == []
    assert (second["verdict"]["status"], second["verdict"]["by"]) == (
        "out_of_scope", "llm")
    assert second["alternatives"][0]["status"] == "needs_llm"  # rules view kept


def test_ground_full_llm_mode_with_rules_alternatives(tmp_path):
    fix = make_corpora(tmp_path)
    client = FakeGroundClient(
        ['[{"idx": 1, "skill": "transfer", "out_of_scope_reason": null, '
         '"ambiguous": false, "detail": "pipette the dye"}, '
         '{"idx": 2, "skill": null, "out_of_scope_reason": "missing_skill", '
         '"ambiguous": false, "detail": "no teleporter"}]'])
    kw = {"sources_dir": fix["sources_dir"], "tier1_path": fix["tier1_path"],
          "client_factory": lambda: client}
    code, session = post(fix["runs"], "/api/ground",
                         {"dataset": "protocolsio", "protocol_id": "4242",
                          "mode": "full_llm"}, **kw)
    assert code == 200 and client.calls == 1
    assert session["grounding_id"] == "g_protocolsio_4242_full_llm"
    assert [s["verdict"]["by"] for s in session["steps"]] == ["llm", "llm"]
    assert session["steps"][0]["verdict"]["action"] == "transfer"
    first_alt, second_alt = (s["alternatives"][0] for s in session["steps"])
    assert first_alt["by"] == "rules"  # rules grounded step 1 on its own
    assert (second_alt["status"], second_alt["by"]) == ("needs_llm", "none")


def test_step_decisions_accept_edit_reject(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    fix = make_corpora(tmp_path)
    kw = {"sources_dir": fix["sources_dir"], "tier1_path": fix["tier1_path"]}
    post(fix["runs"], "/api/ground",
         {"dataset": "bioprot", "protocol_id": "t1", "mode": "rules"}, **kw)
    gid = "g_bioprot_t1_rules"
    code, ok = post(fix["runs"], f"/api/ground/{gid}/steps/1",
                    {"decision": "accept", "note": "looks right"})
    assert code == 200 and ok["decision"]["by"] == "human"
    assert ok["decision"]["action"] == "transfer"  # keeps the verdict skill
    code, ok = post(fix["runs"], f"/api/ground/{gid}/steps/2",
                    {"decision": "edit", "action": "transfer", "note": "manual skill"})
    assert code == 200 and ok["decision"]["action"] == "transfer"
    assert post(fix["runs"], f"/api/ground/{gid}/steps/2",
                {"decision": "edit", "action": "teleport", "note": "x"})[0] == 400
    assert post(fix["runs"], f"/api/ground/{gid}/steps/2",
                {"decision": "reject", "note": ""})[0] == 400  # reject needs a why
    code, ok = post(fix["runs"], f"/api/ground/{gid}/steps/2",
                    {"decision": "reject", "note": "untranslatable"})
    assert code == 200 and ok["decision"]["action"] is None
    assert post(fix["runs"], f"/api/ground/{gid}/steps/9",
                {"decision": "accept"})[0] == 404
    saved = json.loads((fix["runs"] / gid / "grounding.json").read_text(encoding="utf-8"))
    assert [s["decision"]["decision"] for s in saved["steps"]] == ["accept", "reject"]
    assert saved["steps"][0]["verdict"]["by"] == "rules"  # provenance kept
    code, listed = get(fix["runs"], "/api/ground")
    assert (listed[0]["decided"], listed[0]["state"]) == (2, "decided")


def test_ground_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    fix = make_corpora(tmp_path)
    kw = {"sources_dir": fix["sources_dir"], "tier1_path": fix["tier1_path"]}
    assert post(fix["runs"], "/api/ground",
                {"dataset": "bioprot", "protocol_id": "t1", "mode": "magic"},
                **kw)[0] == 400
    assert post(fix["runs"], "/api/ground",
                {"dataset": "nope", "protocol_id": "t1", "mode": "rules"},
                **kw)[0] == 404
    assert post(fix["runs"], "/api/ground",
                {"dataset": "bioprot", "protocol_id": "missing", "mode": "rules"},
                **kw)[0] == 404
    code, err = post(fix["runs"], "/api/ground",
                     {"dataset": "bioprot", "protocol_id": "t1", "mode": "full_llm"},
                     **kw)
    assert code == 502 and "MODEL_API_KEY" in err["error"]
    assert get(fix["runs"], "/api/ground/nope")[0] == 404
    assert get(fix["runs"], "/api/ground/../secret")[0] == 404


def test_page_has_ground_tab(tmp_path):
    code, body = get(tmp_path, "/")
    assert code == 200
    assert b"view-ground" in body and b"full LLM translation" in body
    assert b"Build and certify" in body and b"Ask LLM" in body


def test_ask_llm_adds_and_replaces_alternative(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    fix = make_corpora(tmp_path)
    kw = {"sources_dir": fix["sources_dir"], "tier1_path": fix["tier1_path"]}
    post(fix["runs"], "/api/ground",
         {"dataset": "bioprot", "protocol_id": "t1", "mode": "rules"}, **kw)
    gid = "g_bioprot_t1_rules"
    client = FakeGroundClient(
        ['{"skill": "transfer", "out_of_scope_reason": null, "ambiguous": false, '
         '"detail": "pipette it"}',
         '{"skill": null, "out_of_scope_reason": "missing_skill", '
         '"ambiguous": false, "detail": "no teleporter"}'])
    ask = {"client_factory": lambda: client}
    code, ok = post(fix["runs"], f"/api/ground/{gid}/steps/2/ask", {}, **ask)
    assert code == 200
    (alt,) = ok["step"]["alternatives"]
    assert (alt["status"], alt["action"], alt["by"], alt["asked"]) == (
        "grounded", "transfer", "llm", True)
    assert ok["step"]["verdict"]["status"] == "needs_llm"  # verdict untouched
    code, ok = post(fix["runs"], f"/api/ground/{gid}/steps/2/ask", {}, **ask)
    assert code == 200
    (alt,) = ok["step"]["alternatives"]  # re-asking replaces, not appends
    assert (alt["status"], alt["asked"]) == ("out_of_scope", True)
    saved = json.loads((fix["runs"] / gid / "grounding.json").read_text(encoding="utf-8"))
    assert saved["steps"][1]["alternatives"][0]["asked"] is True


def test_ask_llm_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    fix = make_corpora(tmp_path)
    kw = {"sources_dir": fix["sources_dir"], "tier1_path": fix["tier1_path"]}
    post(fix["runs"], "/api/ground",
         {"dataset": "bioprot", "protocol_id": "t1", "mode": "rules"}, **kw)
    gid = "g_bioprot_t1_rules"
    code, err = post(fix["runs"], f"/api/ground/{gid}/steps/2/ask", {})
    assert code == 502 and "MODEL_API_KEY" in err["error"]
    ask = {"client_factory": lambda: FakeGroundClient(['{"skill": null, '
         '"out_of_scope_reason": "missing_skill", "ambiguous": false, "detail": "x"}'])}
    assert post(fix["runs"], f"/api/ground/{gid}/steps/9/ask", {}, **ask)[0] == 404
    assert post(fix["runs"], "/api/ground/nope/steps/1/ask", {}, **ask)[0] == 404


def test_specify_preview_program_judge_and_gates(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    fix = make_corpora(tmp_path)
    kw = {"sources_dir": fix["sources_dir"], "tier1_path": fix["tier1_path"]}
    post(fix["runs"], "/api/ground",
         {"dataset": "bioprot", "protocol_id": "t1", "mode": "rules"}, **kw)
    gid = "g_bioprot_t1_rules"
    post(fix["runs"], f"/api/ground/{gid}/steps/1", {"decision": "accept", "note": ""})
    code, spec = get(fix["runs"], f"/api/ground/{gid}/specify")
    assert code == 200
    assert (spec["decided"], spec["total"]) == (1, 2)
    assert spec["undecided"] == [2] and spec["blocked"] == []
    (planned,) = spec["program"]["steps"]
    assert (planned["idx"], planned["status"]) == (1, "planned")
    assert [op["skill"] for op in planned["ops"]] == ["aspirate", "dispense"]
    assert all("args" in op for op in planned["ops"])
    (judged,) = spec["judge"]["steps"]
    assert judged["idx"] == 1
    assert {p["pred"] for p in judged["predicates"]} == {"no_spill", "volume"}
    volume = next(p for p in judged["predicates"] if p["pred"] == "volume")
    assert volume["target"] == 1.0
    post(fix["runs"], f"/api/ground/{gid}/steps/2",
         {"decision": "reject", "note": "untranslatable"})
    code, spec = get(fix["runs"], f"/api/ground/{gid}/specify")
    assert code == 200 and spec["blocked"] == [2] and spec["undecided"] == []
    assert get(fix["runs"], "/api/ground/nope/specify")[0] == 404


def test_certify_end_to_end_with_fake_visual(tmp_path, monkeypatch):
    from lab2rob.certify.visual import FakeBackend

    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    runs = tmp_path / "runs"
    runs.mkdir()
    code, session = post(runs, "/api/ground",
                         {"dataset": "bioprot", "protocol_id": "1779", "mode": "rules"})
    assert code == 200
    for step in session["steps"]:
        code, _ = post(runs, f"/api/ground/{session['grounding_id']}/steps/{step['idx']}",
                       {"decision": "accept", "note": ""})
        assert code == 200
    code, spec = get(runs, f"/api/ground/{session['grounding_id']}/specify")
    assert code == 200 and spec["blocked"] == [] and spec["undecided"] == []
    passing = {"passed": True, "detail": "scene looks right", "vessels_ok": True,
               "pipette_held": True, "liquid_visible": True}
    code, result = post(
        runs, f"/api/ground/{session['grounding_id']}/certify", {"seed": 0},
        visual_backend_factory=lambda: FakeBackend([dict(passing)]))
    assert code == 200, result
    assert result["run_id"] == "1779_s0" and result["status"] == "complete"
    assert result["human_flag"] is False
    assert result["predicate_counts"]["failed"] == 0
    assert [v["attempt"] for v in result["visual"]] == [1]
    assert (runs / "1779_s0" / "repair.json").exists()
    assert (runs / session["grounding_id"] / "specify.json").exists()
    saved = json.loads((runs / session["grounding_id"] / "grounding.json")
                       .read_text(encoding="utf-8"))
    assert [r["run_id"] for r in saved["runs"]] == ["1779_s0"]
    code, listed = get(runs, "/api/runs")
    assert code == 200 and [r["run_id"] for r in listed] == ["1779_s0"]
    code, _ = post(runs, "/api/runs/1779_s0/decision",
                   {"decision": "accept", "note": "certified via UI-02"})
    assert code == 200  # the loop ends in the existing run review


def test_certify_gates(tmp_path, monkeypatch):
    from lab2rob.certify.visual import FakeBackend

    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    fix = make_corpora(tmp_path)
    kw = {"sources_dir": fix["sources_dir"], "tier1_path": fix["tier1_path"]}
    post(fix["runs"], "/api/ground",
         {"dataset": "bioprot", "protocol_id": "t1", "mode": "rules"}, **kw)
    gid = "g_bioprot_t1_rules"
    code, err = post(fix["runs"], f"/api/ground/{gid}/certify", {"seed": 0}, **kw)
    assert code == 400 and "undecided" in err["error"]
    post(fix["runs"], f"/api/ground/{gid}/steps/1", {"decision": "accept", "note": ""})
    post(fix["runs"], f"/api/ground/{gid}/steps/2",
         {"decision": "reject", "note": "untranslatable"})
    code, err = post(fix["runs"], f"/api/ground/{gid}/certify", {"seed": 0}, **kw)
    assert code == 400 and "rejected" in err["error"]
    post(fix["runs"], f"/api/ground/{gid}/steps/2",
         {"decision": "edit", "action": "transfer", "note": "manual skill"})
    assert post(fix["runs"], f"/api/ground/{gid}/certify", {"seed": -1}, **kw)[0] == 400
    assert post(fix["runs"], f"/api/ground/{gid}/certify", {"seed": "x"}, **kw)[0] == 400
    code, err = post(fix["runs"], f"/api/ground/{gid}/certify", {"seed": 0}, **kw)
    assert code == 502 and "MODEL_API_KEY" in err["error"]  # visual needs the model
    assert post(fix["runs"], "/api/ground/nope/certify", {"seed": 0})[0] == 404
    passing = {"passed": True, "detail": "ok", "vessels_ok": True,
               "pipette_held": True, "liquid_visible": False}
    code, result = post(
        fix["runs"], f"/api/ground/{gid}/certify", {"seed": 0}, **kw,
        visual_backend_factory=lambda: FakeBackend([dict(passing)] * 2))
    assert code == 200, result  # edited step flows through as human-corrected
    assert result["run_id"] == "t1_s0"
