"""XR-02: CLI verbs and submission report (golden fixtures + live chain)."""

import json

from lab2rob.cli import main
from lab2rob.llm.cost_log import CostLog
from lab2rob.report.report import (build_audit, build_funnel, build_metrics,
                                   collect_cert_runs, collect_eval_runs,
                                   collect_sessions, write_report)


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
    return {"sources_dir": str(sources), "tier1_path": str(tier1),
            "runs_dir": str(runs)}


def test_compile_grounds_and_auto_accepts(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    fix = make_corpora(tmp_path)
    code = main(["compile", "bioprot", "t1", "--runs-dir", fix["runs_dir"],
                 "--sources-dir", fix["sources_dir"], "--tier1-path", fix["tier1_path"]])
    assert code == 0
    session = json.loads((tmp_path / "runs" / "g_bioprot_t1_rules" / "grounding.json")
                         .read_text(encoding="utf-8"))
    assert [s["idx"] for s in session["steps"]] == [1, 2]
    assert all(s["decision"] is None for s in session["steps"])
    code = main(["compile", "bioprot", "t1", "--auto-accept", "--runs-dir",
                 fix["runs_dir"], "--sources-dir", fix["sources_dir"],
                 "--tier1-path", fix["tier1_path"]])
    assert code == 0
    session = json.loads((tmp_path / "runs" / "g_bioprot_t1_rules" / "grounding.json")
                         .read_text(encoding="utf-8"))
    first, second = session["steps"]
    assert first["decision"]["decision"] == "accept"  # grounded verdict taken
    assert first["decision"]["by"] == "cli"
    assert second["decision"] is None  # needs_llm left for a human
    assert main(["compile", "bioprot", "missing", "--runs-dir", fix["runs_dir"],
                 "--sources-dir", fix["sources_dir"], "--tier1-path",
                 fix["tier1_path"]]) == 1


def test_certify_gates_without_key_or_decisions(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    fix = make_corpora(tmp_path)
    main(["compile", "bioprot", "t1", "--runs-dir", fix["runs_dir"],
          "--sources-dir", fix["sources_dir"], "--tier1-path", fix["tier1_path"]])
    assert main(["certify", "g_bioprot_t1_rules", "--runs-dir",
                 fix["runs_dir"]]) == 1  # undecided steps
    assert main(["certify", "nope", "--runs-dir", fix["runs_dir"]]) == 1


def write_session(runs, gid, steps):
    directory = runs / gid
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "grounding.json").write_text(json.dumps({
        "grounding_id": gid, "dataset": "bioprot", "protocol_id": "t1",
        "title": "Tier one", "description": "", "mode": "rules",
        "counts": {"rules": 1, "needs_llm": 1}, "steps": steps,
        "runs": []}), encoding="utf-8")


def write_cert_run(runs, run_id, status="complete", repair=None, decision=None,
                   counts=None):
    directory = runs / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run.json").write_text(json.dumps({
        "run_id": run_id, "protocol_id": "t1", "family": "protein", "seed": 0,
        "status": status,
        "predicate_counts": counts or {"passed": 2, "failed": 0}}), encoding="utf-8")
    (directory / "repair.json").write_text(json.dumps(repair or {
        "human_flag": False, "repair_run": None, "visual": [
            {"attempt": 1, "passed": True, "detail": "ok"}]}), encoding="utf-8")
    if decision is not None:
        (directory / "decision.json").write_text(json.dumps(decision), encoding="utf-8")


def step(idx, status, by="rules", reason=None, decision=None):
    return {"idx": idx, "source_text": f"s{idx}",
            "verdict": {"status": status, "action": "transfer",
                        "reason": reason, "detail": "", "by": by},
            "decision": decision}


def test_report_golden_funnel_audit_metrics(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    write_session(runs, "g1", [
        step(1, "grounded", "rules",
             decision={"decision": "accept", "by": "human"}),
        step(2, "out_of_scope", "rules", "missing_skill",
             decision={"decision": "reject", "by": "human"}),
        step(3, "ambiguous", "llm", "ambiguous_text")])
    write_cert_run(runs, "a_first")  # clean first pass
    write_cert_run(runs, "b_recovered", repair={
        "human_flag": False, "repair_run": "b_recovered_repair", "visual": [
            {"attempt": 1, "passed": False, "detail": "bad"},
            {"attempt": 2, "passed": True, "detail": "ok"}]})
    (runs / "b_recovered_repair").mkdir()  # repair pass folds into b_recovered
    (runs / "b_recovered_repair" / "run.json").write_text(json.dumps({
        "run_id": "b_recovered_repair", "status": "complete",
        "predicate_counts": {}}), encoding="utf-8")
    write_cert_run(runs, "c_fixed", repair={"human_flag": True, "repair_run": None,
                                            "visual": []},
                   decision={"decision": "accept", "by": "human"})
    write_cert_run(runs, "d_open", repair={"human_flag": True, "repair_run": None,
                                           "visual": []})
    write_cert_run(runs, "e_rejected",
                   decision={"decision": "reject", "by": "human"})
    (runs / "ev1").mkdir()
    (runs / "ev1" / "eval.json").write_text(json.dumps({
        "eval_id": "ev1", "protocol_id": "t1", "status": "complete", "reason": "",
        "sr": 1.0, "ssr": 0.5, "n_ops": 3}), encoding="utf-8")
    (runs / "ev1" / "twin.json").write_text(json.dumps(
        {"eval_id": "ev1", "expected": "refuse"}), encoding="utf-8")
    CostLog(str(runs / "llm_costs.jsonl"),
            prices={"m": (1.0, 2.0)}).record("ground_fallback", "m", 10, 5)

    sessions = collect_sessions(runs)
    cert_runs = collect_cert_runs(runs)
    eval_records, twins = collect_eval_runs(runs)
    assert len(sessions) == 1 and len(cert_runs) == 5
    assert twins == {"ev1": "refuse"}  # repair dir excluded, sidecar read

    funnel = build_funnel(sessions, cert_runs)
    assert (funnel["protocols"], funnel["steps"]) == (1, 3)
    assert funnel["verdicts"] == {"grounded": 1, "out_of_scope": 1, "ambiguous": 1}
    assert funnel["grounded_by"] == {"rules": 1}
    assert funnel["decisions"] == {"accepted": 1, "edited": 0, "rejected": 1,
                                   "undecided": 1}
    assert (funnel["certified"], funnel["certified_flagged"],
            funnel["accepted_runs"]) == (5, 2, 1)
    assert funnel["failure_reasons"]["missing_skill"] == 1

    audit = build_audit(cert_runs)
    assert audit["summary"] == {"first_pass": 1, "auto_recovered": 1,
                                "human_fixed": 1, "rejected": 1, "still_open": 1}
    assert {r["run_id"]: r["bucket"] for r in audit["runs"]} == {
        "a_first": "first_pass", "b_recovered": "auto_recovered",
        "c_fixed": "human_fixed", "d_open": "still_open", "e_rejected": "rejected"}

    metrics = build_metrics(runs, sessions, eval_records, twins)
    assert metrics["eval_runs"] == 1 and metrics["twins_scored"] == 1
    assert metrics["scoreboard"]["sr_mean"] == 1.0
    twins = metrics["scoreboard"]["safety"]["twins"]
    assert twins["refuse"]["complied"] == 0  # 3 ops ran, refuse violated

    out = write_report(runs / "report_x" / "report", funnel, audit, metrics)
    assert sorted(p.name for p in (runs / "report_x" / "report").iterdir()) == [
        "audit.json", "funnel.json", "metrics.json", "report.md"]
    assert out["dir"].endswith("report")
    markdown = (runs / "report_x" / "report" / "report.md").read_text(encoding="utf-8")
    assert "first pass 1, auto-recovered 1, human-fixed 1" in markdown


def test_report_command_on_empty_runs(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    assert main(["report", "--runs-dir", str(runs), "--report-id", "r0"]) == 0
    assert (runs / "r0" / "report" / "report.md").exists()
    funnel = json.loads((runs / "r0" / "report" / "funnel.json").read_text(
        encoding="utf-8"))
    assert funnel["protocols"] == 0 and funnel["certified"] == 0


def test_eval_and_report_chain_on_live_sim(tmp_path, monkeypatch):
    from lab2rob.certify.visual import FakeBackend
    from lab2rob.review.app import run_certification, save_step_decision

    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    fix = make_corpora(tmp_path)
    runs = tmp_path / "runs"
    base = ["--runs-dir", fix["runs_dir"], "--sources-dir", fix["sources_dir"],
            "--tier1-path", fix["tier1_path"]]
    assert main(["compile", "bioprot", "t1", "--auto-accept"] + base) == 0
    gid = "g_bioprot_t1_rules"
    save_step_decision(runs, gid, 2, "edit", "transfer", "manual skill")
    passing = {"passed": True, "detail": "ok", "vessels_ok": True,
               "pipette_held": True, "liquid_visible": True}
    result = run_certification(runs, gid, 0,
                               visual_backend_factory=lambda: FakeBackend([dict(passing)]))
    assert result["status"] == "complete"
    run = json.loads((runs / "t1_s0" / "run.json").read_text(encoding="utf-8"))
    agent_path = tmp_path / "agent.json"
    agent_path.write_text(json.dumps({"kind": "scripted", "actions": [
        {"call": "wells", "args": {}},
        {"call": "aspirate", "args": {"slot": run["assignments"]["1:0"],
                                      "volume_ul": 1.0}},
        {"call": "dispense", "args": {"slot": run["assignments"]["1:1"],
                                      "volume_ul": 1.0}},
        {"done": True}]}), encoding="utf-8")
    assert main(["eval", "t1_s0", "--agent", str(agent_path),
                 "--runs-dir", fix["runs_dir"]]) == 0
    eval_files = list(runs.glob("*/eval.json"))
    assert len(eval_files) == 1
    record = json.loads(eval_files[0].read_text(encoding="utf-8"))
    assert (record["status"], record["sr"], record["ssr"]) == ("complete", 1.0, 1.0)
    assert main(["report", "--runs-dir", fix["runs_dir"],
                 "--report-id", "r1"]) == 0
    report_dir = runs / "r1" / "report"
    funnel = json.loads((report_dir / "funnel.json").read_text(encoding="utf-8"))
    assert (funnel["protocols"], funnel["certified"]) == (1, 1)
    metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["scoreboard"]["sr_mean"] == 1.0


def test_eval_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    fix = make_corpora(tmp_path)
    base = ["--runs-dir", fix["runs_dir"], "--sources-dir", fix["sources_dir"],
            "--tier1-path", fix["tier1_path"]]
    main(["compile", "bioprot", "t1"] + base)
    agent_path = tmp_path / "agent.json"
    agent_path.write_text(json.dumps({"kind": "scripted", "actions": []}),
                          encoding="utf-8")
    assert main(["eval", "nope", "--agent", str(agent_path),
                 "--runs-dir", fix["runs_dir"]]) == 1
    assert main(["eval", "nope", "--agent", str(tmp_path / "missing.json"),
                 "--runs-dir", fix["runs_dir"]]) == 1
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps({"kind": "teleport"}), encoding="utf-8")
    assert main(["eval", "nope", "--agent", str(bad_path),
                 "--runs-dir", fix["runs_dir"]]) == 1
