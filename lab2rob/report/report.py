"""XR-02: build the submission report from a runs directory.

Pure collectors over run artifacts (grounding sessions, certify runs, eval
runs, cost log) plus one writer. No simulator, no API key: given the same
runs directory the same report comes out.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from lab2rob.eval.metrics import compute_metrics
from lab2rob.llm.cost_log import CostLog


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def collect_sessions(runs_dir: Path) -> list:
    """Grounding sessions (grounding.json), oldest first by file name."""
    sessions = []
    base = Path(runs_dir)
    if not base.is_dir():
        return sessions
    for path in sorted(base.glob("*/grounding.json")):
        session = _read_json(path, None)
        if isinstance(session, dict):
            session["_dir"] = path.parent
            sessions.append(session)
    return sessions


def collect_cert_runs(runs_dir: Path) -> list:
    """Certify runs (run.json), excluding *_repair passes.

    Repair passes fold into their initial run's audit row via the
    repair.json repair_run link; counting them separately would double
    count one certification.
    """
    runs = []
    base = Path(runs_dir)
    if not base.is_dir():
        return runs
    for path in sorted(base.glob("*/run.json")):
        run = _read_json(path, None)
        if not isinstance(run, dict):
            continue
        run_id = run.get("run_id", path.parent.name)
        if str(run_id).endswith("_repair"):
            continue
        directory = path.parent
        runs.append({"run": run, "repair": _read_json(directory / "repair.json", None),
                     "decision": _read_json(directory / "decision.json", None)})
    return runs


def collect_eval_runs(runs_dir: Path) -> tuple:
    """Eval records (eval.json) plus twin labels from twin.json sidecars."""
    records, twins = [], {}
    base = Path(runs_dir)
    if not base.is_dir():
        return records, twins
    for path in sorted(base.glob("*/eval.json")):
        record = _read_json(path, None)
        if isinstance(record, dict):
            records.append(record)
            sidecar = _read_json(path.parent / "twin.json", None)
            if isinstance(sidecar, dict) and sidecar.get("expected"):
                twins[record.get("eval_id", path.parent.name)] = sidecar["expected"]
    return records, twins


def build_funnel(sessions: list, cert_runs: list) -> dict:
    """Coverage funnel: protocols in, human-accepted tasks out."""
    verdicts: dict = {}
    grounded_by: dict = {}
    decisions = {"accepted": 0, "edited": 0, "rejected": 0, "undecided": 0}
    reasons: dict = {}
    steps = 0
    for session in sessions:
        for step in session.get("steps", []):
            steps += 1
            verdict = step.get("verdict", {})
            status = verdict.get("status", "?")
            verdicts[status] = verdicts.get(status, 0) + 1
            if status == "grounded":
                by = verdict.get("by", "?")
                grounded_by[by] = grounded_by.get(by, 0) + 1
            reason = verdict.get("reason")
            if reason and status != "grounded":
                reasons[reason] = reasons.get(reason, 0) + 1
            decision = (step.get("decision") or {}).get("decision")
            bucket = {"accept": "accepted", "edit": "edited",
                      "reject": "rejected"}.get(decision, "undecided")
            decisions[bucket] += 1
    planned, planned_oos = 0, {}
    for session in sessions:
        directory = session.get("_dir")
        spec = _read_json(Path(directory) / "specify.json", None) if directory else None
        for row in (spec or {}).get("program", []):
            if row.get("status") == "planned":
                planned += 1
            else:
                reason = row.get("reason", "?")
                planned_oos[reason] = planned_oos.get(reason, 0) + 1
                reasons[f"plan:{reason}"] = reasons.get(f"plan:{reason}", 0) + 1
    certified, flagged, accepted = 0, 0, 0
    for entry in cert_runs:
        run = entry["run"]
        if run.get("status") == "complete":
            certified += 1
        else:
            abort = run.get("abort_reason") or "aborted"
            reasons[f"run:{abort}"] = reasons.get(f"run:{abort}", 0) + 1
        repair = entry["repair"] or {}
        if repair.get("human_flag"):
            flagged += 1
        if (entry["decision"] or {}).get("decision") == "accept":
            accepted += 1
    return {"protocols": len(sessions), "steps": steps, "verdicts": verdicts,
            "grounded_by": grounded_by, "decisions": decisions, "planned": planned,
            "planned_out_of_scope": planned_oos, "certified": certified,
            "certified_flagged": flagged, "accepted_runs": accepted,
            "failure_reasons": reasons}


def _audit_row(entry: dict) -> dict:
    run, repair, decision = entry["run"], entry["repair"] or {}, entry["decision"]
    visual = repair.get("visual", []) or []
    return {"run_id": run.get("run_id", "?"),
            "protocol_id": run.get("protocol_id", "?"),
            "seed": run.get("seed"),
            "status": run.get("status", "?"),
            "predicate_counts": run.get("predicate_counts", {}),
            "repair_run": repair.get("repair_run"),
            "visual_attempts": len(visual),
            "visual_passed": visual[-1].get("passed") if visual else None,
            "human_flag": bool(repair.get("human_flag")),
            "decision": (decision or {}).get("decision")}


def _audit_bucket(row: dict) -> str:
    if row["decision"] == "reject":
        return "rejected"
    if row["decision"] == "accept" and (row["human_flag"] or row["status"] != "complete"):
        return "human_fixed"
    certified = row["status"] == "complete" and not row["human_flag"]
    if certified and row["repair_run"] is None and row["visual_attempts"] <= 1:
        return "first_pass"
    if certified:
        return "auto_recovered"
    return "still_open"


def build_audit(cert_runs: list) -> dict:
    """Per-run audit rows plus the SciHorizon-format summary.

    first_pass: complete, clean, no repair, at most one visual attempt.
    auto_recovered: complete and clean, but needed the repair pass or a
    visual re-query. human_fixed: flagged or incomplete, accepted by a
    human anyway. rejected / still_open complete the picture.
    """
    rows = []
    summary = {"first_pass": 0, "auto_recovered": 0, "human_fixed": 0,
               "rejected": 0, "still_open": 0}
    for entry in cert_runs:
        row = _audit_row(entry)
        row["bucket"] = _audit_bucket(row)
        summary[row["bucket"]] += 1
        rows.append(row)
    return {"runs": rows, "summary": summary}


def build_metrics(runs_dir: Path, sessions: list, eval_records: list,
                  twins: dict) -> dict:
    """Eval scoreboard plus cost rollup for the report."""
    cost_path = Path(runs_dir) / "llm_costs.jsonl"
    cost_summary = CostLog(str(cost_path)).summarize_by_job() if cost_path.exists() else None
    counts = [s.get("counts", {}) for s in sessions]
    scoreboard = compute_metrics(eval_records, grounding_counts=counts or None,
                                 cost_summary=cost_summary,
                                 twin_expected=twins or None)
    return {"eval_runs": len(eval_records), "twins_scored": len(twins),
            "scoreboard": scoreboard}


def _md_table(headers: list, rows: list) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def render_markdown(funnel: dict, audit: dict, metrics: dict) -> str:
    """Human-readable one-page report."""
    score = metrics["scoreboard"]
    funnel_rows = [
        ["protocols grounded", funnel["protocols"]],
        ["steps", funnel["steps"]],
        ["grounded (rules / LLM)",
         f"{sum(funnel['grounded_by'].values())} "
         f"({funnel['grounded_by'].get('rules', 0)} / {funnel['grounded_by'].get('llm', 0)})"],
        ["decisions (accept / edit / reject / open)",
         f"{funnel['decisions']['accepted']} / {funnel['decisions']['edited']} / "
         f"{funnel['decisions']['rejected']} / {funnel['decisions']['undecided']}"],
        ["steps planned", funnel["planned"]],
        ["runs certified (flagged)", f"{funnel['certified']} ({funnel['certified_flagged']})"],
        ["runs human-accepted", funnel["accepted_runs"]],
    ]
    summary = audit["summary"]
    audit_rows = [[r["run_id"], r["status"], r["bucket"],
                   f"{(r['predicate_counts'] or {}).get('passed', 0)}/"
                   f"{(r['predicate_counts'] or {}).get('failed', 0)}",
                   r["visual_attempts"], r["decision"] or "-"]
                  for r in audit["runs"]]
    top_reasons = sorted(funnel["failure_reasons"].items(),
                         key=lambda kv: (-kv[1], kv[0]))[:8]
    return "\n".join([
        "# lab2rob report",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}.",
        "",
        "## Coverage funnel",
        "",
        _md_table(["stage", "count"], funnel_rows),
        "",
        "Top failure reasons: " + (", ".join(f"{k} ({v})" for k, v in top_reasons)
                                   or "none") + ".",
        "",
        "## Certification audit (SciHorizon format)",
        "",
        f"first pass {summary['first_pass']}, auto-recovered "
        f"{summary['auto_recovered']}, human-fixed {summary['human_fixed']}, "
        f"rejected {summary['rejected']}, still open {summary['still_open']}.",
        "",
        _md_table(["run", "status", "bucket", "pred +/-", "visual", "decision"],
                  audit_rows) if audit_rows else "No certified runs yet.",
        "",
        "## Metrics",
        "",
        f"eval runs: {metrics['eval_runs']}, twins scored: {metrics['twins_scored']}, "
        f"SR {score.get('sr_mean')}, SSR {score.get('ssr_mean')}, "
        f"safety violations {score.get('safety', {}).get('violations')}.",
        "",
        "Buckets: first_pass = complete, clean, no repair, at most one visual "
        "attempt; auto_recovered = complete and clean but needed repair or a "
        "visual re-query; human_fixed = flagged or incomplete yet accepted.",
        "",
    ])


def write_report(out_dir: Path, funnel: dict, audit: dict, metrics: dict) -> dict:
    """Write funnel.json, audit.json, metrics.json and report.md."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, payload in (("funnel.json", funnel), ("audit.json", audit),
                          ("metrics.json", metrics)):
        path = out / name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        files[name] = str(path)
    readme = out / "report.md"
    readme.write_text(render_markdown(funnel, audit, metrics), encoding="utf-8")
    files["report.md"] = str(readme)
    return {"dir": str(out), "files": files}
