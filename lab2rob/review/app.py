"""Review GUI: certify runs plus per-step protocol grounding review.

Stdlib-only local web app (no installs): run list with review statuses,
per-run detail (predicates, trace ops, visual renders, repair diagnostics),
and a decision form; plus a Ground tab where a cached protocol is grounded
step by step (rules-first with LLM fallback, or full-LLM translation) and
each step is accepted, rejected or edited. Run decisions save to
runs/<run_id>/decision.json; grounding decisions save to
runs/<grounding_id>/grounding.json with rules/LLM/human provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lab2rob.build.scene import SceneRejected, build_scene
from lab2rob.certify.repair import certify_with_repair
from lab2rob.certify.run import RUNS_DIR
from lab2rob.certify.visual import LLMBackend
from lab2rob.ground.llm_ground import (build_context, ground_protocol_llm,
                                       ground_step_llm)
from lab2rob.ground.rules import (Grounding, ProtocolGrounding, Quantity, ground_protocol,
                                 ground_step, parse_steps)
from lab2rob.ingest.bioprot import DEFAULT_TIER1_PATH, load_bioprot_id, load_tier1_file
from lab2rob.ingest.protocols_io import (SOURCES_DIR, ProtocolsIOError, list_cached,
                                         load_cached_protocol)
from lab2rob.llm import (API_KEY_ENV, DailyCapExceeded, JobBudgetExceeded, Lab2RobClient,
                         UnknownJob)
from lab2rob.registry import resolve_skill
from lab2rob.specify.judge import plan_judge
from lab2rob.specify.program import plan_protocol

ID_PATTERN = re.compile(r"[A-Za-z0-9_.\-]+")
MAX_BODY = 65536


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _run_ids(runs_dir: Path) -> list:
    if not runs_dir.is_dir():
        return []
    return sorted(p.name for p in runs_dir.iterdir()
                  if p.is_dir() and ID_PATTERN.fullmatch(p.name)
                  and (p / "run.json").exists())


def review_state(run: dict, repair: dict | None, decision: dict | None) -> str:
    """decided-* > needs_review (flagged, aborted, failed, or unchecked) > passed."""
    if decision is not None:
        return f"decided-{decision.get('decision', 'accept')}"
    counts = run.get("predicate_counts", {})
    if (repair is None or repair.get("human_flag")
            or run.get("status") != "complete" or counts.get("failed", 0)):
        return "needs_review"
    return "passed"


def summarize_runs(runs_dir: Path) -> list:
    rows = []
    for run_id in _run_ids(runs_dir):
        run = _read_json(runs_dir / run_id / "run.json", {})
        repair_path = runs_dir / run_id / "repair.json"
        repair = _read_json(repair_path, None) if repair_path.exists() else None
        decision_path = runs_dir / run_id / "decision.json"
        decision = _read_json(decision_path, None) if decision_path.exists() else None
        rows.append({
            "run_id": run_id,
            "protocol_id": run.get("protocol_id", ""),
            "family": run.get("family", ""),
            "seed": run.get("seed"),
            "status": run.get("status", ""),
            "predicate_counts": run.get("predicate_counts", {}),
            "human_flag": bool(repair and repair.get("human_flag")),
            "decision": (decision or {}).get("decision"),
            "state": review_state(run, repair, decision),
        })
    rows.sort(key=lambda r: (r["state"] != "needs_review", r["run_id"]))
    return rows


def run_detail(runs_dir: Path, run_id: str) -> dict | None:
    """Full review payload for one run, or None when it does not exist."""
    if not ID_PATTERN.fullmatch(run_id):
        return None
    run_dir = runs_dir / run_id
    if not run_dir.is_dir() or not (run_dir / "run.json").exists():
        return None
    run = _read_json(run_dir / "run.json", {})
    repair = _read_json(run_dir / "repair.json", None)
    decision = _read_json(run_dir / "decision.json", None)
    trace = [json.loads(line) for line in
             (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()] if (run_dir / "trace.jsonl").exists() else []
    return {
        "run": run,
        "predicates": _read_json(run_dir / "predicates.json", []),
        "trace": trace,
        "repair": repair,
        "decision": decision,
        "visuals": sorted(p.name for p in run_dir.glob("visual_*.png")),
        "state": review_state(run, repair, decision),
    }


def save_decision(runs_dir: Path, run_id: str, decision: str, note: str) -> dict:
    """Record (or overwrite) the human decision for a run."""
    if decision not in ("accept", "reject"):
        raise ValueError(f"decision must be 'accept' or 'reject', got {decision!r}")
    if not isinstance(note, str) or len(note) > 2000:
        raise ValueError("note must be a string of at most 2000 chars")
    record = {"run_id": run_id, "decision": decision, "note": note,
              "by": "human",
              "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (runs_dir / run_id / "decision.json").write_text(json.dumps(record, indent=2),
                                                     encoding="utf-8")
    return record


MODES = ("rules", "full_llm")
#: Failures from the model side (key, caps, routing, cache) surface as 502.
LLM_ERRORS = (RuntimeError, DailyCapExceeded, JobBudgetExceeded, UnknownJob,
              ProtocolsIOError)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def list_protocols(sources_dir=None, tier1_path=None) -> list:
    """One picker row per cached protocol, both corpora."""
    rows = [{"dataset": "protocolsio", "id": r["id"], "title": r["title"],
             "steps": r["steps"], "license": r["license"]}
            for r in list_cached(Path(sources_dir) if sources_dir else SOURCES_DIR)]
    try:
        tier1 = load_tier1_file(tier1_path or DEFAULT_TIER1_PATH)
    except (OSError, ValueError):
        tier1 = []
    for raw in tier1:
        rows.append({"dataset": "bioprot", "id": raw.protocol_id, "title": raw.title,
                     "steps": raw.number_of_steps, "license": raw.license})
    return rows


def load_raw_protocol(dataset: str, protocol_id, sources_dir=None, tier1_path=None):
    if dataset == "protocolsio":
        return load_cached_protocol(protocol_id,
                                    Path(sources_dir) if sources_dir else SOURCES_DIR)
    if dataset == "bioprot":
        return load_bioprot_id(str(protocol_id), tier1_path or DEFAULT_TIER1_PATH)
    raise KeyError(f"unknown dataset {dataset!r}")


def grounding_id_for(dataset: str, protocol_id, mode: str) -> str:
    gid = f"g_{dataset}_{protocol_id}_{mode}"
    if not ID_PATTERN.fullmatch(gid):
        raise ValueError(f"bad grounding id {gid!r}")
    return gid


def _verdict_json(grounding) -> dict:
    return {"status": grounding.status, "action": grounding.action,
            "reason": grounding.reason, "detail": grounding.detail,
            "by": grounding.by}


def _step_json(grounding, alternatives) -> dict:
    return {"idx": grounding.idx, "source_text": grounding.source_text,
            "comments": list(grounding.comments),
            "quantities": [{"raw": q.raw, "value": q.value, "unit": q.unit,
                            "approximate": q.approximate}
                           for q in grounding.quantities],
            "reagents": list(grounding.reagents),
            "entities": [{"name": e.get("name"), "status": e.get("status"),
                          "scope": e.get("scope"), "reason": e.get("reason")}
                         for e in grounding.entities],
            "verdict": _verdict_json(grounding),
            "alternatives": [_verdict_json(a) for a in alternatives],
            "decision": None}


def start_grounding(runs_dir: Path, dataset: str, protocol_id, mode: str,
                    client_factory=None, sources_dir=None, tier1_path=None) -> dict:
    """Ground a protocol into a reviewable session on disk.

    mode "rules" grounds rules-first with LLM fallback for unresolved steps
    (needs_llm steps stay open when no key is set); mode "full_llm" translates
    the whole protocol with one model call. Alternatives show the other
    engine's verdict: rules for full-LLM steps, rules for fallback-resolved
    steps. Re-running overwrites the session, so grounding is re-doable.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    raw = load_raw_protocol(dataset, protocol_id, sources_dir, tier1_path)
    if client_factory is None:
        llm_available = bool(os.environ.get(API_KEY_ENV))
        client = Lab2RobClient() if llm_available else None
    else:
        llm_available, client = True, client_factory()
    rules_only = [ground_step(p) for p in parse_steps(raw.steps_raw)]
    if mode == "full_llm":
        if client is None:
            raise RuntimeError(f"Set {API_KEY_ENV} for full-LLM translation")
        primary = ground_protocol_llm(raw, client).steps
    else:
        primary = ground_protocol(raw, client).steps
    by_idx = {r.idx: r for r in rules_only}
    steps = []
    for prim in primary:
        alt = []
        if (mode == "full_llm" or prim.by == "llm") and prim.idx in by_idx:
            alt = [by_idx[prim.idx]]
        steps.append(_step_json(prim, alt))
    counts: dict = {}
    for prim in primary:
        key = "llm" if prim.by == "llm" else (
            "rules" if prim.status == "grounded" else prim.status)
        counts[key] = counts.get(key, 0) + 1
    gid = grounding_id_for(dataset, protocol_id, mode)
    session = {"grounding_id": gid, "dataset": dataset,
               "protocol_id": str(protocol_id), "title": raw.title,
               "description": raw.description,
               "source_url": raw.source_url, "license": raw.license,
               "mode": mode, "llm_available": llm_available,
               "created_at": _now(), "counts": counts, "steps": steps, "runs": []}
    (runs_dir / gid).mkdir(parents=True, exist_ok=True)
    (runs_dir / gid / "grounding.json").write_text(json.dumps(session, indent=2),
                                                  encoding="utf-8")
    return session


def list_groundings(runs_dir: Path) -> list:
    rows = []
    if not runs_dir.is_dir():
        return rows
    for path in sorted(runs_dir.glob("*/grounding.json")):
        session = _read_json(path, None)
        if not isinstance(session, dict):
            continue
        steps = session.get("steps", [])
        decided = sum(1 for s in steps
                      if isinstance(s, dict) and s.get("decision"))
        rows.append({"grounding_id": session.get("grounding_id", path.parent.name),
                     "dataset": session.get("dataset", ""),
                     "protocol_id": session.get("protocol_id", ""),
                     "title": session.get("title", ""), "mode": session.get("mode", ""),
                     "steps": len(steps), "decided": decided,
                     "state": ("decided" if steps and decided == len(steps)
                               else "needs_review")})
    rows.sort(key=lambda r: (r["state"] != "needs_review", r["grounding_id"]))
    return rows


def get_grounding(runs_dir: Path, grounding_id: str) -> dict | None:
    if not ID_PATTERN.fullmatch(grounding_id) or grounding_id in (".", ".."):
        return None
    path = runs_dir / grounding_id / "grounding.json"
    if not path.exists():
        return None
    session = _read_json(path, None)
    return session if isinstance(session, dict) else None


def save_step_decision(runs_dir: Path, grounding_id: str, idx: int,
                       decision: str, action: str | None, note: str,
                       by: str = "human") -> dict:
    """Record (or overwrite) the decision for one grounded step."""
    if decision not in ("accept", "reject", "edit"):
        raise ValueError(f"decision must be accept/reject/edit, got {decision!r}")
    if not isinstance(note, str) or len(note) > 2000:
        raise ValueError("note must be a string of at most 2000 chars")
    session = get_grounding(runs_dir, grounding_id)
    if session is None:
        raise KeyError(f"unknown grounding {grounding_id!r}")
    step = next((s for s in session.get("steps", []) if s.get("idx") == idx), None)
    if step is None:
        raise KeyError(f"unknown step {idx!r}")
    if decision == "reject" and not note.strip():
        raise ValueError("a reject needs a note saying why")
    if decision == "edit":
        if not action or not isinstance(action, str):
            raise ValueError("an edit needs the corrected skill name")
        if resolve_skill(action)["reason"] == "unknown_skill":
            raise ValueError(f"unknown skill {action!r}")
        final_action = action
    elif decision == "accept":
        final_action = step["verdict"].get("action")
    else:
        final_action = None
    record = {"decision": decision, "action": final_action, "note": note,
              "by": by, "at": _now()}
    step["decision"] = record
    (runs_dir / grounding_id / "grounding.json").write_text(
        json.dumps(session, indent=2), encoding="utf-8")
    return record


def ask_step_llm(runs_dir: Path, grounding_id: str, idx: int,
                 client_factory=None) -> dict:
    """Ask the model about one step; store the answer as its LLM alternative.

    The step goes through single-step fallback with the full protocol as
    context. Each step keeps at most one asked answer: re-asking replaces
    the old one. The verdict and any human decision are untouched.
    """
    session = get_grounding(runs_dir, grounding_id)
    if session is None:
        raise KeyError(f"unknown grounding {grounding_id!r}")
    step = next((s for s in session.get("steps", []) if s.get("idx") == idx), None)
    if step is None:
        raise KeyError(f"unknown step {idx!r}")
    if client_factory is None:
        if not os.environ.get(API_KEY_ENV):
            raise RuntimeError(f"Set {API_KEY_ENV} to ask the model")
        client = Lab2RobClient()
    else:
        client = client_factory()
    numbered = [(s["idx"], s["source_text"]) for s in session.get("steps", [])]
    shell = Grounding(
        idx=step["idx"], source_text=step["source_text"],
        comments=list(step.get("comments", [])),
        quantities=[Quantity(raw=q["raw"], value=q["value"], unit=q["unit"],
                             approximate=q.get("approximate", False), pint=None)
                    for q in step.get("quantities", [])],
        reagents=list(step.get("reagents", [])),
        entities=[dict(e) for e in step.get("entities", [])],
        status="needs_llm")
    answered = ground_step_llm(
        shell, client,
        build_context(session.get("title", ""), numbered, step["idx"]))
    alternative = _verdict_json(answered)
    alternative["asked"] = True
    step["alternatives"] = [a for a in step.get("alternatives", [])
                            if not a.get("asked")] + [alternative]
    (runs_dir / grounding_id / "grounding.json").write_text(
        json.dumps(session, indent=2), encoding="utf-8")
    return step


def _jsonable(value):
    """Numpy scalars and friends become plain JSON (str when unknown)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def session_grounding(session) -> tuple:
    """Approved steps back into Grounding objects, decisions applied.

    Returns (groundings, undecided idx, rejected idx). Accepted steps keep
    their verdict; edited steps take the human skill as a human-corrected
    grounding (downstream planners re-resolve skills anyway, and nothing
    downstream reads the provenance flag).
    """
    groundings, undecided, blocked = [], [], []
    for step in session.get("steps", []):
        decision = step.get("decision")
        if not decision:
            undecided.append(step["idx"])
            continue
        if decision["decision"] == "reject":
            blocked.append(step["idx"])
            continue
        verdict = step["verdict"]
        action, status = verdict.get("action"), verdict["status"]
        reason, detail, by = (verdict.get("reason"), verdict.get("detail", ""),
                              verdict.get("by", "none"))
        if decision["decision"] == "edit":
            action, status, reason, by = decision["action"], "grounded", None, "human"
            detail = f"human correction: {action}"
            if decision.get("note"):
                detail += f" ({decision['note']})"
        groundings.append(Grounding(
            idx=step["idx"], source_text=step["source_text"],
            comments=list(step.get("comments", [])),
            quantities=[Quantity(raw=q["raw"], value=q["value"], unit=q["unit"],
                                 approximate=q.get("approximate", False), pint=None)
                        for q in step.get("quantities", [])],
            reagents=list(step.get("reagents", [])),
            entities=[dict(e) for e in step.get("entities", [])],
            action=action, status=status, reason=reason, detail=detail, by=by))
    return groundings, undecided, blocked


def _planned_grounding(session, groundings):
    return ProtocolGrounding(protocol_id=session.get("protocol_id", ""),
                             dataset=session.get("dataset", ""),
                             title=session.get("title", ""),
                             description=session.get("description", ""),
                             steps=groundings)


def specify_preview(runs_dir: Path, grounding_id: str) -> dict:
    """Program + judge preview over the session's approved steps (no LLM)."""
    session = get_grounding(runs_dir, grounding_id)
    if session is None:
        raise KeyError(f"unknown grounding {grounding_id!r}")
    groundings, undecided, blocked = session_grounding(session)
    planned = _planned_grounding(session, groundings)
    program, judge = plan_protocol(planned), plan_judge(planned)
    return {
        "grounding_id": grounding_id,
        "decided": len(groundings) + len(blocked),
        "total": len(session.get("steps", [])),
        "undecided": undecided, "blocked": blocked,
        "program": {
            "steps": [{"idx": s.idx, "source_text": s.source_text,
                       "status": s.status, "reason": s.reason, "detail": s.detail,
                       "ops": _jsonable(s.ops), "entities": list(s.entities)}
                      for s in program.steps],
            "out_of_scope": [{"step": o.step, "reason": o.reason, "detail": o.detail}
                             for o in program.out_of_scope],
            "entities": [{"uid": e.uid, "asset": e.asset, "slot": e.slot}
                         for e in program.entities],
        },
        "judge": {
            "steps": [{"idx": s.idx,
                       "predicates": [{"step": p.step, "pred": p.pred,
                                       "args": list(p.args), "target": p.target,
                                       "tol": p.tol} for p in s.predicates]}
                      for s in judge.steps],
            "outcomes": [{"type": o.type, "quantity": o.quantity,
                          "tol_rel": o.tol_rel} for o in judge.outcomes],
        },
    }


def run_certification(runs_dir: Path, grounding_id: str, seed: int = 0,
                      client_factory=None, visual_backend_factory=None) -> dict:
    """Build, certify (execute, predicates, visual, one repair) and record.

    Every step must be decided and none rejected; rejected steps block with
    their idx so the human edits or re-grounds instead of certifying a
    program they already refused. The visual check needs the model, so
    certification without a key (or injected backend) is refused, never
    silently skipped.
    """
    session = get_grounding(runs_dir, grounding_id)
    if session is None:
        raise KeyError(f"unknown grounding {grounding_id!r}")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    groundings, undecided, blocked = session_grounding(session)
    if undecided:
        raise ValueError(f"{len(undecided)} steps undecided: {undecided}")
    if blocked:
        raise ValueError(f"{len(blocked)} steps rejected: {blocked} "
                         f"(edit or re-ground to proceed)")
    planned = _planned_grounding(session, groundings)
    program, judge = plan_protocol(planned), plan_judge(planned)
    built = build_scene(program, seed=seed)
    if visual_backend_factory is not None:
        backend = visual_backend_factory()
    elif os.environ.get(API_KEY_ENV):
        backend = LLMBackend((client_factory or Lab2RobClient)())
    else:
        raise RuntimeError(f"Set {API_KEY_ENV} to certify "
                           f"(the visual check needs the model)")
    out = certify_with_repair(program, judge, built, seed=seed,
                              runs_dir=runs_dir, visual_backend=backend)
    initial, repair = out["initial"], out["repair"]
    final = repair if repair is not None else initial
    counts: dict = {"passed": 0, "failed": 0, "skipped": 0}
    for pred in final.predicates or []:
        key = pred.get("status", "skipped")
        counts[key] = counts.get(key, 0) + 1
    session_dir = runs_dir / grounding_id
    (session_dir / "specify.json").write_text(json.dumps(
        {"seed": seed, "created_at": _now(), "program": program.alignment(),
         "judge": judge.alignment()}, indent=2), encoding="utf-8")
    entry = {"run_id": initial.run_id,
             "repair_run": repair.run_id if repair else None,
             "seed": seed, "status": final.status,
             "human_flag": out["human_flag"], "at": _now()}
    session["runs"] = [r for r in session.get("runs", [])
                       if r.get("run_id") != initial.run_id] + [entry]
    (session_dir / "grounding.json").write_text(json.dumps(session, indent=2),
                                               encoding="utf-8")
    diagnosis = out.get("diagnosis") or {}
    return {"grounding_id": grounding_id, "run_id": initial.run_id,
            "repair_run": entry["repair_run"], "seed": seed,
            "status": final.status, "predicate_counts": counts,
            "human_flag": out["human_flag"],
            "diagnosis": _jsonable(diagnosis.get("why")),
            "visual": [{"attempt": v.get("attempt"), "passed": v.get("passed"),
                        "detail": v.get("detail")} for v in out.get("visual", [])]}


def _split(path: str):
    """Split a request path into (area, run_id, leaf) with traversal rejected."""
    parts = path.split("?", 1)[0].rstrip("/").split("/")
    if len(parts) < 3 or parts[1] not in ("api", "files"):
        return None, None, ""
    if parts[1] == "api":
        # /api/runs/<id>[/decision]
        if len(parts) < 4 or parts[2] != "runs":
            return None, None, ""
        run_id, leaf = parts[3], parts[4] if len(parts) > 4 else ""
    else:
        # /files/<id>/<name>
        if len(parts) != 4:
            return None, None, ""
        run_id, leaf = parts[2], parts[3]
    if not ID_PATTERN.fullmatch(run_id) or run_id in (".", ".."):
        return None, None, ""
    return parts[1], run_id, leaf


def _json(code: int, obj) -> tuple:
    return code, "application/json", json.dumps(obj).encode("utf-8")


def _split_ground(path: str):
    """Split a /api/ground/... path into (grounding_id, extra).

    extra is None (session detail), a step idx (per-step route), the
    "specify"/"certify" leaf, or ("ask", idx) for the Ask-LLM route;
    (None, None) otherwise. The bare /api/ground collection is handled
    separately.
    """
    parts = path.split("?", 1)[0].rstrip("/").split("/")
    if len(parts) < 4 or parts[1:3] != ["api", "ground"]:
        return None, None
    gid = parts[3]
    if not ID_PATTERN.fullmatch(gid) or gid in (".", ".."):
        return None, None
    if len(parts) == 4:
        return gid, None
    if len(parts) == 5 and parts[4] in ("specify", "certify"):
        return gid, parts[4]
    if len(parts) == 6 and parts[4] == "steps" and parts[5].isdigit():
        return gid, int(parts[5])
    if (len(parts) == 7 and parts[4] == "steps" and parts[5].isdigit()
            and parts[6] == "ask"):
        return gid, ("ask", int(parts[5]))
    return None, None


def _read_json_body(body: bytes | None):
    if not body or len(body) > MAX_BODY:
        raise ValueError("body required (max 64 KB)")
    try:
        payload = json.loads(bytes(body).decode("utf-8"))
    except ValueError:
        raise ValueError("body must be JSON") from None
    if not isinstance(payload, dict):
        raise ValueError("body must be a JSON object")
    return payload


def dispatch(method: str, raw_path: str, body: bytes | None, runs_dir: Path,
             client_factory=None, sources_dir=None, tier1_path=None,
             visual_backend_factory=None) -> tuple:
    """Pure request router: (method, path, body) -> (status, content-type, bytes).

    All route logic lives here so tests exercise it without sockets; the
    HTTP handler below is a thin skin over it. client_factory builds the LLM
    client for grounding routes (default: Lab2RobClient when a key is set);
    sources_dir and tier1_path override the protocol corpora (for tests);
    visual_backend_factory builds the certify visual backend (for tests).
    """
    path = raw_path.split("?", 1)[0]
    if method == "GET":
        if path == "/" or path == "/index.html":
            return 200, "text/html; charset=utf-8", PAGE.encode("utf-8")
        if path == "/api/runs":
            return _json(200, summarize_runs(runs_dir))
        if path == "/api/protocols":
            return _json(200, list_protocols(sources_dir, tier1_path))
        if path == "/api/ground":
            return _json(200, list_groundings(runs_dir))
        gid, extra = _split_ground(path)
        if gid is not None and extra is None:
            session = get_grounding(runs_dir, gid)
            if session is None:
                return _json(404, {"error": "unknown grounding"})
            return _json(200, session)
        if gid is not None and extra == "specify":
            try:
                return _json(200, specify_preview(runs_dir, gid))
            except KeyError as e:
                return _json(404, {"error": str(e)})
        area, run_id, _ = _split(path)
        if area == "api" and run_id is not None:
            detail = run_detail(runs_dir, run_id)
            if detail is None:
                return _json(404, {"error": "unknown run"})
            return _json(200, detail)
        if area == "files" and run_id is not None:
            name = path.rsplit("/", 1)[-1]
            allowed = {p.name for p in (runs_dir / run_id).glob("visual_*.png")}
            if name in allowed:
                return 200, "image/png", (runs_dir / run_id / name).read_bytes()
            return _json(404, {"error": "unknown file"})
        return _json(404, {"error": "unknown path"})
    if method == "POST":
        if path == "/api/ground":
            try:
                payload = _read_json_body(body)
                session = start_grounding(
                    runs_dir, payload.get("dataset", ""),
                    payload.get("protocol_id", ""), payload.get("mode", "rules"),
                    client_factory, sources_dir, tier1_path)
            except KeyError as e:
                return _json(404, {"error": str(e)})
            except ValueError as e:
                return _json(400, {"error": str(e)})
            except LLM_ERRORS as e:
                return _json(502, {"error": str(e)})
            return _json(200, session)
        gid, extra = _split_ground(path)
        if gid is not None and isinstance(extra, int):
            try:
                payload = _read_json_body(body)
                record = save_step_decision(runs_dir, gid, extra,
                                            payload.get("decision", ""),
                                            payload.get("action"),
                                            payload.get("note", ""))
            except KeyError as e:
                return _json(404, {"error": str(e)})
            except ValueError as e:
                return _json(400, {"error": str(e)})
            return _json(200, {"ok": True, "decision": record})
        if (gid is not None and isinstance(extra, tuple) and extra[0] == "ask"):
            try:
                _read_json_body(body)
                step = ask_step_llm(runs_dir, gid, extra[1], client_factory)
            except KeyError as e:
                return _json(404, {"error": str(e)})
            except ValueError as e:
                return _json(400, {"error": str(e)})
            except LLM_ERRORS as e:
                return _json(502, {"error": str(e)})
            return _json(200, {"ok": True, "step": step})
        if gid is not None and extra == "certify":
            try:
                payload = _read_json_body(body)
                result = run_certification(runs_dir, gid, payload.get("seed", 0),
                                           client_factory, visual_backend_factory)
            except KeyError as e:
                return _json(404, {"error": str(e)})
            except ValueError as e:
                return _json(400, {"error": str(e)})
            except SceneRejected as e:
                return _json(422, {"error": str(e)})
            except LLM_ERRORS as e:
                return _json(502, {"error": str(e)})
            return _json(200, result)
        area, run_id, leaf = _split(path)
        if area != "api" or run_id is None or leaf != "decision":
            return _json(404, {"error": "unknown path"})
        if run_detail(runs_dir, run_id) is None:
            return _json(404, {"error": "unknown run"})
        try:
            payload = _read_json_body(body)
            record = save_decision(runs_dir, run_id,
                                   payload.get("decision", ""),
                                   payload.get("note", ""))
        except ValueError as e:
            return _json(400, {"error": str(e)})
        return _json(200, {"ok": True, "decision": record})
    return _json(405, {"error": "method not allowed"})


def make_handler(runs_dir: Path, client_factory=None, sources_dir=None,
                   tier1_path=None, visual_backend_factory=None):
    """Thin HTTP skin over dispatch(), bound to one runs directory."""

    class ReviewHandler(BaseHTTPRequestHandler):
        server_version = "lab2rob-review/1"

        def log_message(self, *args):  # keep the console for app output
            pass

        def _serve(self, method: str) -> None:
            body = None
            if method == "POST":
                try:
                    length = int(self.headers.get("Content-Length", 0))
                except ValueError:
                    length = 0
                body = self.rfile.read(min(max(length, 0), MAX_BODY + 1))
            code, content_type, payload = dispatch(
                method, self.path, body, runs_dir, client_factory, sources_dir,
                tier1_path, visual_backend_factory)
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):  # noqa: N802 (stdlib handler naming)
            self._serve("GET")

        def do_POST(self):  # noqa: N802 (stdlib handler naming)
            self._serve("POST")

    return ReviewHandler


def make_server(runs_dir: Path = RUNS_DIR, port: int = 8080,
                  client_factory=None, sources_dir=None, tier1_path=None,
                  visual_backend_factory=None) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(
        ("127.0.0.1", port),
        make_handler(Path(runs_dir), client_factory, sources_dir, tier1_path,
                     visual_backend_factory))


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>lab2rob review</title>
<style>
  :root { --ink: #1f2328; --muted: #59636e; --line: #d8dee4; --bg: #f6f8fa;
    --green: #1a7f37; --red: #cf222e; --amber: #9a6700; --slate: #42526b; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.45 -apple-system, "Segoe UI", system-ui, sans-serif;
    color: var(--ink); background: var(--bg); }
  header { background: #111418; color: #fff; padding: 10px 18px;
    display: flex; align-items: baseline; gap: 12px; }
  header h1 { font-size: 15px; margin: 0; font-weight: 650; }
  header span { font-size: 12px; color: #b6bdc7; font-variant-numeric: tabular-nums; }
  main { display: flex; gap: 14px; padding: 14px 18px; align-items: flex-start; }
  #runs { width: 360px; flex: none; background: #fff; border: 1px solid var(--line);
    border-radius: 6px; overflow: hidden; }
  #runs table { width: 100%; border-collapse: collapse; font-size: 13px; }
  #runs td, #runs th { padding: 7px 10px; border-bottom: 1px solid var(--line);
    text-align: left; }
  #runs tbody tr { cursor: pointer; }
  #runs tbody tr:hover { background: #f0f3f6; }
  #runs tbody tr.sel { background: #e8eef7; }
  .mono { font-family: ui-monospace, Consolas, monospace; font-size: 12px; }
  .pill { display: inline-block; font-size: 11px; font-weight: 650; color: #fff;
    border-radius: 999px; padding: 1px 9px; white-space: nowrap; }
  .needs_review { background: var(--amber); }
  .passed, .decided-accept { background: var(--green); }
  .decided-reject, .failed { background: var(--red); }
  .skipped, .aborted { background: var(--slate); }
  #detail { flex: 1; min-width: 0; background: #fff; border: 1px solid var(--line);
    border-radius: 6px; padding: 14px 16px; }
  #detail h2 { font-size: 16px; margin: 0 0 2px; }
  #detail h3 { font-size: 13px; margin: 18px 0 6px; }
  #detail table { width: 100%; border-collapse: collapse; font-size: 13px; }
  #detail td, #detail th { padding: 5px 8px; border-bottom: 1px solid var(--line);
    text-align: left; vertical-align: top; }
  #detail th { color: var(--muted); font-weight: 600; }
  .num { font-variant-numeric: tabular-nums; }
  .muted { color: var(--muted); }
  #visuals img { max-width: 100%; border: 1px solid var(--line); margin: 0 8px 8px 0;
    max-height: 300px; }
  form { display: flex; gap: 8px; align-items: center; margin-top: 8px; flex-wrap: wrap; }
  button { font-size: 13px; font-weight: 600; border-radius: 6px; padding: 6px 14px;
    cursor: pointer; border: 1px solid transparent; }
  #accept { background: var(--green); color: #fff; }
  #reject { background: #fff; color: var(--red); border-color: var(--red); }
  textarea { flex: 1; min-width: 220px; font: inherit; font-size: 13px;
    border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px; }
  #msg { font-size: 13px; margin-top: 6px; min-height: 18px; }
  nav.tabs { display: flex; gap: 4px; margin-left: 12px; }
  nav.tabs button { background: transparent; color: #b6bdc7; border: none;
    font-weight: 600; padding: 6px 12px; }
  nav.tabs button.sel { color: #fff; background: #2a3139; }
  .hidden { display: none !important; }
  #proto { width: 360px; flex: none; background: #fff; border: 1px solid var(--line);
    border-radius: 6px; padding: 12px; }
  #proto select, #proto input[type=text] { width: 100%; font: inherit; font-size: 13px;
    border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px; margin: 4px 0 10px; }
  #proto label { font-size: 12px; color: var(--muted); font-weight: 600; }
  #proto .row { margin: 4px 0 10px; font-size: 13px; }
  #start { background: #111418; color: #fff; width: 100%; }
  #gmsg { font-size: 13px; margin-top: 6px; min-height: 18px; }
  #gsessions { margin-top: 8px; font-size: 13px; }
  #gsessions div { padding: 6px 8px; border-top: 1px solid var(--line); cursor: pointer; }
  #gsessions div:hover { background: #f0f3f6; }
  #gsessions div.sel { background: #e8eef7; }
  #gdetail { flex: 1; min-width: 0; background: #fff; border: 1px solid var(--line);
    border-radius: 6px; padding: 14px 16px; }
  #gdetail h2 { font-size: 16px; margin: 0 0 2px; }
  .step { border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px;
    margin: 10px 0; }
  .step h4 { margin: 0 0 4px; font-size: 14px; display: flex; gap: 8px;
    align-items: center; flex-wrap: wrap; }
  .step p.src { margin: 4px 0; }
  .chip { display: inline-block; font-size: 12px; background: #eef1f4;
    border-radius: 4px; padding: 1px 8px; margin: 0 6px 4px 0; }
  .grounded, .decided { background: var(--green); }
  .out_of_scope, .ambiguous { background: var(--amber); }
  .needs_llm { background: var(--slate); }
  .by { font-size: 11px; color: var(--muted); border: 1px solid var(--line);
    border-radius: 999px; padding: 1px 9px; white-space: nowrap; }
  details { font-size: 13px; margin: 6px 0; }
  details summary { cursor: pointer; color: var(--muted); }
  #gdetail h3 { font-size: 13px; margin: 18px 0 6px; }
  #gdetail table { width: 100%; border-collapse: collapse; font-size: 13px; }
  #gdetail td, #gdetail th { padding: 5px 8px; border-bottom: 1px solid var(--line);
    text-align: left; vertical-align: top; }
  #gdetail th { color: var(--muted); font-weight: 600; }
  #gdetail input[type=number] { width: 90px; font: inherit; font-size: 13px;
    border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px; }
  #certify { background: #111418; color: #fff; }
  #certify:disabled { opacity: 0.55; cursor: wait; }
  .planned { background: var(--green); }
  .warn { background: #fff8e6; border: 1px solid #e3b341; border-radius: 6px;
    padding: 8px 12px; margin: 10px 0; font-size: 13px; }
  .err { background: #fdeceb; border: 1px solid #e8a19c; border-radius: 6px;
    padding: 8px 12px; margin: 10px 0; font-size: 13px; }
  .ok { background: #e9f7ee; border: 1px solid #7cc491; border-radius: 6px;
    padding: 8px 12px; margin: 10px 0; font-size: 13px; }
</style>
</head>
<body>
<header><h1>lab2rob review</h1><span id="counts"></span>
<nav class="tabs"><button id="tab-runs" class="sel">Runs</button><button id="tab-ground">Ground</button></nav></header>
<main id="view-runs">
  <div id="runs"><table>
    <thead><tr><th>run</th><th>state</th><th>predicates</th></tr></thead>
    <tbody id="rows"></tbody>
  </table></div>
  <div id="detail"><p class="muted">Select a run on the left.</p></div>
</main>
<main id="view-ground" class="hidden">
  <div id="proto">
    <label for="psel">protocol</label>
    <select id="psel"></select>
    <label>mode</label>
    <div class="row">
      <label><input type="radio" name="mode" value="rules" checked> rules first (LLM fills gaps)</label><br>
      <label><input type="radio" name="mode" value="full_llm"> full LLM translation</label>
    </div>
    <button id="start">Ground protocol</button>
    <div id="gmsg" class="muted"></div>
    <label>sessions</label>
    <div id="gsessions"></div>
  </div>
  <div id="gdetail"><p class="muted">Pick a protocol and press Ground, or open a session.</p></div>
</main>
<script>
const $ = s => document.querySelector(s);
function el(tag, text, cls) {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (cls) e.className = cls;
  return e;
}
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.status);
  return r.json();
}
let current = null;
function pill(state) {
  const s = el("span", state.replace(/_/g, " "), "pill " + state);
  return s;
}
async function loadRuns() {
  const runs = await api("/api/runs");
  const need = runs.filter(r => r.state === "needs_review").length;
  $("#counts").textContent = runs.length + " runs, " + need + " need review";
  const tb = $("#rows");
  tb.replaceChildren();
  for (const r of runs) {
    const tr = el("tr");
    if (r.run_id === current) tr.className = "sel";
    tr.appendChild(el("td")).appendChild(el("span", r.run_id, "mono"));
    const c = r.predicate_counts || {};
    const pc = (c.passed || 0) + "/" + (c.failed || 0) + "/" + (c.skipped || 0);
    tr.appendChild(el("td")).appendChild(pill(r.state));
    tr.appendChild(el("td", pc, "num muted"));
    tr.onclick = () => select(r.run_id);
    tr.firstChild.title = r.protocol_id + " " + r.family + " seed " + r.seed;
    tb.appendChild(tr);
  }
}
function kv(table, k, v) {
  const tr = el("tr");
  tr.appendChild(el("td", k, "muted"));
  tr.appendChild(el("td", v));
  table.appendChild(tr);
}
function renderDetail(d) {
  const box = $("#detail");
  box.replaceChildren();
  box.appendChild(el("h2", d.run.run_id, "mono"));
  const sub = el("p", d.run.protocol_id + "  ·  " + d.run.family +
    "  ·  seed " + d.run.seed + "  ·  run " + d.run.status, "muted");
  box.appendChild(sub);
  const h = el("h3", "Decision");
  box.appendChild(h);
  const cur = el("p");
  if (d.decision) {
    cur.appendChild(pill("decided-" + d.decision.decision));
    cur.appendChild(el("span", "  " + (d.decision.note || "(no note)") +
      "  ·  " + d.decision.decided_at + "  ·  by " + d.decision.by, "muted"));
  } else {
    cur.appendChild(el("span", "No decision yet.", "muted"));
  }
  box.appendChild(cur);
  const form = el("form");
  const note = el("textarea");
  note.placeholder = "note (required to reject)";
  note.value = (d.decision && d.decision.note) || "";
  note.rows = 2;
  const acc = el("button", "Accept"); acc.id = "accept";
  const rej = el("button", "Reject"); rej.id = "reject";
  const msg = el("div"); msg.id = "msg";
  acc.onclick = e => { e.preventDefault(); submit("accept", note.value, msg); };
  rej.onclick = e => { e.preventDefault(); submit("reject", note.value, msg); };
  form.append(note, acc, rej);
  box.append(form, msg);
  box.appendChild(el("h3", "Predicates"));
  const pt = el("table");
  const ph = el("tr");
  ["step", "check", "well", "target", "tol", "result", "detail"].forEach(t =>
    ph.appendChild(el("th", t)));
  pt.appendChild(ph);
  for (const p of d.predicates) {
    const tr = el("tr");
    tr.appendChild(el("td", String(p.step), "num"));
    tr.appendChild(el("td", p.pred, "mono"));
    tr.appendChild(el("td", p.well || "—", "mono"));
    tr.appendChild(el("td", p.target === null ? "—" : String(p.target), "num"));
    tr.appendChild(el("td", p.tol === null ? "—" : String(p.tol), "num"));
    tr.appendChild(el("td")).appendChild(pill(p.status));
    tr.appendChild(el("td", p.detail || "", "muted"));
    pt.appendChild(tr);
  }
  box.appendChild(pt);
  if (d.repair) {
    box.appendChild(el("h3", "Repair"));
    const rt = el("table");
    kv(rt, "human flag", String(d.repair.human_flag));
    kv(rt, "diagnosis", d.repair.diagnosis || "—");
    kv(rt, "repair run", d.repair.repair_run || "—");
    (d.repair.visual || []).forEach(v =>
      kv(rt, "visual a" + v.attempt, (v.passed ? "pass" : "FAIL") + " — " + v.detail));
    box.appendChild(rt);
  }
  if (d.visuals.length) {
    box.appendChild(el("h3", "Renders"));
    const vs = el("div"); vs.id = "visuals";
    d.visuals.forEach(n => {
      const img = document.createElement("img");
      img.src = "/files/" + encodeURIComponent(d.run.run_id) + "/" + encodeURIComponent(n);
      img.alt = n;
      img.loading = "lazy";
      vs.appendChild(img);
    });
    box.appendChild(vs);
  }
  box.appendChild(el("h3", "Trace"));
  const tt = el("table");
  const th = el("tr");
  ["#", "op", "slot", "result", "spilled"].forEach(t => th.appendChild(el("th", t)));
  tt.appendChild(th);
  d.trace.filter(r => r.kind === "op").forEach(r => {
    const tr = el("tr");
    tr.appendChild(el("td", r.step + ":" + r.op, "num"));
    tr.appendChild(el("td", r.skill, "mono"));
    tr.appendChild(el("td", r.slot || "—", "mono"));
    const res = r.result || {};
    tr.appendChild(el("td", res.note || JSON.stringify(res), "muted"));
    tr.appendChild(el("td", String((r.state || {}).spilled_ul ?? "—"), "num"));
    tt.appendChild(tr);
  });
  box.appendChild(tt);
}
async function submit(decision, note, msg) {
  msg.textContent = "";
  if (decision === "reject" && !note.trim()) {
    msg.textContent = "A reject needs a note saying why.";
    return;
  }
  try {
    await api("/api/runs/" + encodeURIComponent(current) + "/decision", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({decision, note})});
    await loadRuns();
    await select(current);
  } catch (e) { msg.textContent = "Save failed: " + e.message; }
}
async function select(id) {
  current = id;
  renderDetail(await api("/api/runs/" + encodeURIComponent(id)));
  document.querySelectorAll("#rows tr").forEach(tr =>
    tr.classList.toggle("sel", tr.firstChild.textContent === id));
}
$("#tab-runs").onclick = () => switchTab("runs");
$("#tab-ground").onclick = () => switchTab("ground");
function switchTab(which) {
  $("#tab-runs").classList.toggle("sel", which === "runs");
  $("#tab-ground").classList.toggle("sel", which === "ground");
  $("#view-runs").classList.toggle("hidden", which !== "runs");
  $("#view-ground").classList.toggle("hidden", which !== "ground");
  if (which === "ground") loadGround().catch(e => { $("#gmsg").textContent = e.message; });
}
let gcurrent = null;
async function loadGround() {
  const [protos, sessions] = await Promise.all([api("/api/protocols"), api("/api/ground")]);
  const sel = $("#psel");
  sel.replaceChildren();
  for (const p of protos) {
    const o = document.createElement("option");
    o.value = p.dataset + "|" + p.id;
    o.textContent = "[" + p.dataset + "] " + (p.title || p.id) + " (" + p.steps + " steps)";
    sel.appendChild(o);
  }
  const box = $("#gsessions");
  box.replaceChildren();
  for (const s of sessions) {
    const d = el("div");
    if (s.grounding_id === gcurrent) d.className = "sel";
    d.appendChild(el("span", s.grounding_id, "mono"));
    d.appendChild(el("span", "  " + s.decided + "/" + s.steps + " ", "muted"));
    d.appendChild(pill(s.state));
    d.onclick = () => selectGrounding(s.grounding_id);
    d.title = s.title;
    box.appendChild(d);
  }
  if (!sessions.length) box.appendChild(el("div", "No sessions yet.", "muted"));
}
$("#start").onclick = async () => {
  const msg = $("#gmsg");
  msg.textContent = "Grounding…";
  try {
    const [dataset, id] = $("#psel").value.split("|");
    const mode = document.querySelector("input[name=mode]:checked").value;
    const s = await api("/api/ground", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({dataset, protocol_id: id, mode})});
    gcurrent = s.grounding_id;
    await loadGround();
    renderGrounding(s);
    msg.textContent = "";
  } catch (e) { msg.textContent = "Ground failed: " + e.message; }
};
async function selectGrounding(gid) {
  gcurrent = gid;
  renderGrounding(await api("/api/ground/" + encodeURIComponent(gid)));
  document.querySelectorAll("#gsessions div").forEach(d =>
    d.classList.toggle("sel", d.firstChild.textContent === gid));
}
function verdictLine(v) {
  const bits = [v.status.replace(/_/g, " ")];
  if (v.action) bits.push("skill " + v.action);
  if (v.reason) bits.push(v.reason);
  if (v.detail) bits.push("— " + v.detail);
  return bits.join(" · ");
}
function renderGrounding(s) {
  const box = $("#gdetail");
  box.replaceChildren();
  box.appendChild(el("h2", s.grounding_id, "mono"));
  box.appendChild(el("p", s.title + "  ·  mode " + s.mode +
    (s.llm_available ? "" : "  ·  LLM unavailable (no key)") +
    "  ·  " + JSON.stringify(s.counts), "muted"));
  for (const st of s.steps) box.appendChild(renderStep(s.grounding_id, st));
  if (s.runs && s.runs.length) {
    box.appendChild(el("h3", "Certified runs"));
    const rt = el("table");
    const rh = el("tr");
    ["run", "seed", "status", "review"].forEach(t => rh.appendChild(el("th", t)));
    rt.appendChild(rh);
    for (const r of s.runs) {
      const tr = el("tr");
      const link = el("td");
      const a = el("span", r.run_id, "mono");
      a.style.cursor = "pointer";
      a.style.textDecoration = "underline";
      a.onclick = () => openRun(r.run_id);
      link.appendChild(a);
      tr.appendChild(link);
      tr.appendChild(el("td", String(r.seed), "num"));
      tr.appendChild(el("td", r.status + (r.repair_run ? " (repair " + r.repair_run + ")" : ""), "muted"));
      tr.appendChild(el("td")).appendChild(pill(r.human_flag ? "needs_review" : "passed"));
      rt.appendChild(tr);
    }
    box.appendChild(rt);
  }
  const sec = el("div"); sec.id = "spec";
  sec.appendChild(el("p", "Loading program and judge…", "muted"));
  box.appendChild(sec);
  loadSpecify(s.grounding_id);
}
async function openRun(run_id) {
  switchTab("runs");
  await loadRuns();
  await select(run_id);
}
function renderStep(gid, st) {
  const card = el("div"); card.className = "step";
  const h = el("h4", "Step " + st.idx + " ");
  h.appendChild(pill(st.verdict.status));
  h.appendChild(el("span", st.verdict.by, "by"));
  if (st.decision) {
    h.appendChild(pill("decided-" + (st.decision.decision === "reject" ? "reject" : "accept")));
    h.title = st.decision.decision + (st.decision.action ? " " + st.decision.action : "") +
      " — " + (st.decision.note || "(no note)");
  }
  card.appendChild(h);
  card.appendChild(el("p", st.source_text, "src"));
  const chips = el("div");
  st.quantities.forEach(q => chips.appendChild(el("span", q.raw, "chip mono")));
  st.entities.forEach(e => chips.appendChild(el("span",
    e.name + " (" + e.scope + ")", "chip mono")));
  st.reagents.forEach(r => chips.appendChild(el("span", r, "chip")));
  if (chips.childNodes.length) card.appendChild(chips);
  card.appendChild(el("p", verdictLine(st.verdict), "muted"));
  const rulesAlts = (st.alternatives || []).filter(a => !a.asked);
  const askedAlts = (st.alternatives || []).filter(a => a.asked);
  if (rulesAlts.length) {
    const det = document.createElement("details");
    det.appendChild(el("summary", "rules alternative"));
    rulesAlts.forEach(a => det.appendChild(el("p", verdictLine(a))));
    card.appendChild(det);
  }
  askedAlts.forEach(a => {
    const det = document.createElement("details");
    det.open = true;
    det.appendChild(el("summary", "LLM answer"));
    det.appendChild(el("p", verdictLine(a)));
    card.appendChild(det);
  });
  const form = el("form");
  const skill = document.createElement("input");
  skill.type = "text"; skill.placeholder = "skill (for edit)";
  skill.value = (st.decision && st.decision.action) || st.verdict.action || "";
  skill.style.maxWidth = "160px";
  const note = el("textarea");
  note.placeholder = "note (required to reject)";
  note.value = (st.decision && st.decision.note) || "";
  note.rows = 1;
  const msg = el("div"); msg.id = "msg";
  const acc = el("button", "Accept"); acc.id = "accept";
  const rej = el("button", "Reject"); rej.id = "reject";
  const edt = el("button", "Save edit");
  const ask = el("button", "Ask LLM");
  acc.onclick = e => { e.preventDefault(); submitStep(gid, st.idx, "accept", null, note.value, msg); };
  rej.onclick = e => { e.preventDefault(); submitStep(gid, st.idx, "reject", null, note.value, msg); };
  edt.onclick = e => { e.preventDefault(); submitStep(gid, st.idx, "edit", skill.value.trim(), note.value, msg); };
  ask.onclick = e => { e.preventDefault(); submitAsk(gid, st.idx, msg, ask); };
  form.append(skill, note, acc, rej, edt, ask);
  card.append(form, msg);
  return card;
}
async function submitAsk(gid, idx, msg, btn) {
  msg.textContent = "";
  btn.disabled = true;
  const old = btn.textContent;
  btn.textContent = "Asking…";
  try {
    await api("/api/ground/" + encodeURIComponent(gid) + "/steps/" + idx + "/ask", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({})});
    renderGrounding(await api("/api/ground/" + encodeURIComponent(gid)));
  } catch (e) { msg.textContent = "Ask failed: " + e.message; }
  btn.disabled = false;
  btn.textContent = old;
}
async function submitStep(gid, idx, decision, action, note, msg) {
  msg.textContent = "";
  if (decision === "reject" && !note.trim()) {
    msg.textContent = "A reject needs a note saying why.";
    return;
  }
  if (decision === "edit" && !action) {
    msg.textContent = "An edit needs the corrected skill name.";
    return;
  }
  try {
    await api("/api/ground/" + encodeURIComponent(gid) + "/steps/" + idx, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({decision, action, note})});
    await loadGround();
    renderGrounding(await api("/api/ground/" + encodeURIComponent(gid)));
  } catch (e) { msg.textContent = "Save failed: " + e.message; }
}
async function loadSpecify(gid) {
  try {
    renderSpecify(gid, await api("/api/ground/" + encodeURIComponent(gid) + "/specify"));
  } catch (e) {
    const sec = $("#spec");
    if (sec) sec.replaceChildren(el("p", "Specify preview failed: " + e.message, "muted"));
  }
}
function renderSpecify(gid, sp) {
  const sec = $("#spec");
  if (!sec) return;
  sec.replaceChildren();
  sec.appendChild(el("h3", "Program (" + sp.decided + "/" + sp.total + " steps decided)"));
  if (sp.undecided.length) {
    sec.appendChild(el("div", sp.undecided.length + " steps undecided (" +
      sp.undecided.join(", ") + ") — preview covers decided steps only.", "warn"));
  }
  if (sp.blocked.length) {
    sec.appendChild(el("div", sp.blocked.length + " steps rejected (" +
      sp.blocked.join(", ") + ") — edit or re-ground to certify.", "err"));
  }
  const pt = el("table");
  const ph = el("tr");
  ["step", "skills", "args", "assumed", "result"].forEach(t => ph.appendChild(el("th", t)));
  pt.appendChild(ph);
  for (const s of sp.program.steps) {
    const tr = el("tr");
    tr.appendChild(el("td", String(s.idx), "num"));
    tr.appendChild(el("td", s.ops.map(o => o.skill).join(", ") || "—", "mono"));
    tr.appendChild(el("td", s.ops.map(o => JSON.stringify(o.args)).join("; ") || "—", "muted"));
    const assumed = s.ops.map(o => JSON.stringify(o.assumed))
      .filter(a => a && a !== "{}" && a !== "null").join("; ");
    tr.appendChild(el("td", assumed || "—", "muted"));
    const res = el("td");
    if (s.status === "planned") res.appendChild(pill("planned"));
    else res.appendChild(pill("out_of_scope"));
    if (s.reason) res.appendChild(el("span", " " + s.reason +
      (s.detail ? " — " + s.detail : ""), "muted"));
    tr.appendChild(res);
    pt.appendChild(tr);
  }
  sec.appendChild(pt);
  if (sp.program.entities.length) {
    sec.appendChild(el("p", "entities: " + sp.program.entities.map(
      e => e.uid + "=" + e.asset).join(", "), "muted"));
  }
  sec.appendChild(el("h3", "Judge"));
  const jt = el("table");
  const jh = el("tr");
  ["step", "check", "args", "target", "tol"].forEach(t => jh.appendChild(el("th", t)));
  jt.appendChild(jh);
  for (const s of sp.judge.steps) {
    for (const p of s.predicates) {
      const tr = el("tr");
      tr.appendChild(el("td", String(s.idx), "num"));
      tr.appendChild(el("td", p.pred, "mono"));
      tr.appendChild(el("td", (p.args || []).join(", ") || "—", "mono"));
      tr.appendChild(el("td", p.target === null ? "—" : String(p.target), "num"));
      tr.appendChild(el("td", p.tol === null ? "—" : String(p.tol), "num"));
      jt.appendChild(tr);
    }
  }
  sec.appendChild(jt);
  if (sp.judge.outcomes.length) {
    sec.appendChild(el("p", "outcomes: " + sp.judge.outcomes.map(
      o => o.type + " (" + o.quantity + ", ±" + o.tol_rel + ")").join("; "), "muted"));
  } else {
    sec.appendChild(el("p", "No outcome checks detected.", "muted"));
  }
  sec.appendChild(el("h3", "Certify"));
  sec.appendChild(el("p", "Builds the scene and runs the full certification " +
    "(execute, predicates, visual check, one repair). Needs MODEL_API_KEY.", "muted"));
  const form = el("form");
  const seed = document.createElement("input");
  seed.type = "number"; seed.min = "0"; seed.value = "0"; seed.title = "layout seed";
  const btn = el("button", "Build and certify"); btn.id = "certify";
  const msg = el("div"); msg.id = "msg";
  btn.onclick = e => { e.preventDefault(); submitCertify(gid, seed.value, btn, msg); };
  form.append(el("span", "seed ", "muted"), seed, btn);
  sec.append(form, msg);
}
async function submitCertify(gid, seed, btn, msg) {
  msg.textContent = "";
  btn.disabled = true;
  const old = btn.textContent;
  btn.textContent = "Certifying (simulator running)…";
  try {
    const r = await api("/api/ground/" + encodeURIComponent(gid) + "/certify", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({seed: Number(seed)})});
    const c = r.predicate_counts || {};
    msg.replaceChildren(el("div", "Certified " + r.run_id + ": " + r.status +
      ", predicates " + (c.passed || 0) + "/" + (c.failed || 0) + ", visual " +
      (r.visual.map(v => "a" + v.attempt + (v.passed ? " pass" : " FAIL")).join(", ") || "—") +
      (r.human_flag ? " — needs review." : " — clean."), r.human_flag ? "warn" : "ok"));
    const open = el("button", "Review " + r.run_id + " in Runs tab");
    open.onclick = e => { e.preventDefault(); openRun(r.run_id); };
    msg.appendChild(open);
  } catch (e) { msg.textContent = "Certify failed: " + e.message; }
  btn.disabled = false;
  btn.textContent = old;
}
loadRuns().catch(e => { $("#rows").textContent = "Failed to load runs: " + e.message; });
</script>
</body>
</html>"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Review certified runs and ground protocols in a browser.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    parser.add_argument("--sources-dir", default=None,
                        help="protocols.io cache dir (default: lab2rob/ingest/sources)")
    parser.add_argument("--tier1-path", default=None,
                        help="tier-1 BioProt file (default: lab2rob/ingest/sources/...)")
    args = parser.parse_args(argv)
    server = make_server(Path(args.runs_dir), args.port,
                         sources_dir=args.sources_dir, tier1_path=args.tier1_path)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"lab2rob review: {url}  (runs: {args.runs_dir})")
    print("Open that URL in your browser. Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
