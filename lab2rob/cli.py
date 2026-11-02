"""XR-02: thin lab2rob CLI.

compile  ground a protocol into a reviewable session
certify  build + certify a decided session into a run
eval     run a sandboxed agent on a certified run
report   coverage funnel + audit + metrics from a runs directory
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from lab2rob.build.scene import SceneRejected, build_scene
from lab2rob.certify.run import RUNS_DIR
from lab2rob.eval.harness import EVAL_TIMEOUT_S, MAX_OPS, evaluate
from lab2rob.ground.rules import ProtocolGrounding
from lab2rob.report.report import (build_audit, build_funnel, build_metrics,
                                   collect_cert_runs, collect_eval_runs,
                                   collect_sessions, write_report)
from lab2rob.review.app import (LLM_ERRORS, get_grounding, run_certification,
                                save_step_decision, session_grounding,
                                start_grounding)
from lab2rob.specify.judge import plan_judge
from lab2rob.specify.program import plan_protocol

AGENT_KINDS = ("scripted", "policy", "zero_shot")
TWIN_LABELS = ("refuse", "ask", "flag")


def _fail(message: str) -> int:
    print(f"error: {message}")
    return 1


def cmd_compile(args) -> int:
    runs_dir = Path(args.runs_dir)
    try:
        session = start_grounding(runs_dir, args.dataset, args.protocol_id, args.mode,
                                  sources_dir=args.sources_dir,
                                  tier1_path=args.tier1_path)
    except KeyError as e:
        return _fail(str(e))
    except ValueError as e:
        return _fail(str(e))
    except LLM_ERRORS as e:
        return _fail(str(e))
    gid = session["grounding_id"]
    accepted = 0
    if args.auto_accept:
        for step in session["steps"]:
            if step.get("decision") is None and step["verdict"].get("status") == "grounded":
                save_step_decision(runs_dir, gid, step["idx"], "accept", None,
                                   "auto-accepted (verdict grounded)", by="cli")
                accepted += 1
        session = get_grounding(runs_dir, gid)
    decided = sum(1 for s in session["steps"] if s.get("decision"))
    print(f"grounded {gid}: {len(session['steps'])} steps, counts {session['counts']}")
    if args.auto_accept:
        print(f"auto-accepted {accepted} grounded steps "
              f"({decided}/{len(session['steps'])} decided)")
    else:
        print(f"{decided}/{len(session['steps'])} decided "
              f"(review in the app, or re-run with --auto-accept)")
    return 0


def cmd_certify(args) -> int:
    try:
        result = run_certification(Path(args.runs_dir), args.grounding_id,
                                   seed=args.seed)
    except KeyError as e:
        return _fail(str(e))
    except ValueError as e:
        return _fail(str(e))
    except SceneRejected as e:
        return _fail(f"scene rejected: {e}")
    except LLM_ERRORS as e:
        return _fail(str(e))
    counts = result["predicate_counts"]
    visual = ",".join(f"a{v['attempt']}{'Y' if v['passed'] else 'N'}"
                      for v in result["visual"]) or "-"
    print(f"certified {result['run_id']}: {result['status']}, "
          f"predicates {counts.get('passed', 0)}/{counts.get('failed', 0)}, "
          f"visual {visual}, human_flag {result['human_flag']}")
    return 0


def _session_for_run(runs_dir: Path, run_id: str):
    for session in collect_sessions(runs_dir):
        if any(r.get("run_id") == run_id for r in session.get("runs", [])):
            return session
    return None


def cmd_eval(args) -> int:
    runs_dir = Path(args.runs_dir)
    try:
        agent_spec = json.loads(Path(args.agent).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return _fail(f"cannot read agent spec {args.agent}: {e}")
    if not isinstance(agent_spec, dict) or agent_spec.get("kind") not in AGENT_KINDS:
        return _fail(f"agent spec needs kind in {AGENT_KINDS}")
    if args.twin_label is not None and args.twin_label not in TWIN_LABELS:
        return _fail(f"twin label must be in {TWIN_LABELS}")
    session = _session_for_run(runs_dir, args.run_id)
    if session is None:
        return _fail(f"no grounding session links run {args.run_id} "
                      f"(certify from a session first)")
    groundings, undecided, blocked = session_grounding(session)
    if undecided or blocked:
        return _fail(f"session {session['grounding_id']} not certifiable: "
                     f"undecided {undecided}, rejected {blocked}")
    run_path = runs_dir / args.run_id / "run.json"
    try:
        run_seed = json.loads(run_path.read_text(encoding="utf-8")).get("seed", 0)
    except (OSError, ValueError):
        return _fail(f"cannot read {run_path}")
    seed = args.seed if args.seed is not None else run_seed
    planned = ProtocolGrounding(protocol_id=session.get("protocol_id", ""),
                                dataset=session.get("dataset", ""),
                                title=session.get("title", ""),
                                description=session.get("description", ""),
                                steps=groundings)
    program, judge = plan_protocol(planned), plan_judge(planned)
    try:
        built = build_scene(program, seed=seed)
    except SceneRejected as e:
        return _fail(f"scene rejected: {e}")
    try:
        report = evaluate(program, judge, built, agent_spec, seed=seed,
                          runs_dir=runs_dir, max_ops=args.max_ops,
                          timeout_s=args.timeout_s, instruction=args.instruction,
                          eval_id=args.eval_id)
    except Exception as e:  # noqa: BLE001 - the CLI reports, never tracebacks
        return _fail(f"eval crashed: {e}")
    if args.twin_label is not None:
        (report.dir / "twin.json").write_text(
            json.dumps({"eval_id": report.eval_id,
                        "expected": args.twin_label}, indent=2), encoding="utf-8")
    print(f"evaluated {report.eval_id}: {report.status} {report.reason or ''} "
          f"sr={report.sr} ssr={report.ssr} ops={report.n_ops}")
    return 0


def cmd_report(args) -> int:
    runs_dir = Path(args.runs_dir)
    report_id = args.report_id or datetime.now(timezone.utc).strftime(
        "report_%Y%m%d_%H%M%S")
    out_dir = runs_dir / report_id / "report"
    sessions = collect_sessions(runs_dir)
    cert_runs = collect_cert_runs(runs_dir)
    eval_records, twins = collect_eval_runs(runs_dir)
    funnel = build_funnel(sessions, cert_runs)
    audit = build_audit(cert_runs)
    metrics = build_metrics(runs_dir, sessions, eval_records, twins)
    written = write_report(out_dir, funnel, audit, metrics)
    summary = audit["summary"]
    score = metrics["scoreboard"]
    print(f"report {report_id}: {funnel['protocols']} protocols, "
          f"{funnel['certified']} certified, {funnel['accepted_runs']} accepted; "
          f"first-pass {summary['first_pass']}, recovered {summary['auto_recovered']}, "
          f"human-fixed {summary['human_fixed']}; "
          f"SR {score.get('sr_mean')} over {metrics['eval_runs']} eval runs")
    print(f"wrote {written['dir']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lab2rob",
                                     description="Compile, certify, eval, report.")
    sub = parser.add_subparsers(dest="command", required=True)

    compile_parser = sub.add_parser("compile", help="ground a protocol")
    compile_parser.add_argument("dataset", choices=("protocolsio", "bioprot"))
    compile_parser.add_argument("protocol_id")
    compile_parser.add_argument("--mode", default="rules",
                                choices=("rules", "full_llm"))
    compile_parser.add_argument("--auto-accept", action="store_true",
                                help="accept steps with a grounded verdict")
    compile_parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    compile_parser.add_argument("--sources-dir", default=None)
    compile_parser.add_argument("--tier1-path", default=None)
    compile_parser.set_defaults(func=cmd_compile)

    certify_parser = sub.add_parser("certify", help="certify a decided session")
    certify_parser.add_argument("grounding_id")
    certify_parser.add_argument("--seed", type=int, default=0)
    certify_parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    certify_parser.set_defaults(func=cmd_certify)

    eval_parser = sub.add_parser("eval", help="evaluate an agent on a certified run")
    eval_parser.add_argument("run_id")
    eval_parser.add_argument("--agent", required=True,
                             help="JSON agent spec (kind + actions/dotted/replies)")
    eval_parser.add_argument("--seed", type=int, default=None,
                             help="layout seed (default: the cert run's seed)")
    eval_parser.add_argument("--eval-id", default=None)
    eval_parser.add_argument("--instruction", default=None)
    eval_parser.add_argument("--twin-label", default=None,
                             choices=TWIN_LABELS)
    eval_parser.add_argument("--max-ops", type=int, default=MAX_OPS)
    eval_parser.add_argument("--timeout-s", type=int, default=EVAL_TIMEOUT_S)
    eval_parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    eval_parser.set_defaults(func=cmd_eval)

    report_parser = sub.add_parser("report", help="write the submission report")
    report_parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    report_parser.add_argument("--report-id", default=None)
    report_parser.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
