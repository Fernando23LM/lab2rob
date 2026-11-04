"""lab2rob.sim command line.

replay  one replay of a trajectory in this process; prints status JSON
grade   N replays, each in a fresh process; writes reward.txt + reward.json
freeze  ground (rules) + plan + judge + build + freeze a BioProt protocol,
        or freeze a reviewed grounding session from the runs directory
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def cmd_replay(args) -> int:
    from lab2rob.sim.session import Session, load_trajectory
    traj = load_trajectory(args.traj)
    session = Session(args.instance, level="joints")
    try:
        if traj.get("instance_init_sha") not in ("", session.instance["init_sha"]):
            status = {"success": False, "reason": "trajectory made for another instance"}
        else:
            status = session.replay(traj)
    finally:
        session.close()
    print(json.dumps(status, default=str))
    return 0


def cmd_grade(args) -> int:
    """Verdict: every fresh replay succeeds and all end in one state."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"success": 0, "deterministic": 0, "replays": args.repeat,
              "sim_error": 0, "grader_error": 0, "n_actions": 0}
    statuses = []
    try:
        for _ in range(args.repeat):
            proc = subprocess.run([sys.executable, "-m", "lab2rob.sim", "replay",
                                   "--instance", args.instance, "--traj", args.traj],
                                  capture_output=True, text=True, timeout=args.timeout_s)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip()[-500:])
            statuses.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        shas = {s.get("state_sha") for s in statuses}
        result["deterministic"] = int(len(shas) == 1 and None not in shas)
        result["sim_error"] = int(any(s.get("sim_error") for s in statuses))
        result["n_actions"] = int(statuses[0].get("steps", 0)) if statuses else 0
        result["success"] = int(all(s.get("success") for s in statuses)
                                and result["deterministic"] == 1)
    except Exception as e:  # noqa: BLE001 - a verdict is always written
        result["grader_error"] = 1
        (out / "grade.log").write_text(str(e), encoding="utf-8")
    (out / "reward.txt").write_text(f"{result['success']}\n", encoding="utf-8")
    (out / "reward.json").write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps(result))
    return 0


def cmd_freeze(args) -> int:
    from lab2rob.build.scene import build_scene
    from lab2rob.sim.freeze import FreezeError, freeze_task
    from lab2rob.specify.judge import plan_judge
    from lab2rob.specify.program import plan_protocol

    if args.bioprot:
        from lab2rob.ground.rules import ground_protocol
        from lab2rob.ingest.bioprot import load_bioprot_id
        raw = load_bioprot_id(args.bioprot)
        grounding = ground_protocol(raw)
        source = {"dataset": "bioprot", "protocol_id": args.bioprot,
                  "url": raw.source_url, "license": raw.license}
    else:
        from lab2rob.review.app import (_planned_grounding, get_grounding,
                                        session_grounding)
        session = get_grounding(Path(args.runs_dir), args.grounding_id)
        if session is None:
            print(f"error: unknown grounding {args.grounding_id!r}")
            return 1
        groundings, undecided, blocked = session_grounding(session)
        if undecided or blocked:
            print(f"error: undecided {undecided}, rejected {blocked}")
            return 1
        grounding = _planned_grounding(session, groundings)
        source = {"dataset": session.get("dataset", ""),
                  "protocol_id": session.get("protocol_id", ""),
                  "grounding_id": args.grounding_id}
    program, judge = plan_protocol(grounding), plan_judge(grounding)
    built = build_scene(program, seed=args.seed)
    try:
        report = freeze_task(program, judge, built, args.seed, args.out, source=source)
    except FreezeError as e:
        print(f"not frozen: {e.reason}: {e.detail}")
        return 1
    print(f"frozen {args.out}: {report['n_rows']} rows, {report['n_events']} events, "
          f"replays {[r['success'] for r in report['replays']]}, "
          f"nop {report['nop_success']}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lab2rob.sim")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("replay")
    p.add_argument("--instance", required=True)
    p.add_argument("--traj", required=True)
    p.set_defaults(func=cmd_replay)
    p = sub.add_parser("grade")
    p.add_argument("--instance", required=True)
    p.add_argument("--traj", required=True)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--out", default="/logs/verifier")
    p.add_argument("--timeout-s", type=int, default=2400)
    p.set_defaults(func=cmd_grade)
    p = sub.add_parser("freeze")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--bioprot", help="BioProt protocol id (rules grounding)")
    src.add_argument("--grounding-id", help="reviewed session in --runs-dir")
    p.add_argument("--runs-dir", default=str(Path(__file__).resolve().parents[2] / "runs"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_freeze)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
