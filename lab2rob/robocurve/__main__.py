"""python -m lab2rob.robocurve build-weigh | check"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lab2rob.robocurve.build import (REPO, BuildError, build_weigh_task, check_task,
                                     find_weigh_step)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab2rob.robocurve")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-weigh", help="a weighing task from a protocol's weigh step")
    b.add_argument("--out", required=True, help="the task folder to write (tasks/<name>/)")
    b.add_argument("--protocol", help="protocol id in the corpus sweep (its 'weigh X g' step)")
    b.add_argument("--sweep", default=str(REPO / "runs" / "corpus_sweep_bioprobench"))
    b.add_argument("--target-g", type=float, help="target mass, instead of --protocol")
    b.add_argument("--no-video", action="store_true")
    c = sub.add_parser("check", help="what a task folder still needs before a pull request")
    c.add_argument("task_dir")
    args = ap.parse_args(argv)
    if args.cmd == "check":
        problems = check_task(args.task_dir)
        print("\n".join(f"- {p}" for p in problems) or "ready")
        return 1 if problems else 0
    if args.protocol:
        sweep = Path(args.sweep)
        source = find_weigh_step(args.protocol, sweep / "steps.jsonl", sweep / "protocols.jsonl")
    elif args.target_g is not None:
        source = {"target_g": args.target_g, "reagent": "salt"}
    else:
        ap.error("give --protocol or --target-g")
    try:
        report = build_weigh_task(args.out, source, video=not args.no_video)
    except BuildError as e:
        print(f"not built: {e}")
        return 1
    print(json.dumps({k: report[k] for k in ("out", "run_limit_steps", "score_seconds",
                                             "demo_seconds")}, indent=1))
    print("demo:", json.dumps({k: report["demo"][k] for k in ("score", "reason", "reading_g",
                                                               "steps")}))
    print("still needed:\n" + ("\n".join(f"- {p}" for p in report["problems"]) or "- nothing"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
