"""python -m lab2rob.robocurve weighings | build-weigh | check

weighings    compile a protocol and list its weighing steps, accepted or why not
build-weigh  compile a protocol and build a task folder from one accepted weighing
check        what a task folder still needs before a pull request
"""

from __future__ import annotations

import argparse
import json
import sys

from lab2rob.robocurve.build import BuildError, build_weigh_task, check_task
from lab2rob.robocurve.compile import compile_weighings, load_raw


def _summary(record: dict) -> dict:
    src = record["source"]
    return {"step": src["step"], "target_g": src["target_g"], "reagent": src["reagent"],
            "full_g": src["full_g"], "accepted": record["accepted"], "reason": record["reason"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab2rob.robocurve")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("weighings", help="a protocol's weighing steps, compiled")
    w.add_argument("protocol", help="protocol id (BioProBench, cached protocols.io, or BioProt)")
    b = sub.add_parser("build-weigh", help="build a task folder from a compiled weighing")
    b.add_argument("protocol")
    b.add_argument("--out", required=True, help="the task folder to write (tasks/<name>/)")
    b.add_argument("--weighing", type=int, default=None,
                   help="which accepted weighing (its position in `weighings`); default the first")
    b.add_argument("--no-video", action="store_true")
    c = sub.add_parser("check", help="what a task folder still needs before a pull request")
    c.add_argument("task_dir")
    args = ap.parse_args(argv)

    if args.cmd == "check":
        problems = check_task(args.task_dir)
        print("\n".join(f"- {p}" for p in problems) or "ready")
        return 1 if problems else 0

    records = compile_weighings(load_raw(args.protocol))
    if args.cmd == "weighings":
        for i, record in enumerate(records):
            print(i, json.dumps(_summary(record)))
        return 0 if any(r["accepted"] for r in records) else 1

    if args.weighing is not None:
        if not 0 <= args.weighing < len(records):
            print(f"error: {args.protocol} has {len(records)} weighings")
            return 1
        record = records[args.weighing]
    else:
        record = next((r for r in records if r["accepted"]), None)
    if record is None or not record["accepted"]:
        why = record["reason"] if record else "no weighing step compiles"
        print(f"not built: {why}")
        for i, r in enumerate(records):
            print(" ", i, json.dumps(_summary(r)))
        return 1
    try:
        report = build_weigh_task(args.out, record["source"], video=not args.no_video)
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
