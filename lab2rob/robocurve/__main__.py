"""python -m lab2rob.robocurve candidates | build | check

candidates  compile a protocol for a scene family and list its task candidates, accepted or why not
build       compile a protocol and build a task folder from one accepted candidate
check       what a task folder still needs before a pull request

Families: weigh (weigh out a salt, two arms), centrifuge (load a microcentrifuge balanced and
shut the lid). `weighings` and `build-weigh` are kept as shorthands for --family weigh.
"""

from __future__ import annotations

import argparse
import json
import sys

from lab2rob.robocurve.build import (BuildError, build_centrifuge_task, build_weigh_task,
                                     check_task)
from lab2rob.robocurve.compile import FAMILY_SKILL, compile_candidates, load_raw

BUILDERS = {"weigh": build_weigh_task, "centrifuge": build_centrifuge_task}
SHOWN = ("step", "target_g", "reagent", "full_g", "tubes", "speed", "time_s", "assumed")


def _summary(record: dict) -> dict:
    src = record["source"]
    return {**{k: src[k] for k in SHOWN if k in src}, "accepted": record["accepted"],
            "reason": record["reason"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab2rob.robocurve")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("candidates", "weighings"):
        p = sub.add_parser(name, help="a protocol's task candidates for one family")
        p.add_argument("protocol", help="protocol id (BioProBench, cached protocols.io, or BioProt)")
        if name == "candidates":
            p.add_argument("--family", required=True, choices=sorted(FAMILY_SKILL))
    for name in ("build", "build-weigh"):
        b = sub.add_parser(name, help="build a task folder from a compiled candidate")
        b.add_argument("protocol")
        if name == "build":
            b.add_argument("--family", required=True, choices=sorted(FAMILY_SKILL))
        b.add_argument("--out", required=True, help="the task folder to write (tasks/<name>/)")
        b.add_argument("--candidate", type=int, default=None,
                       help="which candidate (its position in `candidates`); default the first accepted")
        b.add_argument("--no-video", action="store_true")
    c = sub.add_parser("check", help="what a task folder still needs before a pull request")
    c.add_argument("task_dir")
    args = ap.parse_args(argv)

    if args.cmd == "check":
        problems = check_task(args.task_dir)
        print("\n".join(f"- {p}" for p in problems) or "ready")
        return 1 if problems else 0

    family = getattr(args, "family", None) or "weigh"
    records = compile_candidates(load_raw(args.protocol), family)
    if args.cmd in ("candidates", "weighings"):
        for i, record in enumerate(records):
            print(i, json.dumps(_summary(record)))
        return 0 if any(r["accepted"] for r in records) else 1

    if args.candidate is not None:
        if not 0 <= args.candidate < len(records):
            print(f"error: {args.protocol} has {len(records)} candidates for {family}")
            return 1
        record = records[args.candidate]
    else:
        record = next((r for r in records if r["accepted"]), None)
    if record is None or not record["accepted"]:
        why = record["reason"] if record else f"no {family} step compiles"
        print(f"not built: {why}")
        for i, r in enumerate(records):
            print(" ", i, json.dumps(_summary(r)))
        return 1
    try:
        report = BUILDERS[family](args.out, record["source"], video=not args.no_video)
    except BuildError as e:
        print(f"not built: {e}")
        return 1
    print(json.dumps({k: report[k] for k in ("out", "family", "run_limit_steps", "score_seconds",
                                             "demo_seconds")}, indent=1))
    print("demo:", json.dumps({k: report["demo"][k] for k in ("score", "reason", "steps")}))
    print("still needed:\n" + ("\n".join(f"- {p}" for p in report["problems"]) or "- nothing"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
