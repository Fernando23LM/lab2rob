"""One automatic repair, then a human flag.

diagnose() maps a finished run to at most one execution adjustment
(higher approach + slower descent after a blocked move; deeper tip after
an aspirate shortfall) or None when nothing actionable remains.
certify_with_repair() runs the program, spends at most one repair — an
adjusted re-run for execution failures, else a single visual re-query —
saves repair.json with the structured diagnostics, and raises human_flag
when anything still fails.
"""

from __future__ import annotations

import json
from pathlib import Path

from .run import RunReport, certify
from .visual import check_visual

APPROACH_BUMP_M = 0.02
TIP_DEEPEN_MM = 1.0


def diagnose(report: RunReport, trace_rows: list, exec_params: dict):
    """A repair plan for this run, or None when repair cannot help."""
    if report.status == "aborted":
        if report.abort_reason != "move_blocked":
            return None
        return {"kind": "execution",
                "exec": {"approach_m": exec_params["approach_m"] + APPROACH_BUMP_M,
                         "well_speed": exec_params["well_speed"] / 2.0,
                         "tip_extra_mm": exec_params["tip_extra_mm"]},
                "why": f"move blocked ({report.abort_detail}); retry higher and slower"}
    failed_volume = [p for p in report.predicates
                     if p["pred"] == "volume" and p["status"] == "failed"]
    if not failed_volume:
        return None
    for row in trace_rows:
        if row.get("kind") != "op" or row.get("skill") != "aspirate":
            continue
        result = row.get("result", {}) if isinstance(row.get("result"), dict) else {}
        want = row.get("args", {}).get("volume_ul") if isinstance(row.get("args"), dict) else None
        got = result.get("volume_ul")
        if (isinstance(want, (int, float)) and isinstance(got, (int, float))
                and got < want - 1e-6):
            return {"kind": "execution",
                    "exec": {"approach_m": exec_params["approach_m"],
                             "well_speed": exec_params["well_speed"],
                             "tip_extra_mm": exec_params["tip_extra_mm"] + TIP_DEEPEN_MM},
                    "why": (f"step {row.get('step')} drew {got:g} of {want:g} uL; "
                            f"retry with the tip {TIP_DEEPEN_MM:g} mm deeper")}
    return None


def _read_trace(run_dir: Path) -> list:
    path = run_dir / "trace.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _read_exec(run_dir: Path) -> dict:
    path = run_dir / "run.json"
    if not path.exists():
        return {"approach_m": 0.03, "well_speed": 0.04, "tip_extra_mm": 0.0}
    return json.loads(path.read_text(encoding="utf-8")).get("exec", {})


def certify_with_repair(program, judge, built, seed: int = 0, runs_dir=None,
                        visual_backend=None) -> dict:
    """Full certification: run, at most one repair, visual check, human flag."""
    from .run import RUNS_DIR
    runs_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR

    initial = certify(program, judge, built, seed=seed, runs_dir=runs_dir,
                      return_env=visual_backend is not None)
    plan = diagnose(initial, _read_trace(initial.dir), _read_exec(initial.dir))
    repair = None
    spent = False
    if plan is not None:
        repair = certify(program, judge, built, seed=seed, runs_dir=runs_dir,
                         run_suffix="_repair", return_env=visual_backend is not None,
                         **plan["exec"])
        spent = True
    final = repair if repair is not None else initial

    visual_rows = []
    if visual_backend is not None and final.status == "complete" and final.env is not None:
        for attempt in (1, 2):
            verdict = check_visual(final.env, program, visual_backend)
            verdict["attempt"] = attempt
            visual_rows.append({k: v for k, v in verdict.items() if k != "images"})
            for camera, png in verdict["images"].items():
                (final.dir / f"visual_{camera}_a{attempt}.png").write_bytes(png)
            if verdict["passed"] or spent:
                break
            spent = True  # the single repair goes to a visual re-query

    failed_preds = [p for p in final.predicates if p["status"] == "failed"]
    visual_failed = bool(visual_rows) and not visual_rows[-1]["passed"]
    human_flag = final.status != "complete" or bool(failed_preds) or visual_failed
    diagnostics = {
        "initial_run": initial.run_id, "repair_run": repair.run_id if repair else None,
        "diagnosis": plan["why"] if plan else None,
        "repair_exec": plan["exec"] if plan else None,
        "visual": visual_rows,
        "failed_predicates": failed_preds,
        "human_flag": human_flag,
    }
    (initial.dir / "repair.json").write_text(json.dumps(diagnostics, indent=2),
                                             encoding="utf-8")
    return {"initial": initial, "repair": repair, "visual": visual_rows,
            "diagnosis": plan, "human_flag": human_flag, "diagnostics": diagnostics}
