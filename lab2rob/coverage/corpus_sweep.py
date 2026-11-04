"""Corpus sweep: rules-only compile of a large protocol corpus, offline.

For every protocol: ground every step with the rules (no LLM), plan the
program, write the independent judge, and record per step whether it
became executable ops ("planned") and, if not, what blocks it. Then:

- funnel: protocols -> steps -> grounded by rules -> planned;
- blockers: the capability each non-planned step needs (sweep.CAPABILITIES
  over step text + detail), ranked by protocols and steps;
- unlock ranking: greedily, which capability, if added, makes the most
  protocols fully executable, then the next given the first, ...;
- candidates: protocols with judged liquid steps, ranked, and (optional)
  freeze_task on the top K, so the report counts real validated tasks;
- llm pass sizing: how many needs_llm steps stand between near-complete
  protocols and full coverage (what an LLM grounding pass would cost).

Outputs in --out (runs/ is gitignored): protocols.jsonl (per protocol, no
step text), steps.jsonl (step text + verdicts, for local mining only: the
corpus license is non-commercial), summary.json, report.md.

  python -m lab2rob.coverage.corpus_sweep --corpus bioprobench \\
      --out runs/corpus_sweep --workers 4 [--limit N] [--freeze-top 20]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from lab2rob.ground.rules import ground_protocol
from lab2rob.llm.cost_log import DailyCapExceeded, JobBudgetExceeded
from lab2rob.ground.sweep import _capability_key
from lab2rob.specify.judge import LIQUID_ACTIONS, plan_judge
from lab2rob.specify.program import plan_protocol

#: Placeholder-rate estimate per fallback grounding call, from the
#: 42-protocol sweep ($0.94 over 84 calls); see llm/cost_log.py prices.
EST_USD_PER_LLM_CALL = 0.94 / 84
NEAR_COMPLETE_MAX_LLM_STEPS = 3
UNLOCK_ROUNDS = 15
#: A task segment: this many consecutive executable steps, one of them liquid.
SEGMENT_MIN_STEPS = 3


def compile_protocol(protocol, client=None) -> dict:
    """Ground + plan + judge one protocol; a compact, JSON-safe record.

    Rules only by default; with client, steps the rules leave as needs_llm
    go to the LLM fallback (ground_protocol's own routing).
    """
    raw = protocol.raw
    try:
        grounding = ground_protocol(raw, client=client)
        program = plan_protocol(grounding)
        judge = plan_judge(grounding)
    except (DailyCapExceeded, JobBudgetExceeded):
        raise  # a spending cap stops the LLM pass, never silently
    except Exception as e:  # noqa: BLE001 - one bad protocol never stops a sweep
        return {"protocol_id": raw.protocol_id, "error": f"{type(e).__name__}: {e}"[:300]}
    planned = {s.idx: s for s in program.steps}
    steps, run, best_run = [], 0, 0
    for g in grounding.steps:
        p = planned.get(g.idx)
        ok = p is not None and p.status == "planned" and bool(p.ops)
        run = run + 1 if ok else 0
        best_run = max(best_run, run)
        blocker = None
        if not ok and g.status == "grounded" and p is not None and p.reason == "ambiguous":
            blocker = f"{g.action}:unbound"  # the skill exists; its arguments do not
        elif not ok:
            detail = (p.detail if p is not None else "") or g.detail or ""
            blocker = _capability_key(f"{g.source_text} {detail}")
            if blocker == "unclustered" and g.action:
                blocker = g.action
        steps.append({
            "idx": g.idx, "key": protocol.step_keys[g.idx - 1]
            if 0 < g.idx <= len(protocol.step_keys) else None,
            "text": g.source_text, "status": g.status, "reason": g.reason,
            "action": g.action, "planned": ok,
            "plan_reason": None if ok or p is None else p.reason,
            "liquid": ok and g.action in LIQUID_ACTIONS,
            "blocker": blocker,
            "ops": [op["skill"] for op in p.ops] if ok else [],
        })
    judged_volume = sum(1 for js in judge.steps for pr in js.predicates if pr.pred == "volume")
    scene = None
    if any(s["planned"] for s in steps):
        from lab2rob.build.scene import SceneRejected, build_scene
        try:
            built = build_scene(program, seed=0)
            scene = {"ok": True, "family": built.family}
        except SceneRejected as e:
            scene = {"ok": False, "reason": e.reason, "detail": e.detail[:120]}
        except Exception as e:  # noqa: BLE001
            scene = {"ok": False, "reason": type(e).__name__, "detail": str(e)[:120]}
    weighings = _weighings(grounding)
    return {
        "weighings": weighings,
        "protocol_id": raw.protocol_id, "dataset": raw.dataset, "title": raw.title,
        "url": raw.source_url, "license": raw.license, "n_steps": len(steps),
        "n_rules_grounded": sum(s["status"] == "grounded" for s in steps),
        "n_needs_llm": sum(s["status"] == "needs_llm" for s in steps),
        "n_planned": sum(s["planned"] for s in steps),
        "n_liquid": sum(s["liquid"] for s in steps),
        "longest_run": best_run, "judged_volume_checks": judged_volume,
        "segments": len(_segments([(s["planned"], s["liquid"]) for s in steps])),
        "scene": scene,
        "outcomes": [o.type for o in judge.outcomes],
        "blockers": sorted({s["blocker"] for s in steps if s["blocker"]}),
        "steps": steps,
    }


def _weighings(grounding) -> list:
    """Each weighing op, planned for the weighing scene: accepted or why not."""
    from lab2rob.build.scene import SceneRejected, build_scene, weigh_programs
    out = []
    for single in weigh_programs(plan_protocol(grounding, family="weigh")):
        step = next(s for s in single.steps if s.status == "planned")
        args = step.ops[0]["args"]
        row = {"step": step.idx, "mass_g": args.get("mass_g"), "reagent": args.get("reagent"),
               "accepted": False, "reason": ""}
        try:
            build_scene(single, seed=0)
            row["accepted"] = True
        except SceneRejected as e:
            row["reason"] = e.reason
        except Exception as e:  # noqa: BLE001
            row["reason"] = type(e).__name__
        out.append(row)
    return out


def _segments(flags: list, min_steps: int = SEGMENT_MIN_STEPS) -> list:
    """Maximal runs of executable steps, >= min_steps long, with a liquid step.

    flags: (executable, liquid) per step, in order. Returns (start, end) pairs.
    """
    out, start = [], None
    for i, (ok, _) in enumerate(list(flags) + [(False, False)]):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= min_steps and any(liq for _, liq in flags[start:i]):
                out.append((start, i))
            start = None
    return out


def _greedy_unlock(records: list, rounds: int = UNLOCK_ROUNDS) -> list:
    """Capabilities in the order that gives the most protocols a task segment.

    Adding a capability makes its blocked steps executable (counted as liquid
    when the capability is a liquid action). Each round picks the capability
    with the largest gain in protocols holding at least one segment.
    """
    rows = [[(s["planned"], s["liquid"], s["blocker"]) for s in r["steps"]]
            for r in records if r.get("steps")]
    by_cap = defaultdict(list)
    for i, row in enumerate(rows):
        for cap in {b for _, _, b in row if b and b != "unclustered"}:
            by_cap[cap].append(i)

    def has_segment(row, added):
        flags = [(ok or b in added,
                  liq or (b in added and b.split(":")[0] in LIQUID_ACTIONS))
                 for ok, liq, b in row]
        return bool(_segments(flags))

    added: set = set()
    have = {i for i, row in enumerate(rows) if has_segment(row, added)}
    out = []
    for _ in range(rounds):
        best, best_gain = None, set()
        for cap, idxs in by_cap.items():
            if cap in added:
                continue
            gain = {i for i in idxs if i not in have and has_segment(rows[i], added | {cap})}
            if len(gain) > len(best_gain):
                best, best_gain = cap, gain
        if best is None:
            break
        added.add(best)
        have |= best_gain
        out.append({"add": best, "unlocks": len(best_gain), "protocols_with_segment": len(have)})
    return out


def _near_complete(records: list) -> list:
    """Protocols with an executable liquid step whose only non-executable
    steps are a few needs_llm ones (the LLM pass's targets)."""
    return [r for r in records if "error" not in r and r["n_needs_llm"] and r["n_liquid"]
            and r["n_needs_llm"] <= NEAR_COMPLETE_MAX_LLM_STEPS
            and all(s["planned"] or s["status"] == "needs_llm" for s in r["steps"])]


def llm_pass(records: list, protocols_by_id: dict, client, budget_usd: float) -> dict:
    """Re-ground near-complete protocols with the LLM fallback, cheapest first.

    Replaces their records in place and returns what was spent and gained.
    Stops before the budget (measured on the client's cost log when it has
    one, else estimated per call) and on the client's own spending caps.
    """
    near = sorted(_near_complete(records), key=lambda r: (r["n_needs_llm"], r["protocol_id"]))
    index = {r["protocol_id"]: i for i, r in enumerate(records)}
    log = getattr(client, "cost_log", None)
    start = log.spend_total() if log is not None else 0.0
    calls, rows, stopped = 0, [], ""
    for r in near:
        spent = (log.spend_total() - start) if log is not None else calls * EST_USD_PER_LLM_CALL
        if spent + r["n_needs_llm"] * EST_USD_PER_LLM_CALL > budget_usd:
            stopped = "budget"
            break
        try:
            new = compile_protocol(protocols_by_id[r["protocol_id"]], client=client)
        except (DailyCapExceeded, JobBudgetExceeded) as e:
            stopped = type(e).__name__
            break
        calls += r["n_needs_llm"]
        if "error" not in new:
            records[index[r["protocol_id"]]] = new
        rows.append({"protocol_id": r["protocol_id"], "llm_steps": r["n_needs_llm"],
                     "planned_before": r["n_planned"], "planned_after": new.get("n_planned"),
                     "segments_before": r["segments"], "segments_after": new.get("segments"),
                     "error": new.get("error")})
    spent = (log.spend_total() - start) if log is not None else calls * EST_USD_PER_LLM_CALL
    return {"protocols": len(rows), "of_near_complete": len(near), "llm_steps": calls,
            "spent_usd": round(spent, 4), "spend_measured": log is not None,
            "stopped": stopped or "done",
            "planned_gain": sum((x["planned_after"] or 0) - x["planned_before"] for x in rows),
            "segment_gain": sum((x["segments_after"] or 0) - x["segments_before"] for x in rows),
            "rows": rows}


def summarize(records: list, corpus_stats: dict) -> dict:
    ok = [r for r in records if "error" not in r]
    steps = [s for r in ok for s in r["steps"]]
    blocker_steps, blocker_protocols = Counter(), Counter()
    for r in ok:
        for s in r["steps"]:
            if s["blocker"]:
                blocker_steps[s["blocker"]] += 1
        for b in r["blockers"]:
            blocker_protocols[b] += 1
    candidates = sorted((r for r in ok if r["n_liquid"] and r["judged_volume_checks"]
                         and (r.get("scene") or {}).get("ok")),
                        key=lambda r: (-r["n_liquid"], -r["longest_run"], r["protocol_id"]))
    near = _near_complete(ok)
    near_calls = sum(r["n_needs_llm"] for r in near)
    all_llm = sum(r["n_needs_llm"] for r in ok)
    runs = Counter(min(r["longest_run"], 10) for r in ok)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": corpus_stats,
        "protocols": len(ok), "errors": len(records) - len(ok),
        "steps": len(steps),
        "status": dict(Counter(s["status"] for s in steps)),
        "reasons": dict(Counter(s["reason"] or "-" for s in steps).most_common()),
        "grounded_actions": dict(Counter(s["action"] for s in steps
                                         if s["status"] == "grounded").most_common(25)),
        "planned_steps": sum(s["planned"] for s in steps),
        "liquid_steps": sum(s["liquid"] for s in steps),
        "fully_planned_protocols": sum(1 for r in ok if r["n_steps"] and not r["blockers"]),
        "protocols_with_segment": sum(1 for r in ok if r["segments"]),
        "weighing": {
            "ops": sum(len(r.get("weighings", [])) for r in ok),
            "protocols": sum(1 for r in ok if r.get("weighings")),
            "accepted_ops": sum(w["accepted"] for r in ok for w in r.get("weighings", [])),
            "protocols_accepted": sum(1 for r in ok if any(w["accepted"] for w in r.get("weighings", []))),
            "rejected": dict(Counter(w["reason"] for r in ok for w in r.get("weighings", [])
                                     if not w["accepted"]).most_common()),
            "accepted": [{"protocol_id": r["protocol_id"], "title": r["title"], "url": r["url"], **w}
                         for r in ok for w in r.get("weighings", []) if w["accepted"]][:200],
        },
        "segments": sum(r["segments"] for r in ok),
        "scene": dict(Counter("ok" if r["scene"]["ok"] else
                              f"{r['scene']['reason']}: {r['scene']['detail'][:60]}"
                              for r in ok if r.get("scene")).most_common(12)),
        "blocked_unbound": dict(Counter(s["blocker"] for s in steps if s["blocker"]
                                        and s["blocker"].endswith(":unbound")).most_common(12)),
        "protocols_with_planned_step": sum(1 for r in ok if r["n_planned"]),
        "longest_run_hist": {("10+" if k == 10 else str(k)): runs[k] for k in sorted(runs)},
        "blockers_by_protocols": dict(blocker_protocols.most_common(40)),
        "blockers_by_steps": dict(blocker_steps.most_common(40)),
        "unlock": _greedy_unlock(ok),
        "candidates": [{k: r[k] for k in ("protocol_id", "dataset", "title", "url",
                                          "n_steps", "n_planned", "n_liquid", "longest_run",
                                          "judged_volume_checks")}
                       for r in candidates[:200]],
        "n_candidates": len(candidates),
        "llm_pass": {"needs_llm_steps_total": all_llm,
                     "near_complete_protocols": len(near),
                     "near_complete_calls": near_calls,
                     "est_usd_near_complete": round(near_calls * EST_USD_PER_LLM_CALL, 2),
                     "est_usd_all": round(all_llm * EST_USD_PER_LLM_CALL, 2),
                     "note": "estimates use placeholder prices in llm/cost_log.py"},
    }


def freeze_candidates(protocols_by_id: dict, summary: dict, out_dir: Path, top: int) -> list:
    """freeze_task on the top candidates; returns one result row each."""
    from lab2rob.build.scene import SceneRejected, build_scene
    from lab2rob.sim.freeze import FreezeError, freeze_task

    results = []
    for cand in summary["candidates"][:top]:
        protocol = protocols_by_id[cand["protocol_id"]]
        grounding = ground_protocol(protocol.raw)
        program, judge = plan_protocol(grounding), plan_judge(grounding)
        task_dir = out_dir / "frozen" / cand["protocol_id"].replace("/", "_")
        row = {"protocol_id": cand["protocol_id"], "n_liquid": cand["n_liquid"]}
        t0 = time.time()
        try:
            built = build_scene(program, seed=0)
            report = freeze_task(program, judge, built, 0, task_dir, source={
                "dataset": protocol.raw.dataset, "protocol_id": protocol.raw.protocol_id,
                "url": protocol.raw.source_url, "license": protocol.raw.license,
                "noncommercial": protocol.noncommercial})
            row.update(frozen=True, n_rows=report["n_rows"],
                       checks=len(report["oracle"]["checks"]))
        except SceneRejected as e:
            row.update(frozen=False, reason=f"scene_rejected:{e.reason}", detail=e.detail)
        except FreezeError as e:
            row.update(frozen=False, reason=e.reason, detail=str(e.detail)[:200])
        except Exception as e:  # noqa: BLE001 - one bad candidate never stops the run
            row.update(frozen=False, reason=type(e).__name__, detail=str(e)[:200])
        row["seconds"] = round(time.time() - t0, 1)
        results.append(row)
        print(f"freeze {cand['protocol_id']}: {row.get('frozen')} {row.get('reason', '')}",
              flush=True)
    return results


def write_report(out_dir: Path, summary: dict) -> Path:
    s = summary
    pct = (lambda a, b: f"{100 * a / b:.1f}%" if b else "-")
    lines = [
        "# Corpus sweep (rules only)", "",
        f"Generated {s['generated_at']}. Corpus: {s['corpus'].get('name', '?')} "
        f"({s['corpus'].get('records', '?')} records, {s['corpus'].get('duplicates', 0)} "
        f"duplicate URLs and {s['corpus'].get('no_steps', 0)} empty protocols dropped).", "",
        "## Funnel", "",
        "| stage | count | share |", "|---|---|---|",
        f"| protocols | {s['protocols']} | |",
        f"| steps | {s['steps']} | 100% |",
        f"| grounded by rules | {s['status'].get('grounded', 0)} | "
        f"{pct(s['status'].get('grounded', 0), s['steps'])} |",
        f"| needs LLM | {s['status'].get('needs_llm', 0)} | "
        f"{pct(s['status'].get('needs_llm', 0), s['steps'])} |",
        f"| out of scope (rules) | {s['status'].get('out_of_scope', 0)} | "
        f"{pct(s['status'].get('out_of_scope', 0), s['steps'])} |",
        f"| planned into ops | {s['planned_steps']} | {pct(s['planned_steps'], s['steps'])} |",
        f"| planned liquid steps | {s['liquid_steps']} | {pct(s['liquid_steps'], s['steps'])} |",
        "", "| protocols | count |", "|---|---|",
        f"| with at least one planned step | {s['protocols_with_planned_step']} |",
        f"| task candidates (judged liquid step + buildable scene) | {s['n_candidates']} |",
        f"| with a task segment ({SEGMENT_MIN_STEPS}+ consecutive executable steps, "
        f"one liquid) | {s['protocols_with_segment']} |",
        f"| fully planned (every step executable) | {s['fully_planned_protocols']} |",
        "", "Scene builder on protocols with an executable step:", "",
        "| result | protocols |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in s["scene"].items()],
        "", "Longest run of consecutive planned steps per protocol:", "",
        "| run | protocols |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in s["longest_run_hist"].items()],
        "", "## What blocks steps", "",
        "`<skill>:unbound` means the skill exists but the step gives no binding "
        "(usually no volume); `unclustered` is mostly text that is not a robot "
        "action (notes, design, analysis).", "",
        "| capability | protocols | steps |", "|---|---|---|",
        *[f"| {cap} | {n} | {s['blockers_by_steps'].get(cap, 0)} |"
          for cap, n in list(s["blockers_by_protocols"].items())[:30]],
        "", "## Unlock order", "",
        f"Adding capabilities greedily: each row is the one that gives the most "
        f"further protocols a task segment ({SEGMENT_MIN_STEPS}+ consecutive "
        f"executable steps, one liquid). Scenes are not re-checked here.", "",
        f"Protocols with a segment today: {s['protocols_with_segment']}.", "",
        "| add | unlocks | protocols with a segment after |", "|---|---|---|",
        *[f"| {u['add']} | {u['unlocks']} | {u['protocols_with_segment']} |"
          for u in s["unlock"]],
        "", "## Weighing tasks", "",
        f"Weighing steps planned for the balance bench: {s['weighing']['ops']} in "
        f"{s['weighing']['protocols']} protocols; {s['weighing']['accepted_ops']} accepted (in "
        f"{s['weighing']['protocols_accepted']} protocols): a salt within the vial's range.", "",
        "| rejected because | weighings |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in s["weighing"]["rejected"].items()],
        "", "| protocol | title | step | mass | reagent |", "|---|---|---|---|---|",
        *[f"| [{w['protocol_id']}]({w['url']}) | {w['title'][:60]} | {w['step']} | {w['mass_g']:g} g | "
          f"{w['reagent']} |" for w in s["weighing"]["accepted"][:25]],
        "", "## LLM pass sizing", "",
        f"- needs_llm steps in total: {s['llm_pass']['needs_llm_steps_total']} "
        f"(about ${s['llm_pass']['est_usd_all']} at placeholder prices)",
        f"- near-complete protocols (every other step planned, at most "
        f"{NEAR_COMPLETE_MAX_LLM_STEPS} needs_llm steps): "
        f"{s['llm_pass']['near_complete_protocols']}, "
        f"{s['llm_pass']['near_complete_calls']} calls "
        f"(about ${s['llm_pass']['est_usd_near_complete']})",
    ]
    ran = s["llm_pass"].get("run")
    if ran:
        lines += [f"- LLM pass run: {ran['protocols']} of {ran['of_near_complete']} "
                  f"near-complete protocols, {ran['llm_steps']} steps, "
                  f"${ran['spent_usd']} ({'measured' if ran['spend_measured'] else 'estimated'}), "
                  f"stopped: {ran['stopped']}; +{ran['planned_gain']} planned steps, "
                  f"+{ran['segment_gain']} task segments"]
    if s.get("freeze"):
        frozen = [f for f in s["freeze"] if f.get("frozen")]
        reasons = Counter(f.get("reason") for f in s["freeze"] if not f.get("frozen"))
        lines += ["", "## Frozen tasks", "",
                  f"{len(frozen)} of {len(s['freeze'])} top candidates froze into validated "
                  f"tasks (oracle passes, two fresh replays agree, doing nothing fails).", "",
                  "| not frozen because | count |", "|---|---|",
                  *[f"| {k} | {v} |" for k, v in reasons.most_common()]]
    lines += ["", "## Top candidates", "",
              "| protocol | title | steps | planned | liquid | longest run |",
              "|---|---|---|---|---|---|",
              *[f"| [{c['protocol_id']}]({c['url']}) | {c['title'][:70]} | {c['n_steps']} | "
                f"{c['n_planned']} | {c['n_liquid']} | {c['longest_run']} |"
                for c in s["candidates"][:25]]]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(protocols: list, corpus_stats: dict, out_dir: Path, workers: int = 4,
        freeze_top: int = 0, client=None, llm_budget_usd: float = 0.0) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            records = pool.map(compile_protocol, protocols, chunksize=20)
    else:
        records = [compile_protocol(p) for p in protocols]
    llm = None
    if client is not None and llm_budget_usd > 0:
        llm = llm_pass(records, {p.raw.protocol_id: p for p in protocols}, client,
                       llm_budget_usd)
        (out_dir / "llm_pass.json").write_text(json.dumps(llm, indent=2), encoding="utf-8")
    with open(out_dir / "steps.jsonl", "w", encoding="utf-8") as steps_f, \
            open(out_dir / "protocols.jsonl", "w", encoding="utf-8") as prot_f:
        for r in records:
            steps_f.write(json.dumps({"protocol_id": r["protocol_id"],
                                      "steps": r.get("steps", [])}) + "\n")
            prot_f.write(json.dumps({k: v for k, v in r.items() if k != "steps"}) + "\n")
    summary = summarize(records, corpus_stats)
    summary["compile_seconds"] = round(time.time() - t0, 1)
    if llm is not None:
        summary["llm_pass"]["run"] = {k: v for k, v in llm.items() if k != "rows"}
    if freeze_top:
        summary["freeze"] = freeze_candidates({p.raw.protocol_id: p for p in protocols},
                                              summary, out_dir, freeze_top)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out_dir, summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lab2rob.coverage.corpus_sweep")
    parser.add_argument("--corpus", default="bioprobench", choices=("bioprobench",))
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--freeze-top", type=int, default=0)
    parser.add_argument("--llm-budget-usd", type=float, default=0.0,
                        help="re-ground near-complete protocols with the LLM fallback "
                             "(needs MODEL_API_KEY); 0 = rules only")
    args = parser.parse_args(argv)
    from lab2rob.ingest.bioprobench import download, load_corpus
    protocols, stats = load_corpus(download(), limit=args.limit)
    stats["name"] = "BioProBench (protocols.io + Protocol Exchange)"
    client = None
    if args.llm_budget_usd > 0:
        import os
        from lab2rob.llm.client import API_KEY_ENV, default_client
        if not os.environ.get(API_KEY_ENV):
            print(f"error: --llm-budget-usd needs {API_KEY_ENV}")
            return 1
        client = default_client()
    summary = run(protocols, stats, Path(args.out), args.workers, args.freeze_top,
                  client=client, llm_budget_usd=args.llm_budget_usd)
    print(f"{summary['protocols']} protocols, {summary['steps']} steps, "
          f"{summary['planned_steps']} planned, {summary['n_candidates']} candidates, "
          f"{summary['fully_planned_protocols']} fully planned "
          f"({summary['compile_seconds']} s); report in {args.out}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
