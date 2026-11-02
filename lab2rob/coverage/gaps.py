"""ST-01: coverage-gap queue from sweep verdicts plus the refined parse.

Every out-of-scope or ambiguous sweep step lands in the queue under its
reason (missing_skill / not_implemented / missing_asset / unmodeled_physics
/ ambiguous_text) and needed capability. Steps the keyword pass left
unclustered take their capability from the refined parse when one exists:
existing remaps fold into their capability (and flag a keyword gap),
new names are marked proposed, noise stays unresolved. evolve_candidates
ranks the out-of-scope capabilities (noise excluded) for ST-02.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from lab2rob.ground.sweep import _capability_key

REASONS = ("missing_skill", "not_implemented", "missing_asset",
           "unmodeled_physics", "ambiguous_text")
OOS_REASONS = frozenset(REASONS[:4])
MAX_EXAMPLES = 3
EXAMPLE_CHARS = 160


def _refined_lookup(refined_rows: list) -> dict:
    """(protocol_id, step_idx) -> refined row, for unclustered steps."""
    return {(r.get("protocol_id"), r.get("step_idx")): r
            for r in (refined_rows or [])}


def capability_for(text: str, detail: str, refined: dict | None) -> tuple:
    """(capability, proposed, noise): refined parse overrides unclustered."""
    base = _capability_key(f"{text} {detail}")
    if base != "unclustered" or refined is None:
        return base, False, False
    bucket = refined.get("bucket")
    if bucket == "existing":
        return refined.get("capability"), False, False
    if bucket == "new":
        return refined.get("capability"), True, False
    return "unresolved", False, True  # noise or unparsed: no skill gap


def _example(pid: str, step: dict) -> dict:
    return {"protocol_id": pid, "step_idx": step.get("idx"),
            "text": " ".join(str(step.get("text", "")).split())[:EXAMPLE_CHARS],
            "reason": step.get("reason"),
            "detail": str(step.get("detail", ""))[:120]}


def build_queue(records: list, refined_rows: list | None = None) -> dict:
    """Queue payload from sweep records (+ optional refined rows)."""
    lookup = _refined_lookup(refined_rows or [])
    groups: dict[tuple, dict] = {}
    remaps: dict[str, dict] = {}
    steps_total = 0
    for record in records:
        pid = f"{record.get('dataset', '?')}:{record.get('protocol_id', '?')}"
        for step in record.get("steps", []):
            steps_total += 1
            if step.get("status") not in ("out_of_scope", "ambiguous"):
                continue
            reason = step.get("reason") or "ambiguous_text"
            if reason not in REASONS:
                reason = "ambiguous_text"
            refined = lookup.get((pid, step.get("idx")))
            capability, proposed, _ = capability_for(
                step.get("text", ""), step.get("detail", ""), refined)
            group = groups.setdefault(
                (reason, capability),
                {"count": 0, "protocols": set(), "examples": [],
                 "proposed": False})
            group["count"] += 1
            group["protocols"].add(pid)
            group["proposed"] = group["proposed"] or proposed
            if len(group["examples"]) < MAX_EXAMPLES:
                group["examples"].append(_example(pid, step))
            if (refined is not None and refined.get("bucket") == "existing"
                    and _capability_key(
                        f"{step.get('text', '')} {step.get('detail', '')}")
                    == "unclustered"):
                remap = remaps.setdefault(capability, {"count": 0,
                                                       "examples": []})
                remap["count"] += 1
                if len(remap["examples"]) < MAX_EXAMPLES:
                    remap["examples"].append(_example(pid, step))
    by_reason: dict[str, list] = {reason: [] for reason in REASONS}
    for (reason, capability), group in groups.items():
        by_reason[reason].append({
            "capability": capability, "count": group["count"],
            "proposed": group["proposed"],
            "protocols": sorted(group["protocols"]),
            "examples": group["examples"]})
    for entries in by_reason.values():
        entries.sort(key=lambda e: (-e["count"], e["capability"]))
    candidates: dict[str, dict] = {}
    for (reason, capability), group in groups.items():
        if reason not in OOS_REASONS or capability == "unresolved":
            continue
        entry = candidates.setdefault(
            capability, {"capability": capability, "count": 0,
                         "proposed": False, "protocols": set(),
                         "reasons": set(), "examples": []})
        entry["count"] += group["count"]
        entry["proposed"] = entry["proposed"] or group["proposed"]
        entry["protocols"] |= group["protocols"]
        entry["reasons"].add(reason)
        for example in group["examples"]:
            if len(entry["examples"]) < MAX_EXAMPLES:
                entry["examples"].append(example)
    evolve = [{
        "capability": e["capability"], "count": e["count"],
        "proposed": e["proposed"], "protocols": sorted(e["protocols"]),
        "reasons": sorted(e["reasons"]), "examples": e["examples"]}
        for e in candidates.values()]
    evolve.sort(key=lambda e: (-e["count"], e["capability"]))
    fixes = [{"capability": capability, "remapped_steps": r["count"],
              "examples": r["examples"]}
             for capability, r in remaps.items()]
    fixes.sort(key=lambda e: (-e["remapped_steps"], e["capability"]))
    queued = sum(g["count"] for g in groups.values())
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "steps_total": steps_total,
        "steps_queued": queued,
        "by_reason": by_reason,
        "evolve_candidates": evolve,
        "suggested_keyword_fixes": fixes,
    }


def load_sweep_records(sweep_dir: Path) -> list:
    """Per-protocol sweep records (candidates/report files skipped)."""
    records = []
    for path in sorted(glob.glob(str(Path(sweep_dir) / "*.json"))):
        name = Path(path).name
        if name in ("candidates.json", "refined_unclustered.json"):
            continue
        records.append(json.loads(Path(path).read_text(encoding="utf-8")))
    return records


def load_refined(sweep_dir: Path) -> list:
    path = Path(sweep_dir) / "refined_unclustered.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("rows", [])


def write_queue(out_dir: Path, queue: dict) -> dict:
    """queue.json + queue.md; returns the payload."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "queue.json").write_text(json.dumps(queue, indent=2),
                                        encoding="utf-8")
    lines = ["# Coverage-gap queue",
             "",
             f"{queue['steps_queued']} of {queue['steps_total']} sweep steps "
             f"queued on {queue['generated_at']}.",
             "",
             "## Evolve candidates (out-of-scope, noise excluded)",
             ""]
    for i, entry in enumerate(queue["evolve_candidates"], 1):
        tag = " (proposed)" if entry["proposed"] else ""
        lines.append(f"{i}. {entry['capability']}{tag}: {entry['count']} "
                     f"steps, {len(entry['protocols'])} protocols "
                     f"[{', '.join(entry['reasons'])}]")
    lines += ["", "## By reason", ""]
    for reason in REASONS:
        entries = queue["by_reason"][reason]
        total = sum(e["count"] for e in entries)
        lines.append(f"### {reason} ({total} steps)")
        for entry in entries:
            tag = " (proposed)" if entry["proposed"] else ""
            lines.append(f"- {entry['capability']}{tag}: {entry['count']}")
        lines.append("")
    fixes = queue["suggested_keyword_fixes"]
    if fixes:
        lines += ["## Suggested keyword fixes", ""]
        for fix in fixes:
            lines.append(f"- {fix['capability']}: "
                         f"{fix['remapped_steps']} refined steps missed "
                         f"the keywords; add synonyms from these examples:")
            for example in fix["examples"]:
                lines.append(f"  - {example['protocol_id']} "
                             f"s{example['step_idx']}: {example['text']}")
        lines.append("")
    (out_dir / "queue.md").write_text("\n".join(lines), encoding="utf-8")
    return queue


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Build the coverage-gap queue")
    parser.add_argument("--sweep", default="runs/sweep_20251030",
                        help="sweep directory with per-protocol records")
    parser.add_argument("--out", default="lab2rob/coverage",
                        help="output directory for queue.json + queue.md")
    args = parser.parse_args(argv)
    records = load_sweep_records(Path(args.sweep))
    if not records:
        print(f"no sweep records in {args.sweep}")
        return 1
    queue = build_queue(records, load_refined(Path(args.sweep)))
    write_queue(Path(args.out), queue)
    top = queue["evolve_candidates"][:5]
    print(f"queued {queue['steps_queued']} of {queue['steps_total']} steps; "
          f"top evolve candidates:")
    for entry in top:
        print(f"  {entry['count']:4d}  {entry['capability']}"
              f"{' (proposed)' if entry['proposed'] else ''}")
    print(f"wrote {args.out}/queue.json and queue.md")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
