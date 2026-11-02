"""ST-04: sweep cached corpora with full-LLM translation, mine skill candidates.

For each protocol: one ground_protocol_llm call, verdicts saved per protocol.
Then cluster out_of_scope + ambiguous steps by needed capability (keyword map
over step text + model detail) and rank candidates by frequency with example
steps and counts. Output feeds ST-01 (coverage queue) and ST-02 (evolve loop).

Needs MODEL_API_KEY: run in your own terminal (the sandbox has no key):
  python -m lab2rob.ground.sweep --out runs/sweep_20251030 [--limit 3] [--resume]
Per-protocol files land incrementally, so a crash loses nothing --resume cannot
recover; failed protocols are recorded and skipped, never fatal.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

#: Needed-capability keywords, first match wins (distinctive ones first).
#: Matched as stems (dissolv- matches dissolve/dissolved/dissolving).
CAPABILITIES = [
    ("autoclave", ["autoclave"]),
    ("sonicate", ["sonicat"]),
    ("vortex", ["vortex"]),
    ("homogenize", ["homogeniz"]),
    ("lyse", ["lys"]),
    ("digest", ["digest"]),
    ("stain", ["stain"]),
    ("fix", ["fix"]),
    ("mount", ["mount"]),
    ("image", ["imag", "photograph", "microscop"]),
    ("count_cells", ["hemocytometer", "cell count", "count cells"]),
    ("passage", ["passag", "split", "trypsin"]),
    ("seed_cells", ["seed"]),
    ("feed_medium", ["medium change", "media change", "fresh medium",
                     "fresh media", "feed"]),
    ("harvest", ["harvest"]),
    ("coat", ["coat"]),
    ("plate_cells", ["plate the", "plate cells", "plating"]),
    ("scrape", ["scrape"]),
    ("resuspend", ["resuspend"]),
    ("dissolve", ["dissolv"]),
    ("dilute", ["dilut"]),
    ("centrifuge", ["centrifug", "spin", "pellet"]),
    ("filter", ["filtr"]),
    ("weigh", ["weigh"]),
    ("ph_adjust", ["adjust ph", "adjust the ph", "ph adjustment", "calibrat"]),
    ("pour", ["pour"]),
    ("stir", ["stir"]),
    ("shake", ["shak"]),
    ("heat", ["heat", "boil"]),
    ("cool", ["cool", "chill", "ice"]),
    ("cold_store", ["store at", "stored at", "snap freez", "flash freez"]),
    ("freeze", ["freez", "frozen"]),
    ("thaw", ["thaw"]),
    ("dry", ["dry", "desiccat", "evaporat", "air-dry", "air dry"]),
    ("rinse", ["rins"]),
    ("soak", ["soak"]),
    ("wash", ["wash"]),
    ("discard", ["discard", "drain"]),
    ("aliquot", ["aliquot"]),
]

MAX_EXAMPLES = 3
EXAMPLE_CHARS = 160


def _capability_key(text: str) -> str:
    low = text.lower()
    for name, keywords in CAPABILITIES:
        for keyword in keywords:
            if re.search(r"\b" + re.escape(keyword) + r"\w*\b", low):
                return name
    return "unclustered"


def _mineable(step: dict) -> bool:
    return step.get("status") in ("out_of_scope", "ambiguous")


def mine_new_skills(protocols: list) -> list:
    """Rank needed capabilities over out_of_scope + ambiguous steps.

    protocols are sweep dicts (or the same shape loaded from JSON):
    {"protocol_id", "dataset", "steps": [{"idx","text","status","reason",
    "action","detail"}]}. Returns ranking entries with counts, protocol
    lists, and up to MAX_EXAMPLES example steps each.
    """
    buckets: dict[str, dict] = {}
    for protocol in protocols:
        pid = f"{protocol.get('dataset', '?')}:{protocol.get('protocol_id', '?')}"
        for step in protocol.get("steps", []):
            if not _mineable(step):
                continue
            key = _capability_key(f"{step.get('text', '')} {step.get('detail', '')}")
            bucket = buckets.setdefault(key, {"count": 0, "protocols": set(),
                                              "examples": []})
            bucket["count"] += 1
            bucket["protocols"].add(pid)
            if len(bucket["examples"]) < MAX_EXAMPLES:
                text = " ".join(str(step.get("text", "")).split())
                bucket["examples"].append({
                    "protocol_id": pid, "step_idx": step.get("idx"),
                    "text": text[:EXAMPLE_CHARS],
                    "reason": step.get("reason"), "detail": str(step.get("detail", ""))[:120]})
    ranking = [{"capability": key, "count": b["count"],
                "protocols": sorted(b["protocols"]), "examples": b["examples"]}
               for key, b in buckets.items()]
    ranking.sort(key=lambda e: (-e["count"], e["capability"]))
    return ranking


def _protocol_record(protocol_id: str, dataset: str, grounding) -> dict:
    return {
        "protocol_id": protocol_id, "dataset": dataset,
        "title": grounding.title, "n_steps": len(grounding.steps),
        "counts": dict(grounding.counts),
        "steps": [{"idx": s.idx, "text": s.source_text, "status": s.status,
                   "reason": s.reason, "action": s.action, "detail": s.detail}
                  for s in grounding.steps],
    }


def _record_path(out_dir: Path, dataset: str, protocol_id: str) -> Path:
    safe = re.sub(r"[^\w.\-]+", "_", f"{dataset}_{protocol_id}")
    return out_dir / f"{safe}.json"


def sweep_protocols(raws: list, client, out_dir: Path, resume: bool = False) -> list:
    """Ground every raw protocol with one full-LLM call each.

    Writes one JSON per protocol incrementally; with resume, protocols with
    an existing record are loaded back instead of re-grounded. Per-protocol
    failures are recorded and skipped, never fatal.
    """
    from .llm_ground import ground_protocol_llm

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, raw in enumerate(raws, 1):
        path = _record_path(out_dir, raw.dataset, raw.protocol_id)
        if resume and path.exists():
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
                print(f"[{i}/{len(raws)}] {raw.dataset}:{raw.protocol_id} resumed", flush=True)
                continue
            except ValueError:
                pass  # corrupt checkpoint: re-ground below
        try:
            grounding = ground_protocol_llm(raw, client)
            record = _protocol_record(raw.protocol_id, raw.dataset, grounding)
            print(f"[{i}/{len(raws)}] {raw.dataset}:{raw.protocol_id} "
                  f"{len(grounding.steps)} steps {dict(grounding.counts)}", flush=True)
            statuses = {s.status for s in grounding.steps}
            if grounding.steps and statuses <= {"ambiguous", "failed"}:
                print(f"  WARNING: every step came back "
                      f"{'/'.join(sorted(statuses))} -- systematic "
                      f"failure, not data (see the step details)", flush=True)
        except Exception as e:  # noqa: BLE001 - one bad protocol must not kill a sweep
            record = {"protocol_id": raw.protocol_id, "dataset": raw.dataset,
                      "title": raw.title, "n_steps": 0, "steps": [],
                      "error": f"{type(e).__name__}: {e}"[:300]}
            print(f"[{i}/{len(raws)}] {raw.dataset}:{raw.protocol_id} ERROR {e}", flush=True)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)
    return records


def write_report(out_dir: Path, records: list, ranking: list,
               usage: dict | None = None) -> dict:
    """candidates.json + candidates.md; returns the JSON payload.

    usage is a CostLog.summarize_by_job rollup (calls, tokens with reasoning,
    cost per job); included in the JSON and summarized in the Markdown when
    given.
    """
    out_dir = Path(out_dir)
    failed = [r for r in records if "error" in r]
    steps_total = sum(r.get("n_steps", 0) for r in records)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocols_swept": len(records),
        "protocols_failed": len(failed),
        "failed": [f"{r.get('dataset')}:{r.get('protocol_id')}" for r in failed],
        "steps_total": steps_total,
        "steps_mined": sum(e["count"] for e in ranking),
        "ranking": ranking,
    }
    if usage is not None:
        payload["llm_usage"] = usage
    (out_dir / "candidates.json").write_text(json.dumps(payload, indent=2),
                                             encoding="utf-8")
    lines = ["# Mined skill candidates",
             "",
             f"Swept {len(records)} protocols ({steps_total} steps) on "
             f"{payload['generated_at']}; {payload['steps_mined']} out-of-scope "
             f"or ambiguous steps clustered.",
             ""]
    if usage:
        lines.append("## LLM usage")
        for job in sorted(usage):
            row = usage[job]
            lines.append(f"- {job}: {row['calls']} calls, "
                         f"{row['reasoning_tokens']} reasoning tokens, "
                         f"${row['cost_usd']:.4f}")
        lines.append("")
    for i, entry in enumerate(ranking, 1):
        lines.append(f"## {i}. {entry['capability']} ({entry['count']} steps, "
                     f"{len(entry['protocols'])} protocols)")
        for ex in entry["examples"]:
            lines.append(f"- {ex['protocol_id']} s{ex['step_idx']}: {ex['text']}")
        lines.append("")
    (out_dir / "candidates.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def load_corpus(*, protocolsio: bool = True, bioprot: bool = True,
                limit: int | None = None) -> list:
    """Raw protocols from the cached corpora (protocols.io + tier-1 BioProt)."""
    raws = []
    if protocolsio:
        from lab2rob.ingest.protocols_io import iter_cached_protocols
        raws.extend(sorted(iter_cached_protocols(), key=lambda r: r.protocol_id))
    if bioprot:
        from lab2rob.ingest.bioprot import load_tier1_file
        raws.extend(load_tier1_file())
    if limit is not None:
        raws = raws[:max(limit, 0)]
    return raws


def _diff_usage(before: dict, after: dict) -> dict:
    """Per-job usage that happened between two summarize_by_job snapshots.

    The cost ledger is cumulative across runs, so the sweep reports only the
    delta attributable to this run.
    """
    delta = {}
    for job, row in after.items():
        prev = before.get(job, {})
        calls = row["calls"] - prev.get("calls", 0)
        if calls <= 0:
            continue
        delta[job] = {
            "calls": calls,
            "prompt_tokens": row["prompt_tokens"] - prev.get("prompt_tokens", 0),
            "completion_tokens": (row["completion_tokens"]
                                  - prev.get("completion_tokens", 0)),
            "reasoning_tokens": (row["reasoning_tokens"]
                                 - prev.get("reasoning_tokens", 0)),
            "cost_usd": round(row["cost_usd"] - prev.get("cost_usd", 0.0), 6),
        }
    return delta


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Sweep corpora, mine skill candidates")
    parser.add_argument("--out", default=f"runs/sweep_{date.today().isoformat()}",
                        help="output directory (also the resume checkpoint)")
    parser.add_argument("--limit", type=int, default=None,
                        help="sweep only the first N protocols (pilot runs)")
    parser.add_argument("--resume", action="store_true",
                        help="reuse existing per-protocol records instead of re-grounding")
    parser.add_argument("--corpora", default="both",
                        choices=["both", "protocolsio", "bioprot"])
    args = parser.parse_args(argv)
    from lab2rob.llm import Lab2RobClient
    raws = load_corpus(protocolsio=args.corpora in ("both", "protocolsio"),
                       bioprot=args.corpora in ("both", "bioprot"), limit=args.limit)
    if not raws:
        print("no protocols to sweep")
        return 1
    client = Lab2RobClient()  # raises without MODEL_API_KEY — run in your terminal
    out_dir = Path(args.out)
    usage_before = client.cost_log.summarize_by_job()
    records = sweep_protocols(raws, client, out_dir, resume=args.resume)
    ranking = mine_new_skills(records)
    usage = _diff_usage(usage_before, client.cost_log.summarize_by_job())
    payload = write_report(out_dir, records, ranking, usage=usage)
    total_cost = sum(row["cost_usd"] for row in usage.values())
    total_reasoning = sum(row["reasoning_tokens"] for row in usage.values())
    print(f"swept {payload['protocols_swept']} protocols, "
          f"{payload['protocols_failed']} failed, cost ${total_cost:.4f}, "
          f"reasoning tokens {total_reasoning}")
    print("top candidates:")
    for entry in ranking[:10]:
        print(f"  {entry['count']:4d}  {entry['capability']}")
    print(f"wrote {out_dir / 'candidates.json'} and candidates.md")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
