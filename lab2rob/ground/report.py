"""Per-protocol grounding report: rules / LLM / out-of-scope counts + reasons.

Usage: python -m lab2rob.ground.report [tier1.json] [--full-context]
Without MODEL_API_KEY the report runs rules-only; needs_llm steps are listed
as such instead of calling the model. --full-context shows the fallback
model the whole protocol per needs_llm step (costs more tokens).
"""

from __future__ import annotations

import os
import sys

from lab2rob.ground.rules import ground_protocol
from lab2rob.ingest.bioprot import DEFAULT_TIER1_PATH, load_tier1_file
from lab2rob.llm import Lab2RobClient


def report(tier1_path=str(DEFAULT_TIER1_PATH), client=None, full_context=False) -> dict:
    protocols = []
    for raw in load_tier1_file(tier1_path):
        grounding = ground_protocol(raw, client=client, full_context=full_context)
        protocols.append({
            "protocol_id": grounding.protocol_id,
            "dataset": grounding.dataset,
            "title": grounding.title,
            "counts": grounding.counts,
            "steps": [{"idx": s.idx, "text": s.source_text, "status": s.status,
                       "action": s.action, "reason": s.reason, "detail": s.detail,
                       "by": s.by} for s in grounding.steps],
            "description_flags": grounding.description_flags,
        })
    return {"protocols": protocols}


def print_report(data: dict) -> None:
    for protocol in data["protocols"]:
        print(f"== {protocol['dataset']}:{protocol['protocol_id']} {protocol['title']} ==")
        print(f"   counts: {protocol['counts']}")
        for step in protocol["steps"]:
            print(f"   step {step['idx']} [{step['status']}/{step['by']}] "
                  f"{step['action'] or '-'} {step['reason'] or ''} :: {step['detail']}")
            print(f"      {step['text'][:120]}")
        for flag in protocol["description_flags"]:
            print(f"   description: [{flag['reason']}] {flag['text']} :: {flag['detail']}")


def main(argv: list[str]) -> int:
    args = [a for a in argv if a != "--full-context"]
    path = args[1] if len(args) > 1 else str(DEFAULT_TIER1_PATH)
    client = Lab2RobClient() if os.environ.get("MODEL_API_KEY") else None
    if client is None:
        print("(rules-only: MODEL_API_KEY not set, needs_llm steps will be listed)")
    print_report(report(path, client=client,
                        full_context="--full-context" in argv))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
