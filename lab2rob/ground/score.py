"""Score the rules grounder against the hand-annotated WLP-style sample.

Zero LLM tokens: runs ground_step() only, never the fallback. Produces
action/entity/quantity precision/recall, the rules-only share (steps fully
grounded without LLM help), and miss breakdowns that say what to build next.

Usage: python -m lab2rob.ground.score [--sample PATH] [--report PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from .rules import ParsedStep, ground_step

DEFAULT_SAMPLE_PATH = Path(__file__).with_name("wlp_sample.json")
DEFAULT_REPORT_PATH = Path(__file__).with_name("score_report.json")

#: Surface unit spellings -> comparison key. Molar units stay case apart
#: from lengths (mM != mm); unknown spellings fold to lowercase as-is.
UNIT_ALIASES = {
    "ml": "ml", "mL": "ml", "ML": "ml",
    "ul": "ul", "uL": "ul", "µl": "ul", "μl": "ul", "µL": "ul", "μL": "ul",
    "nl": "nl", "nL": "nl", "l": "l", "L": "l", "dl": "dl",
    "g": "g", "mg": "mg", "ug": "ug", "ng": "ng", "kg": "kg",
    "M": "M", "mM": "mM", "uM": "uM", "um": "uM", "nM": "nM",
    "mol": "mol", "mmol": "mmol", "mg/ml": "mg/ml", "g/l": "g/l", "ug/ml": "ug/ml",
    "%": "%", "percent": "%",
    "s": "s", "sec": "s", "secs": "s", "min": "min", "mins": "min",
    "h": "h", "hr": "h", "hrs": "h",
    "c": "c", "°c": "c", "°C": "c",
    "rpm": "rpm", "x": "x",
}
_WS = re.compile(r"\s+")


def norm_entity(name: str) -> str:
    return _WS.sub(" ", str(name).strip().lower())


def norm_unit(unit: str) -> str:
    unit = str(unit).strip()
    return UNIT_ALIASES.get(unit, unit.lower())


def norm_quantity(value, unit) -> tuple:
    return (round(float(value), 9), norm_unit(unit))


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def match_multisets(gold: list, pred: list) -> tuple:
    """Greedy multiset intersection; returns (tp, fp_items, fn_items)."""
    remaining = list(pred)
    tp, fn_items = 0, []
    for item in gold:
        if item in remaining:
            tp += 1
            remaining.remove(item)
        else:
            fn_items.append(item)
    return tp, remaining, fn_items


def load_sample(path: str | Path = DEFAULT_SAMPLE_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def score_record(record: dict) -> dict:
    """Ground one sample step with rules only and compare to gold."""
    grounding = ground_step(ParsedStep(record["step_idx"], record["text"]))
    gold = record["gold"]
    gold_actions = list(gold["action"])
    pred_actions = [grounding.action] if grounding.action else []
    gold_entities = [norm_entity(e) for e in gold["entities"]]
    pred_entities = [norm_entity(r) for r in grounding.reagents]
    pred_entities += [norm_entity(e["name"]) for e in grounding.entities]
    gold_quantities = [norm_quantity(q["value"], q["unit"]) for q in gold["quantities"]]
    pred_quantities = [norm_quantity(q.value, q.unit) for q in grounding.quantities]
    _, act_fp, act_fn = match_multisets(gold_actions, pred_actions)
    _, ent_fp, ent_fn = match_multisets(gold_entities, pred_entities)
    _, qty_fp, qty_fn = match_multisets(gold_quantities, pred_quantities)
    return {
        "id": record["id"],
        "status": grounding.status,
        "reason": grounding.reason,
        "detail": grounding.detail,
        "by": grounding.by,
        "gold_action": gold_actions,
        "pred_action": grounding.action,
        "action_exact": sorted(gold_actions) == sorted(pred_actions),
        "action_fp": act_fp,
        "action_fn": act_fn,
        "entity_tp": len(gold_entities) - len(ent_fn),
        "entity_fp": ent_fp,
        "entity_fn": ent_fn,
        "quantity_tp": len(gold_quantities) - len(qty_fn),
        "quantity_fp": [[v, u] for v, u in qty_fp],
        "quantity_fn": [[v, u] for v, u in qty_fn],
        "provenance": {"gold": "human", "pred": "rules"},
    }


def score_sample(records: list) -> dict:
    rows = [score_record(r) for r in records]
    n = len(rows)
    act_tp = sum(1 for r in rows if r["pred_action"] and r["pred_action"] in r["gold_action"])
    act_fp = sum(len(r["action_fp"]) for r in rows)
    act_fn = sum(len(r["action_fn"]) for r in rows)
    ent_tp = sum(r["entity_tp"] for r in rows)
    ent_fp = sum(len(r["entity_fp"]) for r in rows)
    ent_fn = sum(len(r["entity_fn"]) for r in rows)
    qty_tp = sum(r["quantity_tp"] for r in rows)
    qty_fp = sum(len(r["quantity_fp"]) for r in rows)
    qty_fn = sum(len(r["quantity_fn"]) for r in rows)
    grounded = sum(1 for r in rows if r["status"] == "grounded" and r["by"] == "rules")
    qty_miss_units = Counter(u for r in rows for _, u in r["quantity_fn"])
    entity_misses = Counter(e for r in rows for e in r["entity_fn"])
    action = prf(act_tp, act_fp, act_fn)
    action["accuracy"] = round(sum(1 for r in rows if r["action_exact"]) / n, 4) if n else 0.0
    return {
        "n_steps": n,
        "action": action,
        "entity": prf(ent_tp, ent_fp, ent_fn),
        "quantity": prf(qty_tp, qty_fp, qty_fn),
        "rules_only_share": round(grounded / n, 4) if n else 0.0,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "reason_counts": dict(Counter(r["reason"] or "-" for r in rows)),
        "quantity_miss_units": dict(qty_miss_units),
        "top_entity_misses": dict(entity_misses.most_common(10)),
        "rows": rows,
    }


def print_summary(report: dict) -> None:
    print(f"steps: {report['n_steps']}")
    for key in ("action", "entity", "quantity"):
        m = report[key]
        print(f"{key}: P={m['precision']} R={m['recall']} F1={m['f1']} "
              f"(tp={m['tp']} fp={m['fp']} fn={m['fn']})")
    print(f"action accuracy (exact): {report['action']['accuracy']}")
    print(f"rules-only share: {report['rules_only_share']}")
    print(f"status: {report['status_counts']}")
    print(f"reasons: {report['reason_counts']}")
    print(f"quantity miss units: {report['quantity_miss_units']}")
    print(f"top entity misses: {report['top_entity_misses']}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Score rules grounder on WLP-style sample")
    parser.add_argument("--sample", default=str(DEFAULT_SAMPLE_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    args = parser.parse_args(argv)
    sample = load_sample(args.sample)
    report = {"sample": sample.get("name", ""), **score_sample(sample["records"])}
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print_summary(report)
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
