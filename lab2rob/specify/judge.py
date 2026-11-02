"""Independent judge planner: what must be true, written blind to the program.

plan_judge reads ONLY the grounded protocol — never the planned program —
so it cannot rubber-stamp the planner's choices. For each grounded liquid
step it emits step predicates from the registry vocabulary (volume moved +
no spill); other grounded actions get no predicates because the vocabulary
cannot express them (no time or position checks exist yet). Vessel
references are entity names, or "unbound" when the text names no vessel;
the scene builder resolves them to concrete wells at certify time.

Deliberately NOT emitted: concentration checks from conc quantities (a
stated concentration is usually the source stock, not a target — asserting
it would fail correct runs). Concentrations are checked as outcomes
(outcome.py), where reported-vs-truth semantics are clear.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pint

from lab2rob.ground.rules import VOLUME_UNITS
from lab2rob.schema import OutcomeCheck, StepPredicate

ureg = pint.UnitRegistry()

#: Grounded actions that move liquid and hence get volume + no-spill checks.
#: Learned liquid-moving macros (ST-02) join this set on registration.
LIQUID_ACTIONS = {"transfer", "aspirate", "dispense", "mix_by_pipetting",
                  "serial_dilution", "dilute"}

#: Marker for a vessel the text never names; resolved at certify time.
UNBOUND = "unbound"


@dataclass
class JudgeStep:
    idx: int
    source_text: str
    predicates: list = field(default_factory=list)  # StepPredicate


@dataclass
class PlannedJudge:
    protocol_id: str
    title: str
    steps: list = field(default_factory=list)  # JudgeStep, grounded steps only
    outcomes: list = field(default_factory=list)  # OutcomeCheck

    def alignment(self) -> list:
        """Judge records for export and the GUI: one row per check."""
        rows = []
        for step in self.steps:
            for pred in step.predicates:
                rows.append({
                    "protocol_id": self.protocol_id, "step_idx": step.idx,
                    "source_text": step.source_text, "kind": "predicate",
                    "pred": pred.pred, "args": pred.args,
                    "target": pred.target, "tol": pred.tol,
                })
        for outcome in self.outcomes:
            rows.append({
                "protocol_id": self.protocol_id, "step_idx": None,
                "source_text": None, "kind": "outcome",
                "pred": outcome.type, "args": [outcome.quantity or ""],
                "target": None, "tol": outcome.tol_rel,
            })
        return rows


def save_judge(judge: PlannedJudge, path: str | Path) -> Path:
    """Persist judge records as JSONL, one check per line."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as handle:
        for row in judge.alignment():
            handle.write(json.dumps(row) + "\n")
    return path


def _volume_ul(quantities: list) -> float | None:
    for quantity in quantities:
        if quantity.unit in VOLUME_UNITS:
            try:
                return float(ureg.Quantity(quantity.value, quantity.unit).to("uL").magnitude)
            except Exception:
                continue
    return None


def _tol_ul(volume_ul: float) -> float:
    return max(1.0, round(0.05 * volume_ul, 2))


def _well_ref(grounding) -> str:
    for entity in grounding.entities:
        if entity.get("scope") == "in_scope":
            return entity["name"]
    return UNBOUND


def _judge_liquid_step(grounding) -> list:
    preds = [StepPredicate(step=grounding.idx, pred="no_spill", args=[],
                           target=None, tol=0.0)]
    volume = _volume_ul(grounding.quantities)
    if volume is not None:
        preds.append(StepPredicate(step=grounding.idx, pred="volume",
                                   args=[_well_ref(grounding)], target=volume,
                                   tol=_tol_ul(volume)))
    return preds


def plan_judge(protocol_grounding) -> PlannedJudge:
    """Write step predicates + outcome checks from grounding alone.

    Takes no program input by construction (see test); outcome checks come
    from outcome.plan_outcomes over the same grounding.
    """
    from .outcome import plan_outcomes

    steps = [JudgeStep(idx=g.idx, source_text=g.source_text,
                       predicates=_judge_liquid_step(g)
                       if g.action in LIQUID_ACTIONS else [])
             for g in protocol_grounding.steps if g.status == "grounded"]
    return PlannedJudge(protocol_id=protocol_grounding.protocol_id,
                        title=protocol_grounding.title, steps=steps,
                        outcomes=plan_outcomes(protocol_grounding))
