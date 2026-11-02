"""Scientific outcome checks: reported result vs ground truth, with tolerances.

Detection is heuristic v1: measurement phrases in the protocol title,
description, or step texts select at most one check per outcome type.
Recipe concentrations ("1% SDS") deliberately do NOT trigger — an outcome
is a reported measurement, not a recipe target.
"""

from __future__ import annotations

from lab2rob.schema import OutcomeCheck

#: (trigger phrases, outcome type, quantity label, default relative tolerance)
OUTCOME_PATTERNS = [
    (("ic50", "ic-50"), "ic50", "um", 0.20),
    (("standard curve", "bradford", "linear range",
      "report the concentration", "determine the concentration",
      "measure the concentration"), "concentration", "mg_per_ml", 0.15),
    (("cell viability", "viable cell", "confluence"), "viable_cells", "cells", 0.20),
]


def plan_outcomes(protocol_grounding) -> list:
    """At most one OutcomeCheck per type, from measurement phrases."""
    text = " ".join([protocol_grounding.title or "",
                     getattr(protocol_grounding, "description", "") or ""] +
                    [s.source_text for s in protocol_grounding.steps]).lower()
    checks = []
    for phrases, outcome_type, quantity, tol_rel in OUTCOME_PATTERNS:
        if any(phrase in text for phrase in phrases):
            checks.append(OutcomeCheck(type=outcome_type, quantity=quantity,
                                       tol_rel=tol_rel))
    return checks


def evaluate_outcome(check: OutcomeCheck, episode: dict) -> dict:
    """Score one outcome against an episode trace ({reported, truth}).

    Total function like the registry predicates: missing keys or missing
    tolerances fail with an explanation, never an exception.
    """
    reported, truth = episode.get("reported"), episode.get("truth")
    base = {"type": check.type, "reported": reported, "truth": truth,
            "error_abs": None, "error_rel": None}
    if reported is None or truth is None:
        missing = "reported" if reported is None else "truth"
        return {**base, "passed": False, "detail": f"episode has no {missing} value"}
    try:
        error = abs(float(reported) - float(truth))
    except (TypeError, ValueError):
        return {**base, "passed": False, "detail": "reported/truth are not numbers"}
    rel = error / abs(float(truth)) if float(truth) != 0 else (
        0.0 if error == 0 else float("inf"))
    abs_ok = check.tol_abs is None or error <= check.tol_abs
    rel_ok = check.tol_rel is None or rel <= check.tol_rel
    if check.tol_abs is None and check.tol_rel is None:
        return {**base, "error_abs": error, "error_rel": rel,
                "passed": False, "detail": "check sets no tolerance"}
    passed = abs_ok and rel_ok
    return {**base, "error_abs": error, "error_rel": rel, "passed": passed,
            "detail": f"error {error:.4g} (rel {rel:.4g}) vs "
                      f"tol_abs={check.tol_abs} tol_rel={check.tol_rel}"}
