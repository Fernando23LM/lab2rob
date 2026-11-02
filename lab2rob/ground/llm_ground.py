"""LLM fallback for steps the rules cannot resolve.

Schema-validated JSON output, one retry with the validation error, then the
step is marked ambiguous. A reply cut off at finish "length" is retried once
at the next reasoning effort up with a larger budget, then the step is marked
failed/llm_truncated; truncated JSON is never parsed. Used only for needs_llm
steps (see rules.py).
"""

from __future__ import annotations

import json

from lab2rob.llm import Lab2RobClient, effort_for_job
from lab2rob.registry import resolve_skill, skill_names

SYSTEM = """You ground one wet-lab protocol step to the lab2rob skill registry.
Reply with exactly one JSON object, no other text:
{"skill": <skill name or null>, "out_of_scope_reason": null | "not_implemented" | "missing_skill" | "missing_asset" | "unmodeled_physics", "ambiguous": <true|false>, "detail": <short string>}
Rules: skill must be an implemented or macro skill from the vocabulary below,
or null. If the step needs a declared (unimplemented) skill, set skill to null
and out_of_scope_reason to "not_implemented". If the step is vague or lacks
the amounts needed to execute, set ambiguous true. Never invent skill names."""

REASONS = {"not_implemented", "missing_skill", "missing_asset", "unmodeled_physics"}

#: Structured output for single-step calls: the API honors json_object mode
#: for the contributor model (probe-verified), which guarantees valid JSON
#: syntax. Skill-name validation still runs on top: json_object is not a
#: schema. Chunk calls keep prompt-based JSON: they need a top-level array,
#: which json_object mode cannot express, and the probe showed 4/4 valid.
STEP_RESPONSE_FORMAT = {"type": "json_object"}

#: Truncation-escalation ladder: a reply cut off at finish "length" is
#: retried once at the next effort up with a larger budget, then the step is
#: marked failed/llm_truncated. Truncated JSON is never parsed.
EFFORT_LADDER = ["minimal", "low", "medium"]
#: Reasoning headroom per effort: max_tokens = expected visible output +
#: headroom. The ST-04 pilot measured ~2600-3000 reasoning tokens at default
#: effort; minimal is expected to fit in 1500 (the probe verifies live).
EFFORT_HEADROOM = {"minimal": 1500, "low": 3000, "medium": 6000}
#: Expected visible output: one verdict object, or a base plus one verdict
#: object per chunk step (details are capped at 8 words by the prompt).
STEP_VISIBLE_TOKENS = 256
FULL_VISIBLE_BASE_TOKENS = 128
FULL_VISIBLE_PER_STEP_TOKENS = 128
FULL_MAX_TOKENS_CAP = 8192
#: Full-protocol translation goes out in chunks of this many steps: the ST-04
#: pilot showed 12-28 verdict arrays come back truncated or empty (budget hit
#: mid-array), so no single call ever carries more than a chunk.


def _budget_for(visible: int, effort: str) -> int:
    """max_tokens for a call: expected visible output + effort headroom."""
    return min(FULL_MAX_TOKENS_CAP, visible + EFFORT_HEADROOM[effort])


def _escalate(effort: str) -> str:
    """Next rung up the effort ladder (medium stays medium)."""
    return EFFORT_LADDER[min(EFFORT_LADDER.index(effort) + 1,
                             len(EFFORT_LADDER) - 1)]


def _base_effort() -> str:
    """Job-default effort for grounding calls (minimal per the routing table)."""
    return effort_for_job("ground_fallback") or "minimal"


def _usage(response) -> str:
    """Token counts for failure details ('' when the client hides them)."""
    completion = getattr(response, "completion_tokens", None)
    if completion is None:
        return ""
    reasoning = getattr(response, "reasoning_tokens", 0) or 0
    finish = getattr(response, "finish_reason", None)
    suffix = f" finish {finish}" if finish else ""
    return (f" (last call: {completion} completion tokens, "
            f"{reasoning} reasoning{suffix})")


FULL_CHUNK_STEPS = 8

SYSTEM_FULL = """You ground every step of a wet-lab protocol to the lab2rob skill registry.
Reply with exactly one JSON array, no other text, one object per step in order:
[{"idx": <step number>, "skill": <skill name or null>, "out_of_scope_reason": null | "not_implemented" | "missing_skill" | "missing_asset" | "unmodeled_physics", "ambiguous": <true|false>, "detail": <short string>}]
Rules: skill must be an implemented or macro skill from the vocabulary below,
or null. If a step needs a declared (unimplemented) skill, set skill to null
and out_of_scope_reason to "not_implemented". If a step is vague or lacks the
amounts needed to execute, set ambiguous true. Never invent skill names.
Keep every detail under 8 words. Cover every numbered step exactly once."""


def _vocabulary() -> str:
    return ", ".join(sorted(skill_names()))


def build_context(title: str, steps: list, focus: int) -> str:
    """Numbered full-protocol text with the step being grounded marked >>.

    steps is [(idx, text)]; focus is the idx of the target step. Plain
    data in, plain string out, so CLI and GUI callers share it.
    """
    lines = [f'Protocol "{title}" (grounding step {focus}, marked >>):']
    for idx, text in steps:
        mark = ">>" if idx == focus else "  "
        lines.append(f"{mark} {idx}. {text}")
    return "\n".join(lines)


def _parse(text: str) -> dict:
    try:
        data = json.loads(text)
    except ValueError as e:
        raise ValueError(f"not JSON: {e}") from None
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    return _check(data)


def _check(data: dict) -> dict:
    skill = data.get("skill")
    reason = data.get("out_of_scope_reason")
    if skill is not None and not isinstance(skill, str):
        raise ValueError("skill must be a string or null")
    if reason is not None and reason not in REASONS:
        raise ValueError(f"bad out_of_scope_reason {reason!r}")
    if not isinstance(data.get("ambiguous", False), bool):
        raise ValueError("ambiguous must be a boolean")
    return {"skill": skill, "out_of_scope_reason": reason,
            "ambiguous": bool(data.get("ambiguous", False)),
            "detail": str(data.get("detail", ""))}


def ground_step_llm(grounding, client: Lab2RobClient | None = None,
                    context: str | None = None):
    """Resolve a needs_llm grounding via the model; mutates and returns it.

    context is optional full-protocol text (see build_context) so the model
    can resolve references like "the mixture" or "as above".
    """
    client = client or Lab2RobClient()
    prompt = (f"Step: {grounding.source_text}\n"
              f"Quantities: {[q.raw for q in grounding.quantities]}\n"
              f"Vocabulary: {_vocabulary()}")
    if context:
        prompt += ("\nProtocol context (for reference only; ground ONLY the Step above):\n"
                   + context)
    errors: list[str] = []
    last_usage = ""
    base_effort = _base_effort()
    effort = base_effort
    for attempt in range(2):
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt + "".join(errors)}]
        response = client.complete("ground_fallback", messages,
                                   max_tokens=_budget_for(STEP_VISIBLE_TOKENS, effort),
                                   reasoning_effort=effort,
                                   response_format=STEP_RESPONSE_FORMAT)
        last_usage = _usage(response)
        if getattr(response, "finish_reason", None) == "length":
            if attempt == 1:
                grounding.status, grounding.reason, grounding.by = (
                    "failed", "llm_truncated", "llm")
                grounding.detail = (
                    "LLM reply truncated after retry "
                    f"(effort {base_effort}->{effort}); step marked failed"
                    + last_usage)
                return grounding
            effort = _escalate(effort)
            errors = ["\nYour last reply was cut off (truncated). "
                      "Reply with valid JSON only."]
            continue
        try:
            data = _parse(response.text)
            return _apply(grounding, data)
        except ValueError as e:
            errors = [f"\nYour last reply was invalid ({e}). Reply with valid JSON only."]
            continue
    grounding.status, grounding.reason, grounding.by = "ambiguous", "ambiguous_text", "llm"
    grounding.detail = ("LLM fallback failed twice; "
                        + (errors[0] if errors else "") + last_usage)
    return grounding


def _apply(grounding, data: dict):
    skill = data["skill"]
    if data["ambiguous"] or (skill is None and data["out_of_scope_reason"] is None):
        grounding.status, grounding.reason = "ambiguous", "ambiguous_text"
    elif skill is not None:
        resolution = resolve_skill(skill)
        if resolution["scope"] == "in_scope":
            grounding.status, grounding.action, grounding.reason = "grounded", skill, None
        elif resolution["status"] == "declared":
            grounding.status, grounding.reason = "out_of_scope", "not_implemented"
        else:  # invented skill name counts as a validation failure
            raise ValueError(f"unknown skill {skill!r}")
    else:
        grounding.status, grounding.reason = "out_of_scope", data["out_of_scope_reason"]
    grounding.detail, grounding.by = data["detail"], "llm"
    return grounding


def _parse_full(text: str, want_idxs: list) -> list[dict]:
    try:
        data = json.loads(text)
    except ValueError as e:
        raise ValueError(f"not JSON: {e}") from None
    if not isinstance(data, list):
        raise ValueError("not a JSON array")
    if len(data) != len(want_idxs):
        raise ValueError(f"got {len(data)} verdicts for {len(want_idxs)} steps")
    out = []
    for item, want in zip(data, want_idxs):
        if not isinstance(item, dict) or item.get("idx") != want:
            raise ValueError(f"a verdict must carry idx {want}")
        out.append(_check(item))
    return out


def _fresh_groundings(parsed) -> list:
    from .rules import Grounding, extract_quantities, extract_reagents, match_entities
    return [Grounding(idx=p.idx, source_text=p.text, comments=list(p.comments),
                      quantities=extract_quantities(p.text),
                      reagents=extract_reagents(p.text),
                      entities=match_entities(p.text)) for p in parsed]


def _translate_chunk(client, title: str, chunk, groundings) -> None:
    """One full-LLM call over at most FULL_CHUNK_STEPS steps (global idxs).

    Applies verdicts to the chunk's groundings in place; invented skill names
    count as validation failures and trigger the retry, as before.
    """
    numbered = "\n".join(f"{p.idx}. {p.text}" for p in chunk)
    prompt = (f'Protocol "{title}" (steps {chunk[0].idx}-{chunk[-1].idx} of the protocol):\n'
              f"{numbered}\n"
              f"Vocabulary: {_vocabulary()}")
    visible = FULL_VISIBLE_BASE_TOKENS + FULL_VISIBLE_PER_STEP_TOKENS * len(chunk)
    errors: list[str] = []
    last_usage = ""
    base_effort = _base_effort()
    effort = base_effort
    for attempt in range(2):
        messages = [{"role": "system", "content": SYSTEM_FULL},
                    {"role": "user", "content": prompt + "".join(errors)}]
        response = client.complete("ground_fallback", messages,
                                   max_tokens=_budget_for(visible, effort),
                                   reasoning_effort=effort)
        last_usage = _usage(response)
        if getattr(response, "finish_reason", None) == "length":
            if attempt == 1:
                raise _ChunkFailed(
                    "LLM reply truncated after retry "
                    f"(effort {base_effort}->{effort}); chunk marked failed"
                    + last_usage, status="failed", reason="llm_truncated")
            effort = _escalate(effort)
            errors = ["\nYour last reply was cut off (truncated). "
                      "Reply with valid JSON only."]
            continue
        try:
            data = _parse_full(response.text, [p.idx for p in chunk])
            for grounding, verdict in zip(groundings, data):
                _apply(grounding, verdict)
            return
        except ValueError as e:
            errors = [f"\nYour last reply was invalid ({e}). Reply with valid JSON only."]
            continue
    raise _ChunkFailed("Full-LLM translation failed twice; " + errors[0] + last_usage)


class _ChunkFailed(ValueError):
    """A chunk exhausted its retry; carries status, reason, and detail."""

    def __init__(self, detail, status="ambiguous", reason="ambiguous_text"):
        super().__init__(detail)
        self.status = status
        self.reason = reason


def ground_protocol_llm(raw, client: Lab2RobClient | None = None):
    """Translate a whole protocol with the LLM, in chunks (UI-01 full mode).

    Rules-extracted quantities, reagents and entities still ride along on each
    step; only the skill verdict comes from the model. Each chunk gets one
    retry with the validation error, then its steps are marked ambiguous with
    the reason, never silently accepted; a chunk truncated twice is marked
    failed/llm_truncated instead. Uses the ground_fallback job (same
    grounding model; the routing table is spec-pinned, so no new job name).
    """
    from .rules import ProtocolGrounding, parse_steps, scan_description

    client = client or Lab2RobClient()
    parsed = parse_steps(raw.steps_raw)
    if not parsed:
        return ProtocolGrounding(protocol_id=raw.protocol_id, dataset=raw.dataset,
                                 title=raw.title, description=raw.description,
                                 steps=[], description_flags=[], counts={})
    groundings = _fresh_groundings(parsed)
    for start in range(0, len(parsed), FULL_CHUNK_STEPS):
        chunk = parsed[start:start + FULL_CHUNK_STEPS]
        try:
            _translate_chunk(client, raw.title, chunk,
                             groundings[start:start + len(chunk)])
        except _ChunkFailed as failed:
            for grounding in groundings[start:start + len(chunk)]:
                grounding.status, grounding.reason, grounding.by = (
                    failed.status, failed.reason, "llm")
                grounding.detail = str(failed)
    counts: dict[str, int] = {}
    for grounding in groundings:
        key = "llm" if grounding.by == "llm" else grounding.status
        counts[key] = counts.get(key, 0) + 1
    return ProtocolGrounding(protocol_id=raw.protocol_id, dataset=raw.dataset,
                             title=raw.title, description=raw.description,
                             steps=groundings,
                             description_flags=scan_description(raw.description),
                             counts=counts)
