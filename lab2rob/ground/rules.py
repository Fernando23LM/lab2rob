"""Deterministic grounding: noise strip, pint quantities, lexicon verbs.

Pipeline per step: clean -> quantities -> reagents -> verb -> verdict.
Verdict priority: ambiguity patterns > pH/thermal patterns > verb mapping
(declared -> not_implemented) > pipette-capacity/vessel checks > grounded.
Anything left unresolved is needs_llm (see llm_ground.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pint

from lab2rob.registry import (lexicon_index, load_lexicon, resolve_asset, resolve_skill,
                              shared_lexicon)

ureg = pint.UnitRegistry()

#: Largest open vessel in the sim (culture reservoir, ~36 mL). Steps needing
#: more than this in one vessel are missing_asset.
MAX_VESSEL_ML = 36.0
#: Largest single pipette transfer (culture pipette, 1000 uL). More than this
#: in one action means no pour primitive: missing_skill.
MAX_PIPETTE_UL = 1000.0

MASS_UNITS = {"g", "mg", "ug", "ng", "kg"}
VOLUME_UNITS = {"nl", "ul", "ml", "l", "dl"}
LENGTH_UNITS = {"m", "mm", "cm", "nm"}
CONC_UNITS = {"M", "mM", "uM", "nM", "mol", "mmol", "mg/ml", "g/l", "ug/ml"}
PERCENT_UNITS = {"%", "percent"}
TIME_UNITS = {"s", "min", "h", "seconds", "minutes", "hours"}
#: Spelled-out time units fold to the canonical short form at extraction.
_PLURAL_TIME = {"seconds": "s", "minutes": "min", "hours": "h"}
KNOWN_UNITS = MASS_UNITS | VOLUME_UNITS | LENGTH_UNITS | CONC_UNITS | PERCENT_UNITS | TIME_UNITS

_QTY = re.compile(
    r"(?P<approx>[~≈])?\s*(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>[A-Za-zµμ°%]+(?:/[A-Za-zµμ°%]+)?(?:\s*\([^)]*\))?)"
)
_STEP = re.compile(r"^(\d+)\.\s+(.*)$")
_NOISE_FULL = {"none", "reagents"}
_NOISE_AMOUNT = re.compile(r"^\d+amount$", re.IGNORECASE)
_DELIM = re.compile(r"\s+(?:in|of|with|and|to|at|into|from|for)\s+|\s*,\s*|\s*;\s*|\.\s+|\s*\.\s*$")
_OF_NAME = re.compile(r"\bof\s+([A-Z][\w.\-]*(?:\s+[a-zA-Z][\w.\-]*)*)")
_WITH_NAME = re.compile(r"\bwith\s+([A-Z][\w.\-]*(?:\s+[a-zA-Z][\w.\-]*)*)")
_TRAILING_PREP = re.compile(r"\s+(?:in|of|with|and|to|at|into|from|for)$")
_LEADING_QTY = re.compile(r"^[\d.~≈]+\s*[A-Za-zµμ°/%]+\s*")
_TEMP_STORAGE = re.compile(r"store at|-?\d+\s*°c|heat|warm|freeze|room temperature", re.IGNORECASE)


@dataclass
class ParsedStep:
    idx: int
    text: str
    comments: list = field(default_factory=list)


@dataclass
class Quantity:
    raw: str
    value: float
    unit: str
    approximate: bool
    pint: str | None  # None when pint cannot parse the unit


@dataclass
class Grounding:
    idx: int
    source_text: str
    comments: list = field(default_factory=list)
    quantities: list = field(default_factory=list)
    reagents: list = field(default_factory=list)
    entities: list = field(default_factory=list)
    action: str | None = None
    status: str = "needs_llm"  # grounded | out_of_scope | ambiguous | failed | needs_llm
    reason: str | None = None
    detail: str = ""
    by: str = "none"  # rules | llm | none


@dataclass
class ProtocolGrounding:
    protocol_id: str
    title: str
    description: str = ""
    steps: list = field(default_factory=list)
    description_flags: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    dataset: str = ""


def parse_steps(steps_raw: str) -> list[ParsedStep]:
    steps, pending = [], []
    for raw_line in steps_raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower() in _NOISE_FULL or _NOISE_AMOUNT.match(line):
            continue
        if line == "NOTES" or line.startswith("NOTES "):
            comment = line[5:].strip() if len(line) > 5 else line
            (steps[-1].comments if steps else pending).append(comment)
            continue
        match = _STEP.match(line)
        if match:
            steps.append(ParsedStep(int(match.group(1)), match.group(2).strip(), pending))
            pending = []
        elif steps:
            steps[-1].text += " " + line
        else:
            pending.append(line)
    return steps


def _normalize_unit(unit: str) -> str:
    base = re.split(r"\s*\(", unit, maxsplit=1)[0]
    base = base.replace("µ", "u").replace("μ", "u")
    if base in CONC_UNITS:  # M/mM/uM/nM are case-sensitive (molar vs meter)
        return base
    return _PLURAL_TIME.get(base.lower(), base.lower())


def extract_quantities(text: str) -> list[Quantity]:
    out = []
    for match in _QTY.finditer(text):
        unit = _normalize_unit(match.group("unit"))
        if unit not in KNOWN_UNITS:
            continue
        value = float(match.group("value"))
        try:
            pint_str = str(ureg.Quantity(value, unit))
        except Exception:
            pint_str = None
        out.append(Quantity(raw=match.group(0).strip(), value=value, unit=unit,
                            approximate=match.group("approx") is not None,
                            pint=pint_str))
    return out


def _clean_name(raw: str) -> str:
    name = _LEADING_QTY.sub("", raw.strip(" .")).strip()
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
    while True:
        shorter = _TRAILING_PREP.sub("", name).strip()
        if shorter == name:
            return shorter
        name = shorter


def extract_reagents(text: str) -> list[str]:
    names = []
    for match in _QTY.finditer(text):
        if _normalize_unit(match.group("unit")) not in KNOWN_UNITS:
            continue
        head = _DELIM.split(text[match.end():], maxsplit=1)[0]
        head = _clean_name(head)
        if head and len(head) <= 40 and not head[0].isdigit():
            names.append(head)
    for pattern in (_OF_NAME, _WITH_NAME):
        for match in pattern.finditer(text):
            head = _clean_name(match.group(1))
            if head and len(head) <= 40:
                names.append(head)
    return list(dict.fromkeys(names))


def _lexicon_kinds() -> dict:
    kinds = {}
    for entry in shared_lexicon():  # read only
        kinds.setdefault(entry["canonical"], entry.get("kind", "action"))
    return kinds


def match_verb(text: str) -> tuple | None:
    """Earliest action-kind lexicon phrase in the text (longest wins ties)."""
    index, kinds = lexicon_index(), _lexicon_kinds()
    low = text.lower()
    best = None
    for phrase, canonical in index.items():
        if kinds.get(canonical) == "object":
            continue
        match = re.search(r"\b" + re.escape(phrase) + r"\b", low)
        if match:
            key = (match.start(), -len(phrase))
            if best is None or key < best[0]:
                best = (key, phrase, canonical)
    return (best[1], best[2]) if best else None


def match_entities(text: str) -> list:
    """Object-kind lexicon hits with their registry resolution."""
    index, kinds = lexicon_index(), _lexicon_kinds()
    low = text.lower()
    found = {}
    for phrase, canonical in sorted(index.items()):
        if kinds.get(canonical) != "object":
            continue
        if re.search(r"\b" + re.escape(phrase) + r"\b", low):
            resolution = resolve_asset(canonical)
            found[canonical] = {"name": canonical,
                                **{k: v for k, v in resolution.items() if k != "entry"}}
    return list(found.values())


def _volume_ul(quantity: Quantity) -> float | None:
    if quantity.unit not in VOLUME_UNITS or quantity.pint is None:
        return None
    try:
        return float(ureg.Quantity(quantity.value, quantity.unit).to("uL").magnitude)
    except Exception:
        return None


def _has_mass_or_volume(quantities: list) -> bool:
    return any(q.unit in MASS_UNITS | VOLUME_UNITS for q in quantities)


def ground_step(parsed: ParsedStep) -> Grounding:
    text = parsed.text
    low = text.lower()
    quantities = extract_quantities(text)
    grounding = Grounding(idx=parsed.idx, source_text=text, comments=list(parsed.comments),
                          quantities=quantities, reagents=extract_reagents(text),
                          entities=match_entities(text))
    if "ph" in low and "adjust" in low:
        return _out(grounding, "unmodeled_physics", "pH adjustment/titration is not modeled")
    volumes = [v for q in quantities if (v := _volume_ul(q)) is not None]
    if re.search(r"\baliquot\b", low) and not volumes:
        return _amb(grounding, "aliquot size unspecified")
    if re.search(r"\bprepare\b", low) and not _has_mass_or_volume(quantities):
        return _amb(grounding, "sub-protocol without amounts")
    hit = match_verb(text)
    if hit is None:
        grounding.reason, grounding.detail = "unknown_skill", "no lexicon verb matched"
        return grounding
    resolution = resolve_skill(hit[1])
    grounding.action = hit[1]
    if resolution["scope"] == "out_of_scope":
        return _out(grounding, resolution["reason"], f"{hit[1]} is declared, not implemented")
    if grounding.action in {"transfer", "aspirate", "dispense", "mix_by_pipetting",
                            "serial_dilution", "dilute"} and volumes and max(volumes) > MAX_PIPETTE_UL:
        return _out(grounding, "missing_skill",
                    f"{max(volumes) / 1000:.2g} mL exceeds pipette capacity (1 mL); no pour primitive")
    if volumes and max(volumes) / 1000 > MAX_VESSEL_ML:
        return _out(grounding, "missing_asset",
                    f"no vessel holds {max(volumes) / 1000:.2g} mL "
                    f"(largest: {MAX_VESSEL_ML:.0f} mL culture reservoir)")
    for entity in grounding.entities:
        if entity["scope"] == "out_of_scope":
            return _out(grounding, "missing_asset", f"entity {entity['name']} is not buildable")
    grounding.status, grounding.by = "grounded", "rules"
    return grounding


def _out(grounding: Grounding, reason: str, detail: str) -> Grounding:
    grounding.status, grounding.reason, grounding.detail, grounding.by = (
        "out_of_scope", reason, detail, "rules")
    return grounding


def _amb(grounding: Grounding, detail: str) -> Grounding:
    grounding.status, grounding.reason, grounding.detail, grounding.by = (
        "ambiguous", "ambiguous_text", detail, "rules")
    return grounding


def scan_description(description: str) -> list:
    """Thermal/storage needs in free text are out of scope (no such equipment)."""
    return [{"text": match.group(0), "reason": "unmodeled_physics",
             "detail": "thermal/storage equipment is not modeled"}
            for match in _TEMP_STORAGE.finditer(description or "")]


def ground_protocol(raw, client=None, full_context=False) -> ProtocolGrounding:
    """Ground every step rules-first; LLM fallback only for needs_llm steps.

    full_context shows the fallback model the whole protocol (built once via
    build_context) instead of the lone step; default False keeps calls cheap.
    """
    from .llm_ground import build_context, ground_step_llm

    parsed = parse_steps(raw.steps_raw)
    steps = [ground_step(p) for p in parsed]
    context_for = {}
    if full_context and client is not None and any(s.status == "needs_llm" for s in steps):
        numbered = [(p.idx, p.text) for p in parsed]
        context_for = {s.idx: build_context(raw.title, numbered, s.idx)
                       for s in steps if s.status == "needs_llm"}
    for i, grounding in enumerate(steps):
        if grounding.status == "needs_llm" and client is not None:
            steps[i] = ground_step_llm(grounding, client, context_for.get(grounding.idx))
    counts: dict[str, int] = {}
    for grounding in steps:
        key = "llm" if grounding.by == "llm" else (
            "rules" if grounding.status == "grounded" else grounding.status)
        counts[key] = counts.get(key, 0) + 1
    return ProtocolGrounding(protocol_id=raw.protocol_id, dataset=raw.dataset,
                             title=raw.title, description=raw.description,
                             steps=steps,
                             description_flags=scan_description(raw.description),
                             counts=counts)
