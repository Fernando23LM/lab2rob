"""Program planner: grounded steps -> executable skill sequences.

For each grounded step the planner re-resolves the skill (never trusting the
grounder blindly), expands macros to implemented primitives, and binds
arguments from quantities and entities. Anything it cannot express becomes
out_of_scope with a category (missing_skill | missing_asset |
missing_physics | ambiguous) — never silently dropped.

Binding rules (v1):
- volumes come from the first volume quantity, normalized to uL;
- times come from time quantities (plurals already folded at extraction);
- vessel roles (source/target/well) bind to entity uids when the text names
  a vessel, else stay None for the scene builder (SC-03) to assign;
- defaults are recorded in op["assumed"], never hidden (overnight = 16 h,
  mix cycles = 3, wait/incubate calls split at the sim's per-call caps).
Actions without a binder (move_to coordinates, runtime reports) are
ambiguous: the planner emits only ops it can fully bind.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pint

from lab2rob.ground.rules import MASS_UNITS, TIME_UNITS, VOLUME_UNITS
from lab2rob.registry import resolve_skill
from lab2rob.schema import Entity, OutOfScopeItem

ureg = pint.UnitRegistry()

#: Grounder reasons -> planner out-of-scope categories.
CATEGORIES = {
    "unknown_skill": "missing_skill",
    "not_implemented": "missing_skill",
    "missing_skill": "missing_skill",
    "missing_asset": "missing_asset",
    "unknown_asset": "missing_asset",
    "unmodeled_physics": "missing_physics",
    "ambiguous_text": "ambiguous",
    "needs_llm": "ambiguous",
    "other_scene": "other_scene",
}

#: Per-call sim caps the planner splits longer holds across.
MAX_WAIT_S = 5.0
MAX_INCUBATE_H = 24.0
OVERNIGHT_H = 16.0
DEFAULT_MIX_CYCLES = 3

_TIME_S = {"s": 1.0, "min": 60.0, "h": 3600.0}
_FOLD_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "ten": 10}

_MACROS_PATH = Path(__file__).resolve().parents[1] / "registry" / "macros.yaml"
_macro_entries: dict | None = None


def reset_macro_cache() -> None:
    """Forget cached macro expansions (the learn loop registers per round)."""
    global _macro_entries
    _macro_entries = None


def _macro_steps(action: str) -> list | None:
    """Step list for a macro from macros.yaml (None when not a macro)."""
    global _macro_entries
    if _macro_entries is None:
        import yaml

        try:
            data = yaml.safe_load(_MACROS_PATH.read_text(encoding="utf-8")) or {}
        except OSError:
            data = {}
        _macro_entries = {m["name"]: m for m in data.get("macros", [])
                          if isinstance(m, dict) and "name" in m}
    entry = _macro_entries.get(action)
    return list(entry["steps"]) if isinstance(entry, dict) and entry.get("steps") else None


def _plan_learned(action: str, grounding) -> list:
    """Data-driven binder for learned macros (ST-02).

    Learned macros bottom out in implemented primitives only, so each step
    binds exactly like a hand-planned primitive. Hand binders in _MACROS
    always win; this covers everything else the registry calls a macro.
    """
    steps = _macro_steps(action)
    if not steps:
        raise _Unbindable(f"no macro expansion for {action!r}")
    ops = []
    for step_name in steps:
        resolution = resolve_skill(step_name)
        if resolution["status"] != "implemented":
            raise _Unbindable(f"learned macro {action!r} step {step_name!r} "
                              f"is not implemented")
        ops += _plan_primitive(step_name, grounding, resolution["entry"])
    return ops


@dataclass
class PlannedStep:
    idx: int
    source_text: str
    status: str = "planned"  # planned | out_of_scope
    reason: str | None = None
    detail: str = ""
    ops: list = field(default_factory=list)  # {"skill","args","at","assumed"}
    entities: list = field(default_factory=list)  # entity uids referenced


@dataclass
class PlannedProgram:
    protocol_id: str
    title: str
    steps: list = field(default_factory=list)
    out_of_scope: list = field(default_factory=list)  # OutOfScopeItem
    entities: list = field(default_factory=list)  # schema.Entity

    def alignment(self) -> list:
        """Source-text <-> program pairs for text-pair export and the GUI."""
        return [{
            "protocol_id": self.protocol_id,
            "step_idx": step.idx,
            "source_text": step.source_text,
            "status": step.status,
            "reason": step.reason,
            "skills": [op["skill"] for op in step.ops],
            "ops": len(step.ops),
        } for step in self.steps]


def save_pairs(program: PlannedProgram, path: str | Path) -> Path:
    """Persist alignment records as JSONL, one step per line."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as handle:
        for pair in program.alignment():
            handle.write(json.dumps(pair) + "\n")
    return path


class _Entities:
    """First-seen uid table (e1, e2, ...) over in-scope entity names."""

    def __init__(self):
        self.uids: dict[str, str] = {}
        self.entries: list[Entity] = []

    def uid(self, name: str) -> str:
        if name not in self.uids:
            self.uids[name] = f"e{len(self.uids) + 1}"
            self.entries.append(Entity(uid=self.uids[name], asset=name))
        return self.uids[name]


def _volume_ul(quantities: list) -> float | None:
    for quantity in quantities:
        if quantity.unit in VOLUME_UNITS:
            try:
                return float(ureg.Quantity(quantity.value, quantity.unit).to("uL").magnitude)
            except Exception:
                continue
    return None


def _time_s(quantities: list) -> float | None:
    for quantity in quantities:
        if quantity.unit in TIME_UNITS and quantity.unit in _TIME_S:
            return float(quantity.value) * _TIME_S[quantity.unit]
    return None


#: In-scope entities that are tools or fixtures, never a liquid's source or
#: target: "pipette to mix" or "use a multichannel pipette" names the tool,
#: and binding a vessel role to it left the op without a well.
NON_VESSELS = frozenset({"pipette", "pipette_tip_200ul", "pipette_tip_1000ul",
                         "tip_rack", "yam_arm", "table", "floor"})


def _in_scope_uids(grounding, table: _Entities) -> list:
    return [table.uid(e["name"]) for e in grounding.entities
            if e.get("scope") == "in_scope"]


def _vessel_uids(grounding, table: _Entities) -> list:
    """In-scope entity uids that can hold liquid roles (tools left out)."""
    return [table.uid(e["name"]) for e in grounding.entities
            if e.get("scope") == "in_scope" and e["name"] not in NON_VESSELS]


class _Unbindable(Exception):
    """Raised when no honest argument binding exists; step becomes ambiguous."""


def _op(skill: str, args=None, at=None, assumed=None) -> dict:
    return {"skill": skill, "args": args or {}, "at": at, "assumed": assumed or {}}


def _require_volume(grounding) -> float:
    volume = _volume_ul(grounding.quantities)
    if volume is None:
        raise _Unbindable("no volume mentioned; cannot bind volume_ul")
    return volume


def _plan_transfer(grounding, uids: list) -> list:
    volume = _require_volume(grounding)
    # Two named vessels: source then target. One: it is where the liquid goes
    # ("Add 50 uL of stop solution to each well of the plate"), as the judge
    # reads it too; the reagent comes from a stock the scene provides.
    source = uids[0] if len(uids) > 1 else None
    target = uids[1] if len(uids) > 1 else (uids[0] if uids else None)
    assumed = {}
    if source is None:
        assumed["source"] = "no vessel in text; scene builder assigns"
    if target is None:
        assumed["target"] = "no vessel in text; scene builder assigns"
    return [_op("aspirate", {"volume_ul": volume}, at=source, assumed=assumed),
            _op("dispense", {"volume_ul": volume}, at=target, assumed=assumed)]


def _mix_cycles(text: str) -> tuple:
    match = re.search(r"(\d+)\s*(?:times|cycles?)", text.lower())
    if match:
        return int(match.group(1)), {}
    return DEFAULT_MIX_CYCLES, {"cycles": f"default {DEFAULT_MIX_CYCLES}; text gives no count"}


def _plan_mix(grounding, uids: list) -> list:
    volume = _require_volume(grounding)
    cycles, assumed = _mix_cycles(grounding.source_text)
    well = uids[0] if uids else None
    if well is None:
        assumed["well"] = "no vessel in text; scene builder assigns"
    ops = []
    for _ in range(cycles):
        ops += [_op("aspirate", {"volume_ul": volume}, at=well, assumed=assumed),
                _op("dispense", {"volume_ul": volume}, at=well, assumed=assumed)]
    return ops


def _series_points(text: str) -> int:
    low = text.lower()
    if re.search(r"\d+\s*-\s*\d+\s*-?\s*points?", low):
        raise _Unbindable("dilution series states a point range, not an exact count")
    match = re.search(r"(\d+)\s*-?\s*points?", low)
    if not match:
        raise _Unbindable("dilution series layout not specified")
    return int(match.group(1))


def _fold_factor(text: str) -> int:
    low = text.lower()
    match = re.search(r"(\d+)\s*-?\s*fold", low)
    if match:
        return int(match.group(1))
    match = re.search(r"(two|three|four|five|ten)\s*-?\s*fold", low)
    if match:
        return _FOLD_WORDS[match.group(1)]
    raise _Unbindable("dilution factor not specified")


def _plan_serial(grounding, uids: list) -> list:
    volume = _require_volume(grounding)
    points = _series_points(grounding.source_text)
    fold = _fold_factor(grounding.source_text)
    assumed = {"series": (f"{points}-point {fold}-fold; well layout and prefill "
                          "assigned downstream")}
    ops = []
    for _ in range(max(points - 1, 0)):
        ops += [_op("aspirate", {"volume_ul": volume}, assumed=assumed),
                _op("dispense", {"volume_ul": volume}, assumed=assumed)]
    return ops


def _plan_incubate(grounding) -> list:
    seconds = _time_s(grounding.quantities)
    assumed: dict = {}
    if seconds is None:
        if re.search(r"\bovernight\b", grounding.source_text.lower()):
            seconds, assumed = OVERNIGHT_H * 3600.0, {
                "hours": f"overnight defaulted to {OVERNIGHT_H:g} h"}
        else:
            raise _Unbindable("no duration mentioned; cannot bind hours")
    hours = seconds / 3600.0
    chunks, rest = int(hours // MAX_INCUBATE_H), hours % MAX_INCUBATE_H
    ops = [_op("incubate", {"hours": MAX_INCUBATE_H}, assumed=dict(assumed))
           for _ in range(chunks)]
    if rest > 1e-9:
        ops.append(_op("incubate", {"hours": round(rest, 6)}, assumed=assumed))
    return ops or [_op("incubate", {"hours": 0.0}, assumed=assumed)]


def _plan_wait(grounding) -> list:
    seconds = _time_s(grounding.quantities)
    if seconds is None:
        raise _Unbindable("no duration mentioned; cannot bind seconds")
    chunks, rest = int(seconds // MAX_WAIT_S), seconds % MAX_WAIT_S
    ops = [_op("wait", {"seconds": MAX_WAIT_S}) for _ in range(chunks)]
    if rest > 1e-9:
        ops.append(_op("wait", {"seconds": round(rest, 6)}))
    return ops or [_op("wait", {"seconds": 0.0})]


#: Implemented actions the planner can bind arguments for. Learned macro
#: expansions (ST-02) must stay inside this set; anything else cannot
#: compile no matter how sensible it looks.
BINDABLE_PRIMITIVES = frozenset({"aspirate", "dispense", "incubate", "wait",
                                 "read_plate"})


def _plan_primitive(name: str, grounding, entry: dict | None) -> list:
    if name in ("aspirate", "dispense"):
        return [_op(name, {"volume_ul": _require_volume(grounding)})]
    if name == "incubate":
        return _plan_incubate(grounding)
    if name == "wait":
        return _plan_wait(grounding)
    if name == "read_plate":
        return [_op(name)]
    if entry is not None and entry.get("kind") == "query":
        return [_op(name)]
    raise _Unbindable(f"no argument binder for {name!r}")


def _clean_reagent(name: str) -> str:
    """'sodium chloride (NaCl' -> 'sodium chloride (NaCl)': close or drop a cut bracket."""
    name = " ".join(str(name).split()).strip(" ,;:")
    if name.count("(") > name.count(")"):
        head, _, inner = name.rpartition("(")
        name = f"{head.strip()} ({inner.strip()})" if inner.strip() else head.strip()
    return name


def _plan_weigh(grounding, uids: list) -> list:
    """One weigh_sample op per stated mass, paired with the reagents in order.

    "weigh out 0.9 g sodium chloride and 4.0 g PEG 8000" weighs two things;
    each becomes its own op (the weighing scene holds one vial, so each
    compiles as its own task). A reagent the text does not pair with a mass
    stays None, for the scene builder to reject rather than guess.
    """
    masses = []
    for quantity in grounding.quantities:
        if quantity.unit in MASS_UNITS:
            try:
                masses.append(float(ureg.Quantity(quantity.value, quantity.unit).to("g").magnitude))
            except Exception:
                continue
    if not masses:
        raise _Unbindable("no mass mentioned; cannot bind mass_g")
    reagents = [_clean_reagent(r) for r in grounding.reagents]
    ops = []
    for i, grams in enumerate(masses):
        if len(reagents) == len(masses):
            reagent, assumed = reagents[i], {}
        elif len(reagents) == 1 and len(masses) == 1:
            reagent, assumed = reagents[0], {}
        else:
            reagent = None
            assumed = {"reagent": f"{len(masses)} masses, {len(reagents)} reagents in the text: unpaired"}
        ops.append(_op("weigh_sample", {"mass_g": round(grams, 6), "reagent": reagent},
                       assumed=assumed))
    return ops


_SPEED = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(rpm|r\.p\.m\.|x\s*g|×\s*g|rcf|g)\b", re.I)
_COUNTS = {"two": 2, "three": 3, "four": 4, "six": 6, "eight": 8}
_TUBES = re.compile(r"\b(\d+|two|three|four|six|eight)\s+(?:sample\s+|micro\w*\s+|1\.5\s*ml\s+)?tubes\b", re.I)
DEFAULT_TUBES = 2


def _plan_centrifuge(grounding, uids: list) -> list:
    """Load the tubes balanced and shut the lid; speed and time are recorded.

    Nothing spins in the scene, so neither is required. The tube count comes
    from the text ("spin the two tubes") or defaults to DEFAULT_TUBES, recorded.
    """
    text = grounding.source_text
    speed = _SPEED.search(text)
    tubes = _TUBES.search(text)
    assumed = {}
    if tubes:
        word = tubes.group(1).lower()
        count = _COUNTS.get(word) or int(word)
    else:
        count = DEFAULT_TUBES
        assumed["tubes"] = f"the text gives no tube count: {DEFAULT_TUBES} sample tubes"
    args = {"tubes": count, "time_s": _time_s(grounding.quantities),
            "speed": " ".join(speed.group(0).split()) if speed else None}
    return [_op("centrifuge", args, assumed=assumed)]


_MACROS = {"transfer": _plan_transfer, "mix_by_pipetting": _plan_mix,
           "serial_dilution": _plan_serial, "weigh_sample": _plan_weigh,
           "centrifuge": _plan_centrifuge}

#: Skills each scene family runs. A step whose ops belong to another family is
#: out of scope for this scene ("other_scene") and compiles as its own task.
FAMILY_SKILLS = {"weigh": frozenset({"weigh_sample"}), "centrifuge": frozenset({"centrifuge"})}
FAMILIES = ("pipetting", "weigh", "centrifuge")


def _scene_of(step, action: str | None) -> str | None:
    """The scene a step belongs to, by its grounded action (or its ops)."""
    skills = {op["skill"] for op in step.ops} | ({action} if action else set())
    for family, owned in FAMILY_SKILLS.items():
        if skills & owned:
            return family
    return "pipetting" if step.status == "planned" and step.ops else None


def _limit_to_family(steps: list, groundings: list, family: str) -> None:
    """Steps of another scene, planned or not, are out of scope for this one."""
    for step, grounding in zip(steps, groundings):
        scene = _scene_of(step, getattr(grounding, "action", None))
        if scene is None or scene == family:
            continue
        if step.status == "planned" or step.reason == "ambiguous":
            step.status, step.reason = "out_of_scope", "other_scene"
            step.detail = f"runs in the {scene} scene; compiled as its own task"
            step.ops = []


def _check_macro_bottoms_out(name: str, entry: dict | None, seen=None) -> None:
    """Every macro step must resolve to implemented primitives (gen_skills
    validates this statically; the planner re-checks the loaded registry)."""
    seen = seen or set()
    if name in seen:
        raise _Unbindable(f"macro cycle at {name!r}")
    seen = seen | {name}
    for sub in (entry or {}).get("steps", []):
        resolution = resolve_skill(sub)
        if resolution["scope"] != "in_scope":
            raise _Unbindable(f"macro {name!r} references out-of-scope {sub!r}")
        if resolution["status"] == "macro":
            _check_macro_bottoms_out(sub, resolution["entry"], seen)


def plan_step(grounding, table: _Entities | None = None) -> PlannedStep:
    """Plan one grounded step; unplannable steps become categorized out_of_scope."""
    table = table if table is not None else _Entities()
    step = PlannedStep(idx=grounding.idx, source_text=grounding.source_text)
    if grounding.status != "grounded":
        step.status = "out_of_scope"
        if grounding.status == "needs_llm":
            # Never mapped to any skill (no LLM pass ran): ambiguous, not missing.
            step.reason = "ambiguous"
            step.detail = grounding.detail or "unresolved: no LLM pass ran"
            return step
        reason = grounding.reason or grounding.status
        step.reason = CATEGORIES.get(reason, "ambiguous")
        step.detail = grounding.detail or reason
        return step
    resolution = resolve_skill(grounding.action or "")
    if resolution["scope"] != "in_scope":
        step.status = "out_of_scope"
        step.reason = CATEGORIES.get(resolution["reason"], "missing_skill")
        step.detail = f"{grounding.action} is {resolution['status'] or 'unknown'}"
        return step
    for entity in grounding.entities:
        if entity.get("scope") == "out_of_scope":
            step.status = "out_of_scope"
            step.reason = "missing_asset"
            step.detail = f"entity {entity['name']} is not buildable"
            return step
    step.entities = _in_scope_uids(grounding, table)
    try:
        if resolution["status"] == "macro":
            _check_macro_bottoms_out(grounding.action, resolution["entry"])
            binder = _MACROS.get(grounding.action)
            if binder is None:
                step.ops = _plan_learned(grounding.action, grounding)
            else:
                step.ops = binder(grounding, _vessel_uids(grounding, table))
        else:
            step.ops = _plan_primitive(grounding.action, grounding, resolution["entry"])
    except _Unbindable as exc:
        step.status, step.reason, step.detail = "out_of_scope", "ambiguous", str(exc)
        step.ops = []
    return step


def _load_empty_dispenses(steps: list) -> None:
    """A dispense the pipette cannot cover draws the shortfall first.

    "Dispense 25 uL into each spot" or "Add 150 uL of block" plans as a lone
    dispense; run as written, an empty pipette delivers nothing. Tracking
    what the pipette holds across the program, each such dispense gets an
    aspirate of the missing volume from a scene stock just before it,
    recorded in the op's assumed fields like every other default.
    """
    held = 0.0
    for step in steps:
        if step.status != "planned":
            continue
        ops = []
        for op in step.ops:
            volume = op["args"].get("volume_ul")
            if op["skill"] == "dispense" and isinstance(volume, (int, float)):
                short = float(volume) - held
                if short > 1e-9:
                    ops.append(_op("aspirate", {"volume_ul": round(short, 6)}, at=None,
                                   assumed={"source": "dispense with an empty pipette: "
                                                      "reagent drawn from a scene stock"}))
                    held += short
                held = max(0.0, held - float(volume))
            elif op["skill"] == "aspirate" and isinstance(volume, (int, float)):
                held += float(volume)
            ops.append(op)
        step.ops = ops


def plan_protocol(protocol_grounding, family: str = "pipetting") -> PlannedProgram:
    """Plan every step of a grounded protocol into one program for one scene.

    family picks the scene the program runs in: "pipetting" (the default:
    the protein and culture benches), "weigh" (the balance bench) or "centrifuge". Steps
    another scene runs are out of scope here with reason "other_scene".
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    table = _Entities()
    steps = [plan_step(grounding, table) for grounding in protocol_grounding.steps]
    _limit_to_family(steps, list(protocol_grounding.steps), family)
    _load_empty_dispenses(steps)
    # Entities only another scene's steps named stay out of this scene.
    used = {uid for step in steps if step.reason != "other_scene" for uid in step.entities}
    table.entries = [e for e in table.entries if e.uid in used]
    for step in steps:
        if step.reason == "other_scene":
            step.entities = [uid for uid in step.entities if uid in used]
    out_of_scope = [OutOfScopeItem(step=s.idx, reason=s.reason or "ambiguous",
                                   detail=s.detail)
                    for s in steps if s.status != "planned"]
    return PlannedProgram(protocol_id=protocol_grounding.protocol_id,
                          title=protocol_grounding.title, steps=steps,
                          out_of_scope=out_of_scope, entities=table.entries)
