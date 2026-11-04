"""Protocol -> Robocurve task candidates, through the compiler.

compile_candidates grounds a protocol (rules), plans it for one scene family,
writes the independent judge, and builds one scene per op of that family's
skill. Each accepted op is a task candidate whose parameters come from the
compiler: the program sets what the demo does, the judge sets what is
checked. Rejections keep their reason.
"""

from __future__ import annotations

import re

from lab2rob.build.scene import SceneRejected, build_scene, skill_programs
from lab2rob.ground.rules import ground_protocol
from lab2rob.specify.judge import plan_judge
from lab2rob.specify.program import plan_protocol

#: Scene family -> the skill whose ops become tasks.
FAMILY_SKILL = {"weigh": "weigh_sample", "centrifuge": "centrifuge"}


def _source(raw, step, texts, op, judge_preds) -> dict:
    src = {"protocol_id": raw.protocol_id, "dataset": raw.dataset, "title": raw.title,
           "url": raw.source_url, "license": raw.license, "step": step.idx,
           "step_text": texts.get(step.idx, "")[:200]}
    args = op["args"]
    if op["skill"] == "weigh_sample":
        check = next((p for p in judge_preds if p.pred == "mass_on_balance"
                      and abs(p.target - args["mass_g"]) < 1e-9), None)
        src.update(target_g=args["mass_g"], reagent=args["reagent"], full_g=check.tol if check else None,
                   checked=check is not None)
    elif op["skill"] == "centrifuge":
        preds = {p.pred for p in judge_preds}
        src.update(tubes=args["tubes"], speed=args.get("speed"), time_s=args.get("time_s"),
                   assumed=dict(op.get("assumed", {})),
                   checked={"rotor_balanced", "lid_closed"} <= preds)
    return src


def compile_candidates(raw, family: str, client=None) -> list:
    """One record per op: {"accepted", "reason", "source", "program", "judge", "built"}."""
    skill = FAMILY_SKILL[family]
    grounding = ground_protocol(raw, client=client)
    program = plan_protocol(grounding, family=family)
    judge = plan_judge(grounding)
    preds = {js.idx: js.predicates for js in judge.steps}
    texts = {g.idx: g.source_text for g in grounding.steps}
    out = []
    for single in skill_programs(program, skill):
        step = next(s for s in single.steps if s.status == "planned")
        source = _source(raw, step, texts, step.ops[0], preds.get(step.idx, []))
        record = {"accepted": False, "reason": "", "source": source, "program": single,
                  "judge": judge, "built": None}
        if not source.pop("checked"):
            record["reason"] = "the judge has no check for this step"
        else:
            try:
                record["built"] = build_scene(single, seed=0)
                record["accepted"] = True
            except SceneRejected as e:
                record["reason"] = f"{e.reason}: {e.detail}"
        out.append(record)
    return out


# ------------------------------------------------------------------ prep_spin
#: The prep bench sets out this many reagent troughs (plus water for the balance tube).
PREP_MAX_REAGENTS = 2
PREP_VOLUME_UL = (2.0, 1000.0)    # one addition: above a pipette's floor, inside a 1.5 mL tube
PREP_TOTAL_UL = 1000.0
PIPETTING_OPS = {"aspirate", "dispense"}


def _is_addition(step) -> bool:
    return (step.status == "planned" and bool(step.ops)
            and all(op["skill"] in PIPETTING_OPS for op in step.ops)
            and any(op["skill"] == "dispense" for op in step.ops))


#: A "reagent" the rules read that is really labware or a unit phrase: not a liquid to pour.
NOT_REAGENT = re.compile(r"\b(column|tube|plate|well|flask|vial|dish|filter|bead|cell|pellet|"
                         r"supernatant|sample|per)s?\b", re.I)


_TRAILING = re.compile(r"\s+(?:and|then|to|for|at|in|on|with|into|until)\b.*$", re.I)


def _reagents(grounding) -> list:
    """Reagent names the rules read, without stray punctuation or the words that follow a name."""
    names = [_TRAILING.sub("", str(n)).strip(" ()[],;.:") for n in getattr(grounding, "reagents", []) or []]
    return list(dict.fromkeys(n for n in names if re.search(r"[A-Za-z]{2}", n)))


def _volumes(grounding) -> int:
    from lab2rob.ground.rules import VOLUME_UNITS
    return sum(1 for q in getattr(grounding, "quantities", []) if q.unit in VOLUME_UNITS)


def compile_prep_spin(raw, client=None) -> list:
    """Multi-step candidates: the pipetting steps that run straight into a centrifugation.

    For each centrifuge op, the run of addition steps immediately before it (planned in
    the pipetting scene) becomes one task: add each reagent to the sample tube, make a
    balance tube to match, load both opposite and shut the lid. Every addition needs one
    named reagent and the judge's volume check; the spin needs the judge's rotor checks
    and must pass the centrifuge scene's own rejections (tubes it can take).
    """
    grounding = ground_protocol(raw, client=client)
    pipetting = {s.idx: s for s in plan_protocol(grounding, family="pipetting").steps}
    spins = plan_protocol(grounding, family="centrifuge")
    judge = plan_judge(grounding)
    preds = {js.idx: js.predicates for js in judge.steps}
    texts = {g.idx: g.source_text for g in grounding.steps}
    grounded = {g.idx: g for g in grounding.steps}
    out = []
    for single in skill_programs(spins, "centrifuge"):
        spin = next(s for s in single.steps if s.status == "planned")
        run, i = [], spin.idx - 1
        while i in pipetting and _is_addition(pipetting[i]):
            run.insert(0, pipetting[i])
            i -= 1
        if not run:
            continue  # a centrifugation on its own: the centrifuge family's task
        source = _source(raw, spin, texts, spin.ops[0], preds.get(spin.idx, []))
        record = {"accepted": False, "reason": "", "source": source, "program": single,
                  "judge": judge, "built": None, "additions": [s.idx for s in run]}
        out.append(record)
        if not source.pop("checked"):
            record["reason"] = "the judge has no check for the centrifugation"
            continue
        if len(run) > PREP_MAX_REAGENTS:
            record["reason"] = (f"over_capacity: {len(run)} additions before the spin; the bench sets "
                                f"out {PREP_MAX_REAGENTS} reagent troughs")
            continue
        additions, problem = [], ""
        for step in run:
            names = _reagents(grounded[step.idx])
            volume = round(sum(op["args"].get("volume_ul", 0.0) for op in step.ops
                               if op["skill"] == "dispense"), 6)
            check = next((p for p in preds.get(step.idx, []) if p.pred == "volume"
                          and abs(p.target - volume) < 1e-6), None)
            if _volumes(grounded[step.idx]) > 1:
                problem = f"ambiguous: step {step.idx} gives {_volumes(grounded[step.idx])} volumes"
            elif len(names) == 1 and NOT_REAGENT.search(names[0]):
                problem = f"ambiguous: step {step.idx} adds {names[0]!r}, not a named reagent"
            elif len(names) != 1:
                problem = f"ambiguous: step {step.idx} names {len(names)} reagents ({', '.join(names) or 'none'})"
            elif check is None:
                problem = f"the judge has no volume check for step {step.idx}"
            elif not PREP_VOLUME_UL[0] <= volume <= PREP_VOLUME_UL[1]:
                problem = (f"out_of_range: step {step.idx} adds {volume:g} uL "
                           f"(the bench takes {PREP_VOLUME_UL[0]:g} to {PREP_VOLUME_UL[1]:g} uL)")
            if problem:
                break
            additions.append({"name": names[0], "volume_ul": volume, "tol_ul": float(check.tol),
                              "step": step.idx, "step_text": texts.get(step.idx, "")[:200]})
        total = sum(a["volume_ul"] for a in additions)
        if not problem and total > PREP_TOTAL_UL:
            problem = f"over_capacity: {total:g} uL in the sample tube (at most {PREP_TOTAL_UL:g})"
        if problem:
            record["reason"] = problem
            continue
        source["additions"] = additions
        try:
            record["built"] = build_scene(single, seed=0)
            record["accepted"] = True
        except SceneRejected as e:
            record["reason"] = f"{e.reason}: {e.detail}"
    return out


#: Every family a task can be compiled for; prep_spin spans two scenes (pipetting, then a spin).
FAMILIES = (*FAMILY_SKILL, "prep_spin")


def compile_family(raw, family: str, client=None) -> list:
    if family == "prep_spin":
        return compile_prep_spin(raw, client)
    return compile_candidates(raw, family, client)


def compile_weighings(raw, client=None) -> list:
    return compile_candidates(raw, "weigh", client)


def load_raw(protocol_id: str):
    """A RawProtocol by id: BioProBench corpus, cached protocols.io, or BioProt tier 1."""
    if protocol_id.startswith(("Protocol.io-", "Protocol-exchange-")):
        from lab2rob.ingest.bioprobench import download, load_corpus
        protocols, _ = load_corpus(download())
        for protocol in protocols:
            if protocol.raw.protocol_id == protocol_id:
                return protocol.raw
        raise KeyError(f"{protocol_id} not in the BioProBench corpus")
    try:
        from lab2rob.ingest.protocols_io import load_cached_protocol
        return load_cached_protocol(protocol_id)
    except KeyError:
        from lab2rob.ingest.bioprot import load_bioprot_id
        return load_bioprot_id(protocol_id)
