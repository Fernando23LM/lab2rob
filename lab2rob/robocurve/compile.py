"""Protocol -> Robocurve task candidates, through the compiler.

compile_candidates grounds a protocol (rules), plans it for one scene family,
writes the independent judge, and builds one scene per op of that family's
skill. Each accepted op is a task candidate whose parameters come from the
compiler: the program sets what the demo does, the judge sets what is
checked. Rejections keep their reason.
"""

from __future__ import annotations

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
