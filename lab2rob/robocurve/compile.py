"""Protocol -> Robocurve weighing tasks, through the compiler.

compile_weighings grounds a protocol (rules), plans it for the weighing
scene, writes the independent judge, and builds one scene per weighing op.
Each accepted weighing is a task candidate whose target, reagent and
full-score tolerance come from the compiler: the program sets what the demo
does, the judge sets what is checked. Rejections keep their reason.
"""

from __future__ import annotations

from lab2rob.build.scene import SceneRejected, build_scene, weigh_programs
from lab2rob.ground.rules import ground_protocol
from lab2rob.specify.judge import plan_judge
from lab2rob.specify.program import plan_protocol


def compile_weighings(raw, client=None) -> list:
    """One record per weighing op: {"accepted", "reason", "source", "program", "judge", "built"}."""
    grounding = ground_protocol(raw, client=client)
    program = plan_protocol(grounding, family="weigh")
    judge = plan_judge(grounding)
    checks = {js.idx: [p for p in js.predicates if p.pred == "mass_on_balance"]
              for js in judge.steps}
    texts = {g.idx: g.source_text for g in grounding.steps}
    out = []
    for single in weigh_programs(program):
        step = next(s for s in single.steps if s.status == "planned")
        op = step.ops[0]
        grams, reagent = op["args"]["mass_g"], op["args"]["reagent"]
        check = next((p for p in checks.get(step.idx, []) if abs(p.target - grams) < 1e-9), None)
        source = {"protocol_id": raw.protocol_id, "dataset": raw.dataset, "title": raw.title,
                  "url": raw.source_url, "license": raw.license, "step": step.idx,
                  "step_text": texts.get(step.idx, "")[:200], "target_g": grams,
                  "reagent": reagent, "full_g": check.tol if check else None}
        record = {"accepted": False, "reason": "", "source": source, "program": single,
                  "judge": judge, "built": None}
        if check is None:
            record["reason"] = "the judge has no balance check for this mass"
        else:
            try:
                record["built"] = build_scene(single, seed=0)
                record["accepted"] = True
            except SceneRejected as e:
                record["reason"] = f"{e.reason}: {e.detail}"
        out.append(record)
    return out


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
