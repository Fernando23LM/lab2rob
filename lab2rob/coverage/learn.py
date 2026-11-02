"""ST-02: learn missing skills from the coverage-gap queue.

Propose / validate / smoke loop in the evolve-loop pattern: an LLM
callable proposes a macro expansion for a queued capability, a tentative
registration plus a real pipeline smoke test (ground the queued example,
plan, build, certify) scores it, and the first passing expansion is kept
and registered. Rounds live as plain files under runs/learn_<capability>/
so a run can stop and resume.

Why not drive biolab/evolve practice() directly: practice() learns task
policies (skills.py + policy.py code) over TASKS entries, the wrong shape
for registry skills, and pointing it at a new task would mean adding code
to the pre-existing nested repo. Skill rounds reuse the loop pattern (LLM
callable, keep-best, file-per-round) through lab2rob's own cached,
cost-logged client on job "evolve".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from lab2rob.build.scene import build_scene
from lab2rob.certify.run import RUNS_DIR, certify
from lab2rob.ground.rules import VOLUME_UNITS, ground_step, parse_steps
from lab2rob.llm import Lab2RobClient
from lab2rob.registry import resolve_skill, skill_names
from lab2rob.schema import StepPredicate
from lab2rob.specify.judge import JudgeStep, PlannedJudge
from lab2rob.specify.program import PlannedProgram, plan_step

REGISTRY_DIR = Path(__file__).resolve().parents[1] / "registry"
QUEUE_PATH = Path(__file__).parent / "queue.json"

#: Gap reasons a learned macro could plausibly cover.
LEARNABLE_REASONS = {"missing_skill", "not_implemented"}
MAX_ROUNDS = 3
MAX_ATTEMPTS = 3
#: Proposal budget: the contributor model reasons ~1000 tokens first, which
#: starved the 1024 default exactly like grounding before UI-04.
PROPOSE_MAX_TOKENS = 2048
NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]{1,39}")

SYSTEM = """You design one lab2rob macro skill: a named expansion into simulator primitives.
Reply with exactly one JSON object, no other text, in one of these shapes:
{"skill": <snake_case name>, "description": <short string>, "params": [<names>], "steps": [<primitive names in order>], "smoke_text": <one in-scope demonstration sentence>}
{"cannot_express": <short reason why no honest expansion exists>}
Rules: steps must come ONLY from the action primitives below, 1-8 of
them. Never invent primitive names. The smoke sentence must use the
skill's verb with pipette-scale amounts (the pipette moves at most 1 mL);
bulk-scale queued examples motivate the skill but are not themselves
simulatable. If the capability needs hardware, senses, or physics the
primitives lack, reply cannot_express instead of faking it."""


def implemented_primitives() -> list:
    """Bindable action primitives for the proposal prompt.

    Only actions the planner can bind arguments for (queries and
    coordinate-needing motion excluded): anything else in an expansion
    could never compile.
    """
    from lab2rob.specify.program import BINDABLE_PRIMITIVES

    actions = []
    for name in skill_names():
        resolution = resolve_skill(name)
        if resolution["status"] != "implemented":
            continue
        if name in BINDABLE_PRIMITIVES:
            actions.append(name)
    return sorted(actions)


def ranked_candidates(queue: dict, limit: int = MAX_ATTEMPTS) -> list:
    """Evolve candidates with learnable reasons, most frequent first."""
    out = []
    for entry in queue.get("evolve_candidates", []):
        reasons = set(entry.get("reasons", []))
        if reasons and reasons <= LEARNABLE_REASONS and entry.get("examples"):
            out.append(entry)
        if len(out) >= limit:
            break
    return out


def _parse_proposal(text: str) -> dict:
    try:
        data = json.loads(text)
    except ValueError as e:
        raise ValueError(f"not JSON: {e}") from None
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    if "cannot_express" in data:
        return {"cannot_express": str(data["cannot_express"])}
    skill = data.get("skill", "")
    if not isinstance(skill, str) or not NAME_PATTERN.fullmatch(skill):
        raise ValueError(f"bad skill name {skill!r}")
    params = data.get("params", [])
    steps = data.get("steps", [])
    if not isinstance(params, list) or not all(isinstance(p, str) for p in params):
        raise ValueError("params must be a string list")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 8:
        raise ValueError("steps must be 1-8 primitive names")
    resolution = resolve_skill(skill)
    if resolution["reason"] != "unknown_skill" and resolution["status"] != "declared":
        raise ValueError(f"{skill!r} is already registered ({resolution['status']})")
    from lab2rob.specify.program import BINDABLE_PRIMITIVES

    for step_name in steps:
        step = resolve_skill(step_name)
        if step["status"] != "implemented":
            raise ValueError(f"step {step_name!r} is not implemented "
                             f"({step['status'] or step['reason']})")
        if step_name not in BINDABLE_PRIMITIVES:
            raise ValueError(f"step {step_name!r} has no planner binder; "
                             f"use only {sorted(BINDABLE_PRIMITIVES)}")
    smoke_text = data.get("smoke_text", "")
    if not isinstance(smoke_text, str) or not (1 <= len(smoke_text) <= 300):
        raise ValueError("smoke_text must be a 1-300 char demonstration sentence")
    return {"skill": skill, "description": str(data.get("description", "")),
            "params": params, "steps": steps, "smoke_text": smoke_text}


def propose_expansion(candidate: dict, primitives: list, client,
                      prior_error: str = "") -> dict:
    """One proposal round; raises ValueError when the reply is unusable."""
    examples = "\n".join(f"- {e['text']}" for e in candidate.get("examples", [])[:6])
    user = (f"Capability: {candidate['capability']} "
            f"({candidate.get('count', '?')} queued steps)\n"
            f"Example steps:\n{examples}\n"
            f"Primitives: {', '.join(primitives)}" + prior_error)
    text = client.complete(
        "evolve", [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": user}],
        max_tokens=PROPOSE_MAX_TOKENS).text
    return _parse_proposal(text)


def smoke_test(skill: str, smoke_text: str, smoke_dir: Path,
               seed: int = 0) -> dict:
    """Ground the proposal's demonstration sentence and certify it.

    Runs the real pipeline (ground, plan through the generic learned-macro
    binder, build, certify) against no-spill + volume checks. The sentence
    must ground to the skill with a volume quantity, else the proposal's
    own demonstration fails and the round is lost.
    """
    parsed = parse_steps(f"1. {smoke_text}")
    grounding = ground_step(parsed[0]) if parsed else None
    quantities = ([q for q in grounding.quantities if q.unit in VOLUME_UNITS]
                  if grounding else [])
    if (grounding is None or grounding.status != "grounded"
            or grounding.action != skill or not quantities):
        return {"passed": False,
                "detail": f"smoke sentence did not ground to {skill} with a volume"}
    planned = plan_step(grounding)
    if planned.status != "planned":
        return {"passed": False,
                "detail": f"planner: {planned.reason}: {planned.detail}"}
    import pint

    ureg = pint.UnitRegistry()
    volume = float(ureg.Quantity(quantities[0].value,
                                 quantities[0].unit).to("uL").magnitude)
    program = PlannedProgram(protocol_id=f"learn_{skill}", title=grounding.source_text,
                             steps=[planned])
    judge = PlannedJudge(
        protocol_id=program.protocol_id, title=grounding.source_text,
        steps=[JudgeStep(idx=1, source_text=grounding.source_text, predicates=[
            StepPredicate(step=1, pred="no_spill", args=[], target=None, tol=0.0),
            StepPredicate(step=1, pred="volume", args=["unbound"],
                          target=volume, tol=max(1.0, round(0.05 * volume, 2)))])])
    try:
        built = build_scene(program, seed=seed)
    except Exception as e:  # noqa: BLE001 - smoke reports, never raises
        return {"passed": False, "detail": f"scene rejected: {e}"}
    report = certify(program, judge, built, seed=seed, runs_dir=smoke_dir)
    failed = [p for p in report.predicates if p["status"] == "failed"]
    if report.status != "complete" or failed:
        return {"passed": False, "detail": f"run {report.status}: "
                f"{len(failed)} predicates failed", "run_id": report.run_id}
    return {"passed": True, "detail": f"{len(planned.ops)} ops certified",
            "run_id": report.run_id}


def _remove_declared(text: str, skill: str) -> str:
    """Drop one entry block from declared.yaml, byte-preserving otherwise."""
    start = text.find(f"  - name: {skill}\n")
    if start < 0:
        return text
    rest = text.find("  - name: ", start + 1)
    end = rest if rest > 0 else len(text)
    return text[:start] + text[end:]


def _snapshot_registry(registry_dir: Path) -> dict:
    return {name: (path.read_text(encoding="utf-8") if path.exists() else None)
            for name, path in (("macros.yaml", registry_dir / "macros.yaml"),
                               ("declared.yaml", registry_dir / "declared.yaml"),
                               ("skills.yaml", registry_dir / "skills.yaml"))}


def _restore_registry(registry_dir: Path, snapshot: dict) -> None:
    from lab2rob.specify.program import reset_macro_cache

    for name, text in snapshot.items():
        path = registry_dir / name
        if text is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(text, encoding="utf-8")
    reset_macro_cache()


def register_skill(proposal: dict, registry_dir: Path = REGISTRY_DIR) -> dict:
    """Promote a learned expansion into the registry overlays + regenerate.

    Removes the name from declared.yaml when present (the generator rejects
    macro/declared collisions), appends the macro entry, and regenerates
    skills.yaml. Raises ValueError when the result does not validate.
    registry_dir must match the gen_skills paths (tests monkeypatch them).
    """
    import yaml

    from lab2rob.registry import gen_skills
    from lab2rob.specify.program import reset_macro_cache

    registry_dir = Path(registry_dir)
    skill = proposal["skill"]
    declared_path = registry_dir / "declared.yaml"
    if declared_path.exists():
        before = declared_path.read_text(encoding="utf-8")
        after = _remove_declared(before, skill)
        if after != before:
            remaining = yaml.safe_load(after).get("declared_skills", [])
            assert all(s["name"] != skill for s in remaining)
            declared_path.write_text(after, encoding="utf-8")
    entry = (f"  - name: {skill}\n"
             f"    kind: action\n"
             f"    description: {proposal.get('description') or f'Learned skill: {skill}.'}\n"
             f"    params: [{', '.join(proposal.get('params', []))}]\n"
             f"    steps: [{', '.join(proposal['steps'])}]\n"
             f"    learned: true\n")
    macros_path = registry_dir / "macros.yaml"
    macros_text = macros_path.read_text(encoding="utf-8")
    if f"  - name: {skill}\n" in macros_text:
        raise ValueError(f"{skill!r} is already a macro")
    if not macros_text.endswith("\n"):
        macros_text += "\n"
    macros_path.write_text(macros_text + entry, encoding="utf-8")
    if gen_skills.main([]) != 0:
        raise ValueError("skills.yaml regeneration failed")
    if gen_skills.main(["--check"]) != 0:
        raise ValueError("regenerated skills.yaml does not validate")
    reset_macro_cache()
    return {"skill": skill, "steps": proposal["steps"]}


def learn_candidate(candidate: dict, client, run_dir: Path,
                    rounds: int = MAX_ROUNDS, seed: int = 0,
                    registry_dir: Path = REGISTRY_DIR) -> dict:
    """Propose/validate/smoke rounds for one capability; keeps the first pass.

    Each round tentatively registers its proposal (rolling back on any
    failure) so the smoke test runs the real pipeline: ground, plan, build,
    certify. A passing round keeps its registration.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    primitives = implemented_primitives()
    snapshot = _snapshot_registry(Path(registry_dir))
    prior_error = ""
    for attempt in range(rounds):
        folder = run_dir / f"round_{attempt:03d}"
        folder.mkdir(exist_ok=True)
        try:
            proposal = propose_expansion(candidate, primitives, client, prior_error)
        except ValueError as e:
            prior_error = f"\nYour last reply was invalid ({e}). Reply with valid JSON only."
            (folder / "error.json").write_text(json.dumps(
                {"round": attempt, "error": str(e)}, indent=2), encoding="utf-8")
            continue
        (folder / "proposal.json").write_text(json.dumps(proposal, indent=2),
                                              encoding="utf-8")
        if "cannot_express" in proposal:
            return {"capability": candidate["capability"], "learned": False,
                    "reason": proposal["cannot_express"], "rounds": attempt + 1}
        try:
            register_skill(proposal, registry_dir)
        except ValueError as e:
            prior_error = f"\nYour last expansion failed registration ({e}). Propose another."
            (folder / "error.json").write_text(json.dumps(
                {"round": attempt, "error": str(e)}, indent=2), encoding="utf-8")
            _restore_registry(Path(registry_dir), snapshot)
            continue
        smoke = smoke_test(proposal["skill"], proposal["smoke_text"],
                           folder / "smoke", seed=seed)
        (folder / "smoke.json").write_text(json.dumps(smoke, indent=2),
                                           encoding="utf-8")
        if smoke["passed"]:
            best = {"capability": candidate["capability"], "learned": True,
                    "proposal": proposal, "smoke": smoke, "rounds": attempt + 1}
            (run_dir / "best.json").write_text(json.dumps(best, indent=2),
                                               encoding="utf-8")
            return best
        _restore_registry(Path(registry_dir), snapshot)
        prior_error = (f"\nYour last expansion ({proposal['steps']}) failed: "
                       f"{smoke['detail']}. Propose a better expansion.")
    return {"capability": candidate["capability"], "learned": False,
            "reason": prior_error or "no valid proposal", "rounds": rounds}


def learn_from_queue(queue: dict, client, runs_dir: Path,
                     max_attempts: int = MAX_ATTEMPTS, rounds: int = MAX_ROUNDS,
                     only: str | None = None,
                     registry_dir: Path = REGISTRY_DIR,
                     register: bool = True) -> dict:
    """Try ranked candidates until one learns; report every attempt."""
    snapshot = _snapshot_registry(Path(registry_dir))
    attempts = []
    for candidate in ranked_candidates(queue, limit=max(1, max_attempts)):
        if only is not None and candidate["capability"] != only:
            continue
        result = learn_candidate(candidate, client,
                                 Path(runs_dir) / f"learn_{candidate['capability']}",
                                 rounds=rounds, registry_dir=registry_dir)
        attempts.append(result)
        if result["learned"]:
            if not register:
                _restore_registry(Path(registry_dir), snapshot)
            return {"learned": result, "attempts": attempts}
    return {"learned": None, "attempts": attempts}


def main(argv: list[str]) -> int:
    from lab2rob.llm import DailyCapExceeded, JobBudgetExceeded

    parser = argparse.ArgumentParser(
        description="Learn a missing skill from the coverage-gap queue.")
    parser.add_argument("--queue", default=str(QUEUE_PATH))
    parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    parser.add_argument("--candidate", default=None,
                        help="only try this capability (default: ranked order)")
    parser.add_argument("--rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument("--no-register", action="store_true",
                        help="learn but roll the registration back (dry run)")
    args = parser.parse_args(argv)
    try:
        queue = json.loads(Path(args.queue).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"error: cannot read queue {args.queue}: {e}")
        return 1
    client = Lab2RobClient()
    try:
        outcome = learn_from_queue(queue, client, Path(args.runs_dir),
                                   max_attempts=args.max_attempts, rounds=args.rounds,
                                   only=args.candidate,
                                   register=not args.no_register)
    except (RuntimeError, DailyCapExceeded, JobBudgetExceeded) as e:
        print(f"error: {e}")
        return 1
    for attempt in outcome["attempts"]:
        status = "learned" if attempt["learned"] else f"skipped ({attempt['reason']})"
        print(f"{attempt['capability']}: {status} in {attempt['rounds']} rounds")
    learned = outcome["learned"]
    if learned is None:
        print("no capability learned")
        return 1
    print(f"learned {learned['proposal']['skill']}: {learned['proposal']['steps']} "
          f"({learned['smoke']['detail']})")
    if args.no_register:
        print("(dry run: registration rolled back)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
