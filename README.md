# lab2rob — protocol-to-task compiler for wet-lab robotics

lab2rob turns real, published wet-lab protocols (protocols.io, Bio-protocol,
BioProt) into **executable, certified robot benchmark tasks** in a MuJoCo lab
with two I2RT YAM arms. Each task ships with a scene, a reference program, an
**independent judge** written without seeing the program, a
**scientific-outcome check**, and a **safety twin** (an unsafe variant the
agent should refuse or flag). Steps the compiler cannot handle become
categorized coverage gaps, and a self-evolving loop can learn new skills to
close them.

Built for the Meta Global AI Developer Hackathon (Indie track).

## How it works

```
 protocols.io / Bio-protocol / BioProt
              │
         [1] INGEST ──────────────── raw text + URL + license
              │
         [2] GROUND  rules first ──► units, verbs, labware synonyms
              │      LLM fallback ─► only unresolved steps
              ▼
  grounded steps + entities ──► registry check (55 skills, 31 assets)
              │                              │ unsupported step
              │                              ▼
              │                        [7] COVERAGE GAP ──► evolve loop
              │                              learns skill ──► registry
        ┌─────┴──────┐
   [3a] PROGRAM   [3b] JUDGE      (judge never sees the program)
        └─────┬──────┘
         [4] BUILD  MuJoCo scene from registry (reach-checked)
              │
         [5] CERTIFY  run → predicates → VLM visual check
              │       one automatic repair → else human review
              ▼
         [6] SAFETY TWIN  unsafe variant + expected behavior
              │
         certified task ──► [8] SANDBOXED EVAL
                                 SR, SSR, outcome error,
                                 safety violations, cost
```

Every certified task is also training data: reference programs replay into
robot demonstrations (LeRobot), judges into success labels, compiler stages
into text pairs.

## Quickstart

```bash
conda env create -n lab2rob python=3.11
conda activate lab2rob
pip install mujoco scipy numpy Pillow pint pyyaml openai pydantic imageio
export MODEL_API_KEY=<your Meta Model API key>   # LLM stages only
```

Compile a protocol, certify it, evaluate an agent, write the report:

```bash
python -m lab2rob.cli compile bioprot 1779 --auto-accept
python -m lab2rob.cli certify g_bioprot_1779_rules
echo '{"kind": "scripted"}' > agent.json
python -m lab2rob.cli eval 1779_s0 --agent agent.json
python -m lab2rob.cli report --report-id demo1   # runs/demo1/report/report.md
```

Review and ground protocols in a browser (stdlib only, no install):

```bash
python -m lab2rob.review.app --port 8000
```

Run the suite: `python -m pytest lab2rob/tests/ -q` (257 tests).
All tests run offline with fake transports; live LLM calls need the key and
stay under a $12/day cap with per-call cost logging.

## Metrics

| metric | value |
|---|---|
| registry | 55 skills (20 implemented, 4 macros, 31 declared), 31 assets |
| 42-protocol sweep | 514 steps; 346 out-of-scope/ambiguous mined into 32 gap clusters |
| LLM grounding fallback | 84 calls, $0.94 for the sweep |
| learned skill | `dilute` macro in 3 evolve rounds, $0.27 |
| certification audit | 1 first-pass (check1 demo); SciHorizon-format buckets in every report |
| camera perception | 2.7 mm pose error, 5/5 vision grasps = truth grasps |
| total LLM spend | $4.20 of $150 budget |

## Comparison to prior work

### SciHorizon-eLab (closest)

Three-stage compiler (ground → plan → MuJoCo certify) with an independent
judge planner and VLM keyframe review; 51 scenarios → 300 tasks; audit of 100
commands: 58% first pass, 81% after recovery.

- We adopt: independent judge, validators after every LLM stage, ordered
  step success (SSR), certify = compile ∧ execute ∧ predicates ∧ visual.
- We differ: real published protocols (not hand-curated scenarios), coverage
  reported on real text with a reason per failure, long-horizon workflows,
  biomedical focus, scientific-outcome checks, learned skills for gaps,
  bimanual steps, zero-shot agents, and a safety twin per task.

### ProtoAct

Protocols to schema-constrained JSON actions; 22 annotated cell-culture
protocols. Small scale, no closed-loop workflows. We compile to full
simulated tasks with execution and judging.

### Pipette

43+ open USD wet-lab assets, 11-task benchmark, Isaac/USD on a single Franka
arm. We are MuJoCo-native with bimanual YAM arms and compile from protocol
text rather than shipping fixed tasks.

## Demo video

`demo_assets/demo.mp4` is a silent reel: protocol in, certified task out, the
robot running a protein assay and a cell-culture week, vision-based tube
grasping, and a safety twin being refused. `demo_assets/SHOTS.md` is the shot
list and narration script for the full ~3-minute narrated cut.

## Limitations

- Skill vocabulary is small (20 implemented primitives + 4 macros); most real
  protocol steps are still out of scope (centrifuge, wash, incubate, …).
- Liquids are bookkeeping (volume + concentration), not fluid simulation.
- Only one task has camera perception; the rest still use ground-truth poses.
- sam-3.1 returns empty replies, so the visual job runs on the contributor
  VLM until the endpoint works.
- AutoBio instrument meshes (`lab2rob/build/autobio/`) are unlicensed and
  demo-only; they must go before any public release (task XR-04).

## Next steps

- Grow the sweep to 30+ protocols with WLP precision/recall scoring.
- Learn the top gap skills (centrifuge, wash, rinse) through the evolve loop.
- Camera perception for the remaining tasks; LabOP/Autoprotocol export.
- Calibrate the safety twins against human refusal labels.

## Layout

- `lab2rob/` — the compiler: `ingest/`, `ground/`, `specify/`, `build/`,
  `certify/`, `safety/`, `coverage/`, `eval/`, `export/`, `review/`, `llm/`,
  `registry/`, `perception/`, `report/`, `cli.py`
- `Code-as-a-Policy-Evolving-Skill/biolab/` — the MuJoCo simulator, tasks,
  reference policies, and evolve loop (pre-existing)
- `runs/` — per-run inputs, traces, costs (never committed)
- `demo_assets/` — demo reel, shot list, narration script
