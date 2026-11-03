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

Run the suite: `python -m pytest lab2rob/tests/ -q` (271 tests).
All tests run offline with fake transports; live LLM calls need the key and
stay under a $12/day cap with per-call cost logging.

## Frozen scenes and trajectories (`lab2rob/sim/`)

A certified program can be frozen into a graded task in the style of
Embodied First Exam's privileged mode: a frozen scene (`instance.json`, with
model and initial-state hashes), a reference trajectory (`oracle.npz`), and
a grader that replays a trajectory in fresh processes and checks the end
state against the independent judge.

```bash
python -m lab2rob.sim freeze --bioprot 1779 --out runs/frozen/1779_s0
python -m lab2rob.sim grade --instance runs/frozen/1779_s0/instance.json \
  --traj runs/frozen/1779_s0/oracle.npz --repeat 2 --out runs/frozen/1779_s0/verdict
```

`freeze` only writes a task whose oracle passes every check, replays to the
same final state in two fresh processes, and whose empty trajectory fails.

An agent drives the scene through `Session(instance, level=...)`. Whatever
it calls, the session records rows of 15 targets per 20 ms control step
(6 joints + gripper per arm, then the pipette plunger in uL) plus
instrument events, so every level is graded by the same replay:

| level | the agent uses |
|---|---|
| `joints` | `step(row, n)` only: joint and gripper targets and the plunger, with `ik`/`fk` helpers |
| `skills` | `move_to`, `move_joints`, `set_gripper`, `aspirate`/`dispense(volume, slot=...)`, `move_tip_into`, `withdraw`, `wait` |
| `mixed` | both, in one program: skills for the easy parts, raw rows for the rest |

Instruments (`read_plate`, `report_concentration`, `incubate`, `inspect`)
work at every level. The plunger moves liquid through biolab's own model:
liquid only when the tip is in a well below its surface, air otherwise, and
what is pushed out over no well is spilled. The sandboxed skill harness in
`eval/` is unchanged and still runs model + skill agents.

## Corpus sweep (BioProBench, offline)

`ingest/bioprobench.py` loads the BioProBench protocol corpus (about 9.6k
protocols.io and 0.9k Protocol Exchange protocols with sectioned steps) at a
pinned revision into `runs/corpora/` (never committed: the compilation is
CC BY-NC 4.0, and every record carries that license plus its source's and a
non-commercial flag). Only steps, title, URL and abstract are used; the
corpus's `description`/`classification` fields are misaligned with their
protocols for a sizeable share of records.

`coverage/corpus_sweep.py` compiles every protocol with the rules only (no
LLM, about 5 minutes for the whole corpus on 4 cores), builds the scene, and
reports the funnel, what blocks each step, a greedy unlock order (which
capability gives the most further protocols a task segment of 3+ consecutive
executable steps), the size of a targeted LLM pass, and the result of
freezing the top candidates into validated tasks.

```bash
python -m lab2rob.coverage.corpus_sweep --out runs/corpus_sweep --workers 4 --freeze-top 40
```

With `MODEL_API_KEY` set, `--llm-budget-usd 10` adds a targeted LLM pass:
only protocols that already have an executable liquid step and whose other
non-executable steps are at most 3 the rules could not ground are
re-grounded with the LLM fallback, cheapest first, stopping before the
budget (and at the client's daily cap). The latest sweep's report is
`lab2rob/coverage/bioprobench_sweep.md`.

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
  `registry/`, `perception/`, `report/`, `sim/`, `cli.py`
- `Code-as-a-Policy-Evolving-Skill/biolab/` — the MuJoCo simulator, tasks,
  reference policies, and evolve loop (pre-existing)
- `runs/` — per-run inputs, traces, costs (never committed)
- `demo_assets/` — demo reel, shot list, narration script
