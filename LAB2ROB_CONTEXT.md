# lab2rob — Project Context for the Coding Agent

Read this file fully before writing code. It explains what we are building, what already exists, how the
code must be organized, and which mistakes from prior work (especially SciHorizon-eLab) we must avoid.

---

## 1. Goal in one paragraph

lab2rob is a **protocol-to-task compiler for biomedical wet-lab robotics**. It takes real, published
wet-lab protocols (protocols.io, Bio-protocol, BioProt) and turns them into **executable, certified robot
benchmark tasks** in our MuJoCo lab simulator with two I2RT YAM arms. Each task has a scene, a reference
program, an **independent judge**, a **scientific-outcome check**, and a **safety twin** (an unsafe variant
the agent should refuse or flag). Steps the compiler cannot handle are logged as coverage gaps, and our
self-evolving code-as-policy loop can learn new skills to close them.

**Long-term goal: every output of lab2rob must be usable as training data**, not only as a benchmark.
Certified tasks become a data engine: reference programs replayed under randomization produce robot
demonstrations (for VLA / imitation learning), judges produce success labels (for reward models and state
verifiers), and the compiler's own stages produce text pairs (for training a small protocol parser). Design
every module so its outputs are saved in a structured, versioned, license-tracked form. See section 6b.

Built for the **Meta Global AI Developer Hackathon** (Indie track, theme "Push the Frontier").
**Submission deadline: Nov 4, 2026, 9:00 PM ET.** Budget: **$150 in Meta Model API credits.**

---

## 2. What already exists (do not rebuild)

Repository: `github.com/Fernando23LM/Code-as-a-Policy-Evolving-Skill`

- `biolab/scene.py` — builds the MuJoCo model (two YAM arms, racks, vials, cameras).
- `biolab/env.py` — `TubeRackEnv`, `SnapCapEnv`, `WeighEnv`, etc.: `reset`, `perceive`, `move_to`,
  `set_gripper`, `home`, `render`, `evaluate`.
- `biolab/judge.py` — success checks (must stay out of the agent's view).
- `biolab/api.py` — functions a policy may call + reference policies. **This is the skill vocabulary.**
- `biolab/liquid.py` — liquids as bookkeeping (volume + concentration per well), not fluid simulation.
- `biolab/protein.py`, `biolab/culture.py` — protein assay and cell culture tasks with hidden parameters.
- `biolab/evolve/` — self-evolving loop: each round the LLM rewrites a skill library, lessons and a policy;
  best version kept; final policy scored once on held-out layouts.
- `biolab/capx_env/` — CaP-X integration.
- `cluster/` — Slurm scripts for Purdue's cluster.
- Existing tasks: `tube_rack`, `snap_cap`, `weigh_salt`, `weigh_handover`, `weigh_pour`, `protein_assay`,
  `cell_culture`, `drug_response`.

### Known limitations of the existing code (must be addressed or documented)
1. `perceive()` returns **ground-truth poses** (privileged information).
2. Generated policy code is **not sandboxed**: in CaP-X it receives the raw `env` and could read the judge
   through `env.sim`. The lab2rob eval harness must run policies in a separate process exposing only the
   API functions.
3. CaP-X seeds each trial with its index; practice/test seed split must be explicit.
4. YAM reach limits: gripper can point straight down only ~0.10–0.35 m from its base and below z = 0.16 m;
   otherwise pass `tilt`. Scene builder must check reachability.

---

## 3. Prior work — what to learn and what NOT to repeat

### 3.1 SciHorizon-eLab (arXiv 2609.30971) — closest prior work
Code: `github.com/SciHorizon-elab/SciHorizon-elab` (check license before reusing anything).

**What they built**
- Three-stage compiler: (1) semantic grounding of the protocol into ordered steps + simulator environment,
  (2) an execution-planning agent that writes a program from 30 registered atomic skills, and an
  **independently prompted condition-planning agent that writes step-level success predicates without
  seeing the program**, (3) simulation certification in **MuJoCo**: compile → execute → predicate check →
  visual review of keyframes by a VLM. A task is certified only if all four pass.
- SciVLABench: 51 source scenarios → 300 certified tasks, 5 operation families (liquid handling, mixing,
  solid handling/weighing, thermal control, apparatus/workspace), 15 human-agent coordination tasks.
- Metrics: task success rate (SR), ordered step-wise success rate (SSR), premature violation rate, response
  latency, held-out object-binding gap.
- Results: best policy averages ~50% SR (ACT); π0.5 generalizes best to unseen objects.
- Audit of 100 commands: 58% certified first pass, 81% after automatic recovery, 19% needed human fixes
  (1–2 min each).

**ADOPT these ideas**
- Independent judge generation: the judge planner must never see the generated program.
- Deterministic validators after every LLM stage (schema, registered skills only, entity references resolve,
  step order preserved).
- Consistent task-level entity IDs across steps, assets, skills and predicates.
- Ordered step-level evaluation (SSR) in addition to full-task success.
- Certification = compile ∧ execute ∧ predicates ∧ visual check.
- Report our audit in the same format (first pass / auto-recovered / human-fixed) for direct comparison.
- Randomization of poses/layouts that preserves task semantics.

**DO NOT repeat these weaknesses**
1. **Hand-curated scenarios instead of real protocols.** Their 51 scenarios were curated from literature,
   automated-lab systems, robot benchmarks and safety procedures. → We ingest **real published protocols**
   and keep a link from every task step to the source text.
2. **Audit only on in-scope inputs.** Their audit used only commands covered by their asset/skill libraries.
   → We report **coverage on real protocols**: % of steps compiled, and a categorized reason for every
   failure (missing asset, missing skill, unmodeled physics, ambiguous text).
3. **Very short tasks.** Most evaluated tasks have 2–4 steps. → We keep **long-horizon workflows**
   (multi-step, simulated hours/days, hidden parameters like cell doubling time or dye response).
4. **Chemistry teaching-lab bias** (alcohol lamps, glass rods, pH paper). → We focus on **biomedical wet-lab
   work**: tubes, plates, pipetting, cell culture, assays, centrifuge loading.
5. **Success = simulator states only.** → Every task also gets a **scientific-outcome check** when the
   protocol has a measurable result (reported concentration, IC50, viable cell count).
6. **Fixed skill library.** Anything outside their 30 skills is out of scope forever. → Unsupported steps go
   to a **coverage-gap queue** that the evolve loop can turn into new learned skills.
7. **Single-arm only** (7D end-effector command). → We support **bimanual** steps (hold + uncap, handover,
   steady + pipette).
8. **Only trained per-task policies evaluated.** → Our harness also evaluates **agents that read the
   protocol zero-shot** (code-as-policy, self-evolving).
9. **Safety only in a small subset.** → Every certified task gets a **safety twin** and every run reports
   safety violations.

### 3.2 Other prior work to cite and differentiate from
- **ProtoAct** (arXiv 2608.01690): protocols → schema-constrained JSON action functions; 22 annotated
  cell-culture protocols (BioP2E). Small scale; not full closed-loop workflows.
- **Pipette** (arXiv 2606.12936): 43+ open USD wet-lab assets, 11-task benchmark, sim data augmentation,
  natural-language scene construction. Isaac/USD, single Franka arm by default.
- **AutoBio**: MuJoCo-native biology-lab simulator with MJCF assets and a quasi-static liquid plugin.
  **Best source of extra MuJoCo assets** (check license).
- **LabUtopia** (NeurIPS 2025): 30 tasks, 200+ USD assets (CC-BY-NC-4.0), Isaac Sim.
- **BioPlanner / BioProt** (EMNLP 2023): protocols.io protocols paired with pseudocode and admissible
  actions; repo `github.com/bioplanner/bioplanner`. **Use as a structured input source.**
- **ProtoPilot**: protocol → Opentrons code for liquid handlers. Shows pipetting on liquid handlers is
  largely solved; our value is the manual "glue" work between instruments.
- **SEAR** (sear.bot) and **ASPIRE** (NVIDIA): self-evolving code-as-policy with skill libraries; the
  inspiration for `biolab/evolve/`.

---

## 3b. Input sources and standards

### Protocol sources (in priority order)
| Source | Use | Notes |
|---|---|---|
| **protocols.io** (API, JSON) | Primary source | Large, step-structured, versioned with DOIs, mostly CC-BY. Quality is uneven and steps are free text (inconsistent units/names, notes mixed with instructions). Filter: CC-BY, biology categories, uses our assets. Store protocol id + version + DOI. |
| **STAR Protocols** (Cell Press) | High-quality subset | Peer-reviewed, open access, consistent step-by-step structure. |
| **Bio-protocol** | High-quality subset | Peer-reviewed, structured sections. Check license per article. |
| **BioProt** (BioPlanner) | Structured input | protocols.io protocols with pseudocode + admissible actions. |
| **Opentrons Protocol Library** | Exact parameters for liquid steps | Executable Python with volumes and labware; liquid handling only. |
| Manufacturer kit manuals (BCA, Bradford, media) | Testing only | Standardized but usually copyrighted: do not redistribute. |
| JoVE / Nature Protocols / Protocol Exchange | Reference only | Mostly paywalled; JoVE videos can serve as visual reference. |

Ingestion rules: never redistribute non-open text; store license, URL, version and retrieval date for every
protocol; deduplicate near-identical protocols before splitting train/test (splits are by protocol).

### Annotated corpora (use before spending tokens)
- **Wet Lab Protocol (WLP) corpus:** hundreds of protocols.io protocols annotated with actions, reagents,
  equipment, amounts and relations. Use it to **build and evaluate `ground/rules.py`** (precision/recall of
  action, entity and quantity extraction) before calling any LLM.
- **BioProt:** pseudocode targets to evaluate step → action mapping.
- **ProtoAct BioP2E:** if released, an extra evaluation set for cell-culture protocols.

### Standards to align with (naming, not necessarily full adoption)
- **LabOP** (Laboratory Open Protocol language): open standard for machine-readable protocols that can be
  compiled to different executors. Borrow its concepts (primitives, parameters, samples/containers).
- **Autoprotocol:** JSON standard for lab instructions from cloud-lab automation. Use compatible action and
  parameter names where possible (e.g. `pipette`/`transfer`, `spin`, `incubate`, `seal`, `cover`).
- **Ontologies:** map lexicon entries to IDs: **OBI** (lab operations, instruments), **ChEBI** (chemicals),
  **UO** (units). Add an `ontology_id` field to entities and actions.
- Keep our own JSON schema as the source of truth, but provide an export (`export/labop.py` or
  `export/autoprotocol.py`, post-hackathon) so outputs are reusable by others.

---

## 4. Architecture

```
 protocols.io / Bio-protocol / BioProt
              │
         [1] INGEST ──────────────── cache raw text + URL + license
              │
         [2] GROUND  rules first ──► units, verbs, labware synonyms
              │      LLM fallback ─► only unresolved steps
              ▼
        grounded steps + entities ──► asset/skill registry check
              │                              │ unsupported step
              │                              ▼
              │                        [7] COVERAGE GAP ──► evolve loop
              │                              learns new skill ──► registry
        ┌─────┴──────┐
   [3a] PROGRAM   [3b] JUDGE        (judge never sees the program)
   planner         planner + outcome checks
        └─────┬──────┘
         [4] BUILD  scene from registry (MuJoCo, YAM arms)
              │
         [5] CERTIFY  run program → predicates → SAM 3.1 visual check
              │       one automatic repair → else human review
              ▼
         [6] SAFETY TWIN  unsafe variant + expected behavior
              │
         certified task ──► [8] EVAL HARNESS (sandboxed)
                                 metrics: SR, SSR, outcome error,
                                 safety violations, coverage, cost
```

### Repository layout (new code goes in `lab2rob/`)
```
lab2rob/
├── llm/
│   ├── client.py        # OpenAI SDK → https://api.meta.ai/v1, model routing per job
│   ├── cache.py         # hash(model + prompt) → saved response; never pay twice
│   └── cost_log.py      # tokens + $ per call; hard daily cap
├── ingest/
│   ├── protocols_io.py  # fetch + store raw protocol, license, URL
│   ├── bioprot.py       # load BioProt protocols + pseudocode
│   └── sources/         # cached raw protocols (JSON)
├── registry/
│   ├── assets.yaml      # every object scene.py can build: dimensions, affordances
│   ├── skills.yaml      # generated from biolab/api.py (+ learned skills)
│   ├── predicates.py    # in_slot, capped, volume, concentration, no_spill, ...
│   └── lexicon.yaml     # "Eppendorf" → tube_1_5ml, "spin" → centrifuge, ...
├── schema.py            # pydantic models: GroundedStep, Entity, Task, Judge, SafetyTwin
├── ground/
│   ├── rules.py         # regex + pint units, verb templates, synonym matching
│   └── llm_ground.py    # fallback for unresolved steps, schema-validated
├── specify/
│   ├── program.py       # grounded step → sequence of registered skills
│   ├── judge.py         # independent step predicates (no access to program)
│   └── outcome.py       # scientific-result checks
├── build/
│   └── scene.py         # task spec → biolab scene (layout seeds, reach check)
├── certify/
│   ├── run.py           # execute reference program, record trace
│   ├── visual.py        # render cameras → SAM 3.1 → objects where the spec says?
│   └── repair.py        # one automatic retry with structured diagnostics
├── safety/
│   └── twins.py         # unsafe variant + label: refuse / ask / flag
├── coverage/
│   └── gaps.py          # unsupported steps → categorized queue for evolve loop
├── eval/
│   ├── harness.py       # runs agents/policies in a separate process, API-only access
│   └── metrics.py       # SR, SSR, outcome error, safety, coverage, cost
├── export/
│   ├── lerobot.py       # replay certified programs under randomization → LeRobot dataset (A)
│   ├── verifier.py      # judge predicates on rendered frames → (image, question, label) (B)
│   ├── text_pairs.py    # aligned compiler records: source → steps → program → judge (C)
│   ├── trajectories.py  # evolve/eval agent traces + safety-twin decisions (D)
│   └── manifest.py      # versioning, splits by protocol, provenance, licenses
├── review/
│   └── app.py           # minimal accept / edit / reject UI
└── cli.py               # lab2rob compile | certify | eval | report

runs/                    # one folder per run: inputs, LLM calls, outputs, traces, costs
datasets/                # released, versioned training datasets (see section 6b)
```

### Core schema (starting point; refine in `schema.py`)
```json
{
  "id": "bradford_standards_01",
  "source": {"url": "...", "license": "CC-BY-4.0", "steps_used": [3, 4, 5]},
  "instruction": "Prepare six protein standards in row A ...",
  "entities": [{"uid": "plate_1", "asset": "plate_2x4", "slot": "bench_center"}],
  "initial_state": {"stock_tube": {"volume_uL": 500, "conc_mg_mL": 2.0}},
  "steps": [{"idx": 1, "source_text": "...", "skills": ["pick", "aspirate", "dispense"]}],
  "judge": [{"step": 1, "pred": "volume", "args": ["plate_1.A1"], "target": 100, "tol": 10}],
  "outcome": {"type": "reported_value", "quantity": "concentration", "tol_rel": 0.10},
  "safety": [{"pred": "no_spill"}],
  "safety_twin": {"instruction": "...", "violation": "...", "expected": "refuse"},
  "out_of_scope": [{"step": 6, "reason": "missing_asset", "detail": "plate reader incubation"}],
  "difficulty": {"level": "trained_technician", "est_human_min": 8},
  "certification": {"compiled": true, "executed": true, "predicates": true, "visual": true}
}
```

---

## 5. Models and budget (Meta Model API, $150 total)

Endpoint `https://api.meta.ai/v1`, OpenAI-compatible, key in env var `MODEL_API_KEY`.

| Job | Model | Reasoning effort |
|---|---|---|
| Coding (Muse Code) | `muse-spark-1.3-contributor` | — |
| Grounding public protocols (fallback only) | `muse-spark-1.3-contributor` | `minimal` |
| Anything private (partner protocols, unpublished data) | `muse-spark-1.3` (standard; not used for training) | — |
| Program planner, independent judge planner | `muse-spark-1.3` | `low` |
| Visual certification | `muse-spark-1.3-contributor` (vision input) | `minimal` |
| Evolve loop | `muse-spark-1.3-contributor` | `low` |
| Hard cases only (final audit, tricky safety twins) | `muse-wonder-1.0`, capped at ~$15 | `medium` |

Rules:
- Every call goes through `llm/client.py` → cached and cost-logged. **Hard cap: $12/day.**
- Reasoning effort is set per job in the routing layer (`minimal` for grounding, `low` for
  planners, `medium` for escalation only). Budgets are expected visible output + headroom
  (1500/3000/6000). A reply cut off at finish `length` is retried once at the next effort
  up, then the step is marked `failed`/`llm_truncated`; truncated JSON is never parsed.
- Keep long, stable prompt prefixes (API docs, registry, schema) at the start so they hit the prompt cache.
- Contributor tier = Meta may train on prompts/completions. **Never send private or partner data there.**
- Rate limits are per team (contributor tier: 100 requests/min).

---

## 6. Engineering rules

1. **Code before model.** If a step can be done deterministically (units, synonyms, verb templates, schema
   checks, reachability), do it in code. The LLM is a fallback. Track the share of steps compiled by rules
   vs. LLM — it is a reported metric.
2. **Validate every LLM output** with pydantic + deterministic checks; on failure, send the error back once,
   then mark as failed with a reason. Never silently accept malformed output.
3. **Never invent assets or skills.** Only use entries in `registry/`. Anything else → `out_of_scope` with a
   reason category.
4. **The judge never sees the program**, and policies never see the judge.
5. **No privileged information at eval time.** Policies get the API (and camera images), not `env.sim`.
6. **Liquids are bookkeeping** (`biolab/liquid.py`). Do not attempt fluid simulation.
7. **Traceability:** every task step links to its source protocol text; every asset records its license
   (most public lab assets are non-commercial — track this).
8. **Reproducibility:** everything a run produces goes to `runs/<run_id>/` (inputs, prompts, responses,
   costs, traces, renders). Fixed seeds; explicit practice/held-out split.
9. **Hackathon rule:** all new code lives in `lab2rob/`; the commit at hackathon start is tagged so new
   work is clearly separated from pre-existing work.
10. **Safety twins** describe procedural violations only (wrong order, skipped containment/labeling, wrong
    sample). Never generate content that explains how to cause harm.

---

## 6b. Outputs as training data (long-term requirement)

lab2rob produces four kinds of training data. The hackathon only needs dataset A for one or two tasks, but
**every module must already save what B–D need**, so nothing has to be regenerated later.

### A. Robot demonstrations (for VLA / imitation learning: SmolVLA, π0.5, ACT, Diffusion Policy)
- Replay each certified reference program under randomized seeds (poses, layouts, initial volumes, lighting,
  camera jitter, textures) and record episodes.
- **Format: LeRobot dataset** (parquet + mp4, the format used at MARS Lab and by SmolVLA/π0.5 fine-tuning).
  Per frame:
  - observations: RGB from all cameras (overhead, front, left/right wrist), optional depth, proprioception
    (joint positions/velocities of **both** arms + gripper widths);
  - action: **bimanual joint-position targets (both arms + grippers) at a fixed control rate** (e.g. 10–30 Hz).
    Also store end-effector poses so either action space can be used later;
  - `task`: the natural-language instruction; plus a per-frame **subtask index** linking to the grounded
    step (step-level language annotation comes for free from the compiler).
- Per episode metadata: task id, seed, randomization parameters, step boundaries (frame indices), judge
  result per step, outcome value, safety violations, success flag.
- **Keep failed episodes too**, labeled with the failing step and reason. They are valuable for reward
  models, verifiers and recovery learning; filter them out only at training time.
- Add **perturbation episodes**: inject a fault mid-episode (dropped tube, wrong slot) and record the
  reference recovery when one exists.

### B. State-verifier / reward-model data
- From every rendered frame at a step boundary, save (images, question, label) tuples generated from the
  judge predicates: "is tube_1 capped?" → yes/no, "volume in A1?" → value. Fully automatic labels.
- Use to train a small lab-state verifier (answers whether a step succeeded from images only).

### C. Compiler text pairs (to train a small protocol parser and cut API cost)
- Save every stage as aligned records: protocol source text → grounded steps → entities → skill program →
  judge predicates → out-of-scope decisions, with a flag saying whether each output came from **rules**, the
  **LLM**, or a **human correction**. Human corrections are the most valuable records.

### D. Agent trajectories (for training code-as-policy agents)
- From the evolve loop and eval harness: prompt, generated code, execution feedback, scores, lessons, kept
  or discarded. Also safety-twin pairs with the agent's decision (execute / refuse / ask) and the label.

### Rules for all datasets
1. **Split by protocol, not by seed.** Held-out test sets contain protocols (and ideally object types)
   never seen in training, otherwise results leak.
2. **Every record carries provenance:** source protocol URL + license, asset licenses, code commit hash,
   model id used, run id. Non-commercial assets/protocols must be flagged so they can be excluded from a
   commercial dataset later.
3. **Check the Meta Model API terms** on using model outputs to train other models before training on
   dataset C/D generated with Muse Spark; record the generating model on every record so it can be filtered.
4. **Versioned and immutable:** `datasets/<name>/v<N>/` with a `manifest.json` (counts, splits, schema
   version, generation config). Never overwrite a released version.
5. **No privileged state in observations.** Simulator ground truth may be stored as labels/metadata, never
   inside the observation fields a policy trains on.
6. **Sim-to-real readiness:** match camera names, control rate and action space to the real I2RT YAM setup
   (the benchmark's recording kit) so real demonstrations can be mixed into the same dataset later.

---

## 7. Build order and priorities

**Must have (in order)**
1. `llm/` (client, cache, cost log) + `schema.py` + `registry/` (assets, skills from `biolab/api.py`,
   predicates, lexicon).
2. `ingest/` for ~30 protocols (favor ones overlapping existing tasks: pipetting, standards, plates, cell
   passaging, weighing, tube handling) + BioProt loader. Sources and filters: section 3b.
3. `ground/` (rules first, LLM fallback). Build and score the rules against the WLP corpus first.
4. `specify/` (program planner, independent judge planner, outcome checks).
5. `build/` + `certify/` (execute, predicates, SAM 3.1 visual check, one repair attempt).
6. `safety/twins.py`.
7. `eval/harness.py` (sandboxed) + `eval/metrics.py`.
8. `export/lerobot.py` + `export/manifest.py` for 1–2 certified tasks; all other stages already save the
   records that `export/text_pairs.py` and `export/trajectories.py` will need later.
9. Report: coverage funnel, certification audit table, metrics.

**Stretch (only if must-haves are done)**
- `coverage/` → evolve loop learns 1–2 new skills from the gap queue.
- SAM 3.1 camera-based perception replacing `perceive()` for one task.

**Keep minimal:** `review/app.py`.

---

## 8. Definition of done (Nov 4)

- ~30 real protocols processed end-to-end.
- **Coverage funnel:** protocols → steps → % grounded by rules / by LLM → % in scope → specs built →
  scenes certified → human-accepted, with categorized failure reasons.
- **Certification audit table** in SciHorizon's format (first pass / auto-recovered / human-fixed).
- 5–8 certified simulated tasks, each with judge, outcome check (where applicable) and safety twin.
- At least one agent (e.g. Muse Spark code-as-policy) evaluated on the certified tasks with SR, SSR,
  outcome error and safety violations.
- A LeRobot-format dataset (e.g. 100+ randomized episodes, successes and labeled failures) for at least
  one certified task, with a manifest, loadable by LeRobot's dataset loader.
- README: architecture diagram, metrics, comparison to SciHorizon/ProtoAct/Pipette, limitations
  (no fluid simulation, perception, sandboxing status), next steps.
- ~3-minute demo video: protocol in → certified task out → robot attempt → safety twin refused.
