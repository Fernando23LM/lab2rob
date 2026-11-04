# lab2rob: published lab protocols in, validated robot tasks out

**Track:** Indie · **Theme:** Push the Frontier

## The problem

Robots for wet labs are judged on hand-made tasks: a few dozen scenarios someone
wrote by hand, mostly two to four steps, rarely tied to how biologists actually work.
Meanwhile thousands of real protocols sit on protocols.io as free text. There is no
way to turn that text into robot tasks you can trust: tasks that are physically
runnable, scored by something other than the program that solves them, and honest
about what they leave out.

## What lab2rob does

lab2rob is a compiler from published wet-lab protocols to validated robot benchmark
tasks, in MuJoCo with two I2RT YAM arms.

1. **Ground** each step: rules first (units, verbs, labware), a Meta model only for
   what the rules cannot read.
2. **Plan** the step into registered robot skills, and separately **judge** it: an
   independent check written from the protocol text without seeing the program.
3. **Build** a scene for it, or reject it with a reason (missing instrument, wrong
   vessel, mass out of range...).
4. **Freeze** a task only if its reference solution passes the judge, replays to the
   same state in two fresh processes, and doing nothing fails.
5. **Chain** steps across scenes: the pipetting steps that run straight into a
   centrifugation compile into one multi-step task, judged at the spin, so the
   order the steps happen in counts.
6. **Package** it as a benchmark task with a 20 Hz runtime, a scorer a person could
   reproduce with a balance or a photo, a demo video and a setup screenshot.

## How it uses Meta's models

- **Grounding fallback:** steps the rules cannot read go to Muse Spark, schema-validated,
  retried once, and cost-logged under a daily cap (84 calls, $0.94 for a 42-protocol
  sweep).
- **Visual certification:** a Meta vision model checks rendered keyframes against what
  the program expects before a task is certified.
- **Learning missing skills:** a self-evolving code-as-policy loop on Muse Spark learned
  the `dilute` skill from the coverage-gap queue in 3 rounds ($0.27).
- **Spending only where it pays:** the corpus sweep sizes a targeted LLM pass before
  spending (24 near-complete protocols, 46 calls) instead of sending 122k steps to a
  model. Total spend: $4.20 of the $150 budget.

## Results

- **10,149 protocols** (BioProBench: protocols.io and Protocol Exchange), 246,169 steps,
  compiled offline in about 100 s. 27% of steps grounded by rules alone, 12.7%
  executable, every other step with a categorized reason. The report ranks which
  missing capability unlocks the most protocols.
- **37 of the 40 top pipetting candidates** freeze into validated tasks, up from 2,
  after the validation gate exposed pipette-aiming and planning bugs.
- **Three scene families**: pipetting, weighing (68 simulated salt grains on a spring
  balance) and a microcentrifuge (LabUtopia's model converted to MuJoCo with a hinged,
  latching lid). 15,519 centrifuge steps reach the compiler and 10,466 are accepted;
  5,042 are rejected because they spin tubes or plates the microcentrifuge cannot take.
- **Multi-step tasks across scenes**: pipetting into a tube and then spinning it is the
  most common run of steps that crosses scenes in the corpus. 2,054 spins follow one or
  more addition steps (in 1,072 protocols). 534 of them pass the compiler and the scene
  builder (in 360 protocols), 33 with two reagents. The rest are rejected with a reason:
  1,064 because a step adds several liquids at once or names none, 331 because they
  spin labware the microcentrifuge cannot take, 110 because there are more additions
  than the bench has troughs.
- **Three benchmark-ready tasks**, compiled from published steps, each needing both arms:
  - weigh out 0.90 g of NaCl passed between the arms (from a SARS-CoV-2 wastewater
    protocol);
  - load a microcentrifuge balanced and shut its lid, where a lid shut on an unbalanced
    rotor scores 0 (from a DNA-extraction protocol);
  - **multi-step**: pipette 60 uL of lysis solution then 20 uL of protein precipitation
    solution into a sample tube, fill a balance tube with water to match, load both
    opposite and shut the lid (three consecutive steps of an extracellular-vesicle RNA
    isolation protocol). Scored at the spin: shutting the lid on an unbalanced load
    scores 0, and each missing or wrong reagent loses its share. The demo takes 8,132
    steps (6.8 minutes at 20 Hz).

## What is new

- Tasks come from **real published protocols**, with every task step linked to its
  source text, not from hand-curated scenarios.
- **Coverage is reported on real text**, with a reason for every step that does not
  compile, not only on inputs a library already covers.
- **The judge never sees the program**, and a task exists only if it validates
  (judge, two identical replays, idle fails).
- **Multi-step tasks come from the protocol's own order**: the compiler chains the
  steps the protocol runs back to back, and the scorer checks the state at the moment
  the order matters (the spin), not only the end state.
- **Robot-realistic**: the weighing and centrifuge tasks use only the arms' 14 joint and
  gripper targets at 20 Hz, so they can be run on the real arms. The multi-step task
  pipettes, so it adds the plunger as a 15th action (see Limitations).

## Try it

- Repository: lab2rob (see README: results at a glance, quickstart, task builder).
- Videos: `demo_assets/reel_tasks.mp4` (the three compiled tasks),
  `demo_assets/task_weigh_demo.mp4`, `demo_assets/task_centrifuge_demo.mp4`,
  `demo_assets/task_prep_spin_demo.mp4` (multi-step),
  `demo_assets/demo.mp4` (pipetting, cell culture, safety twin).

```bash
python -m lab2rob.coverage.corpus_sweep --out runs/sweep --workers 4
python -m lab2rob.robocurve build Protocol-exchange-1154 --family centrifuge --out tasks/centrifuge
python -m lab2rob.robocurve build Protocol-exchange-758 --family prep_spin --out tasks/prep-spin
```

## Limitations

Liquids are bookkeeping, not fluid simulation. The pipetting tasks, including the
multi-step one, use a commanded plunger that a real gripper-held pipette cannot
reproduce. The multi-step tasks keep only the pipetting and the spin from their steps
(mixing and incubating are left out), and take one or two reagents, one per step. The weighing bench models
coarse salt only, and the centrifuge's slots are vertical and nothing spins. The
centrifuge meshes are CC BY-NC 4.0 (LabUtopia).
