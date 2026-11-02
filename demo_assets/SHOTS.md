# Demo video: shot list + narration script (~3 minutes)

`demo.mp4` in this folder is the silent 47-second reel (built by
`make_demo.py` from sim clips). The full narrated cut below needs a human:
screen-record the review app and read the narration over the clips.

## Shots

| # | time | visual | narration |
|---|---|---|---|
| 1 | 0:00–0:20 | Title card → BioProt protocol text on screen | "Lab protocols live as text. Robots need tasks. lab2rob is the compiler between them." |
| 2 | 0:20–0:50 | Review app: grounding a protocol, per-step verdicts, Ask-LLM on an ambiguous step | "It grounds each step: units, verbs, labware. Anything out of scope gets a reason, not silence." |
| 3 | 0:50–1:20 | `lab2rob certify` running; protein-assay sim clip | "Grounded steps become a program, an independent judge, and a scene — then the robot runs it in simulation." |
| 4 | 1:20–1:50 | Cell-culture week clip; predicates passing; VLM verdict | "Long horizons, predicate checks, and a vision model confirming what the cameras see." |
| 5 | 1:50–2:20 | Tube grasp from camera poses (vision overlay); 2.7 mm / 5-of-5 caption | "Where truth isn't available, it perceives: camera-based tube grasping with zero success gap." |
| 6 | 2:20–2:45 | Safety twin: unsafe instruction → REFUSED banner | "And every task ships a safety twin — an unsafe variant the agent must refuse." |
| 7 | 2:45–3:00 | Report page: funnel, audit buckets; closing card | "Coverage, audit, and metrics in one report. lab2rob: protocols in, certified tasks out." |

## Recording notes

- Review app: `python -m lab2rob.review.app --port 8000`, ground BioProt
  1779 (CTAB mix) with `--mode rules`, then full-LLM for contrast.
- Sim clips: `MUJOCO_GL=egl python
  Code-as-a-Policy-Evolving-Skill/reports/make_videos.py`.
- Twin refusal: `python -m lab2rob.cli eval <run> --agent agent.json
  --twin-label refuse` and capture the verdict.
- Keep each CLI command on screen long enough to read; 1080p capture,
  downscale to 720p for upload.
