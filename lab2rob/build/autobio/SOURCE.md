# AutoBio instrument models (vendored, DEMO ONLY)

Source: https://github.com/autobio-bench/AutoBio
Commit: 08504213356cbfe9a82bcd33a43cd282945a45df (branch `main`), retrieved 2026-10-30.
Files: `autobio/model/instrument/<name>.xml` and `autobio/assets/instrument/<name>/*` for
`centrifuge_tiangen_tgear_mini` and `vortex_mixer_genie_2`, unmodified.

License: NONE. The repository has no LICENSE file (license question open as
https://github.com/autobio-bench/AutoBio/issues/10 since 2026-08-24). Without a license these
files may not be redistributed. The project owner allowed them for the hackathon demo only
(decision 2026-10-30, Kanvas ST-05):

- placed only with `build_scene(..., demo_props=True)`;
- scenes holding them report `BuiltScene.redistributable == False`;
- never ship them, or data rendered with them, in a released dataset until AutoBio publishes a
  license. Remove this folder before making the repository public if that has not happened.

Cite: AutoBio: A Simulation and Benchmark for Robotic Automation in Digital Biology Laboratory
(arXiv 2505.14030, ICLR 2026).
