"""Demo-only instrument props from AutoBio (ST-05).

AutoBio (github.com/autobio-bench/AutoBio) ships MuJoCo models of real lab
instruments but has NO license file, so these models may not be
redistributed. The human allowed them for the hackathon demo only:

- they are never placed unless build_scene(..., demo_props=True);
- every placement is recorded with REDISTRIBUTABLE = False, so dataset
  exports must drop scenes that contain them until AutoBio publishes a license.

The vendored MJCF and meshes in autobio/ are unmodified (pinned commit in
autobio/SOURCE.md). At load time the AutoBio-only plugin (a knob detent), the
model options, the vortexer's motor, and all joints and joint couplings are
stripped: the props are rigid scenery in their modeled pose (centrifuge lid
open), and the env's actuator and joint lists stay the arms'.
The centrifuge and vortex skills stay declared; placing a prop does not make
those steps compilable.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import mujoco

AUTOBIO_DIR = Path(__file__).parent / "autobio"
LICENSE = ("UNLICENSED: AutoBio (github.com/autobio-bench/AutoBio) has no LICENSE file; "
           "hackathon demo only, not redistributable")
REDISTRIBUTABLE = False

#: Registry asset -> AutoBio instrument model.
MODELS = {
    "centrifuge_mini": "centrifuge_tiangen_tgear_mini",
    "vortexer": "vortex_mixer_genie_2",
}

#: Spots at the far end of the bench, past the arms' reach (the props are
#: scenery; no implemented skill operates them). Yaw turns the fronts to the arms.
DEMO_SPOTS = ((0.33, -0.40, -math.pi / 2), (0.36, -0.20, -math.pi / 2))
DEMO_HALF_M = 0.13  # generous half footprint, for keeping labware clear


def _sanitized_xml(model: str) -> str:
    path = AUTOBIO_DIR / "model" / "instrument" / f"{model}.xml"
    xml = path.read_text(encoding="utf-8")
    xml = re.sub(r"<extension>.*?</extension>|<extension\s*/>", "", xml, flags=re.S)
    xml = re.sub(r"<plugin[^>]*/>", "", xml)
    xml = re.sub(r"<option[^>]*/>", "", xml)
    xml = re.sub(r"<actuator>.*?</actuator>|<actuator\s*/>", "", xml, flags=re.S)
    assets = (AUTOBIO_DIR / "assets").resolve().as_posix()
    return re.sub(r'assetdir="[^"]*"', f'assetdir="{assets}"', xml)


def load_model(asset: str) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_string(_sanitized_xml(MODELS[asset]))
    for element in [*spec.equalities, *spec.joints]:
        spec.delete(element)
    return spec


@dataclass
class PlacedInstrument:
    uid: str
    asset: str
    xy: tuple
    yaw: float = 0.0

    def add_to(self, spec: mujoco.MjSpec) -> None:
        frame = spec.worldbody.add_frame(
            pos=[*self.xy, 0.0], quat=[math.cos(self.yaw / 2), 0, 0, math.sin(self.yaw / 2)])
        spec.attach(load_model(self.asset), prefix=f"{self.uid}/", frame=frame)

    def wells(self) -> list:
        return []

    def corners(self):
        import numpy as np
        h = DEMO_HALF_M
        return np.array(self.xy) + np.array([(-h, -h), (h, -h), (h, h), (-h, h)])

    def record(self) -> dict:
        return {"uid": self.uid, "asset": self.asset, "xy": [round(v, 4) for v in self.xy],
                "yaw": round(self.yaw, 4), "model": f"autobio/{MODELS[self.asset]}",
                "license": LICENSE, "redistributable": REDISTRIBUTABLE}
