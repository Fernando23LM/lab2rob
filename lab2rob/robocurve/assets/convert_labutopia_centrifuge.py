#!/usr/bin/env python3
"""Convert LabUtopia's Centrifuge_02 (USD, Isaac Sim) into MuJoCo meshes.

    pip install usd-core
    python convert_labutopia_centrifuge.py Centrifuge_02.usd OUT_DIR

Writes binary STL files in metres, in the frames the runtime's centrifuge
scene expects: ``body.stl`` and ``panel.stl`` in the asset's own frame, and
``lid.stl`` relative to the lid's hinge (so the hinge joint rotates it).
Only geometry is kept: no materials, textures or physics (the USD has none
of the latter; collision shapes are built by hand in the scene).

Source: LabUtopia (https://rui-li023.github.io/labutopia-site), dataset
huggingface.co/datasets/Ruinwalker/LabUtopia-Dataset,
Instruments/Centrifuge/Centrifuge_02/Centrifuge_02.usd, CC BY-NC 4.0.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

#: Hinge of the lid in the asset frame (metres): the open lid's inner face
#: (x = -0.063) at the height of the well's rim (z = 0.087).
HINGE = np.array([-0.063, 0.0, 0.087])
PARTS = {"body": ["mesh_000"], "panel": ["mesh_001", "mesh_002"], "lid": ["mesh_003"]}


def _triangles(mesh, xf) -> np.ndarray:
    from pxr import Usd  # noqa: F401
    points = np.array([xf.Transform(p) for p in mesh.GetPointsAttr().Get()]) * 0.01
    counts = mesh.GetFaceVertexCountsAttr().Get()
    index = mesh.GetFaceVertexIndicesAttr().Get()
    tris, i = [], 0
    for n in counts:
        face = index[i:i + n]
        for k in range(1, n - 1):  # fan triangulation (the faces are convex polygons)
            tris.append((face[0], face[k], face[k + 1]))
        i += n
    return points[np.array(tris)]


def _write_stl(path: Path, tris: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(b"converted from LabUtopia Centrifuge_02 (CC BY-NC 4.0)".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            normal = np.cross(b - a, c - a)
            norm = np.linalg.norm(normal)
            normal = normal / norm if norm > 0 else normal
            f.write(struct.pack("<12fH", *normal, *a, *b, *c, 0))


def convert(usd_path, out_dir) -> dict:
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(usd_path))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for part, names in PARTS.items():
        tris = []
        for name in names:
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/World/group_000/{name}"))
            tris.append(_triangles(mesh, mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())))
        tris = np.concatenate(tris)
        if part == "lid":
            tris = tris - HINGE
        _write_stl(out / f"{part}.stl", tris.astype(np.float32))
        report[part] = {"triangles": int(len(tris)), "min": tris.reshape(-1, 3).min(0).round(4).tolist(),
                        "max": tris.reshape(-1, 3).max(0).round(4).tolist()}
    return report


if __name__ == "__main__":
    import json
    print(json.dumps(convert(sys.argv[1], sys.argv[2]), indent=1))
