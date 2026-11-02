"""Labware built from Opentrons labware definitions (ST-05).

Plates, tube racks and tip racks are MuJoCo bodies made of boxes and
cylinders, sized from the vendored Opentrons JSON in labware_defs/opentrons
(Apache-2.0, pinned commit in labware_defs/opentrons/SOURCE.md). Full
definitions give real footprints (96/384-well plates); tube and tip racks are
compact cuts of a definition (a few cells at the real pitch and well size), so
they fit the small bench area the right arm reaches.

Liquid model: a well is biolab.liquid.Well (square opening). A round well
becomes the square of equal area, and its floor sits where the rated volume
fills it to the rim, so capacity matches the definition (conical bottoms are
flattened). The collision geometry follows the same floor.

Placement: placed labware joins a family layout through with_labware(); its
wells join the layout's wells, so the pipette, liquid bookkeeping, perceive()
and the renderer all see them. find_spot() picks free bench space inside the
right arm's straight-down envelope, clear of the family's own labware.
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

DEFS_DIR = Path(__file__).parent / "labware_defs" / "opentrons"
LICENSE = "Apache-2.0 (Opentrons labware definitions); geometry built by lab2rob"

#: Gripper height above the rim when a held pipette approaches a well:
#: certify.run.APPROACH_M (0.03) + biolab.protein.GRIP_ABOVE_TIP (0.08).
PIPETTE_CLEARANCE_M = 0.11

FOOTPRINT_MARGIN_M = 0.012  # free space kept around every footprint
READY_XY = (-0.03, -0.27)  # pipette hangs here at reset (biolab.protein.READY)
READY_KEEPOUT_M = 0.035


@dataclass(frozen=True)
class LabwareKind:
    asset: str  # registry name
    definition: str  # Opentrons definition file stem
    holder: str  # plate | tube_rack | tip_rack
    grid: tuple | None = None  # (columns, rows) compact cut; None = full plate
    rgba: tuple = (0.92, 0.92, 0.9, 1)

    @property
    def liquid(self) -> bool:
        return self.holder != "tip_rack"


KINDS = {k.asset: k for k in (
    LabwareKind("plate_96_well", "corning_96_wellplate_360ul_flat", "plate"),
    LabwareKind("plate_384_well", "corning_384_wellplate_112ul_flat", "plate"),
    LabwareKind("microcentrifuge_tube_1_5ml",
                "opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap", "tube_rack",
                grid=(2, 2), rgba=(0.55, 0.6, 0.68, 1)),
    LabwareKind("centrifuge_tube_15ml", "opentrons_15_tuberack_falcon_15ml_conical",
                "tube_rack", grid=(2, 1), rgba=(0.55, 0.6, 0.68, 1)),
    LabwareKind("centrifuge_tube_50ml", "opentrons_6_tuberack_falcon_50ml_conical",
                "tube_rack", grid=(1, 1), rgba=(0.55, 0.6, 0.68, 1)),
    LabwareKind("tip_rack", "opentrons_96_filtertiprack_200ul", "tip_rack",
                grid=(4, 3), rgba=(0.25, 0.3, 0.4, 1)),
    LabwareKind("pipette_tip_200ul", "opentrons_96_filtertiprack_200ul", "tip_rack",
                grid=(4, 3), rgba=(0.25, 0.3, 0.4, 1)),
    LabwareKind("pipette_tip_1000ul", "opentrons_96_tiprack_1000ul", "tip_rack",
                grid=(4, 2), rgba=(0.25, 0.3, 0.4, 1)),
)}


@functools.lru_cache(maxsize=None)
def load_definition(stem: str) -> dict:
    return json.loads((DEFS_DIR / f"{stem}.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class WellGeom:
    """One cell in the labware's local frame (origin: footprint centre, z = 0 table)."""
    name: str
    xy: tuple  # local centre
    half: float  # half side of the equal-area square opening
    floor: float  # liquid floor height
    rim: float  # top of the well
    radius: float  # true round radius (0 for square wells), for visuals


@dataclass(frozen=True)
class Geometry:
    half_x: float
    half_y: float
    height: float
    pitch: float
    wells: tuple  # WellGeom, in Opentrons order (A1, B1, ..., A2, ...)


def _square_half(well: dict) -> tuple:
    if well["shape"] == "circular":
        radius = well["diameter"] / 2000.0
        return math.sqrt(math.pi) * radius / 2.0, radius
    return math.sqrt(well["xDimension"] * well["yDimension"]) / 2000.0, 0.0


@functools.lru_cache(maxsize=None)
def geometry(asset: str) -> Geometry:
    """Local geometry of an asset's labware, in metres."""
    kind = KINDS[asset]
    d = load_definition(kind.definition)
    columns = d["ordering"]
    if kind.grid is not None:
        n_cols, n_rows = kind.grid
        columns = [col[:n_rows] for col in columns[:n_cols]]
    names = [name for col in columns for name in col]
    first = d["wells"][columns[0][0]]
    pitch = (abs(d["wells"][columns[1][0]]["x"] - first["x"]) if len(columns) > 1
             else abs(d["wells"][columns[0][1]]["y"] - first["y"]) if len(columns[0]) > 1
             else first.get("diameter", first.get("xDimension")) + 4.0) / 1000.0
    xs = [d["wells"][n]["x"] / 1000.0 for n in names]
    ys = [d["wells"][n]["y"] / 1000.0 for n in names]
    if kind.grid is None:  # full plate: the definition's own footprint
        cx, cy = d["dimensions"]["xDimension"] / 2000.0, d["dimensions"]["yDimension"] / 2000.0
        half_x, half_y = cx, cy
    else:  # compact cut: one pitch of rack around the outer cells
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half_x = (max(xs) - min(xs)) / 2 + pitch / 2
        half_y = (max(ys) - min(ys)) / 2 + pitch / 2
    wells, rims = [], []
    for name, x, y in zip(names, xs, ys):
        w = d["wells"][name]
        half, radius = _square_half(w)
        if kind.grid is None:
            rim = (w["z"] + w["depth"]) / 1000.0
        else:  # compact racks stand on the bench: well bottom just above a 3 mm base
            rim = 0.003 + w["depth"] / 1000.0
        depth = min(w["depth"] / 1000.0, w["totalLiquidVolume"] / ((2 * half * 1000) ** 2) / 1000.0)
        wells.append(WellGeom(name, (x - cx, y - cy), half, rim - depth, rim, radius))
        rims.append(rim)
    height = d["dimensions"]["zDimension"] / 1000.0 if kind.grid is None else max(rims)
    return Geometry(half_x, half_y, height, pitch, tuple(wells))


def _rotate(xy, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]])


def _yaw_quat(yaw: float) -> list:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


@dataclass
class PlacedLabware:
    uid: str  # program entity uid (body and well names derive from it)
    asset: str
    xy: tuple
    yaw: float = 0.0

    @property
    def geom(self) -> Geometry:
        return geometry(self.asset)

    @property
    def kind(self) -> LabwareKind:
        return KINDS[self.asset]

    def well_name(self, name: str) -> str:
        return f"{self.uid}:{name}"

    @property
    def first_well(self) -> str:
        return self.well_name(self.geom.wells[0].name)

    def corners(self) -> np.ndarray:
        g = self.geom
        local = [(-g.half_x, -g.half_y), (g.half_x, -g.half_y), (g.half_x, g.half_y), (-g.half_x, g.half_y)]
        return np.array([np.array(self.xy) + _rotate(c, self.yaw) for c in local])

    def wells(self) -> list:
        """biolab Wells for the liquid bookkeeping (none for tip racks)."""
        if not self.kind.liquid:
            return []
        from biolab.liquid import Well  # biolab is on sys.path via build.scene._biolab()
        return [Well(self.well_name(w.name), np.array(self.xy) + _rotate(w.xy, self.yaw), w.floor,
                     w.half, w.rim - w.floor, yaw=self.yaw)
                for w in self.geom.wells]

    def add_to(self, spec: mujoco.MjSpec) -> None:
        g, kind = self.geom, self.kind
        body = spec.worldbody.add_body(name=self.uid, pos=[*self.xy, 0], quat=_yaw_quat(self.yaw))
        box = dict(type=mujoco.mjtGeom.mjGEOM_BOX, rgba=list(kind.rgba))
        if kind.holder == "tip_rack":
            _add_tip_rack(body, g, box)
        else:
            _add_wells_block(body, g, box)
            if kind.holder == "tube_rack":
                _add_tube_shells(body, g)


def _add_wells_block(body, g: Geometry, box: dict) -> None:
    """A block with one square opening per well: a base under the floors, then walls on a grid."""
    floor = min(w.floor for w in g.wells)
    body.add_geom(size=[g.half_x, g.half_y, floor / 2], pos=[0, 0, floor / 2], **box)
    top = max(w.rim for w in g.wells)
    zc, hz = (floor + top) / 2, (top - floor) / 2
    xs = sorted({round(w.xy[0], 6) for w in g.wells})
    ys = sorted({round(w.xy[1], 6) for w in g.wells})
    half = g.wells[0].half
    # Walls between neighbouring columns/rows and an outer frame to the footprint edge.
    for i in range(len(xs) + 1):
        lo = -g.half_x if i == 0 else xs[i - 1] + half
        hi = g.half_x if i == len(xs) else xs[i] - half
        if hi > lo:
            body.add_geom(size=[(hi - lo) / 2, g.half_y, hz], pos=[(hi + lo) / 2, 0, zc], **box)
    for j in range(len(ys) + 1):
        lo = -g.half_y if j == 0 else ys[j - 1] + half
        hi = g.half_y if j == len(ys) else ys[j] - half
        if hi > lo:
            body.add_geom(size=[g.half_x, (hi - lo) / 2, hz], pos=[0, (hi + lo) / 2, zc], **box)


def _add_tube_shells(body, g: Geometry) -> None:
    """Translucent tube bodies and rims (visual only; the block holds the liquid)."""
    for w in g.wells:
        r = w.radius or w.half
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r * 1.06, (w.rim - 0.003) / 2 + 0.001],
                      pos=[*w.xy, (w.rim + 0.003) / 2 + 0.001], contype=0, conaffinity=0,
                      rgba=[0.9, 0.93, 0.97, 0.35])
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r * 1.15, 0.0015],
                      pos=[*w.xy, w.rim + 0.0015], contype=0, conaffinity=0, rgba=[0.95, 0.95, 0.97, 0.9])


def _add_tip_rack(body, g: Geometry, box: dict) -> None:
    """A rack box with tips standing in it (tips are visual; no tip model or attach skill yet)."""
    deck = 0.6 * g.height
    body.add_geom(size=[g.half_x, g.half_y, deck / 2], pos=[0, 0, deck / 2], **box)
    for w in g.wells:
        r = w.radius or w.half
        length = g.height - deck + 0.012
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r * 0.9, length / 2],
                      pos=[*w.xy, g.height - length / 2], contype=0, conaffinity=0,
                      rgba=[0.97, 0.9, 0.55, 0.9])


# -- placement ---------------------------------------------------------------

def _aabb(points: np.ndarray, margin: float = 0.0) -> tuple:
    return (points[:, 0].min() - margin, points[:, 0].max() + margin,
            points[:, 1].min() - margin, points[:, 1].max() + margin)


def _overlaps(a: tuple, b: tuple) -> bool:
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def layout_obstacles(layout) -> list:
    """Bench boxes taken by the family's work area and by placed labware.

    The family's wells count as one box (walls included), so nothing lands in
    the pipette's path between its reservoir and its plate.
    """
    base_wells = super(_WithLabware, layout).wells() if isinstance(layout, _WithLabware) else layout.wells()
    boxes = []
    if base_wells:
        pts = np.array([[w.center[0] + sx * (w.half + 0.008), w.center[1] + sy * (w.half + 0.008)]
                        for w in base_wells for sx in (-1, 1) for sy in (-1, 1)])
        boxes.append(_aabb(pts))
    for placed in getattr(layout, "labware", []):
        boxes.append(_aabb(placed.corners()))
    ready = (READY_XY[0] - READY_KEEPOUT_M, READY_XY[0] + READY_KEEPOUT_M,
             READY_XY[1] - READY_KEEPOUT_M, READY_XY[1] + READY_KEEPOUT_M)
    return boxes + [ready]


def find_spot(asset: str, uid: str, obstacles: list, envelope: dict, rng=None, near=None,
              table=(-0.33, 0.53, -0.58, 0.58)) -> PlacedLabware | None:
    """Free spot for the labware: every corner inside the reach annulus, in front
    of the arm base, clear of obstacles; None when the bench is full.

    Candidates are scanned on a 1 cm grid, closest to `near` first (default:
    straight ahead of the right arm, mid-envelope); rng jitters the chosen spot
    by up to 5 mm / 0.1 rad when that stays valid.
    """
    base = np.array(envelope["base"])
    lo, hi = envelope["min_m"] + 0.01, envelope["max_m"] - 0.005
    near = np.array(near if near is not None else base + [(lo + hi) / 2, 0.0])

    def valid(placed: PlacedLabware) -> bool:
        corners = placed.corners()
        dists = np.linalg.norm(corners - base, axis=1)
        if dists.min() < lo or dists.max() > hi or corners[:, 0].min() < base[0] + 0.05:
            return False
        box = _aabb(corners, FOOTPRINT_MARGIN_M)
        if box[0] < table[0] or box[1] > table[1] or box[2] < table[2] or box[3] > table[3]:
            return False
        return not any(_overlaps(box, o) for o in obstacles)

    xs = np.arange(base[0] - envelope["max_m"], base[0] + envelope["max_m"] + 1e-9, 0.01)
    ys = np.arange(base[1] - envelope["max_m"], base[1] + envelope["max_m"] + 1e-9, 0.01)
    candidates = sorted((math.hypot(x - near[0], y - near[1]), round(x, 3), round(y, 3))
                        for x in xs for y in ys)
    for _, x, y in candidates:
        for yaw in (0.0, math.pi / 2):
            placed = PlacedLabware(uid, asset, (x, y), yaw)
            if not valid(placed):
                continue
            if rng is not None:
                jittered = PlacedLabware(uid, asset, (x + rng.uniform(-0.005, 0.005),
                                                      y + rng.uniform(-0.005, 0.005)),
                                         yaw + rng.uniform(-0.1, 0.1))
                if valid(jittered):
                    return jittered
            return placed
    return None


# -- layouts with extra labware ---------------------------------------------

class _WithLabware:
    """Mixin: a family layout plus placed labware (wells join the family's)."""

    labware: list

    def wells(self):
        return super().wells() + [w for placed in self.labware for w in placed.wells()]

    def add_to(self, spec) -> None:
        for placed in self.labware:
            placed.add_to(spec)
        super().add_to(spec)  # adds liquid geoms for self.wells(), extras included


@functools.lru_cache(maxsize=None)
def _layout_class(base: type) -> type:
    return type(f"{base.__name__}WithLabware", (_WithLabware, base), {})


def with_labware(layout, placed: PlacedLabware):
    """A copy of the layout with one more piece of labware (isinstance of the family layout)."""
    if isinstance(layout, _WithLabware):
        base = type(layout).__mro__[2]
        existing = list(layout.labware)
    else:
        base, existing = type(layout), []
    out = object.__new__(_layout_class(base))
    out.__dict__.update({k: v for k, v in vars(layout).items() if k != "labware"})
    out.labware = existing + [placed]
    return out
