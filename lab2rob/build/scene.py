"""Scene builder: planned program -> placed biolab scene + reach check.

Family is picked by capability (skills, entity kinds, volumes), vessels are
assigned from the family's pools (named vessels first, unbound roles from
defaults), and every interaction point is reach-checked against the YAM
envelope from the asset registry. Unbuildable specs raise SceneRejected
with a reason — never a half-built scene.

Coverage policy (the equipment question): only implemented, containable
assets are placed. Named labware with a real model (96/384-well plates,
1.5/15/50 mL tubes, tip racks; build/labware.py) is placed on free bench
space beside the family's labware and its wells join the liquid
bookkeeping. Declared vessels map through SUBSTITUTIONS (recorded);
instruments that need no physical model (incubator, reader, ...) are noted
as unplaced; anything else declared is rejected as missing_asset. Named
declared vessels rarely reach this stage — grounding usually drops those
steps first — so a grounding-time substitution pass is tracked as follow-up
work on ST-01.

biolab lives in a nested checkout (not installed); _biolab() puts it on
sys.path once. Proper packaging is an infra follow-up, not SC-03.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lab2rob.build import instruments, labware
from lab2rob.registry import load_assets

_BIOLAB_ROOT = Path(__file__).resolve().parents[2] / "Code-as-a-Policy-Evolving-Skill"

#: Declared vessels and their buildable stand-ins (recorded, never silent).
SUBSTITUTIONS = {
    "reagent_reservoir_50ml": "reservoir_block",
    "petri_dish_90mm": "culture_dish",
}

#: Instruments with no physical model that their skills do not need placed
#: (incubate/read_plate are API calls, not placements).
NON_PLACEABLE = {"centrifuge_mini", "incubator_co2", "plate_reader", "vortexer",
                 "ph_meter"}

#: Always present via the scene/env, never assigned to program roles.
SCENE_PROVIDED = {"yam_arm", "table", "floor", "pipette"}

RIGHT_ARM = "right"


class SceneRejected(Exception):
    """The spec cannot become a scene; reason + detail say why."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass
class BuiltScene:
    family: str
    seed: int
    layout: object  # biolab layout instance (ProteinLayout, ...)
    assignments: dict = field(default_factory=dict)  # "step:op" -> slot name
    slots: dict = field(default_factory=dict)  # slot -> {"pos":[x,y,z], ...}
    vessels: dict = field(default_factory=dict)  # uid -> {"asset":..,"slot":..}
    substitutions: list = field(default_factory=list)  # {"from":..,"to":..}
    unplaced: list = field(default_factory=list)  # equipment notes
    aspirate_sources: list = field(default_factory=list)  # slots, in order
    labware: list = field(default_factory=list)  # placed extra labware records
    instruments: list = field(default_factory=list)  # demo-only AutoBio props

    @property
    def redistributable(self) -> bool:
        """False when the scene holds an unlicensed demo prop (keep out of released datasets)."""
        return all(r.get("redistributable", True) for r in self.instruments)


def _biolab():
    """Import the nested biolab checkout (once) and return its modules."""
    root = str(_BIOLAB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    import biolab.culture as culture
    import biolab.protein as protein
    import biolab.scene as scene
    return protein, culture, scene


def reach_envelope() -> dict:
    """YAM straight-down envelope + arm base, from registry and scene truth."""
    _, _, scene = _biolab()
    dims = next(a for a in load_assets() if a["name"] == "yam_arm")["dimensions"]
    return {"min_m": dims["straight_down_reach_min_m"],
            "max_m": dims["straight_down_reach_max_m"],
            "height_max_m": dims["straight_down_height_max_m"],
            "base": scene.ARM_BASE[RIGHT_ARM][:2]}


def _family_table():
    """Family -> layout class, pipette cap, and named well pools."""
    protein, culture, _ = _biolab()
    return {
        "protein": {
            "layout_cls": protein.ProteinLayout,
            "pipette_ul": protein.PipetteEnv.PIPETTE_CAPACITY,
            "plate": [f"{row}{i}" for row in "AB" for i in (1, 2, 3, 4)],
            "trough": list(protein.RESERVOIRS),
        },
        "culture": {
            "layout_cls": culture.CultureLayout,
            "pipette_ul": culture.CultureEnv.PIPETTE_CAPACITY,
            "plate": list(culture.DISHES),
            "trough": list(culture.RESERVOIRS),
        },
    }


def _asset_entries() -> dict:
    return {a["name"]: a for a in load_assets()}


def _resolve_named_asset(asset: str, entries: dict, built: BuiltScene) -> str | None:
    """Map a program entity asset to a placeable one.

    Returns None for scene-provided/unplaced assets (recorded, not assigned);
    raises SceneRejected for assets with no physical stand-in.
    """
    if asset in SCENE_PROVIDED:
        return None
    if asset in NON_PLACEABLE:
        built.unplaced.append(f"{asset}: no physical model needed")
        return None
    entry = entries.get(asset)
    if entry is None:
        raise SceneRejected("missing_asset", f"unknown asset {asset!r}")
    if entry["status"] == "implemented":
        target, target_entry = asset, entry
    elif asset in SUBSTITUTIONS:
        target = SUBSTITUTIONS[asset]
        target_entry = entries[target]
        built.substitutions.append({"from": asset, "to": target})
    else:
        raise SceneRejected("missing_asset",
                            f"{asset} is declared with no buildable stand-in")
    affordances = target_entry.get("affordances", [])
    if "insertable" in affordances:
        raise SceneRejected("missing_asset",
                            f"{asset} needs a tube rack; pipetting families have none")
    if "containable" not in affordances:
        raise SceneRejected("missing_asset",
                            f"{asset} cannot hold liquid in this scene")
    return target


def _program_signals(program) -> tuple:
    skills, volumes, assets = set(), [], set()
    uid_asset = {e.uid: e.asset for e in program.entities}
    for step in program.steps:
        if step.status != "planned":
            continue
        for op in step.ops:
            skills.add(op["skill"])
            if isinstance(op["args"].get("volume_ul"), (int, float)):
                volumes.append(float(op["args"]["volume_ul"]))
            if op.get("at") in uid_asset:
                assets.add(uid_asset[op["at"]])
    return skills, volumes, assets


def pick_family(program) -> str:
    """Choose the scene family by capability; reject conflicts."""
    skills, volumes, _ = _program_signals(program)
    max_volume = max(volumes) if volumes else 0.0
    # incubate lives in the culture/drug APIs only; a scene must run every op.
    needs_dishes = bool({"inspect", "incubate"} & skills) or any(
        e.asset == "culture_dish" for e in program.entities)
    needs_plate = bool({"read_plate", "report_concentration", "report_ic50"} & skills)
    if needs_dishes and needs_plate:
        # A drug family would satisfy both; v1 has protein + culture only.
        raise SceneRejected("family_conflict",
                            "program needs both culture dishes and a readable plate")
    if needs_dishes:
        return "culture"
    if needs_plate:
        return "protein"
    if max_volume > 1000.0:
        raise SceneRejected("over_capacity",
                            f"{max_volume:g} uL exceeds the largest pipette (1000 uL)")
    if max_volume > 200.0:
        return "culture"
    return "protein"


def _slot_positions(layout) -> dict:
    return _slot_positions_of(layout.wells())


def _slot_positions_of(wells) -> dict:
    return {w.name: {"pos": [float(w.center[0]), float(w.center[1]),
                                  float(w.floor + w.depth)]}
            for w in wells}


def _labware_slot(uid: str, asset: str, place, envelope: dict) -> str:
    """The uid's first well on its own placed labware; rejects full benches and tall rims."""
    spot = place(uid, asset)
    if spot is None:
        raise SceneRejected("over_capacity",
                            f"no free bench space within reach for {asset} ({uid})")
    rim = max(w.rim for w in spot.geom.wells)
    if rim + labware.PIPETTE_CLEARANCE_M >= envelope["height_max_m"]:
        raise SceneRejected(
            "unreachable",
            f"{asset} rim at z={rim:.3f} m: a held pipette puts the gripper at "
            f"{rim + labware.PIPETTE_CLEARANCE_M:.3f} m, past the straight-down "
            f"max {envelope['height_max_m']} m (needs tilt)")
    return spot.first_well


def check_reach(points: dict, envelope: dict | None = None) -> list:
    """Points {name: (x, y, z)} outside the straight-down envelope.

    Pure geometry, no simulator: planar distance from the arm base within
    [min_m, max_m] and interaction height below height_max_m.
    """
    envelope = envelope or reach_envelope()
    base_x, base_y = envelope["base"]
    violations = []
    for name, (x, y, z) in points.items():
        dist = math.hypot(x - base_x, y - base_y)
        if not envelope["min_m"] <= dist <= envelope["max_m"]:
            violations.append(
                f"{name} at ({x:.3f}, {y:.3f}) is {dist:.3f} m from the "
                f"{RIGHT_ARM} base (envelope {envelope['min_m']}-"
                f"{envelope['max_m']} m)")
        elif z >= envelope["height_max_m"]:
            violations.append(
                f"{name} at z={z:.3f} m needs tilt (straight-down max "
                f"{envelope['height_max_m']} m)")
    return violations


def _family_of_layout(layout) -> str:
    protein, culture, _ = _biolab()
    if isinstance(layout, protein.ProteinLayout):
        return "protein"
    if isinstance(layout, culture.CultureLayout):
        return "culture"
    raise ValueError(f"unsupported layout type {type(layout).__name__}")


def build_scene(program, seed: int = 0, layout=None, demo_props: bool = False) -> BuiltScene:
    """Place a planned program's vessels and reach-check the result.

    layout overrides sampling (tests/advanced use) and implies the family.
    demo_props also sets out the AutoBio instruments the program names
    (demo only: unlicensed, so the scene is marked not redistributable).
    """
    families = _family_table()
    family = _family_of_layout(layout) if layout is not None else pick_family(program)
    spec = families[family]
    if layout is None:
        layout = spec["layout_cls"].sample(np.random.default_rng(seed))
    built = BuiltScene(family=family, seed=seed, layout=layout)
    built.slots = _slot_positions(layout)
    entries = _asset_entries()
    uid_asset = {e.uid: e.asset for e in program.entities}

    envelope = reach_envelope()
    labware_rng = np.random.default_rng([seed, 1])  # separate stream: family layouts stay as before
    placed: dict[str, labware.PlacedLabware] = {}

    def place(uid: str, asset: str, near=None) -> labware.PlacedLabware | None:
        """Put the uid's labware on free bench space (once); None when none is left."""
        if uid in placed:
            return placed[uid]
        spot = labware.find_spot(asset, uid, labware.layout_obstacles(built.layout),
                                 envelope, labware_rng, near=near)
        if spot is None:
            return None
        built.layout = labware.with_labware(built.layout, spot)
        placed[uid] = spot
        built.slots.update(_slot_positions_of(spot.wells()))
        built.labware.append({"uid": uid, "asset": asset,
                              "xy": [round(float(v), 4) for v in spot.xy],
                              "yaw": round(float(spot.yaw), 4),
                              "definition": spot.kind.definition,
                              "license": entries[asset].get("license", "")})
        return spot

    free = {"plate": [w for w in spec["plate"] if w in built.slots],
            "trough": [w for w in spec["trough"] if w in built.slots]}
    assigned: dict[str, str] = {}  # uid -> slot
    resolved: dict[str, str | None] = {}  # uid -> placeable asset (once each)

    def take(uid: str | None, prefer: str) -> str:
        if uid is not None and uid in assigned:
            return assigned[uid]
        for pool in (prefer, "plate" if prefer == "trough" else "trough"):
            if free[pool]:
                slot = free[pool].pop(0)
                if uid is not None:
                    assigned[uid] = slot
                return slot
        raise SceneRejected("over_capacity",
                            f"no free {prefer} slot left in the {family} scene")

    for step in program.steps:
        if step.status != "planned":
            continue
        for op_idx, op in enumerate(step.ops):
            if op["skill"] not in ("aspirate", "dispense"):
                continue
            volume = op["args"].get("volume_ul")
            if isinstance(volume, (int, float)) and volume > spec["pipette_ul"]:
                raise SceneRejected(
                    "over_capacity",
                    f"{volume:g} uL exceeds the {family} pipette ({spec['pipette_ul']:g} uL)")
            uid = op.get("at")
            if uid is not None and uid not in uid_asset:
                uid = None  # unknown ref degrades to an unbound role
            if uid is not None:
                if uid not in resolved:
                    resolved[uid] = _resolve_named_asset(uid_asset[uid], entries, built)
                if resolved[uid] is None:
                    continue  # scene-provided or unplaced equipment
                built.vessels.setdefault(uid, {"asset": resolved[uid], "slot": None})
            prefer = "trough" if op["skill"] == "aspirate" else "plate"
            if uid is not None and resolved[uid] in labware.KINDS:
                slot = _labware_slot(uid, resolved[uid], place, envelope)
                assigned[uid] = slot
            else:
                slot = take(uid, prefer)
            if uid is not None:
                built.vessels[uid]["slot"] = slot
            built.assignments[f"{step.idx}:{op_idx}"] = slot
            if op["skill"] == "aspirate" and slot not in built.aspirate_sources:
                built.aspirate_sources.append(slot)

    # Named labware no op touches (tip racks, a plate only mentioned) is set out
    # towards the back of the reach, away from the work; a full bench is
    # noted, never a rejection.
    back = np.array(envelope["base"]) + [0.12, -0.22]
    for entity in program.entities:
        if (entity.uid not in placed and entity.asset in labware.KINDS
                and entries.get(entity.asset, {}).get("status") == "implemented"
                and place(entity.uid, entity.asset, near=back) is None):
            built.unplaced.append(f"{entity.asset}: no free bench space for {entity.uid}")

    if demo_props:
        _place_demo_props(program, built)

    used = {slot: tuple(built.slots[slot]["pos"]) for slot in set(built.assignments.values())}
    violations = check_reach(used)
    if violations:
        raise SceneRejected("unreachable", "; ".join(violations))
    return built


def _place_demo_props(program, built: BuiltScene) -> None:
    """AutoBio instruments named by the program, at the far end of the bench."""
    spots = iter(instruments.DEMO_SPOTS)
    seen = set()
    for entity in program.entities:
        if entity.asset not in instruments.MODELS or entity.asset in seen:
            continue
        seen.add(entity.asset)
        spot = next(spots, None)
        if spot is None:
            built.unplaced.append(f"{entity.asset}: no demo spot left for {entity.uid}")
            continue
        prop = instruments.PlacedInstrument(entity.uid, entity.asset, spot[:2], spot[2])
        built.layout = labware.with_labware(built.layout, prop)
        built.instruments.append(prop.record())


def to_mujoco(built: BuiltScene):
    """Compile the placed scene to a MuJoCo model (env wiring is SC-04)."""
    _, _, scene = _biolab()
    return scene.build_spec(built.layout).compile()
