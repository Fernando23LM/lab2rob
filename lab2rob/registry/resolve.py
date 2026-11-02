"""Grounded-name -> registry-status mapping: the grounder contract.

IG-03 wires this in: a step matching a declared entry is reported as
``out_of_scope: not_implemented`` rather than ``unknown``. Macros are in
scope (the program planner expands them); only implemented and macro skills
may appear in a compiled program.
"""

from __future__ import annotations

from pathlib import Path

import yaml

DIR = Path(__file__).parent

NOT_IMPLEMENTED = "not_implemented"
UNKNOWN_SKILL = "unknown_skill"
UNKNOWN_ASSET = "unknown_asset"


def _load(name: str, key: str) -> list:
    return yaml.safe_load((DIR / name).read_text(encoding="utf-8"))[key]


def resolve_skill(name: str, skills=None) -> dict:
    """Classify a skill name: implemented | macro | declared | missing."""
    skills = skills if skills is not None else _load("skills.yaml", "skills")
    entry = next((s for s in skills if s["name"] == name), None)
    if entry is None:
        return {"name": name, "status": None, "scope": "out_of_scope",
                "reason": UNKNOWN_SKILL, "entry": None}
    if entry["status"] in ("implemented", "macro"):
        return {"name": name, "status": entry["status"], "scope": "in_scope",
                "reason": None, "entry": entry}
    return {"name": name, "status": "declared", "scope": "out_of_scope",
            "reason": NOT_IMPLEMENTED, "entry": entry}


def resolve_asset(name: str, assets=None) -> dict:
    """Classify an asset name: implemented | declared | missing."""
    assets = assets if assets is not None else _load("assets.yaml", "assets")
    entry = next((a for a in assets if a["name"] == name), None)
    if entry is None:
        return {"name": name, "status": None, "scope": "out_of_scope",
                "reason": UNKNOWN_ASSET, "entry": None}
    if entry["status"] == "implemented":
        return {"name": name, "status": "implemented", "scope": "in_scope",
                "reason": None, "entry": entry}
    return {"name": name, "status": "declared", "scope": "out_of_scope",
            "reason": NOT_IMPLEMENTED, "entry": entry}
