"""Grounded-name -> registry-status mapping: the grounder contract.

IG-03 wires this in: a step matching a declared entry is reported as
``out_of_scope: not_implemented`` rather than ``unknown``. Macros are in
scope (the program planner expands them); only implemented and macro skills
may appear in a compiled program.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import yaml

DIR = Path(__file__).parent

NOT_IMPLEMENTED = "not_implemented"
UNKNOWN_SKILL = "unknown_skill"
UNKNOWN_ASSET = "unknown_asset"

#: path -> (content digest, parsed document). Grounding resolves names once
#: per step, and parsing the YAML each time dominated a corpus sweep (~0.3 s
#: per step); the digest keeps runtime edits (learned skills) visible.
_PARSED: dict = {}


def shared_yaml(path: Path):
    """Parsed YAML file, reparsed only when its content changes.

    The SAME object is returned to every caller: read it, never mutate it
    (load_yaml gives a private copy).
    """
    data = Path(path).read_bytes()
    digest = hashlib.sha1(data).hexdigest()
    hit = _PARSED.get(str(path))
    if hit is None or hit[0] != digest:
        hit = (digest, yaml.safe_load(data.decode("utf-8")))
        _PARSED[str(path)] = hit
    return hit[1]


def load_yaml(path: Path):
    """A private copy of a parsed YAML file (safe to mutate)."""
    return copy.deepcopy(shared_yaml(path))


def _load(name: str, key: str) -> list:
    return shared_yaml(DIR / name)[key]


def resolve_skill(name: str, skills=None) -> dict:
    """Classify a skill name: implemented | macro | declared | missing."""
    skills = skills if skills is not None else _load("skills.yaml", "skills")
    entry = next((s for s in skills if s["name"] == name), None)
    entry = copy.deepcopy(entry)
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
    entry = copy.deepcopy(entry)
    if entry is None:
        return {"name": name, "status": None, "scope": "out_of_scope",
                "reason": UNKNOWN_ASSET, "entry": None}
    if entry["status"] == "implemented":
        return {"name": name, "status": "implemented", "scope": "in_scope",
                "reason": None, "entry": entry}
    return {"name": name, "status": "declared", "scope": "out_of_scope",
            "reason": NOT_IMPLEMENTED, "entry": entry}
