"""Asset + skill registries: the compiler's closed vocabulary.

Nothing outside these files may appear in a scene or a program (rule 3);
anything else becomes out_of_scope with a reason.
"""

from pathlib import Path

import yaml

from .resolve import NOT_IMPLEMENTED, resolve_asset, resolve_skill

DIR = Path(__file__).parent


def load_skills():
    """Skill entries from skills.yaml (generated: implemented + macro + declared)."""
    return yaml.safe_load((DIR / "skills.yaml").read_text(encoding="utf-8"))["skills"]


def load_assets():
    """Asset entries from assets.yaml (implemented + declared)."""
    return yaml.safe_load((DIR / "assets.yaml").read_text(encoding="utf-8"))["assets"]


def load_lexicon():
    """Lexicon entries: canonical name -> kind, synonyms, ontology id."""
    return yaml.safe_load((DIR / "lexicon.yaml").read_text(encoding="utf-8"))["lexicon"]


def lexicon_index():
    """Lowercased synonym -> canonical name. Duplicate synonyms are an error."""
    index = {}
    for entry in load_lexicon():
        for synonym in entry.get("synonyms", []):
            key = synonym.lower()
            if key in index and index[key] != entry["canonical"]:
                raise ValueError(f"Synonym {synonym!r} maps to both "
                                 f"{index[key]!r} and {entry['canonical']!r}")
            index[key] = entry["canonical"]
    return index


def skill_names(status=None):
    return [s["name"] for s in load_skills()
            if status is None or s["status"] == status]


def asset_names(status=None):
    return [a["name"] for a in load_assets()
            if status is None or a["status"] == status]


__all__ = ["NOT_IMPLEMENTED", "asset_names", "lexicon_index", "load_assets",
           "load_lexicon", "load_skills", "resolve_asset", "resolve_skill",
           "skill_names"]
