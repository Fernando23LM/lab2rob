"""Load BioProt protocols into lab2rob schema objects.

Tier-1 starts from a curated JSON file (raw BioProt text plus human test
hints); loading straight from the BioProt repo by id plugs into
:func:`load_bioprot_id` later. Source URL, license field and raw text are
always preserved; parsing the raw text into steps is the grounder's job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from lab2rob.schema import GroundedStep, Source, Task

DEFAULT_TIER1_PATH = Path(__file__).resolve().parent / "sources" / "tier1_protocols.json"


@dataclass
class RawProtocol:
    protocol_id: str
    title: str
    description: str
    steps_raw: str
    source_url: str
    license: str
    number_of_steps: int
    dataset: str = "bioprot"
    pseudocode: str = ""
    pseudocode_is_human_edited: bool = False
    #: Human-written test expectations. NEVER a parser input (see tests).
    hints: dict = field(default_factory=dict)

    def to_task(self, steps: list[GroundedStep]) -> Task:
        """Wrap parsed steps in a Task skeleton (skills filled in later)."""
        return Task(
            id=f"{self.dataset}_{self.protocol_id}",
            source=Source(url=self.source_url, license=self.license,
                          steps_used=[s.idx for s in steps]),
            instruction=f"{self.title}: {self.description}",
            steps=steps,
        )


def _raw_from_dict(data: dict) -> RawProtocol:
    source = data.get("source", {})
    return RawProtocol(
        protocol_id=str(data["protocol_id"]),
        dataset=str(data.get("dataset", "bioprot")),
        title=data.get("title", ""),
        description=data.get("description", ""),
        steps_raw=data.get("steps_raw", ""),
        source_url=source.get("url", ""),
        license=source.get("license", ""),
        number_of_steps=int(data.get("number_of_steps", 0)),
        pseudocode=data.get("pseudocode_reference", ""),
        pseudocode_is_human_edited=bool(data.get("pseudocode_is_human_edited", False)),
        hints=dict(data.get("expected_grounding_hints", {})),
    )


def load_tier1_file(path: str | Path = DEFAULT_TIER1_PATH) -> list[RawProtocol]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [_raw_from_dict(p) for p in data.get("protocols", [])]


def load_bioprot_id(bioprot_id: str, tier1_path: str | Path = DEFAULT_TIER1_PATH,
                    repo_root: str | Path | None = None) -> RawProtocol:
    """Load one protocol by BioProt id: tier-1 file now, repo checkout later."""
    if repo_root is not None:
        raise NotImplementedError(
            "Loading from a BioProt repo checkout is not implemented yet; "
            "use the tier-1 file")
    for protocol in load_tier1_file(tier1_path):
        if protocol.protocol_id == str(bioprot_id):
            return protocol
    raise KeyError(f"BioProt id {bioprot_id!r} not in {tier1_path}")
