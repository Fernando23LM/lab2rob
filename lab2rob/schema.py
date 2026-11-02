"""Shared data contract for every lab2rob stage: the certified Task object.

Grounding produces steps + entities, the planners add skills + judge +
outcome, and build/certify/eval consume them — all validated here so
malformed LLM output is rejected at stage boundaries instead of crashing
the simulator three stages later. Starting point: section 4 of
LAB2ROB_CONTEXT.md.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Source(BaseModel):
    """Where the task came from. URL + license are mandatory (rule 7)."""

    url: str
    license: str
    steps_used: list[int] = Field(default_factory=list)
    version: Optional[str] = None
    retrieved: Optional[str] = None  # ISO date


class Entity(BaseModel):
    """One task-level object; uid is stable across steps, skills, predicates."""

    uid: str
    asset: str
    slot: Optional[str] = None
    ontology_id: Optional[str] = None  # OBI / ChEBI / UO identifier


class GroundedStep(BaseModel):
    """One protocol step in simulator terms. Skills start empty: grounding
    runs before program planning, which fills them in."""

    idx: int
    source_text: str
    skills: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)  # entity uids used


class StepPredicate(BaseModel):
    """One check of the independent judge against a single step."""

    step: int
    pred: str
    args: list[str] = Field(default_factory=list)
    target: Optional[float] = None
    tol: Optional[float] = None


class OutcomeCheck(BaseModel):
    """Scientific-result check, when the protocol has a measurable result."""

    type: str
    quantity: Optional[str] = None
    tol_rel: Optional[float] = None
    tol_abs: Optional[float] = None


class SafetyPredicate(BaseModel):
    """Invariant that must hold on every run (e.g. no_spill)."""

    pred: str
    args: list[str] = Field(default_factory=list)


TwinLabel = Literal["refuse", "ask", "flag"]


class SafetyTwin(BaseModel):
    """Unsafe variant of the task plus the behavior the agent should show."""

    instruction: str
    violation: str
    expected: TwinLabel


class OutOfScopeItem(BaseModel):
    """A protocol step the compiler could not handle, with a reason category."""

    step: int
    reason: str
    detail: str = ""


class Difficulty(BaseModel):
    level: str = "trained_technician"
    est_human_min: Optional[float] = None


class Certification(BaseModel):
    """The four certification gates; all must pass (section 3.1)."""

    compiled: bool = False
    executed: bool = False
    predicates: bool = False
    visual: bool = False

    def certified(self) -> bool:
        return self.compiled and self.executed and self.predicates and self.visual


class Task(BaseModel):
    id: str = Field(min_length=1)
    source: Source
    instruction: str = Field(min_length=1)
    entities: list[Entity] = Field(default_factory=list)
    initial_state: dict[str, Any] = Field(default_factory=dict)
    steps: list[GroundedStep] = Field(min_length=1)
    judge: list[StepPredicate] = Field(default_factory=list)
    outcome: Optional[OutcomeCheck] = None
    safety: list[SafetyPredicate] = Field(default_factory=list)
    safety_twin: Optional[SafetyTwin] = None
    out_of_scope: list[OutOfScopeItem] = Field(default_factory=list)
    difficulty: Difficulty = Field(default_factory=Difficulty)
    certification: Certification = Field(default_factory=Certification)

    @field_validator("steps")
    @classmethod
    def _unique_step_ids(cls, steps):
        idxs = [s.idx for s in steps]
        if len(set(idxs)) != len(idxs):
            raise ValueError(f"Step idx values must be unique, got {idxs}")
        return steps

    @field_validator("entities")
    @classmethod
    def _unique_entity_ids(cls, entities):
        uids = [e.uid for e in entities]
        if len(set(uids)) != len(uids):
            raise ValueError(f"Entity uids must be unique, got {uids}")
        return entities

    @model_validator(mode="after")
    def _references_resolve(self):
        idxs = {s.idx for s in self.steps}
        for pred in self.judge:
            if pred.step not in idxs:
                raise ValueError(
                    f"Judge predicate {pred.pred!r} refers to unknown step {pred.step}"
                )
        uids = {e.uid for e in self.entities}
        for step in self.steps:
            for uid in step.entities:
                if uid not in uids:
                    raise ValueError(
                        f"Step {step.idx} refers to unknown entity {uid!r}"
                    )
        return self
