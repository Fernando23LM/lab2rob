"""Certify: run programs in built scenes, trace, grade, see, repair (SC-04/05)."""

from .repair import certify_with_repair, diagnose
from .run import RunReport, certify, evaluate_predicates
from .visual import FakeBackend, LLMBackend, check_visual

__all__ = ["FakeBackend", "LLMBackend", "RunReport", "certify",
           "certify_with_repair", "check_visual", "diagnose",
           "evaluate_predicates"]
