"""Step predicates for independent judges: small, total, simulator-free checks.

Each predicate reads a plain state mapping and returns a CheckResult; a
missing key is a failure with an explanation, never an exception, so a judge
run cannot crash on a partial trace.

State protocol (all keys optional)::

    {
      "tubes": {"tube_1": {"slot": ["rack_dest", 0, 1], "lid_angle_rad": 0.0}},
      "wells": {"A1": {"volume_ul": 100.0, "conc": 1.0}},
      "spilled_ul": 0.0,
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CheckResult:
    passed: bool
    detail: str


def _tube(state: dict, tube: str):
    try:
        return state["tubes"][tube]
    except (KeyError, TypeError):
        return None


def _well(state: dict, well: str):
    try:
        return state["wells"][well]
    except (KeyError, TypeError):
        return None


def in_slot(state: dict, tube: str, rack: str, ix: int, iy: int) -> CheckResult:
    """The tube sits in the given rack cell."""
    info = _tube(state, tube)
    if info is None:
        return CheckResult(False, f"no state for tube {tube!r}")
    slot = info.get("slot")
    if slot is None:
        return CheckResult(False, f"{tube} is not in any slot")
    want = [rack, ix, iy]
    if list(slot) == want:
        return CheckResult(True, f"{tube} in {rack}[{ix},{iy}]")
    return CheckResult(False, f"{tube} in {slot}, want {want}")


def capped(state: dict, tube: str, tol_rad: float = 0.05) -> CheckResult:
    """A snap-cap lid is shut (angle within tolerance of zero)."""
    info = _tube(state, tube)
    if info is None:
        return CheckResult(False, f"no state for tube {tube!r}")
    angle = info.get("lid_angle_rad")
    if angle is None:
        return CheckResult(False, f"{tube} has no lid angle recorded")
    if abs(angle) <= tol_rad:
        return CheckResult(True, f"{tube} lid shut ({angle:.3f} rad)")
    return CheckResult(False, f"{tube} lid open at {angle:.3f} rad")


def lid_open(state: dict, tube: str, min_angle_rad: float) -> CheckResult:
    """A snap-cap lid is open past the given angle."""
    info = _tube(state, tube)
    if info is None:
        return CheckResult(False, f"no state for tube {tube!r}")
    angle = info.get("lid_angle_rad")
    if angle is None:
        return CheckResult(False, f"{tube} has no lid angle recorded")
    if angle >= min_angle_rad:
        return CheckResult(True, f"{tube} lid open at {angle:.3f} rad")
    return CheckResult(False, f"{tube} lid at {angle:.3f} rad, want >= {min_angle_rad}")


def volume(state: dict, well: str, target_ul: float, tol_ul: float) -> CheckResult:
    """A well holds the target volume within tolerance."""
    info = _well(state, well)
    if info is None:
        return CheckResult(False, f"no state for well {well!r}")
    actual = info.get("volume_ul")
    if actual is None:
        return CheckResult(False, f"{well} has no volume recorded")
    if abs(actual - target_ul) <= tol_ul:
        return CheckResult(True, f"{well} holds {actual:.1f} uL")
    return CheckResult(False, f"{well} holds {actual:.1f} uL, want {target_ul} +- {tol_ul}")


def concentration(state: dict, well: str, target: float, tol_rel: float = 0.10) -> CheckResult:
    """A well holds the target concentration within relative tolerance."""
    info = _well(state, well)
    if info is None:
        return CheckResult(False, f"no state for well {well!r}")
    actual = info.get("conc")
    if actual is None:
        return CheckResult(False, f"{well} has no concentration recorded")
    tol = tol_rel * abs(target) if target else tol_rel
    if abs(actual - target) <= tol:
        return CheckResult(True, f"{well} at {actual:.4g}")
    return CheckResult(False, f"{well} at {actual:.4g}, want {target} +- {tol_rel * 100:.0f}%")


def no_spill(state: dict, tol_ul: float = 0.0) -> CheckResult:
    """Nothing (past tolerance) was spilled outside labware."""
    spilled = state.get("spilled_ul", 0.0) if isinstance(state, dict) else 0.0
    try:
        spilled = float(spilled)
    except (TypeError, ValueError):
        return CheckResult(False, "spilled_ul is not a number")
    if spilled <= tol_ul:
        return CheckResult(True, f"{spilled:.1f} uL spilled")
    return CheckResult(False, f"{spilled:.1f} uL spilled, tolerance {tol_ul}")


def mass_on_balance(state: dict, target_g: float, tol_g: float) -> CheckResult:
    """The balance reads the target mass within tolerance, with nothing else on it."""
    reading = state.get("balance_g") if isinstance(state, dict) else None
    if reading is None:
        return CheckResult(False, "no balance reading recorded")
    if state.get("pan_touched"):
        return CheckResult(False, f"something other than the sample touches the pan ({reading:.2f} g)")
    if abs(float(reading) - target_g) <= tol_g:
        return CheckResult(True, f"balance reads {reading:.2f} g")
    return CheckResult(False, f"balance reads {reading:.2f} g, want {target_g} +- {tol_g}")


PREDICATES = {
    "in_slot": in_slot,
    "capped": capped,
    "lid_open": lid_open,
    "volume": volume,
    "concentration": concentration,
    "no_spill": no_spill,
    "mass_on_balance": mass_on_balance,
}
