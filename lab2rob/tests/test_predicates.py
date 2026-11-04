"""FD-05: predicates judge fake states correctly and never crash on gaps."""

from lab2rob.registry.predicates import (
    PREDICATES,
    capped,
    concentration,
    in_slot,
    lid_open,
    no_spill,
    volume,
)

STATE = {
    "tubes": {
        "tube_1": {"slot": ["rack_dest", 0, 1], "lid_angle_rad": 0.01},
        "tube_2": {"slot": None, "lid_angle_rad": 1.9},
    },
    "wells": {
        "A1": {"volume_ul": 100.0, "conc": 1.0},
        "B2": {"volume_ul": 40.0, "conc": 0.0},
    },
    "spilled_ul": 0.0,
}


def test_predicate_names():
    assert sorted(PREDICATES) == ["capped", "concentration", "in_slot",
                                  "lid_closed", "lid_open", "mass_on_balance", "no_spill",
                                  "rotor_balanced", "volume"]


def test_in_slot():
    assert in_slot(STATE, "tube_1", "rack_dest", 0, 1).passed is True
    assert in_slot(STATE, "tube_1", "rack_dest", 0, 0).passed is False
    assert in_slot(STATE, "tube_2", "rack_dest", 0, 1).passed is False
    assert in_slot(STATE, "ghost", "rack_dest", 0, 1).passed is False
    assert in_slot({}, "tube_1", "rack_dest", 0, 1).passed is False


def test_capped_and_lid_open():
    assert capped(STATE, "tube_1").passed is True
    assert capped(STATE, "tube_2").passed is False
    assert lid_open(STATE, "tube_2", 1.4).passed is True
    assert lid_open(STATE, "tube_1", 1.4).passed is False
    assert capped({}, "tube_1").passed is False


def test_volume():
    assert volume(STATE, "A1", 100, 10).passed is True
    assert volume(STATE, "A1", 100, 5).passed is True
    assert volume(STATE, "B2", 100, 10).passed is False
    assert volume(STATE, "ghost", 100, 10).passed is False


def test_concentration():
    assert concentration(STATE, "A1", 1.0).passed is True
    assert concentration(STATE, "A1", 1.0, tol_rel=0.01).passed is True
    assert concentration(STATE, "B2", 1.0).passed is False
    assert concentration(STATE, "ghost", 1.0).passed is False


def test_no_spill():
    assert no_spill(STATE).passed is True
    assert no_spill({**STATE, "spilled_ul": 12.0}).passed is False
    assert no_spill({**STATE, "spilled_ul": 12.0}, tol_ul=15.0).passed is True
    assert no_spill({}).passed is True
