from __future__ import annotations

import pytest

from bayesfilter.highdim.genut_algorithm import resolve_genut_algorithm


def test_default_resolves_to_dual_cap_family():
    import bayesfilter.highdim as highdim

    assert "resolve_genut_algorithm" in highdim.__all__
    assert highdim.resolve_genut_algorithm is resolve_genut_algorithm
    selection = resolve_genut_algorithm()
    assert selection.algorithm == "dual_cap"
    assert selection.requested_name == "default"
    controls = selection.apply(
        {
            "higher_moment_correction_steps": 4,
            "higher_moment_strength": 0.2,
            "epsilon": 2.0,
        }
    )
    assert controls["pairwise_moment_correction_steps"] == 4
    assert controls["pairwise_particle_rms_cap"] == 2.0
    assert controls["coordinatewise_standardized_cap"] == 0.98
    assert controls["coordinatewise_standardized_cap_power"] == 8


def test_explicit_diagonal_and_none_do_not_inherit_caps():
    base = {
        "higher_moment_correction_steps": 4,
        "higher_moment_strength": 0.2,
    }
    diagonal = resolve_genut_algorithm("diagonal").apply(base)
    none = resolve_genut_algorithm("none").apply(base)
    assert diagonal["higher_moment_correction_steps"] == 4
    assert diagonal["pairwise_moment_correction_steps"] == 0
    assert diagonal["coordinatewise_standardized_cap"] == 0.0
    assert none["higher_moment_correction_steps"] == 0
    assert none["higher_moment_strength"] == 0.0


def test_route_specific_options_fail_closed_without_being_inferred():
    selection = resolve_genut_algorithm("bounded_teacher")
    assert selection.requires_route_specific_inputs
    assert selection.apply({"higher_moment_correction_steps": 4})[
        "coordinatewise_standardized_cap"
    ] == 0.0


def test_unknown_algorithm_fails_closed():
    with pytest.raises(ValueError, match="unknown GenUT algorithm"):
        resolve_genut_algorithm("not-a-route")
