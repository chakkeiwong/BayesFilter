from __future__ import annotations

import pytest

from bayesfilter.highdim.ledh_tuning_registry import (
    ROUTES,
    require_active_route_tuning,
    route_for_model,
)


def test_every_registered_model_has_a_route_specific_control_family() -> None:
    assert len({route.model_id for route in ROUTES}) == len(ROUTES)
    assert all(route.control_family_id for route in ROUTES)
    assert all(route.tunable_controls for route in ROUTES)
    assert all(route.tuner_program for route in ROUTES)


def test_ot_and_tp_routes_do_not_share_control_vocabulary() -> None:
    lgssm = route_for_model("canonical_lgssm_m3")
    actual_sv = route_for_model("actual_sv")
    assert lgssm.tunable_controls == ("sinkhorn_steps", "balance_steps")
    assert "sinkhorn_steps" not in actual_sv.tunable_controls
    assert "lookahead_steps" in actual_sv.tunable_controls


def test_missing_model_specific_tuner_fails_closed() -> None:
    with pytest.raises(ValueError, match="no implemented scope tuner"):
        require_active_route_tuning(
            "predator_prey", selected_scope_sha256="present_but_not_enough"
        )


def test_missing_selected_scope_fails_closed() -> None:
    with pytest.raises(ValueError, match="no selected scope tuning artifact"):
        require_active_route_tuning("canonical_lgssm_m3", selected_scope_sha256=None)
