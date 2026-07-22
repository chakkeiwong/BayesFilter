from __future__ import annotations

import pytest

import bayesfilter.highdim as highdim


def test_fixed_variant_is_the_hmc_default_route() -> None:
    decision = highdim.zhao_cui_hmc_route_policy()

    assert decision["policy_id"] == "zhao_cui_fixed_variant_hmc_default_v1"
    assert decision["default_route_id"] == "fixed_variant_zhao_cui_source_route"
    assert decision["selected_route_id"] == "fixed_variant_zhao_cui_source_route"
    assert decision["status"] == "default_hmc_fixed_variant"
    assert decision["hmc_eligible"] is True
    assert decision["fail_closed"] is True
    assert highdim.require_zhao_cui_hmc_route() == "fixed_variant_zhao_cui_source_route"


@pytest.mark.parametrize(
    "route_id",
    (
        "adaptive_author_full_sol",
        "adaptive_author_pre_sol",
        "diagnostic_historical_retained_grid",
        "multistate_nonlinear_fixed_design_tt_score_path",
        "zhao_cui_fixed_adjacent_state_squared_tt_v1",
    ),
)
def test_non_fixed_routes_are_historical_and_blocked_for_hmc(route_id: str) -> None:
    decision = highdim.zhao_cui_hmc_route_policy(route_id)

    assert decision["status"] == "blocked_hmc_route_not_fixed_variant"
    assert decision["hmc_eligible"] is False
    assert decision["historical_diagnostic_only"] is True
    with pytest.raises(ValueError, match="fixed-variant source route"):
        highdim.require_zhao_cui_hmc_route(route_id)


def test_unknown_route_also_fails_closed() -> None:
    decision = highdim.zhao_cui_hmc_route_policy("future_unregistered_route")

    assert decision["hmc_eligible"] is False
    assert decision["historical_diagnostic_only"] is False
    with pytest.raises(ValueError, match="fixed-variant source route"):
        highdim.require_zhao_cui_hmc_route("future_unregistered_route")


@pytest.mark.parametrize(
    "route_id",
    (
        "zhao_cui_exact_transformed_sv_fixed_branch_tt",
        "zhao_cui_ksc_mixture_fixed_branch_tt",
        "zhao_cui_sir_d18_local_complete_data_manual_component",
    ),
)
def test_model_specific_fixed_variant_routes_are_hmc_eligible(route_id: str) -> None:
    decision = highdim.zhao_cui_hmc_route_policy(route_id)

    assert decision["selected_route_id"] == route_id
    assert decision["hmc_eligible"] is True
    assert decision["historical_diagnostic_only"] is False
    assert highdim.require_zhao_cui_hmc_route(route_id) == route_id
