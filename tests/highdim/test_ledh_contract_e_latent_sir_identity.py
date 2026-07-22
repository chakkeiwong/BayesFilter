from __future__ import annotations

from dataclasses import replace

import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_latent_sir_tf as candidate
from bayesfilter.highdim import ledh_contract_e_reset_tf as reset
from bayesfilter.highdim.ledh_contract_e_identity import (
    CONTRACT_E_DERIVATIVE_COMPOSITION_ID,
    CONTRACT_E_RESET_CONTRACT_ID,
    _PRODUCTION_FACTORY,
    _require_factory_identity,
    issue_contract_e_route_identity,
    issue_latent_sir_contract_e_route_identity,
    issue_latent_sir_two_node_contract_e_route_identity,
)
from tests.highdim.test_ledh_contract_e_latent_sir_tf import (
    _prepared_austria_for_identity_test,
)


def _identity(prepared=None):
    return issue_latent_sir_contract_e_route_identity(
        prepared_inputs=(prepared or _prepared_austria_for_identity_test())
    )


def test_repository_owned_sir_issuer_binds_route_and_prepared_inputs() -> None:
    identity = _identity()
    payload = identity.to_dict()
    assert payload["factory_scope"] == "production"
    assert payload["route_specification_id"] == candidate.CANONICAL_ROUTE_SPECIFICATION_ID
    assert payload["reset_contract_id"] == CONTRACT_E_RESET_CONTRACT_ID
    assert payload["derivative_composition_id"] == CONTRACT_E_DERIVATIVE_COMPOSITION_ID
    assert payload["admitted"] is False
    assert {row["name"] for row in payload["prepared_input_records"]} == {
        "observations",
        "initial_noise",
        "transition_noise",
        "fixed_reset_mask",
        "residual_design",
        "prepared_ridge",
        "epsilon",
        "scaling",
    }
    assert len(payload["dependency_records"]) >= 50
    assert payload["source_dependency_closure_sha256"]


def test_two_node_issuer_is_separate_from_austria_identity() -> None:
    prepared = _prepared_austria_for_identity_test()
    prepared["observations"] = prepared["observations"][:, :2]
    prepared["initial_noise"] = prepared["initial_noise"][:, :, :4]
    prepared["transition_noise"] = prepared["transition_noise"][:, :, :, :4]
    prepared["residual_design"] = prepared["residual_design"][:, :, :, :4]
    identity = issue_latent_sir_two_node_contract_e_route_identity(
        prepared_inputs=prepared
    )
    assert identity.route_specification_id == candidate.TWO_NODE_ROUTE_SPECIFICATION_ID
    assert identity.row_id == "zhao_cui_sir_two_node_spatial_latent_preclip"
    assert identity.identity_sha256 != _identity().identity_sha256


def test_omitted_or_changed_prepared_input_fails_or_changes_identity() -> None:
    prepared = _prepared_austria_for_identity_test()
    missing = dict(prepared)
    missing.pop("residual_design")
    with pytest.raises(ValueError, match="prepared input field mismatch"):
        _identity(missing)

    baseline = _identity(prepared)
    changed = dict(prepared)
    changed["prepared_ridge"] = prepared["prepared_ridge"] * 2.0
    mutation = _identity(changed)
    assert mutation.prepared_input_sha256 != baseline.prepared_input_sha256
    assert mutation.identity_sha256 != baseline.identity_sha256


def test_generic_api_rejects_callable_substitution_and_wrong_jit() -> None:
    prepared = _prepared_austria_for_identity_test()

    @tf.function(jit_compile=True)
    def substituted(*args):
        return candidate.latent_sir_contract_e_canonical_value_and_score_tf(*args)

    with pytest.raises(ValueError, match="exact registered symbol"):
        issue_contract_e_route_identity(
            route_specification_id=candidate.CANONICAL_ROUTE_SPECIFICATION_ID,
            callables={
                "reset": reset.contract_e_chol_cloud_forward_tf,
                "value": substituted,
                "gradient": candidate.latent_sir_contract_e_canonical_value_and_score_tf,
            },
            prepared_inputs=prepared,
        )

    wrong_jit = tf.function(
        candidate.latent_sir_contract_e_canonical_value_and_score_tf.python_function,
        jit_compile=False,
    )
    with pytest.raises(ValueError, match="exact registered symbol|jit_compile"):
        issue_contract_e_route_identity(
            route_specification_id=candidate.CANONICAL_ROUTE_SPECIFICATION_ID,
            callables={
                "reset": reset.contract_e_chol_cloud_forward_tf,
                "value": wrong_jit,
                "gradient": candidate.latent_sir_contract_e_canonical_value_and_score_tf,
            },
            prepared_inputs=prepared,
        )


def test_factory_and_issued_identity_are_immutable() -> None:
    with pytest.raises(AttributeError, match="immutable"):
        _PRODUCTION_FACTORY._route_specifications = {}  # type: ignore[attr-defined]
    identity = _identity()
    forged = replace(identity, identity_sha256="0" * 64)
    with pytest.raises(ValueError, match="digest|issuance"):
        _require_factory_identity(forged)


def test_monkeypatched_registered_symbol_fails_closed(monkeypatch) -> None:
    original = candidate.latent_sir_contract_e_canonical_value_and_score_tf

    @tf.function(jit_compile=True)
    def replacement(*args):
        return original(*args)

    monkeypatch.setattr(
        candidate, "latent_sir_contract_e_canonical_value_and_score_tf", replacement
    )
    with pytest.raises(
        ValueError, match="registered symbol|monkeypatched|local functions"
    ):
        issue_latent_sir_contract_e_route_identity(
            prepared_inputs=_prepared_austria_for_identity_test()
        )
