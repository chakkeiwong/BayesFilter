"""Regression checks for portable q=20 bridge identity metadata."""

import math

from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge


def test_q20_bridge_weight_sum_is_canonical_and_repeatable() -> None:
    first = make_q20_tempered_bridge(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh_strict"
    )
    second = make_q20_tempered_bridge(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh_strict"
    )
    facts = first.source_facts()
    weights = tuple(float(value) for value in facts["covariance_weights"])
    assert facts["covariance_weight_sum"] == math.fsum(weights)
    assert facts["covariance_weight_sum"] == 3.0
    assert first.signature == second.signature


def test_q20_backend_identity_propagates_through_bridge_and_fixed_beta_adapter() -> None:
    strict = make_q20_tempered_bridge(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh_strict"
    )
    factor = make_q20_tempered_bridge(
        20,
        jit_compile=True,
        principal_sqrt_backend="tensorflow_eigh_strict_factor_cached",
    )

    # The mathematical target program is intentionally shared, but the
    # backend-specific component, bridge, and fixed-beta identities must not
    # collide in tuning or HMC handoffs.
    assert strict.target_signature == factor.target_signature
    assert strict.component_target_adapter_signature
    assert factor.component_target_adapter_signature
    assert (
        strict.component_target_adapter_signature
        != factor.component_target_adapter_signature
    )
    assert strict.signature != factor.signature
    strict_adapter = strict.fixed_beta_adapter(0.0)
    factor_adapter = factor.fixed_beta_adapter(0.0)
    assert strict_adapter.adapter_signature() != factor_adapter.adapter_signature()
    assert (
        strict.signature_payload()["component_target_adapter_signature"]
        == strict.component_target_adapter_signature
    )
    assert (
        factor_adapter.adapter_signature_payload()[
            "component_target_adapter_signature"
        ]
        == factor.component_target_adapter_signature
    )
