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
