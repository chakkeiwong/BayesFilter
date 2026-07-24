from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.adapters.bgs import (
    BGSConstrainedLikelihoodResult,
    BGSPosteriorAdapter,
    BGS_STATUS_DESCRIPTOR_FAILURE,
    BGS_STATUS_LIKELIHOOD_SCORE_NONFINITE,
    BGS_STATUS_LIKELIHOOD_VALUE_NONFINITE,
    BGS_STATUS_NONFINITE_UNCONSTRAINED,
    BGS_STATUS_STATE_SPACE_FAILURE,
    BGS_STATUS_TRANSFORM_OUTSIDE_OPEN_SUPPORT,
    PARAMETER_DIMENSION,
    constrained_log_prior_and_score,
    log_abs_det_jacobian,
    theta_from_unconstrained,
    unconstrained_from_theta,
)
from bayesfilter.inference.posterior_adapter import value_score_capability
from bayesfilter.inference.hmc_verification import (
    TARGET_STATUS_TELEMETRY_FIELDS,
    target_status_telemetry_has_failure,
)


ZERO_PRIOR = -327.5320442180172
ZERO_JACOBIAN = 5.732110076250844
RAMP_PRIOR = -494.49922994338436
RAMP_JACOBIAN = -0.2866878549150067


@tf.function(
    input_signature=(tf.TensorSpec((PARAMETER_DIMENSION,), tf.float64),),
    autograph=False,
)
def _quadratic_likelihood(theta):
    return BGSConstrainedLikelihoodResult(
        -0.5 * tf.reduce_sum(tf.square(theta)),
        -theta,
    )


def test_transform_roundtrip_and_frozen_prior_values():
    for u, expected_prior, expected_jacobian in (
        (tf.zeros((46,), tf.float64), ZERO_PRIOR, ZERO_JACOBIAN),
        (
            tf.linspace(tf.constant(-1.25, tf.float64), tf.constant(1.25, tf.float64), 46),
            RAMP_PRIOR,
            RAMP_JACOBIAN,
        ),
    ):
        theta = theta_from_unconstrained(u)
        recovered = unconstrained_from_theta(theta)
        prior, score = constrained_log_prior_and_score(theta)
        np.testing.assert_allclose(recovered.numpy(), u.numpy(), atol=2.0e-15)
        np.testing.assert_allclose(prior.numpy(), expected_prior, atol=2.0e-12)
        np.testing.assert_allclose(
            log_abs_det_jacobian(u).numpy(), expected_jacobian, atol=2.0e-13
        )
        assert score.shape == (46,)
        assert bool(tf.reduce_all(tf.math.is_finite(score)).numpy())


def test_analytical_constrained_prior_score_matches_tape():
    theta = theta_from_unconstrained(
        tf.linspace(tf.constant(-0.8, tf.float64), tf.constant(0.9, tf.float64), 46)
    )
    with tf.GradientTape() as tape:
        tape.watch(theta)
        value, _score = constrained_log_prior_and_score(theta)
    tape_score = tape.gradient(value, theta)
    _value, analytical = constrained_log_prior_and_score(theta)
    np.testing.assert_allclose(
        analytical.numpy(), tape_score.numpy(), rtol=2.0e-12, atol=2.0e-11
    )


def test_adapter_identity_score_and_capability_are_conservative():
    adapter = BGSPosteriorAdapter(
        _quadratic_likelihood,
        evidence_path="docs/plans/bgs-phase04-test-evidence.md",
    )
    u = tf.linspace(tf.constant(-0.5, tf.float64), tf.constant(0.5, tf.float64), 46)
    components = adapter.components(u)
    value, score, status = adapter.log_prob_and_grad_status(u)
    np.testing.assert_allclose(value.numpy(), components.posterior_value.numpy(), atol=0.0)
    np.testing.assert_allclose(score.numpy(), components.posterior_score.numpy(), atol=0.0)
    np.testing.assert_allclose(
        value.numpy(),
        (
            components.signed_log_likelihood
            + components.constrained_log_prior
            + components.log_abs_det_jacobian
        ).numpy(),
        atol=0.0,
    )
    capability = value_score_capability(adapter)
    assert capability.value_score_authority == "debug_only"
    assert capability.xla_hmc_ready is False
    assert capability.full_chain_xla_diagnostic_ready is False
    assert adapter.parameter_dim == 46
    assert len(adapter.parameter_names) == 46
    assert len(adapter.adapter_signature()) == 64
    assert int(status["status_code"].numpy()) == 0
    assert bool(status["valid_pre_regularized_score"].numpy()) is True
    assert bool(status["innovation_metrics_available"].numpy()) is False
    assert adapter.supports_retained_value_score_status is True


def _status_likelihood(
    *,
    descriptor_success=True,
    numerical_state_space_success=True,
    likelihood_value_finite=True,
    likelihood_score_finite=True,
):
    def likelihood(theta):
        value = tf.constant(-1.0, tf.float64)
        score = tf.ones((PARAMETER_DIMENSION,), tf.float64)
        if not likelihood_value_finite:
            value = tf.constant(float("nan"), tf.float64)
        if not likelihood_score_finite:
            score = tf.fill((PARAMETER_DIMENSION,), tf.constant(float("nan"), tf.float64))
        return BGSConstrainedLikelihoodResult(
            value,
            score,
            descriptor_success,
            numerical_state_space_success,
            likelihood_value_finite,
            likelihood_score_finite,
        )

    return likelihood


def test_nonfinite_and_support_rounding_inputs_fail_closed_without_nan_outputs():
    adapter = BGSPosteriorAdapter(
        _status_likelihood(),
        evidence_path="docs/plans/bgs-phase04-test-evidence.md",
    )
    for u, required_bits in (
        (
            tf.concat((
                tf.constant([float("inf")], tf.float64),
                tf.zeros((PARAMETER_DIMENSION - 1,), tf.float64),
            ), axis=0),
            BGS_STATUS_NONFINITE_UNCONSTRAINED
            | BGS_STATUS_TRANSFORM_OUTSIDE_OPEN_SUPPORT,
        ),
        (
            tf.concat((
                tf.constant([1.0e6], tf.float64),
                tf.zeros((PARAMETER_DIMENSION - 1,), tf.float64),
            ), axis=0),
            BGS_STATUS_TRANSFORM_OUTSIDE_OPEN_SUPPORT,
        ),
    ):
        value, score, status = adapter.log_prob_and_grad_status(u)
        code = int(status["status_code"].numpy())
        assert code & required_bits == required_bits
        assert np.isneginf(value.numpy())
        np.testing.assert_array_equal(score.numpy(), np.zeros(PARAMETER_DIMENSION))
        assert bool(status["valid_pre_regularized_score"].numpy()) is False


def test_component_failures_remain_distinguishable_in_status_telemetry():
    cases = (
        ({"descriptor_success": False}, BGS_STATUS_DESCRIPTOR_FAILURE),
        ({"numerical_state_space_success": False}, BGS_STATUS_STATE_SPACE_FAILURE),
        ({"likelihood_value_finite": False}, BGS_STATUS_LIKELIHOOD_VALUE_NONFINITE),
        ({"likelihood_score_finite": False}, BGS_STATUS_LIKELIHOOD_SCORE_NONFINITE),
    )
    for overrides, expected_bit in cases:
        adapter = BGSPosteriorAdapter(
            _status_likelihood(**overrides),
            evidence_path="docs/plans/bgs-phase04-test-evidence.md",
        )
        value, score, status = adapter.log_prob_and_grad_status(
            tf.zeros((PARAMETER_DIMENSION,), tf.float64)
        )
        assert int(status["status_code"].numpy()) & expected_bit
        assert np.isneginf(value.numpy())
        np.testing.assert_array_equal(score.numpy(), np.zeros(PARAMETER_DIMENSION))


def test_status_telemetry_satisfies_shared_verification_schema():
    adapter = BGSPosteriorAdapter(
        _status_likelihood(),
        evidence_path="docs/plans/bgs-phase04-test-evidence.md",
    )
    for u, expected_failure in (
        (tf.zeros((PARAMETER_DIMENSION,), tf.float64), False),
        (tf.fill((PARAMETER_DIMENSION,), tf.constant(1.0e6, tf.float64)), True),
    ):
        status = adapter.target_status_telemetry(u)
        shared = {
            name: np.asarray(status[name].numpy()).reshape((1,))
            for name in TARGET_STATUS_TELEMETRY_FIELDS
        }
        assert target_status_telemetry_has_failure(
            shared, expected_shape=(1,)
        ) is expected_failure


def test_bgs_adapter_is_available_from_public_namespaces():
    import bayesfilter
    import bayesfilter.adapters as adapters

    assert bayesfilter.BGSPosteriorAdapter is BGSPosteriorAdapter
    assert adapters.BGSPosteriorAdapter is BGSPosteriorAdapter


def test_adapter_source_has_no_numpy_scipy_callbacks_or_sampler():
    source = inspect.getsource(
        __import__("bayesfilter.adapters.bgs", fromlist=["bgs"])
    )
    for forbidden in (
        "import numpy",
        "from numpy",
        "import scipy",
        "from scipy",
        "numpy_function",
        "py_function",
        "vectorized_map",
        "HamiltonianMonteCarlo",
        "sample_chain",
    ):
        assert forbidden not in source
