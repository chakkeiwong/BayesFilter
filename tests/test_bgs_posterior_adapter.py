from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.adapters.bgs import (
    BGSBatchPosteriorAdapter,
    BGSConstrainedLikelihoodResult,
    BGSPosteriorAdapter,
    BGS_STATUS_DESCRIPTOR_FAILURE,
    BGS_STATUS_LIKELIHOOD_SCORE_NONFINITE,
    BGS_STATUS_LIKELIHOOD_VALUE_NONFINITE,
    BGS_STATUS_NONFINITE_UNCONSTRAINED,
    BGS_STATUS_POSTERIOR_NONFINITE,
    BGS_STATUS_STATE_SPACE_FAILURE,
    BGS_STATUS_TRANSFORM_OUTSIDE_OPEN_SUPPORT,
    PARAMETER_DIMENSION,
    constrained_log_prior_contributions,
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


ZERO_PRIOR = -330.33138659665684
ZERO_JACOBIAN = 5.732110076250844
RAMP_PRIOR = -497.04752136904375
RAMP_JACOBIAN = -0.2866878549150067
MODE_PRIOR = -42.147492848382186
MODE_THETA = np.array([
    0.38745973756849544, 1.4832828480352025, 0.3001850232423397,
    0.8657017105333411, 0.22144019707804824, 0.12011259010436091,
    0.3816496548082241, 0.5291405580121559, 0.13810847716163144,
    0.1758002774325605, 0.4974104660611901, 1.030432878603705,
    0.6619429604586082, 0.1341901179793779, 0.8613713124163744,
    10.093680600575706, 0.6015131978279178, 0.7468119068332796,
    0.2498528549728866, 0.6257824893125601, 1.5777171414028768,
    0.7807836120152734, 1.4770320679022462, 0.006968178173500131,
    0.19323015459259787, 0.10010956178263745, 0.7379396622168786,
    0.3511749767915141, 0.9671083866643949, 0.5702834853287866,
    0.9779228684648676, 0.7725527668328076, 0.9742653997029259,
    0.8315978723169639, 0.9101441999588041, 0.9101411568417116,
    0.840758952546921, 0.33898828528328434, 0.26626526829844244,
    0.017846651571437697, 0.8151628959949924, 3.0837477684048733,
    0.149, 0.32351068206498257, 0.6950038527132324,
    -0.7970421298061325,
], dtype=np.float64)
MODE_DYNARE_PRIOR_CONTRIBUTIONS = np.array([
    -2.2648477892984022, -6.403915208254642, -1.4897589870952888,
    -4.737783611847867, -0.5802960391322971, 1.1385787875674094,
    -2.2187501847688944, -3.2197608111518847, 0.7670547229801372,
    0.09180755226762083, -3.0297609096842724, -0.7220853921801568,
    -2.2227201417006905, 0.9990040771674851, 0.1804624940451074,
    -9.576168810608992, 0.735628801638732, -0.22625503533986357,
    1.5738465094752496, 0.6336843687140465, -2.276249784141577,
    -0.6301423703572144, 0.46313562067644165, -0.7095084523872033,
    1.1457229412033574, 1.9528869574188499, 1.2982843504645594,
    0.4052313682367972, -2.7942626454076507, 0.5235502569306947,
    -3.424039821334447, -0.01699091397969621, -3.181025862846555,
    -0.3857275428245175, -1.259806140714991, -1.259756541587301,
    -0.4588194888259509, 0.3780766974731967, 1.4321177402140295,
    1.2093407899761142, -0.6353022648921385, -0.9224453775610708,
    -1.1482789553347765, -0.6371584982437963, 1.047205930020482,
    -1.691495233350355,
], dtype=np.float64)


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


def test_frozen_dynare_prior_contributions_match_all_46_rows():
    theta = tf.constant(MODE_THETA, tf.float64)
    contributions = constrained_log_prior_contributions(theta)
    value, _score = constrained_log_prior_and_score(theta)
    np.testing.assert_allclose(
        contributions.numpy(),
        MODE_DYNARE_PRIOR_CONTRIBUTIONS,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(value.numpy(), MODE_PRIOR, rtol=0.0, atol=2.0e-13)


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
    assert adapter.capability_mode == "debug_graph"
    assert getattr(adapter, "batch_rank_policy", None) is None


def test_target_xla_graph_chain_mode_has_bound_identity_and_reviewed_capability():
    debug = BGSPosteriorAdapter(
        _quadratic_likelihood,
        evidence_path="docs/plans/bgs-phase05-06-test-evidence.md",
    )
    admitted = BGSPosteriorAdapter(
        _quadratic_likelihood,
        evidence_path="docs/plans/bgs-phase05-06-test-evidence.md",
        capability_mode="target_xla_graph_chain",
        likelihood_signature="a" * 64,
    )
    capability = value_score_capability(admitted)
    assert admitted.capability_mode == "target_xla_graph_chain"
    assert admitted.adapter_signature() != debug.adapter_signature()
    assert capability.value_score_authority == "reviewed_gradient_tape_xla_exception"
    assert capability.xla_hmc_ready is False
    assert capability.full_chain_xla_diagnostic_ready is False
    assert capability.target_scope == "bgs_d296_synthetic_transformed_target"
    assert "no posterior convergence claim" in capability.nonclaims


def test_target_graph_chain_mode_has_bound_graph_native_capability():
    debug = BGSPosteriorAdapter(
        _quadratic_likelihood,
        evidence_path="docs/plans/bgs-graph-native-test-evidence.md",
    )
    admitted = BGSPosteriorAdapter(
        _quadratic_likelihood,
        evidence_path="docs/plans/bgs-graph-native-test-evidence.md",
        capability_mode="target_graph_chain",
        likelihood_signature="b" * 64,
    )
    capability = value_score_capability(admitted)
    assert admitted.capability_mode == "target_graph_chain"
    assert admitted.adapter_signature() != debug.adapter_signature()
    assert capability.value_score_authority == "graph_native"
    assert capability.xla_hmc_ready is False
    assert capability.full_chain_xla_diagnostic_ready is False
    assert capability.runtime_backend == (
        "tensorflow_tfp_bayesfilter_qr_target_graph_chain"
    )
    assert capability.target_scope == "bgs_d296_synthetic_transformed_target"
    assert "no posterior convergence claim" in capability.nonclaims


@pytest.mark.parametrize("mode", ("target_graph_chain", "target_xla_graph_chain"))
def test_target_chain_modes_require_reviewed_likelihood_signature(mode):
    for signature in (None, "", "A" * 64, "a" * 63):
        with pytest.raises(ValueError, match="likelihood_signature"):
            BGSPosteriorAdapter(
                _quadratic_likelihood,
                evidence_path="docs/plans/bgs-phase05-06-test-evidence.md",
                capability_mode=mode,
                likelihood_signature=signature,
            )


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


def test_retained_target_status_preserves_exact_bgs_validity_veto():
    cases = (
        ({}, tf.zeros((PARAMETER_DIMENSION,), tf.float64)),
        (
            {},
            tf.fill((PARAMETER_DIMENSION,), tf.constant(1.0e6, tf.float64)),
        ),
        (
            {"descriptor_success": False},
            tf.zeros((PARAMETER_DIMENSION,), tf.float64),
        ),
        (
            {"numerical_state_space_success": False},
            tf.zeros((PARAMETER_DIMENSION,), tf.float64),
        ),
        (
            {"likelihood_value_finite": False},
            tf.zeros((PARAMETER_DIMENSION,), tf.float64),
        ),
        (
            {"likelihood_score_finite": False},
            tf.zeros((PARAMETER_DIMENSION,), tf.float64),
        ),
    )
    for overrides, u in cases:
        adapter = BGSPosteriorAdapter(
            _status_likelihood(**overrides),
            evidence_path="docs/plans/bgs-phase04-test-evidence.md",
        )
        value, _score, full = adapter.log_prob_and_grad_status(u)
        retained = adapter.retained_target_status_telemetry(value)
        assert bool(retained["valid_pre_regularized_score"].numpy()) == bool(
            full["valid_pre_regularized_score"].numpy()
        )
        assert (int(retained["status_code"].numpy()) != 0) == (
            int(full["status_code"].numpy()) != 0
        )
        if int(full["status_code"].numpy()) != 0:
            assert int(retained["status_code"].numpy()) == (
                BGS_STATUS_POSTERIOR_NONFINITE
            )


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
    assert bayesfilter.BGSBatchPosteriorAdapter is BGSBatchPosteriorAdapter
    assert adapters.BGSBatchPosteriorAdapter is BGSBatchPosteriorAdapter


def test_batch_adapter_admits_full_chain_xla_capability():
    @tf.function(
        input_signature=(tf.TensorSpec((None, PARAMETER_DIMENSION), tf.float64),),
        autograph=False,
    )
    def batch_likelihood(theta):
        return BGSConstrainedLikelihoodResult(
            -0.5 * tf.reduce_sum(tf.square(theta), axis=1),
            -theta,
        )

    adapter = BGSBatchPosteriorAdapter(
        batch_likelihood,
        evidence_path="docs/plans/bgs-full-chain-xla-cpu-gpu-comparison-plan-2026-07-28.md",
        likelihood_signature="c" * 64,
        safe_theta=MODE_THETA,
    )
    capability = value_score_capability(adapter)
    assert capability.value_score_authority == "graph_native"
    assert capability.xla_hmc_ready is True
    assert capability.full_chain_xla_diagnostic_ready is True
    assert adapter.batch_rank_policy == "rank2_required"


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
    batch_method = inspect.getsource(BGSBatchPosteriorAdapter.log_prob_and_grad_status)
    assert "for " not in batch_method
    for forbidden in (
        "map_fn",
        "vectorized_map",
        "numpy_function",
        "py_function",
        ".jacobian(",
        ".batch_jacobian(",
    ):
        assert forbidden not in batch_method
