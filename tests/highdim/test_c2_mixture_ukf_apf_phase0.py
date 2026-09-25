"""Phase 0 checks for the generic mixture-UKF/APF proposal endpoint.

These are CPU reference checks.  They establish API shape, linear-Gaussian
UKF/Kalman parity, APF w/a algebra, complete-mixture density semantics, and the
C2 signed-observation zero-gain negative control.  They do not establish C2
posterior accuracy, proposal efficiency, or production readiness.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.highdim.c2_mixture_ukf_apf_tf import (
    BatchedUKFConfig,
    complete_gaussian_mixture_log_density,
    compile_k1_apf_proposal,
    gaussian_log_density,
    make_batched_ukf_kernel,
    require_valid_ukf_result,
)


DTYPE = tf.float64
ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "docs/benchmarks/fixtures/c2_mixture_ukf_lgssm_phase0_v1.json"


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _t(value: object) -> tf.Tensor:
    return tf.convert_to_tensor(value, dtype=DTYPE)


def _linear_kernel(*, jit_compile: bool, batch_size: int = 4):
    fixture = _fixture()
    transition_matrix = _t(fixture["transition_matrix"])
    transition_offset = _t(fixture["transition_offset"])
    observation_matrix = _t(fixture["observation_matrix"])
    observation_offset = _t(fixture["observation_offset"])

    def transition_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.einsum("ij,bpj->bpi", transition_matrix, points) + transition_offset

    def observation_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.einsum("ij,bpj->bpi", observation_matrix, points) + observation_offset

    return make_batched_ukf_kernel(
        batch_size=batch_size,
        state_dim=2,
        observation_dim=2,
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        config=BatchedUKFConfig(alpha=1.0, beta=2.0, kappa=0.0, jitter=0.0),
        jit_compile=jit_compile,
    )


def _linear_inputs() -> tuple[tf.Tensor, ...]:
    fixture = _fixture()
    means = _t(fixture["ancestor_means"])
    covariances = _t(fixture["ancestor_covariances"])
    batch_size = int(means.shape[0])
    process = tf.broadcast_to(
        _t(fixture["process_covariance"]), [batch_size, 2, 2]
    )
    observation = _t(fixture["observation"])
    observation_covariance = tf.broadcast_to(
        _t(fixture["observation_covariance"]), [batch_size, 2, 2]
    )
    return means, covariances, process, observation, observation_covariance


def _exact_linear_moments() -> dict[str, tf.Tensor]:
    fixture = _fixture()
    means, covariances, process, observation, observation_covariance = _linear_inputs()
    transition_matrix = _t(fixture["transition_matrix"])
    transition_offset = _t(fixture["transition_offset"])
    observation_matrix = _t(fixture["observation_matrix"])
    observation_offset = _t(fixture["observation_offset"])
    predicted_mean = tf.einsum("ij,bj->bi", transition_matrix, means) + transition_offset
    predicted_covariance = tf.einsum(
        "ij,bjk,lk->bil", transition_matrix, covariances, transition_matrix
    ) + process
    innovation_mean = tf.einsum("ij,bj->bi", observation_matrix, predicted_mean) + observation_offset
    innovation_covariance = tf.einsum(
        "ij,bjk,lk->bil", observation_matrix, predicted_covariance, observation_matrix
    ) + observation_covariance
    cross_covariance = tf.einsum(
        "bij,lj->bil", predicted_covariance, observation_matrix
    )
    innovation = observation[None, :] - innovation_mean
    gain = tf.transpose(
        tf.linalg.solve(
            innovation_covariance,
            tf.transpose(cross_covariance, [0, 2, 1]),
        ),
        [0, 2, 1],
    )
    posterior_mean = predicted_mean + tf.einsum("bio,bo->bi", gain, innovation)
    posterior_covariance = 0.5 * (
        predicted_covariance
        - tf.matmul(
            tf.matmul(gain, innovation_covariance), gain, transpose_b=True
        )
        + tf.transpose(
            predicted_covariance
            - tf.matmul(
                tf.matmul(gain, innovation_covariance), gain, transpose_b=True
            ),
            [0, 2, 1],
        )
    )
    chol = tf.linalg.cholesky(innovation_covariance)
    solved = tf.linalg.cholesky_solve(
        chol, innovation[:, :, tf.newaxis]
    )[:, :, 0]
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=1)
    log_likelihood = -0.5 * (
        2.0 * math.log(2.0 * math.pi) + logdet
        + tf.reduce_sum(innovation * solved, axis=1)
    )
    return {
        "predicted_mean": predicted_mean,
        "predicted_covariance": predicted_covariance,
        "innovation_mean": innovation_mean,
        "innovation_covariance": innovation_covariance,
        "cross_covariance": cross_covariance,
        "posterior_mean": posterior_mean,
        "posterior_covariance": posterior_covariance,
        "log_likelihood": log_likelihood,
    }


def test_linear_gaussian_ukf_matches_independent_kalman_moments() -> None:
    kernel = _linear_kernel(jit_compile=False)
    result = kernel(*_linear_inputs())
    expected = _exact_linear_moments()
    for name, value in expected.items():
        key = "predicted_observation_mean" if name == "innovation_mean" else name
        key = "innovation_log_likelihood" if name == "log_likelihood" else key
        tf.debugging.assert_near(result[key], value, atol=2e-12, rtol=2e-12)
    tf.debugging.assert_near(
        result["gain"],
        tf.transpose(
            tf.linalg.solve(
                expected["innovation_covariance"],
                tf.transpose(expected["cross_covariance"], [0, 2, 1]),
            ),
            [0, 2, 1],
        ),
        atol=2e-12,
        rtol=2e-12,
    )
    tf.debugging.assert_equal(result["sigma_point_count"], tf.constant(5, tf.int32))
    tf.debugging.assert_equal(result["valid_rows"], tf.ones([4], tf.bool))
    assert bool(result["finite"].numpy())


def test_fixed_signature_and_xla_match_reference_kernel() -> None:
    eager_kernel = _linear_kernel(jit_compile=False)
    xla_kernel = _linear_kernel(jit_compile=True)
    eager = eager_kernel(*_linear_inputs())
    compiled = xla_kernel(*_linear_inputs())
    for key in (
        "predicted_mean",
        "predicted_covariance",
        "innovation_covariance",
        "posterior_mean",
        "posterior_covariance",
        "innovation_log_likelihood",
    ):
        tf.debugging.assert_near(compiled[key], eager[key], atol=2e-11, rtol=2e-11)
    signature = xla_kernel.input_signature
    assert signature is not None
    assert [tuple(spec.shape.as_list()) for spec in signature] == [
        (4, 2),
        (4, 2, 2),
        (4, 2, 2),
        (2,),
        (4, 2, 2),
    ]
    assert bool(compiled["finite"].numpy())


def test_apf_w_over_a_identity_holds_pointwise_for_selected_ancestors() -> None:
    fixture = _fixture()
    means, covariances, process, observation, observation_covariance = _linear_inputs()
    kernel = _linear_kernel(jit_compile=False)
    proposal = compile_k1_apf_proposal(
        kernel,
        prior_means=means,
        prior_covariances=covariances,
        process_covariances=process,
        observation=observation,
        observation_covariances=observation_covariance,
        log_parent_weights=_t(fixture["log_parent_weights"]),
        seed=(20260903, 17),
    )
    ancestor = proposal["ancestor_indices"]
    parent = tf.gather(means, ancestor)
    transition_matrix = _t(fixture["transition_matrix"])
    transition_offset = _t(fixture["transition_offset"])
    transition_chol = tf.linalg.cholesky(_t(fixture["process_covariance"]))
    transition_mean = tf.einsum("ij,bj->bi", transition_matrix, parent) + transition_offset
    transition_log_q = gaussian_log_density(
        proposal["samples"],
        transition_mean,
        tf.broadcast_to(transition_chol, [4, 2, 2]),
    )
    observation_matrix = _t(fixture["observation_matrix"])
    observation_offset = _t(fixture["observation_offset"])
    observation_chol = tf.linalg.cholesky(_t(fixture["observation_covariance"]))
    observation_mean = tf.einsum(
        "ij,bj->bi", observation_matrix, proposal["samples"]
    ) + observation_offset
    observation_log_density = gaussian_log_density(
        tf.broadcast_to(observation[None, :], [4, 2]),
        observation_mean,
        tf.broadcast_to(observation_chol, [4, 2, 2]),
    )
    log_target = (
        tf.gather(_t(fixture["log_parent_weights"]), ancestor)
        + transition_log_q
        + observation_log_density
    )
    log_weight = log_target - proposal["selected_log_q"] - tf.gather(
        proposal["log_ancestor_probabilities"], ancestor
    )
    tf.debugging.assert_near(
        tf.gather(proposal["log_ancestor_probabilities"], ancestor)
        + proposal["selected_log_q"]
        + log_weight,
        log_target,
        atol=2e-12,
        rtol=2e-12,
    )
    tf.debugging.assert_near(
        tf.reduce_logsumexp(proposal["log_ancestor_probabilities"]),
        tf.constant(0.0, DTYPE),
        atol=2e-14,
    )
    assert bool(proposal["finite"].numpy())


def test_complete_mixture_density_uses_all_components_and_is_permutation_invariant() -> None:
    fixture = _fixture()
    means, covariances, process, observation, observation_covariance = _linear_inputs()
    result = _linear_kernel(jit_compile=False)(
        means, covariances, process, observation, observation_covariance
    )
    base_mean = result["posterior_mean"]
    base_chol = result["posterior_cholesky"]
    points = base_mean + tf.constant([[0.4, -0.25], [-0.2, 0.3], [0.1, 0.05], [-0.3, -0.1]], DTYPE)
    offsets = tf.constant([[[0.0, 0.0], [0.7, -0.2]], [[0.0, 0.0], [0.7, -0.2]], [[0.0, 0.0], [0.7, -0.2]], [[0.0, 0.0], [0.7, -0.2]]], DTYPE)
    component_means = base_mean[:, None, :] + offsets
    component_chol = tf.broadcast_to(base_chol[:, None, :, :], [4, 2, 2, 2])
    component_weights = tf.constant([[0.35, 0.65]] * 4, DTYPE)
    complete = complete_gaussian_mixture_log_density(
        points, component_means, component_chol, component_weights
    )
    component_logs = tf.stack(
        [
            gaussian_log_density(points, component_means[:, 0, :], base_chol),
            gaussian_log_density(points, component_means[:, 1, :], base_chol),
        ],
        axis=1,
    )
    independent = tf.reduce_logsumexp(
        component_logs + tf.math.log(component_weights), axis=1
    )
    tf.debugging.assert_near(complete, independent, atol=2e-12, rtol=2e-12)
    permuted = complete_gaussian_mixture_log_density(
        points,
        tf.gather(component_means, [1, 0], axis=1),
        tf.gather(component_chol, [1, 0], axis=1),
        tf.gather(component_weights, [1, 0], axis=1),
    )
    tf.debugging.assert_near(complete, permuted, atol=2e-12, rtol=2e-12)
    selected_only = component_logs[:, 0]
    assert float(tf.reduce_max(tf.abs(complete - selected_only)).numpy()) > 1e-3


def test_ancestor_permutation_preserves_ukf_moments_and_lookahead() -> None:
    fixture = _fixture()
    inputs = _linear_inputs()
    kernel = _linear_kernel(jit_compile=False)
    base = kernel(*inputs)
    permutation = tf.constant([2, 0, 3, 1], tf.int32)
    permuted_inputs = (
        tf.gather(inputs[0], permutation),
        tf.gather(inputs[1], permutation),
        tf.gather(inputs[2], permutation),
        inputs[3],
        tf.gather(inputs[4], permutation),
    )
    permuted = kernel(*permuted_inputs)
    inverse = tf.argsort(permutation)
    for key in (
        "predicted_mean",
        "innovation_log_likelihood",
        "posterior_mean",
        "posterior_covariance",
    ):
        tf.debugging.assert_near(
            tf.gather(permuted[key], inverse), base[key], atol=2e-12, rtol=2e-12
        )
    base_log_a = tf.nn.log_softmax(_t(fixture["log_parent_weights"]) + base["innovation_log_likelihood"])
    permuted_log_a = tf.nn.log_softmax(_t(fixture["log_parent_weights"]) + permuted["innovation_log_likelihood"])
    tf.debugging.assert_near(tf.gather(permuted_log_a, inverse), base_log_a, atol=2e-12)


def test_c2_raw_signed_observation_has_zero_gain_but_transformed_map_has_signal() -> None:
    means, covariances, process, _observation, _observation_covariance = _linear_inputs()
    observation_covariance = tf.broadcast_to(
        tf.constant([[0.25]], DTYPE), [4, 1, 1]
    )
    process_one = process
    raw_kernel = make_batched_ukf_kernel(
        batch_size=4,
        state_dim=2,
        observation_dim=1,
        transition_fn=lambda points: points,
        observation_fn=lambda points: tf.zeros([4, 5, 1], DTYPE),
        config=BatchedUKFConfig(),
        jit_compile=False,
    )
    raw_plus = raw_kernel(means, covariances, process_one, tf.constant([1.2], DTYPE), observation_covariance)
    raw_minus = raw_kernel(means, covariances, process_one, tf.constant([-1.2], DTYPE), observation_covariance)
    tf.debugging.assert_near(raw_plus["cross_covariance"], tf.zeros([4, 2, 1], DTYPE), atol=2e-13)
    tf.debugging.assert_near(raw_plus["posterior_mean"], raw_minus["posterior_mean"], atol=2e-13)

    transformed_kernel = make_batched_ukf_kernel(
        batch_size=4,
        state_dim=2,
        observation_dim=1,
        transition_fn=lambda points: points,
        observation_fn=lambda points: tf.exp(0.5 * points[..., :1]),
        config=BatchedUKFConfig(),
        jit_compile=False,
    )
    transformed = transformed_kernel(
        means,
        covariances,
        process_one,
        tf.constant([1.2], DTYPE),
        observation_covariance,
    )
    assert float(tf.reduce_max(tf.abs(transformed["cross_covariance"])).numpy()) > 1e-4


def test_invalid_covariance_is_reported_and_rejected_at_host_boundary() -> None:
    means, covariances, process, observation, observation_covariance = _linear_inputs()
    invalid = tf.tensor_scatter_nd_update(
        covariances,
        tf.constant([[0, 0, 0]], tf.int32),
        tf.constant([-0.5], DTYPE),
    )
    result = _linear_kernel(jit_compile=True)(
        means, invalid, process, observation, observation_covariance
    )
    assert not bool(result["finite"].numpy())
    assert not bool(result["valid_rows"][0].numpy())
    with pytest.raises(ValueError, match="invalid row"):
        require_valid_ukf_result(result)
