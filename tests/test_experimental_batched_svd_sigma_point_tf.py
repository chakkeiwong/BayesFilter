from __future__ import annotations

import ast
import inspect
import os
import textwrap
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

import bayesfilter.nonlinear.experimental_batched_svd_sigma_point_tf as svd_tf
from bayesfilter.nonlinear.experimental_batched_svd_sigma_point_tf import (
    TFBatchedStructuralFirstDerivatives,
    TFBatchedStructuralStateSpace,
    _checked_batched_principal_sqrt_factor_first_derivatives,
    tf_batched_svd_sigma_point_value_and_score,
)
from docs.benchmarks.benchmark_experimental_batched_svd_sigma_point_cpu_gpu import (
    _batched_model_and_derivatives,
    _scalar_model_and_derivatives,
    _scalar_score,
    _stable_fixture,
    _to_tensors,
)


BACKENDS = ("tf_svd_ukf", "tf_svd_cubature")
OLD_BACKENDS = ("tf_svd_ukf", "tf_svd_cubature", "tf_svd_cut4")


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        placement_floor=0.0,
        innovation_floor=1.0e-12,
        rank_tolerance=1.0e-12,
        spectral_gap_tolerance=1.0e-10,
        fixed_null_tolerance=1.0e-10,
        jitter=0.0,
    )


def _fixture(*, batch_size: int = 3, time_steps: int = 5) -> dict[str, tf.Tensor]:
    return _to_tensors(
        _stable_fixture(
            batch_size=batch_size,
            time_steps=time_steps,
            state_dim=2,
            obs_dim=2,
            parameter_dim=2,
        )
    )


def _value_and_score(
    tensors: dict[str, tf.Tensor],
    *,
    backend: str,
) -> tuple[tf.Tensor, tf.Tensor]:
    return _value_and_score_with_observations(
        tensors["observations"], tensors, backend=backend
    )


def _value_and_score_with_observations(
    observations: tf.Tensor,
    tensors: dict[str, tf.Tensor],
    *,
    backend: str,
) -> tuple[tf.Tensor, tf.Tensor]:
    model, derivatives = _batched_model_and_derivatives(tensors)
    value, score, _diagnostics = tf_batched_svd_sigma_point_value_and_score(
        observations,
        model,
        derivatives,
        backend=backend,
        placement_floor=tf.constant(0.0, dtype=tf.float64),
        innovation_floor=tf.constant(1.0e-12, dtype=tf.float64),
        rank_tolerance=tf.constant(1.0e-12, dtype=tf.float64),
        spectral_gap_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
        fixed_null_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
        jitter=tf.constant(0.0, dtype=tf.float64),
    )
    return value, score


def _model_row(
    tensors: dict[str, tf.Tensor], row: int
) -> dict[str, tf.Tensor]:
    batch_size = int(tensors["initial_mean"].shape[0])
    return {
        name: (
            tensor[row : row + 1]
            if tensor.shape.rank and tensor.shape[0] == batch_size
            else tensor
        )
        for name, tensor in tensors.items()
    }


def _value_score_diagnostics(
    tensors: dict[str, tf.Tensor],
    *,
    backend: str,
) -> tuple[tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]:
    model, derivatives = _batched_model_and_derivatives(tensors)
    value, score, diagnostics = tf_batched_svd_sigma_point_value_and_score(
        tensors["observations"],
        model,
        derivatives,
        backend=backend,
        placement_floor=tf.constant(0.0, dtype=tf.float64),
        innovation_floor=tf.constant(1.0e-12, dtype=tf.float64),
        rank_tolerance=tf.constant(1.0e-12, dtype=tf.float64),
        spectral_gap_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
        fixed_null_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
        jitter=tf.constant(0.0, dtype=tf.float64),
    )
    return value, score, dict(diagnostics)


def _repeated_positive_model_and_derivatives() -> tuple[
    tf.Tensor,
    TFBatchedStructuralStateSpace,
    TFBatchedStructuralFirstDerivatives,
]:
    batch_size = 2
    parameter_dim = 1
    state_dim = 1
    innovation_dim = 1
    observation_dim = 1
    observations = tf.constant([[0.0]], dtype=tf.float64)

    def transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        return previous + innovation

    def observe(states: tf.Tensor) -> tf.Tensor:
        return states

    def transition_state_jacobian(
        previous: tf.Tensor,
        innovation: tf.Tensor,
    ) -> tf.Tensor:
        del innovation
        return tf.ones(
            [tf.shape(previous)[0], tf.shape(previous)[1], state_dim, state_dim],
            dtype=tf.float64,
        )

    def transition_innovation_jacobian(
        previous: tf.Tensor,
        innovation: tf.Tensor,
    ) -> tf.Tensor:
        del previous
        return tf.ones(
            [tf.shape(innovation)[0], tf.shape(innovation)[1], state_dim, innovation_dim],
            dtype=tf.float64,
        )

    def d_transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        return tf.zeros(
            [tf.shape(previous)[0], parameter_dim, tf.shape(previous)[1], state_dim],
            dtype=tf.float64,
        )

    def observation_state_jacobian(states: tf.Tensor) -> tf.Tensor:
        return tf.ones(
            [tf.shape(states)[0], tf.shape(states)[1], observation_dim, state_dim],
            dtype=tf.float64,
        )

    def d_observation(states: tf.Tensor) -> tf.Tensor:
        return tf.zeros(
            [tf.shape(states)[0], parameter_dim, tf.shape(states)[1], observation_dim],
            dtype=tf.float64,
        )

    model = TFBatchedStructuralStateSpace(
        initial_mean=tf.zeros([batch_size, state_dim], dtype=tf.float64),
        initial_covariance=tf.ones([batch_size, state_dim, state_dim], dtype=tf.float64),
        innovation_covariance=tf.ones([batch_size, innovation_dim, innovation_dim], dtype=tf.float64),
        observation_covariance=tf.ones([batch_size, observation_dim, observation_dim], dtype=tf.float64),
        transition_fn=transition,
        observation_fn=observe,
    )
    derivatives = TFBatchedStructuralFirstDerivatives(
        d_initial_mean=tf.zeros([batch_size, parameter_dim, state_dim], dtype=tf.float64),
        d_initial_covariance=tf.zeros(
            [batch_size, parameter_dim, state_dim, state_dim],
            dtype=tf.float64,
        ),
        d_innovation_covariance=tf.zeros(
            [batch_size, parameter_dim, innovation_dim, innovation_dim],
            dtype=tf.float64,
        ),
        d_observation_covariance=tf.zeros(
            [batch_size, parameter_dim, observation_dim, observation_dim],
            dtype=tf.float64,
        ),
        transition_state_jacobian_fn=transition_state_jacobian,
        transition_innovation_jacobian_fn=transition_innovation_jacobian,
        d_transition_fn=d_transition,
        observation_state_jacobian_fn=observation_state_jacobian,
        d_observation_fn=d_observation,
    )
    return observations, model, derivatives


def _lagged_linear_model_and_derivatives(
    theta: tf.Tensor,
    *,
    previous_coeff: float = 0.35,
) -> tuple[
    tf.Tensor,
    TFBatchedStructuralStateSpace,
    TFBatchedStructuralFirstDerivatives,
]:
    batch_size = 1
    parameter_dim = 1
    state_dim = 1
    innovation_dim = 1
    observation_dim = 1
    transition_state = tf.constant(0.72, dtype=tf.float64)
    transition_innovation = tf.constant(0.18, dtype=tf.float64)
    previous_weight = tf.constant(previous_coeff, dtype=tf.float64)
    innovation_weight = tf.constant(0.11, dtype=tf.float64)
    next_weight = tf.constant(1.20, dtype=tf.float64)
    theta = tf.ensure_shape(tf.convert_to_tensor(theta, dtype=tf.float64), [batch_size, 1])
    observations = tf.constant([[0.12], [-0.03]], dtype=tf.float64)

    def transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        return transition_state * previous + transition_innovation * innovation

    def observe(states: tf.Tensor) -> tf.Tensor:
        return next_weight * states

    def observe_lagged(
        previous: tf.Tensor,
        innovation: tf.Tensor,
        predicted: tf.Tensor,
    ) -> tf.Tensor:
        return (
            previous_weight * previous
            + innovation_weight * innovation
            + next_weight * predicted
        )

    def transition_state_jacobian(
        previous: tf.Tensor,
        innovation: tf.Tensor,
    ) -> tf.Tensor:
        del innovation
        return tf.fill(
            [tf.shape(previous)[0], tf.shape(previous)[1], state_dim, state_dim],
            transition_state,
        )

    def transition_innovation_jacobian(
        previous: tf.Tensor,
        innovation: tf.Tensor,
    ) -> tf.Tensor:
        del previous
        return tf.fill(
            [tf.shape(innovation)[0], tf.shape(innovation)[1], state_dim, innovation_dim],
            transition_innovation,
        )

    def d_transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        return tf.zeros(
            [tf.shape(previous)[0], parameter_dim, tf.shape(previous)[1], state_dim],
            dtype=tf.float64,
        )

    def observation_state_jacobian(states: tf.Tensor) -> tf.Tensor:
        return tf.fill(
            [tf.shape(states)[0], tf.shape(states)[1], observation_dim, state_dim],
            next_weight,
        )

    def d_observation(states: tf.Tensor) -> tf.Tensor:
        return tf.zeros(
            [tf.shape(states)[0], parameter_dim, tf.shape(states)[1], observation_dim],
            dtype=tf.float64,
        )

    def lagged_previous_jacobian(
        previous: tf.Tensor,
        innovation: tf.Tensor,
        predicted: tf.Tensor,
    ) -> tf.Tensor:
        del innovation, predicted
        return tf.fill(
            [tf.shape(previous)[0], tf.shape(previous)[1], observation_dim, state_dim],
            previous_weight,
        )

    def lagged_innovation_jacobian(
        previous: tf.Tensor,
        innovation: tf.Tensor,
        predicted: tf.Tensor,
    ) -> tf.Tensor:
        del previous, predicted
        return tf.fill(
            [tf.shape(innovation)[0], tf.shape(innovation)[1], observation_dim, innovation_dim],
            innovation_weight,
        )

    def lagged_next_jacobian(
        previous: tf.Tensor,
        innovation: tf.Tensor,
        predicted: tf.Tensor,
    ) -> tf.Tensor:
        del previous, innovation
        return tf.fill(
            [tf.shape(predicted)[0], tf.shape(predicted)[1], observation_dim, state_dim],
            next_weight,
        )

    def d_lagged_observation(
        previous: tf.Tensor,
        innovation: tf.Tensor,
        predicted: tf.Tensor,
    ) -> tf.Tensor:
        del innovation, predicted
        return tf.zeros(
            [tf.shape(previous)[0], parameter_dim, tf.shape(previous)[1], observation_dim],
            dtype=tf.float64,
        )

    model = TFBatchedStructuralStateSpace(
        initial_mean=theta,
        initial_covariance=tf.constant([[[0.20]]], dtype=tf.float64),
        innovation_covariance=tf.constant([[[0.10]]], dtype=tf.float64),
        observation_covariance=tf.constant([[[0.07]]], dtype=tf.float64),
        transition_fn=transition,
        observation_fn=observe,
        lagged_observation_fn=observe_lagged,
    )
    derivatives = TFBatchedStructuralFirstDerivatives(
        d_initial_mean=tf.ones([batch_size, parameter_dim, state_dim], dtype=tf.float64),
        d_initial_covariance=tf.zeros(
            [batch_size, parameter_dim, state_dim, state_dim],
            dtype=tf.float64,
        ),
        d_innovation_covariance=tf.zeros(
            [batch_size, parameter_dim, innovation_dim, innovation_dim],
            dtype=tf.float64,
        ),
        d_observation_covariance=tf.zeros(
            [batch_size, parameter_dim, observation_dim, observation_dim],
            dtype=tf.float64,
        ),
        transition_state_jacobian_fn=transition_state_jacobian,
        transition_innovation_jacobian_fn=transition_innovation_jacobian,
        d_transition_fn=d_transition,
        observation_state_jacobian_fn=observation_state_jacobian,
        d_observation_fn=d_observation,
        lagged_observation_previous_jacobian_fn=lagged_previous_jacobian,
        lagged_observation_innovation_jacobian_fn=lagged_innovation_jacobian,
        lagged_observation_next_jacobian_fn=lagged_next_jacobian,
        d_lagged_observation_fn=d_lagged_observation,
    )
    return observations, model, derivatives


def _lagged_value_score(
    theta: tf.Tensor,
    *,
    previous_coeff: float = 0.35,
) -> tuple[tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]:
    observations, model, derivatives = _lagged_linear_model_and_derivatives(
        theta,
        previous_coeff=previous_coeff,
    )
    value, score, diagnostics = tf_batched_svd_sigma_point_value_and_score(
        observations,
        model,
        derivatives,
        backend="tf_principal_sqrt_ukf",
        placement_floor=tf.constant(0.0, dtype=tf.float64),
        innovation_floor=tf.constant(1.0e-12, dtype=tf.float64),
        rank_tolerance=tf.constant(1.0e-12, dtype=tf.float64),
        spectral_gap_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
        fixed_null_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
        jitter=tf.constant(0.0, dtype=tf.float64),
    )
    return value, score, dict(diagnostics)


def _scalar_rows(
    tensors: dict[str, tf.Tensor],
    *,
    backend: str,
) -> tuple[tf.Tensor, tf.Tensor]:
    values = []
    scores = []
    batch_size = int(tensors["initial_mean"].shape[0])
    for row in range(batch_size):
        model, derivatives = _scalar_model_and_derivatives(tensors, row)
        value, score = _scalar_score(
            tensors["observations"],
            model,
            derivatives,
            backend=backend,
            args=_args(),
        )
        values.append(value)
        scores.append(score)
    return (
        tf.constant(values, dtype=tf.float64),
        tf.constant(np.stack(scores, axis=0), dtype=tf.float64),
    )


def _permute_tensors(
    tensors: dict[str, tf.Tensor],
    permutation: tf.Tensor,
) -> dict[str, tf.Tensor]:
    permuted = {}
    for name, tensor in tensors.items():
        if name == "observations":
            permuted[name] = tensor
        else:
            permuted[name] = tf.gather(tensor, permutation, axis=0)
    return permuted


@pytest.mark.parametrize("backend", BACKENDS)
def test_experimental_batched_svd_value_and_score_matches_scalar_rows(
    backend: str,
) -> None:
    tensors = _fixture()
    batched_value, batched_score = _value_and_score(tensors, backend=backend)
    scalar_value, scalar_score = _scalar_rows(tensors, backend=backend)

    np.testing.assert_allclose(
        batched_value.numpy(),
        scalar_value.numpy(),
        rtol=1.0e-8,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        batched_score.numpy(),
        scalar_score.numpy(),
        rtol=1.0e-7,
        atol=1.0e-7,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_experimental_batched_svd_singleton_matches_scalar_row(backend: str) -> None:
    tensors = _fixture(batch_size=1)
    batched_value, batched_score = _value_and_score(tensors, backend=backend)
    scalar_value, scalar_score = _scalar_rows(tensors, backend=backend)

    np.testing.assert_allclose(batched_value.numpy(), scalar_value.numpy(), atol=1.0e-8)
    np.testing.assert_allclose(
        batched_score.numpy(),
        scalar_score.numpy(),
        rtol=1.0e-7,
        atol=1.0e-7,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_experimental_batched_svd_row_permutation_preserves_order(
    backend: str,
) -> None:
    tensors = _fixture()
    base_value, base_score = _value_and_score(tensors, backend=backend)
    permutation = tf.constant([2, 0, 1], dtype=tf.int32)
    permuted_value, permuted_score = _value_and_score(
        _permute_tensors(tensors, permutation),
        backend=backend,
    )

    np.testing.assert_allclose(
        permuted_value.numpy(),
        tf.gather(base_value, permutation).numpy(),
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        permuted_score.numpy(),
        tf.gather(base_score, permutation).numpy(),
        rtol=1.0e-7,
        atol=1.0e-7,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_experimental_batched_svd_graph_and_cpu_xla_parity(backend: str) -> None:
    tensors = _fixture(batch_size=2, time_steps=3)
    eager_value, eager_score = _value_and_score(tensors, backend=backend)

    @tf.function(reduce_retracing=True)
    def graph_call() -> tuple[tf.Tensor, tf.Tensor]:
        return _value_and_score(tensors, backend=backend)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def xla_call() -> tuple[tf.Tensor, tf.Tensor]:
        return _value_and_score(tensors, backend=backend)

    graph_value, graph_score = graph_call()
    xla_value, xla_score = xla_call()

    np.testing.assert_allclose(graph_value.numpy(), eager_value.numpy(), atol=1.0e-8)
    np.testing.assert_allclose(
        graph_score.numpy(),
        eager_score.numpy(),
        rtol=1.0e-7,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(xla_value.numpy(), eager_value.numpy(), atol=1.0e-8)
    np.testing.assert_allclose(
        xla_score.numpy(),
        eager_score.numpy(),
        rtol=1.0e-7,
        atol=1.0e-7,
    )


def test_row_aligned_observations_match_independent_model_rows() -> None:
    tensors = _fixture(batch_size=3, time_steps=4)
    shared = tensors["observations"]
    row_aligned = tf.stack(
        [shared + tf.cast(row, tf.float64) * 0.025 for row in range(3)], axis=0
    )

    value, score = _value_and_score_with_observations(
        row_aligned, tensors, backend="tf_principal_sqrt_ukf"
    )
    expected = [
        _value_and_score_with_observations(
            row_aligned[row],
            _model_row(tensors, row),
            backend="tf_principal_sqrt_ukf",
        )
        for row in range(3)
    ]

    np.testing.assert_array_equal(
        value.numpy(), np.asarray([row_value.numpy()[0] for row_value, _ in expected])
    )
    np.testing.assert_array_equal(
        score.numpy(), np.stack([row_score.numpy()[0] for _, row_score in expected])
    )


def test_shared_observations_match_explicit_row_repetition_bitwise() -> None:
    tensors = _fixture(batch_size=3, time_steps=4)
    shared = tensors["observations"]
    repeated = tf.repeat(shared[tf.newaxis, :, :], repeats=3, axis=0)

    shared_value, shared_score = _value_and_score_with_observations(
        shared, tensors, backend="tf_principal_sqrt_ukf"
    )
    repeated_value, repeated_score = _value_and_score_with_observations(
        repeated, tensors, backend="tf_principal_sqrt_ukf"
    )

    np.testing.assert_array_equal(shared_value.numpy(), repeated_value.numpy())
    np.testing.assert_array_equal(shared_score.numpy(), repeated_score.numpy())


def test_row_aligned_observation_shapes_fail_closed() -> None:
    tensors = _fixture(batch_size=3, time_steps=4)
    shared = tensors["observations"]

    with pytest.raises(ValueError, match="model batch, time, observation"):
        _value_and_score_with_observations(
            tf.repeat(shared[tf.newaxis, :, :], repeats=2, axis=0),
            tensors,
            backend="tf_principal_sqrt_ukf",
        )
    with pytest.raises(ValueError, match="observation dimension"):
        _value_and_score_with_observations(
            tf.zeros([3, 4, 3], tf.float64),
            tensors,
            backend="tf_principal_sqrt_ukf",
        )
    with pytest.raises(ValueError, match="rank 1, 2, or 3"):
        _value_and_score_with_observations(
            tf.zeros([1, 3, 4, 2], tf.float64),
            tensors,
            backend="tf_principal_sqrt_ukf",
        )


def test_row_aligned_observations_cpu_xla_parity() -> None:
    tensors = _fixture(batch_size=2, time_steps=3)
    observations = tf.stack(
        (tensors["observations"], tensors["observations"] + 0.025), axis=0
    )
    eager_value, eager_score = _value_and_score_with_observations(
        observations, tensors, backend="tf_principal_sqrt_ukf"
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def xla_call(y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return _value_and_score_with_observations(
            y, tensors, backend="tf_principal_sqrt_ukf"
        )

    xla_value, xla_score = xla_call(observations)

    np.testing.assert_allclose(xla_value.numpy(), eager_value.numpy(), atol=1.0e-8)
    np.testing.assert_allclose(
        xla_score.numpy(), eager_score.numpy(), rtol=1.0e-7, atol=1.0e-7
    )


def test_experimental_batched_svd_shape_mismatch_fails_closed() -> None:
    tensors = _fixture()
    bad_tensors = dict(tensors)
    bad_tensors["d_initial_mean"] = bad_tensors["d_initial_mean"][:2]

    with pytest.raises(ValueError, match="has shape|expected|derivative batch dimension"):
        _batched_model_and_derivatives(bad_tensors)


def test_experimental_batched_svd_cpu_only_hides_gpu() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    assert tf.config.list_physical_devices("GPU") == []


def test_principal_sqrt_helper_repeated_positive_reconstructs_derivative() -> None:
    covariance = tf.eye(3, batch_shape=[2], dtype=tf.float64)
    d_covariance = tf.constant(
        [
            [
                [[0.5, 0.1, -0.2], [0.1, -0.3, 0.4], [-0.2, 0.4, 0.7]],
                [[1.0, 0.0, 0.2], [0.0, 0.6, -0.1], [0.2, -0.1, 0.8]],
            ],
            [
                [[0.2, -0.2, 0.3], [-0.2, 0.9, 0.0], [0.3, 0.0, -0.4]],
                [[0.4, 0.5, 0.0], [0.5, 0.1, -0.3], [0.0, -0.3, 0.2]],
            ],
        ],
        dtype=tf.float64,
    )

    placement = _checked_batched_principal_sqrt_factor_first_derivatives(
        covariance,
        d_covariance,
        singular_floor=tf.constant(0.0, dtype=tf.float64),
        fixed_null_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
        label="test repeated-positive placement",
    )
    lyapunov = (
        placement.factor[:, tf.newaxis, :, :] @ placement.d_factor
        + placement.d_factor @ placement.factor[:, tf.newaxis, :, :]
        - d_covariance
    )
    factor_reconstruction = (
        tf.matmul(placement.d_factor, placement.factor[:, tf.newaxis, :, :], transpose_b=True)
        + tf.matmul(placement.factor[:, tf.newaxis, :, :], placement.d_factor, transpose_b=True)
        - d_covariance
    )

    assert float(tf.reduce_max(tf.linalg.norm(lyapunov, axis=[-2, -1])).numpy()) <= 1.0e-10
    assert (
        float(tf.reduce_max(tf.linalg.norm(factor_reconstruction, axis=[-2, -1])).numpy())
        <= 1.0e-10
    )
    np.testing.assert_allclose(
        placement.derivative_reconstruction_residual.numpy(),
        np.zeros([2]),
        atol=1.0e-10,
    )


def test_principal_sqrt_helper_graph_and_cpu_xla_factor_derivative_parity() -> None:
    covariance = tf.constant(
        [
            [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 9.0]],
            [[3.0, 0.25, 0.0], [0.25, 4.0, 0.1], [0.0, 0.1, 6.0]],
        ],
        dtype=tf.float64,
    )
    d_covariance = tf.constant(
        [
            [
                [[0.5, 0.1, -0.2], [0.1, -0.3, 0.4], [-0.2, 0.4, 0.7]],
                [[1.0, 0.0, 0.2], [0.0, 0.6, -0.1], [0.2, -0.1, 0.8]],
            ],
            [
                [[0.2, -0.2, 0.3], [-0.2, 0.9, 0.0], [0.3, 0.0, -0.4]],
                [[0.4, 0.5, 0.0], [0.5, 0.1, -0.3], [0.0, -0.3, 0.2]],
            ],
        ],
        dtype=tf.float64,
    )

    def call() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        placement = _checked_batched_principal_sqrt_factor_first_derivatives(
            covariance,
            d_covariance,
            singular_floor=tf.constant(0.0, dtype=tf.float64),
            fixed_null_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
            label="test compiled principal-sqrt placement",
        )
        return (
            placement.factor,
            placement.d_factor,
            placement.derivative_reconstruction_residual,
        )

    eager_factor, eager_d_factor, eager_residual = call()

    @tf.function(reduce_retracing=True)
    def graph_call() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return call()

    @tf.function(jit_compile=True, reduce_retracing=True)
    def xla_call() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return call()

    graph_factor, graph_d_factor, graph_residual = graph_call()
    xla_factor, xla_d_factor, xla_residual = xla_call()

    np.testing.assert_allclose(graph_factor.numpy(), eager_factor.numpy(), atol=1.0e-11)
    np.testing.assert_allclose(graph_d_factor.numpy(), eager_d_factor.numpy(), atol=1.0e-11)
    np.testing.assert_allclose(graph_residual.numpy(), eager_residual.numpy(), atol=1.0e-11)
    np.testing.assert_allclose(xla_factor.numpy(), eager_factor.numpy(), atol=1.0e-11)
    np.testing.assert_allclose(xla_d_factor.numpy(), eager_d_factor.numpy(), atol=1.0e-11)
    np.testing.assert_allclose(xla_residual.numpy(), eager_residual.numpy(), atol=1.0e-11)


def test_principal_sqrt_helper_scaled_reconstruction_guard_allows_large_rhs() -> None:
    covariance = tf.constant(
        [
            [[4.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 16.0]],
        ],
        dtype=tf.float64,
    )
    d_covariance = tf.constant(
        [
            [
                [
                    [1.0e4, 2.0e3, -1.0e3],
                    [2.0e3, -3.0e4, 4.0e3],
                    [-1.0e3, 4.0e3, 7.0e4],
                ],
            ],
        ],
        dtype=tf.float64,
    )

    placement = _checked_batched_principal_sqrt_factor_first_derivatives(
        covariance,
        d_covariance,
        singular_floor=tf.constant(0.0, dtype=tf.float64),
        fixed_null_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
        lyapunov_tolerance=tf.constant(1.0e-12, dtype=tf.float64),
        label="test scaled principal-sqrt reconstruction",
    )

    assert np.isfinite(placement.d_factor.numpy()).all()
    assert float(tf.reduce_max(
        placement.derivative_reconstruction_residual).numpy()) <= 1.0e-5



def test_principal_sqrt_helper_scaled_reconstruction_guard_rejects_bad_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    covariance = tf.constant(
        [
            [[4.0, 0.0], [0.0, 9.0]],
        ],
        dtype=tf.float64,
    )
    d_covariance = tf.constant(
        [
            [
                [[1.0, 0.2], [0.2, -0.3]],
            ],
        ],
        dtype=tf.float64,
    )

    def wrong_sylvester_solve(factor: tf.Tensor, rhs: tf.Tensor) -> tf.Tensor:
        del factor
        return tf.zeros_like(rhs)

    monkeypatch.setattr(
        svd_tf,
        "symmetric_sylvester_solve",
        wrong_sylvester_solve,
    )

    with pytest.raises(
        tf.errors.InvalidArgumentError,
        match="blocked_principal_sqrt_reconstruction",
    ):
        placement = _checked_batched_principal_sqrt_factor_first_derivatives(
            covariance,
            d_covariance,
            singular_floor=tf.constant(0.0, dtype=tf.float64),
            fixed_null_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
            lyapunov_tolerance=tf.constant(1.0e-12, dtype=tf.float64),
            label="test bad principal-sqrt reconstruction",
        )
        _ = placement.d_factor.numpy()


def test_principal_sqrt_helper_roundoff_repair_cpu_xla_records_diagnostics() -> None:
    covariance = tf.constant(
        [
            [[1.0, 0.0], [0.0, -1.0e-20]],
            [[2.0, 0.1], [0.1, 1.5]],
        ],
        dtype=tf.float64,
    )
    d_covariance = tf.zeros([2, 1, 2, 2], dtype=tf.float64)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def xla_call() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        placement = _checked_batched_principal_sqrt_factor_first_derivatives(
            covariance,
            d_covariance,
            singular_floor=tf.constant(0.0, dtype=tf.float64),
            fixed_null_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
            label="test roundoff repair placement",
        )
        return (
            placement.factor,
            placement.roundoff_repair_count,
            placement.classified_invalid_count,
            placement.min_eigenvalue,
        )

    factor, repair_count, invalid_count, min_eigenvalue = xla_call()

    assert np.isfinite(factor.numpy()).all()
    np.testing.assert_array_equal(repair_count.numpy(), np.asarray([1, 0], dtype=np.int32))
    np.testing.assert_array_equal(invalid_count.numpy(), np.asarray([0, 0], dtype=np.int32))
    assert float(min_eigenvalue.numpy()[0]) < 0.0


def test_principal_sqrt_helper_repairs_low_margin_strict_rows_cpu_xla() -> None:
    covariance = tf.constant(
        [
            [[1.0, 0.0], [0.0, 1.0e-16]],
            [[1.0, 0.0], [0.0, 2.0e-14]],
        ],
        dtype=tf.float64,
    )
    d_covariance = tf.zeros([2, 1, 2, 2], dtype=tf.float64)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def xla_call() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        placement = _checked_batched_principal_sqrt_factor_first_derivatives(
            covariance,
            d_covariance,
            singular_floor=tf.constant(0.0, dtype=tf.float64),
            fixed_null_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
            label="test low-margin strict row repair",
        )
        return (
            placement.implemented_covariance,
            placement.roundoff_repair_count,
            placement.classified_invalid_count,
            placement.min_eigenvalue,
        )

    implemented, repair_count, invalid_count, min_eigenvalue = xla_call()

    np.testing.assert_allclose(
        np.diag(implemented.numpy()[0]),
        np.asarray([1.0 + 1.0e-14, 2.01e-14], dtype=np.float64),
        atol=1.0e-18,
    )
    np.testing.assert_allclose(
        np.diag(implemented.numpy()[1]),
        np.asarray([1.0 + 1.0e-14, 3.0e-14], dtype=np.float64),
        atol=1.0e-18,
    )
    np.testing.assert_array_equal(repair_count.numpy(), np.asarray([1, 0], dtype=np.int32))
    np.testing.assert_array_equal(invalid_count.numpy(), np.asarray([0, 0], dtype=np.int32))
    assert 0.0 < float(min_eigenvalue.numpy()[0]) < 1.0e-14


@pytest.mark.parametrize(
    "covariance",
    (
        tf.constant([[[1.0, 0.0], [0.0, np.nan]]], dtype=tf.float64),
        tf.constant([[[1.0, 0.0], [0.0, -1.0e-3]]], dtype=tf.float64),
    ),
)
def test_principal_sqrt_helper_classifies_invalid_covariance_cpu_xla(
    covariance: tf.Tensor,
) -> None:
    d_covariance = tf.zeros([1, 1, 2, 2], dtype=tf.float64)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def xla_call() -> tuple[tf.Tensor, tf.Tensor]:
        placement = _checked_batched_principal_sqrt_factor_first_derivatives(
            covariance,
            d_covariance,
            singular_floor=tf.constant(0.0, dtype=tf.float64),
            fixed_null_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
            label="test classified invalid placement",
        )
        return placement.factor, placement.classified_invalid_count

    factor, invalid_count = xla_call()

    assert np.isfinite(factor.numpy()).all()
    np.testing.assert_array_equal(invalid_count.numpy(), np.asarray([1], dtype=np.int32))


def test_principal_sqrt_nonfinite_covariance_never_reaches_eigensolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    covariance = tf.constant(
        [[[2.0, 0.1], [0.1, 1.5]], [[1.0, 0.0], [0.0, np.nan]]],
        tf.float64,
    )
    d_covariance = tf.zeros([2, 1, 2, 2], tf.float64)
    original = svd_tf._batched_psd_eigh

    def finite_only_eigh(values: tf.Tensor, floor: tf.Tensor):
        tf.debugging.assert_all_finite(values, "unsafe covariance reached eigensolver")
        return original(values, floor)

    monkeypatch.setattr(svd_tf, "_batched_psd_eigh", finite_only_eigh)
    placement = _checked_batched_principal_sqrt_factor_first_derivatives(
        covariance,
        d_covariance,
        singular_floor=tf.constant(0.0, tf.float64),
        fixed_null_tolerance=tf.constant(1.0e-10, tf.float64),
        label="test pre-eigensolver nonfinite classification",
        factor_backend="tensorflow_newton_schulz",
    )

    np.testing.assert_array_equal(
        placement.classified_invalid_count.numpy(), np.asarray([0, 1], np.int32)
    )
    assert np.isfinite(placement.factor.numpy()).all()


def test_principal_sqrt_helper_classifies_nonfinite_derivative_rhs_cpu_xla() -> None:
    covariance = tf.constant([[[2.0, 0.1], [0.1, 1.5]]], dtype=tf.float64)
    d_covariance = tf.constant(
        [[[[0.0, np.nan], [np.nan, 0.0]]]],
        dtype=tf.float64,
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def xla_call() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        placement = _checked_batched_principal_sqrt_factor_first_derivatives(
            covariance,
            d_covariance,
            singular_floor=tf.constant(0.0, dtype=tf.float64),
            fixed_null_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
            label="test nonfinite derivative RHS placement",
        )
        return (
            placement.factor,
            placement.d_factor,
            placement.classified_invalid_count,
            placement.derivative_rhs_nonfinite_count,
        )

    factor, d_factor, invalid_count, rhs_nonfinite_count = xla_call()

    assert np.isfinite(factor.numpy()).all()
    assert np.isfinite(d_factor.numpy()).all()
    np.testing.assert_array_equal(invalid_count.numpy(), np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(
        rhs_nonfinite_count.numpy(),
        np.asarray([1], dtype=np.int32),
    )


def test_principal_sqrt_value_score_rejects_invalid_innovation_covariance_cpu_xla() -> None:
    observations, model, derivatives = _repeated_positive_model_and_derivatives()
    bad_model = TFBatchedStructuralStateSpace(
        initial_mean=model.initial_mean,
        initial_covariance=model.initial_covariance,
        innovation_covariance=tf.fill(
            tf.shape(model.innovation_covariance),
            tf.constant(-1.0e-3, dtype=tf.float64),
        ),
        observation_covariance=model.observation_covariance,
        transition_fn=model.transition_fn,
        observation_fn=model.observation_fn,
        deterministic_residual_fn=model.deterministic_residual_fn,
        lagged_observation_fn=model.lagged_observation_fn,
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def xla_call() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        value, score, diagnostics = tf_batched_svd_sigma_point_value_and_score(
            observations,
            bad_model,
            derivatives,
            backend="tf_principal_sqrt_ukf",
            placement_floor=tf.constant(0.0, dtype=tf.float64),
            innovation_floor=tf.constant(1.0e-12, dtype=tf.float64),
        )
        return (
            value,
            score,
            diagnostics["principal_sqrt_target_classified_invalid_count"],
            diagnostics["principal_sqrt_target_valid_count"],
            diagnostics["principal_sqrt_target_row_class_code"],
        )

    value, score, invalid_count, valid_count, row_class_code = xla_call()

    np.testing.assert_allclose(
        value.numpy(),
        np.full([2], -1.0e100, dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(score.numpy(), np.zeros([2, 1]), atol=0.0)
    np.testing.assert_array_equal(invalid_count.numpy(), np.ones([2], dtype=np.int32))
    np.testing.assert_array_equal(valid_count.numpy(), np.zeros([2], dtype=np.int32))
    np.testing.assert_array_equal(row_class_code.numpy(), np.full([2], 2, dtype=np.int32))


def test_principal_sqrt_value_score_rejects_nonfinite_derivative_rhs_cpu_xla() -> None:
    observations, model, derivatives = _repeated_positive_model_and_derivatives()
    bad_derivatives = TFBatchedStructuralFirstDerivatives(
        d_initial_mean=derivatives.d_initial_mean,
        d_initial_covariance=tf.fill(
            tf.shape(derivatives.d_initial_covariance),
            tf.constant(np.nan, dtype=tf.float64),
        ),
        d_innovation_covariance=derivatives.d_innovation_covariance,
        d_observation_covariance=derivatives.d_observation_covariance,
        transition_state_jacobian_fn=derivatives.transition_state_jacobian_fn,
        transition_innovation_jacobian_fn=derivatives.transition_innovation_jacobian_fn,
        d_transition_fn=derivatives.d_transition_fn,
        observation_state_jacobian_fn=derivatives.observation_state_jacobian_fn,
        d_observation_fn=derivatives.d_observation_fn,
        lagged_observation_previous_jacobian_fn=(
            derivatives.lagged_observation_previous_jacobian_fn
        ),
        lagged_observation_innovation_jacobian_fn=(
            derivatives.lagged_observation_innovation_jacobian_fn
        ),
        lagged_observation_next_jacobian_fn=(
            derivatives.lagged_observation_next_jacobian_fn
        ),
        d_lagged_observation_fn=derivatives.d_lagged_observation_fn,
        name=derivatives.name,
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def xla_call() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        value, score, diagnostics = tf_batched_svd_sigma_point_value_and_score(
            observations,
            model,
            bad_derivatives,
            backend="tf_principal_sqrt_ukf",
            placement_floor=tf.constant(0.0, dtype=tf.float64),
            innovation_floor=tf.constant(1.0e-12, dtype=tf.float64),
        )
        return (
            value,
            score,
            diagnostics["principal_sqrt_target_classified_invalid_count"],
            diagnostics["principal_sqrt_target_derivative_rhs_nonfinite_count"],
            diagnostics["principal_sqrt_target_row_class_code"],
        )

    value, score, invalid_count, rhs_nonfinite_count, row_class_code = xla_call()

    np.testing.assert_allclose(
        value.numpy(),
        np.full([2], -1.0e100, dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(score.numpy(), np.zeros([2, 1]), atol=0.0)
    np.testing.assert_array_equal(invalid_count.numpy(), np.ones([2], dtype=np.int32))
    np.testing.assert_array_equal(
        rhs_nonfinite_count.numpy(),
        np.ones([2], dtype=np.int32),
    )
    np.testing.assert_array_equal(row_class_code.numpy(), np.full([2], 2, dtype=np.int32))


def test_principal_sqrt_value_score_valid_rows_report_valid_class_cpu_xla() -> None:
    observations, model, derivatives = _repeated_positive_model_and_derivatives()

    @tf.function(jit_compile=True, reduce_retracing=True)
    def xla_call() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        value, score, diagnostics = tf_batched_svd_sigma_point_value_and_score(
            observations,
            model,
            derivatives,
            backend="tf_principal_sqrt_ukf",
            placement_floor=tf.constant(0.0, dtype=tf.float64),
            innovation_floor=tf.constant(1.0e-12, dtype=tf.float64),
        )
        return (
            value,
            score,
            diagnostics["principal_sqrt_target_classified_invalid_count"],
            diagnostics["principal_sqrt_target_valid_count"],
            diagnostics["principal_sqrt_target_row_class_code"],
        )

    value, score, invalid_count, valid_count, row_class_code = xla_call()

    assert np.isfinite(value.numpy()).all()
    assert np.isfinite(score.numpy()).all()
    np.testing.assert_array_equal(invalid_count.numpy(), np.zeros([2], dtype=np.int32))
    np.testing.assert_array_equal(valid_count.numpy(), np.ones([2], dtype=np.int32))
    np.testing.assert_array_equal(row_class_code.numpy(), np.zeros([2], dtype=np.int32))


def test_principal_sqrt_helper_uses_compiled_factor_and_sylvester_derivative() -> None:
    from bayesfilter.nonlinear import experimental_batched_svd_sigma_point_tf as module

    source = inspect.getsource(
        module._checked_batched_principal_sqrt_factor_first_derivatives
    )
    tree = ast.parse(textwrap.dedent(source))

    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "_principal_sqrt_factor" in called_names
    assert "_symmetric_sylvester_factor_solve" in called_names
    assert "symmetric_principal_sqrt" in inspect.getsource(module._principal_sqrt_factor)
    assert "symmetric_sylvester_solve" in inspect.getsource(
        module._symmetric_sylvester_factor_solve
    )
    assert "roundoff_repair_count" in source
    assert "classified_invalid_count" in source
    assert "assert_equal" not in source
    assert "_principal_sqrt_frechet_derivative_from_eigh" not in source
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr == "eigh"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "linalg"
        for node in ast.walk(tree)
    )


def test_newton_schulz_principal_root_matches_eigh_and_reconstructs_cpu_xla() -> None:
    covariance = tf.constant(
        [
            [[4.0, 0.2, 0.0], [0.2, 2.0, 0.1], [0.0, 0.1, 1.0]],
            [[3.0, -0.1, 0.2], [-0.1, 5.0, 0.0], [0.2, 0.0, 2.0]],
        ],
        tf.float64,
    )
    expected = svd_tf._tensorflow_native_principal_sqrt(covariance)
    eager = svd_tf._tensorflow_newton_schulz_principal_sqrt(covariance)

    @tf.function(
        input_signature=[tf.TensorSpec([2, 3, 3], tf.float64)],
        jit_compile=True,
    )
    def compiled(values: tf.Tensor) -> tf.Tensor:
        return svd_tf._tensorflow_newton_schulz_principal_sqrt(values)

    xla = compiled(covariance)
    np.testing.assert_allclose(eager.numpy(), expected.numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(xla.numpy(), eager.numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        (xla @ tf.linalg.matrix_transpose(xla)).numpy(),
        covariance.numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_principal_sqrt_dispatcher_passes_repeated_positive_while_svd_fails() -> None:
    observations, model, derivatives = _repeated_positive_model_and_derivatives()

    with pytest.raises(tf.errors.InvalidArgumentError, match="blocked_weak_spectral_gap"):
        tf_batched_svd_sigma_point_value_and_score(
            observations,
            model,
            derivatives,
            backend="tf_svd_ukf",
            spectral_gap_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
        )

    value, score, diagnostics = tf_batched_svd_sigma_point_value_and_score(
        observations,
        model,
        derivatives,
        backend="tf_principal_sqrt_ukf",
        spectral_gap_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
    )

    assert np.isfinite(value.numpy()).all()
    assert np.isfinite(score.numpy()).all()
    assert diagnostics["backend"].numpy() == b"tf_principal_sqrt_ukf"
    assert diagnostics["derivative_branch"].numpy() == b"strict_spd_principal_sqrt"
    np.testing.assert_allclose(
        diagnostics["factor_derivative_reconstruction_residual"].numpy(),
        np.zeros([2]),
        atol=1.0e-10,
    )


def test_principal_sqrt_ukf_nondegenerate_parity_with_svd_ukf() -> None:
    tensors = _fixture(batch_size=2, time_steps=3)
    svd_value, svd_score, _diagnostics = _value_score_diagnostics(tensors, backend="tf_svd_ukf")
    sqrt_value, sqrt_score, diagnostics = _value_score_diagnostics(
        tensors,
        backend="tf_principal_sqrt_ukf",
    )

    np.testing.assert_allclose(sqrt_value.numpy(), svd_value.numpy(), atol=1.0e-7)
    np.testing.assert_allclose(
        sqrt_score.numpy(),
        svd_score.numpy(),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    assert diagnostics["backend"].numpy() == b"tf_principal_sqrt_ukf"


def test_principal_sqrt_ukf_graph_and_cpu_xla_smoke() -> None:
    tensors = _fixture(batch_size=2, time_steps=3)
    eager_value, eager_score, _diagnostics = _value_score_diagnostics(
        tensors,
        backend="tf_principal_sqrt_ukf",
    )

    @tf.function(reduce_retracing=True)
    def graph_call() -> tuple[tf.Tensor, tf.Tensor]:
        value, score, _diagnostics = _value_score_diagnostics(
            tensors,
            backend="tf_principal_sqrt_ukf",
        )
        return value, score

    @tf.function(jit_compile=True, reduce_retracing=True)
    def xla_call() -> tuple[tf.Tensor, tf.Tensor]:
        value, score, _diagnostics = _value_score_diagnostics(
            tensors,
            backend="tf_principal_sqrt_ukf",
        )
        return value, score

    graph_value, graph_score = graph_call()
    xla_value, xla_score = xla_call()

    np.testing.assert_allclose(graph_value.numpy(), eager_value.numpy(), atol=1.0e-7)
    np.testing.assert_allclose(graph_score.numpy(), eager_score.numpy(), rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(xla_value.numpy(), eager_value.numpy(), atol=1.0e-7)
    np.testing.assert_allclose(xla_score.numpy(), eager_score.numpy(), rtol=1.0e-5, atol=1.0e-6)


def test_lagged_observation_contract_selects_runtime_path_and_changes_value() -> None:
    theta = tf.constant([[0.04]], dtype=tf.float64)
    lagged_value, lagged_score, diagnostics = _lagged_value_score(
        theta,
        previous_coeff=0.35,
    )
    current_value, _current_score, current_diagnostics = _lagged_value_score(
        theta,
        previous_coeff=0.0,
    )

    assert diagnostics["observation_contract"].numpy() == (
        b"lagged_previous_innovation_predicted"
    )
    assert bool(diagnostics["observation_contract_runtime_selected"].numpy()) is True
    assert current_diagnostics["observation_contract"].numpy() == (
        b"lagged_previous_innovation_predicted"
    )
    assert np.isfinite(lagged_value.numpy()).all()
    assert np.isfinite(lagged_score.numpy()).all()
    assert abs(float(lagged_value.numpy()[0] - current_value.numpy()[0])) > 1.0e-6


def test_lagged_observation_contract_score_matches_finite_difference() -> None:
    theta = tf.constant([[0.04]], dtype=tf.float64)
    value, score, _diagnostics = _lagged_value_score(theta)
    step = tf.constant(1.0e-5, dtype=tf.float64)
    plus_value, _plus_score, _plus_diagnostics = _lagged_value_score(theta + step)
    minus_value, _minus_score, _minus_diagnostics = _lagged_value_score(theta - step)
    finite_difference = (plus_value - minus_value) / (2.0 * step)

    assert np.isfinite(value.numpy()).all()
    np.testing.assert_allclose(
        score.numpy()[:, 0],
        finite_difference.numpy(),
        rtol=5.0e-5,
        atol=5.0e-6,
    )


def test_lagged_observation_contract_cpu_xla_smoke() -> None:
    theta = tf.constant([[0.04]], dtype=tf.float64)
    eager_value, eager_score, eager_diagnostics = _lagged_value_score(theta)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def xla_call(theta_arg: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        value, score, diagnostics = _lagged_value_score(theta_arg)
        with tf.control_dependencies(
            [
                tf.debugging.assert_equal(
                    diagnostics["observation_contract"],
                    tf.constant("lagged_previous_innovation_predicted"),
                ),
                tf.debugging.assert_equal(
                    diagnostics["observation_contract_runtime_selected"],
                    tf.constant(True),
                ),
            ]
        ):
            return tf.identity(value), tf.identity(score)

    xla_value, xla_score = xla_call(theta)

    assert eager_diagnostics["observation_contract"].numpy() == (
        b"lagged_previous_innovation_predicted"
    )
    np.testing.assert_allclose(xla_value.numpy(), eager_value.numpy(), atol=1.0e-8)
    np.testing.assert_allclose(xla_score.numpy(), eager_score.numpy(), rtol=1.0e-5, atol=1.0e-6)


def test_lagged_observation_contract_source_contract() -> None:
    from bayesfilter.nonlinear import experimental_batched_svd_sigma_point_tf as module

    source = inspect.getsource(module.tf_batched_svd_sigma_point_value_and_score_with_rule)
    tree = ast.parse(textwrap.dedent(source))

    assert "tf.while_loop" in source
    assert "observation_contract" in source
    assert "lagged_previous_innovation_predicted" in source
    assert "d_lagged_observation_fn" in source
    assert not any(isinstance(node, (ast.For, ast.AsyncFor)) for node in ast.walk(tree))


@pytest.mark.parametrize("backend", OLD_BACKENDS)
def test_old_backend_dispatcher_smoke_still_passes(backend: str) -> None:
    tensors = _fixture(batch_size=2, time_steps=3)
    value, score, diagnostics = _value_score_diagnostics(tensors, backend=backend)

    assert np.isfinite(value.numpy()).all()
    assert np.isfinite(score.numpy()).all()
    assert diagnostics["backend"].numpy() == backend.encode()
