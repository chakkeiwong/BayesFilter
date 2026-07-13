"""Testing helpers for the multidimensional lower-triangular LGSSM fixture.

This module materializes the Phase 1/2 synthetic benchmark contract. It does
not train NeuTra, run HMC, compute posterior diagnostics, or certify target
score correctness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf

from bayesfilter.linear.kalman_svd_derivatives_tf import (
    SVD_LINEAR_SCORE_STATUS_VALID_PRE_REGULARIZED,
    tf_svd_linear_gaussian_score_first_order_graph_status,
)
from bayesfilter.linear.types_tf import TFLinearGaussianStateSpaceFirstDerivatives
from bayesfilter.linear.types_tf import TFLinearGaussianStateSpace


DEFAULT_ARTIFACT_ROOT = Path(
    "docs/plans/artifacts/multidim-triangular-lgssm-neutra-hmc-2026-07-08"
)
DEFAULT_CONTRACT_PATH = DEFAULT_ARTIFACT_ROOT / "lower_triangular_lgssm_contract_v1.json"
DEFAULT_SYNTHETIC_DATA_PATH = (
    DEFAULT_ARTIFACT_ROOT / "lower_triangular_lgssm_synthetic_data_v1_seed20260708.json"
)


@dataclass(frozen=True)
class LowerTriangularLGSSMMaterialization:
    """TensorFlow tensors for the Phase 1 lower-triangular LGSSM contract."""

    raw_parameters: tf.Tensor
    transition_matrix: tf.Tensor
    process_std: tf.Tensor
    observation_std: tf.Tensor
    stationary_covariance: tf.Tensor
    model: TFLinearGaussianStateSpace
    derivatives: TFLinearGaussianStateSpaceFirstDerivatives | None = None


def load_lower_triangular_lgssm_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    """Load the Phase 1 lower-triangular LGSSM contract JSON."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_lower_triangular_lgssm_synthetic_data(
    path: str | Path = DEFAULT_SYNTHETIC_DATA_PATH,
) -> dict[str, Any]:
    """Load the Phase 2 lower-triangular LGSSM synthetic data JSON."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def raw_truth_from_contract(
    contract: dict[str, Any],
    *,
    dtype: tf.dtypes.DType = tf.float64,
) -> tf.Tensor:
    """Return the Phase 1 raw truth vector from a contract manifest."""

    rho_max = tf.constant(float(contract["transform"]["rho_max"]), dtype=dtype)
    lower_scale = tf.constant(float(contract["transform"]["lower_scale"]), dtype=dtype)
    truth = contract["truth_template"]
    diagonal = tf.convert_to_tensor(truth["diag_A"], dtype=dtype)
    lower_values = tf.convert_to_tensor(
        [
            truth["lower_A"]["a21"],
            truth["lower_A"]["a31"],
            truth["lower_A"]["a32"],
            truth["lower_A"]["a41"],
            truth["lower_A"]["a42"],
            truth["lower_A"]["a43"],
        ],
        dtype=dtype,
    )
    process_std = tf.convert_to_tensor(truth["process_std"], dtype=dtype)
    observation_std = tf.convert_to_tensor(truth["observation_std"], dtype=dtype)
    return tf.concat(
        [
            tf.atanh(diagonal / rho_max),
            tf.atanh(lower_values / lower_scale),
            tf.math.log(process_std),
            tf.math.log(observation_std),
        ],
        axis=0,
    )


def stationary_covariance_from_transition_tf(
    transition_matrix: Any,
    transition_covariance: Any,
) -> tf.Tensor:
    """Solve ``P = A P A' + Q`` for a stationary covariance matrix."""

    transition = tf.convert_to_tensor(transition_matrix, dtype=tf.float64)
    covariance = tf.convert_to_tensor(transition_covariance, dtype=tf.float64)
    if transition.shape.rank != 2 or covariance.shape.rank != 2:
        raise ValueError("transition_matrix and transition_covariance must be rank 2")
    state_dim = transition.shape[0]
    if state_dim is None or transition.shape[1] != state_dim:
        raise ValueError("transition_matrix must be square with static shape")
    if tuple(covariance.shape.as_list()) != (int(state_dim), int(state_dim)):
        raise ValueError("transition_covariance must match transition_matrix shape")
    dim = int(state_dim)
    system = tf.eye(dim * dim, dtype=tf.float64) - _kron_tf(transition, transition)
    solution = tf.linalg.solve(
        system,
        tf.reshape(covariance, (dim * dim, 1)),
    )
    matrix = tf.reshape(solution, (dim, dim))
    return _symmetrize(matrix)


def materialize_lower_triangular_lgssm_from_raw(
    raw_parameters: Any,
    contract: dict[str, Any] | None = None,
) -> LowerTriangularLGSSMMaterialization:
    """Materialize the Phase 1 lower-triangular LGSSM from raw parameters."""

    manifest = load_lower_triangular_lgssm_contract() if contract is None else contract
    raw = tf.convert_to_tensor(raw_parameters, dtype=tf.float64)
    if raw.shape.rank != 1:
        raise ValueError("raw_parameters must be rank 1")
    parameter_dim = raw.shape[0]
    if parameter_dim is None:
        raise ValueError("raw_parameters must have static dimension")
    expected_dim = int(manifest["static_shape"]["parameter_dim"])
    if int(parameter_dim) != expected_dim:
        raise ValueError("raw_parameters length must match contract parameter_dim")
    transition = lower_triangular_transition_from_raw(raw, manifest)
    process_std = tf.exp(raw[10:14])
    observation_std = tf.exp(raw[14:18])
    process_covariance = tf.linalg.diag(tf.square(process_std))
    observation_covariance = tf.linalg.diag(tf.square(observation_std))
    stationary_covariance = stationary_covariance_from_transition_tf(
        transition,
        process_covariance,
    )
    state_dim = int(manifest["static_shape"]["state_dim"])
    observation_dim = int(manifest["static_shape"]["observation_dim"])
    model = TFLinearGaussianStateSpace(
        initial_mean=tf.zeros((state_dim,), dtype=tf.float64),
        initial_covariance=stationary_covariance,
        transition_offset=tf.zeros((state_dim,), dtype=tf.float64),
        transition_matrix=transition,
        transition_covariance=process_covariance,
        observation_offset=tf.zeros((observation_dim,), dtype=tf.float64),
        observation_matrix=tf.eye(observation_dim, state_dim, dtype=tf.float64),
        observation_covariance=observation_covariance,
    )
    return LowerTriangularLGSSMMaterialization(
        raw_parameters=raw,
        transition_matrix=transition,
        process_std=process_std,
        observation_std=observation_std,
        stationary_covariance=stationary_covariance,
        model=model,
        derivatives=None,
    )


def materialize_lower_triangular_lgssm_with_first_derivatives(
    raw_parameters: Any,
    contract: dict[str, Any] | None = None,
) -> LowerTriangularLGSSMMaterialization:
    """Materialize the Phase 1 LGSSM and manual first derivatives."""

    manifest = load_lower_triangular_lgssm_contract() if contract is None else contract
    materialized = materialize_lower_triangular_lgssm_from_raw(raw_parameters, manifest)
    raw = materialized.raw_parameters
    state_dim = int(manifest["static_shape"]["state_dim"])
    observation_dim = int(manifest["static_shape"]["observation_dim"])
    parameter_dim = int(manifest["static_shape"]["parameter_dim"])
    d_transition = transition_first_derivatives_from_raw(raw, manifest)
    d_process_covariance = process_covariance_first_derivatives_from_raw(raw, state_dim)
    d_observation_covariance = observation_covariance_first_derivatives_from_raw(
        raw,
        observation_dim,
    )
    d_stationary = stationary_covariance_first_derivatives_tf(
        materialized.transition_matrix,
        materialized.model.transition_covariance,
        materialized.stationary_covariance,
        d_transition,
        d_process_covariance,
    )
    derivatives = TFLinearGaussianStateSpaceFirstDerivatives(
        d_initial_mean=tf.zeros((parameter_dim, state_dim), dtype=tf.float64),
        d_initial_covariance=d_stationary,
        d_transition_offset=tf.zeros((parameter_dim, state_dim), dtype=tf.float64),
        d_transition_matrix=d_transition,
        d_transition_covariance=d_process_covariance,
        d_observation_offset=tf.zeros((parameter_dim, observation_dim), dtype=tf.float64),
        d_observation_matrix=tf.zeros(
            (parameter_dim, observation_dim, state_dim),
            dtype=tf.float64,
        ),
        d_observation_covariance=d_observation_covariance,
    )
    return LowerTriangularLGSSMMaterialization(
        raw_parameters=raw,
        transition_matrix=materialized.transition_matrix,
        process_std=materialized.process_std,
        observation_std=materialized.observation_std,
        stationary_covariance=materialized.stationary_covariance,
        model=materialized.model,
        derivatives=derivatives,
    )


def lower_triangular_transition_from_raw(
    raw_parameters: Any,
    contract: dict[str, Any],
) -> tf.Tensor:
    """Return the lower-triangular transition matrix defined by Phase 1."""

    raw = tf.convert_to_tensor(raw_parameters, dtype=tf.float64)
    rho_max = tf.constant(float(contract["transform"]["rho_max"]), dtype=tf.float64)
    lower_scale = tf.constant(
        float(contract["transform"]["lower_scale"]),
        dtype=tf.float64,
    )
    diagonal = rho_max * tf.tanh(raw[0:4])
    lower = lower_scale * tf.tanh(raw[4:10])
    zeros = tf.zeros_like(diagonal)
    row0 = tf.stack([diagonal[0], zeros[0], zeros[0], zeros[0]])
    row1 = tf.stack([lower[0], diagonal[1], zeros[0], zeros[0]])
    row2 = tf.stack([lower[1], lower[2], diagonal[2], zeros[0]])
    row3 = tf.stack([lower[3], lower[4], lower[5], diagonal[3]])
    return tf.stack([row0, row1, row2, row3], axis=0)


def transition_first_derivatives_from_raw(
    raw_parameters: Any,
    contract: dict[str, Any],
) -> tf.Tensor:
    """Return first derivatives of ``A`` with respect to raw parameters."""

    raw = tf.convert_to_tensor(raw_parameters, dtype=tf.float64)
    rho_max = tf.constant(float(contract["transform"]["rho_max"]), dtype=tf.float64)
    lower_scale = tf.constant(
        float(contract["transform"]["lower_scale"]),
        dtype=tf.float64,
    )
    parameter_dim = int(contract["static_shape"]["parameter_dim"])
    state_dim = int(contract["static_shape"]["state_dim"])
    indices = tf.constant(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 1, 0],
            [5, 2, 0],
            [6, 2, 1],
            [7, 3, 0],
            [8, 3, 1],
            [9, 3, 2],
        ],
        dtype=tf.int32,
    )
    diagonal_derivatives = rho_max * (1.0 - tf.square(tf.tanh(raw[0:4])))
    lower_derivatives = lower_scale * (1.0 - tf.square(tf.tanh(raw[4:10])))
    updates = tf.concat([diagonal_derivatives, lower_derivatives], axis=0)
    return tf.tensor_scatter_nd_update(
        tf.zeros((parameter_dim, state_dim, state_dim), dtype=tf.float64),
        indices,
        updates,
    )


def process_covariance_first_derivatives_from_raw(
    raw_parameters: Any,
    state_dim: int,
) -> tf.Tensor:
    """Return first derivatives of diagonal process covariance ``Q``."""

    raw = tf.convert_to_tensor(raw_parameters, dtype=tf.float64)
    parameter_dim = int(raw.shape[0])
    log_q = raw[10:14]
    updates = 2.0 * tf.exp(2.0 * log_q)
    indices = tf.constant(
        [[10, 0, 0], [11, 1, 1], [12, 2, 2], [13, 3, 3]],
        dtype=tf.int32,
    )
    return tf.tensor_scatter_nd_update(
        tf.zeros((parameter_dim, state_dim, state_dim), dtype=tf.float64),
        indices,
        updates,
    )


def observation_covariance_first_derivatives_from_raw(
    raw_parameters: Any,
    observation_dim: int,
) -> tf.Tensor:
    """Return first derivatives of diagonal observation covariance ``R``."""

    raw = tf.convert_to_tensor(raw_parameters, dtype=tf.float64)
    parameter_dim = int(raw.shape[0])
    log_r = raw[14:18]
    updates = 2.0 * tf.exp(2.0 * log_r)
    indices = tf.constant(
        [[14, 0, 0], [15, 1, 1], [16, 2, 2], [17, 3, 3]],
        dtype=tf.int32,
    )
    return tf.tensor_scatter_nd_update(
        tf.zeros((parameter_dim, observation_dim, observation_dim), dtype=tf.float64),
        indices,
        updates,
    )


def stationary_covariance_first_derivatives_tf(
    transition_matrix: Any,
    transition_covariance: Any,
    stationary_covariance: Any,
    d_transition_matrix: Any,
    d_transition_covariance: Any,
) -> tf.Tensor:
    """Return first derivatives of the discrete stationary covariance solve."""

    transition = tf.convert_to_tensor(transition_matrix, dtype=tf.float64)
    covariance = tf.convert_to_tensor(transition_covariance, dtype=tf.float64)
    stationary = tf.convert_to_tensor(stationary_covariance, dtype=tf.float64)
    d_transition = tf.convert_to_tensor(d_transition_matrix, dtype=tf.float64)
    d_covariance = tf.convert_to_tensor(d_transition_covariance, dtype=tf.float64)
    del covariance
    parameter_dim = int(d_transition.shape[0])
    state_dim = int(transition.shape[0])
    system = tf.eye(state_dim * state_dim, dtype=tf.float64) - _kron_tf(
        transition,
        transition,
    )
    rhs = (
        d_transition @ stationary @ tf.transpose(transition)
        + transition[tf.newaxis, :, :] @ stationary[tf.newaxis, :, :] @ tf.linalg.matrix_transpose(d_transition)
        + d_covariance
    )
    batched_system = tf.broadcast_to(
        system,
        (parameter_dim, state_dim * state_dim, state_dim * state_dim),
    )
    solution = tf.linalg.solve(
        batched_system,
        tf.reshape(rhs, (parameter_dim, state_dim * state_dim, 1)),
    )
    return tf.reshape(solution, (parameter_dim, state_dim, state_dim))


def lyapunov_residual_tf(
    transition_matrix: Any,
    stationary_covariance: Any,
    transition_covariance: Any,
) -> tf.Tensor:
    """Return the residual ``P - A P A' - Q``."""

    transition = tf.convert_to_tensor(transition_matrix, dtype=tf.float64)
    stationary = tf.convert_to_tensor(stationary_covariance, dtype=tf.float64)
    covariance = tf.convert_to_tensor(transition_covariance, dtype=tf.float64)
    return stationary - transition @ stationary @ tf.transpose(transition) - covariance


def lower_triangular_lgssm_observations_from_fixture(
    fixture: dict[str, Any] | None = None,
) -> tf.Tensor:
    """Return Phase 2 observations as a TensorFlow tensor."""

    data = load_lower_triangular_lgssm_synthetic_data() if fixture is None else fixture
    return tf.convert_to_tensor(data["observations"], dtype=tf.float64)


def gaussian_raw_prior_log_prob_and_score(
    raw_parameters: Any,
    contract: dict[str, Any],
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return the friendly Phase 1 Gaussian prior value and score."""

    raw = tf.convert_to_tensor(raw_parameters, dtype=tf.float64)
    center = raw_truth_from_contract(contract)
    scales = tf.constant(
        [0.50] * 4 + [0.60] * 6 + [0.35] * 4 + [0.35] * 4,
        dtype=tf.float64,
    )
    standardized = (raw - center) / scales
    value = -0.5 * tf.reduce_sum(tf.square(standardized))
    score = -(raw - center) / tf.square(scales)
    return value, score


def lower_triangular_lgssm_log_prob_and_score(
    raw_parameters: Any,
    observations: Any,
    contract: dict[str, Any] | None = None,
    *,
    jitter: float = 1.0e-9,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return log posterior, score, likelihood value, and likelihood score."""

    value, score, likelihood_value, likelihood_score, _status = (
        lower_triangular_lgssm_log_prob_score_status(
            raw_parameters,
            observations,
            contract,
            jitter=jitter,
        )
    )
    return value, score, likelihood_value, likelihood_score


def lower_triangular_lgssm_log_prob_score_status(
    raw_parameters: Any,
    observations: Any,
    contract: dict[str, Any] | None = None,
    *,
    jitter: float = 1.0e-9,
    singular_floor: float = 1.0e-12,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]:
    """Return posterior value/score plus SVD/eigh graph-status telemetry.

    The historical QR derivative backend is not HMC-admissible for this target:
    it builds a Python-unrolled derivative graph. Serious LGSSM HMC uses the
    SVD/eigh graph-status score authority and treats nonzero status as a hard
    gradient veto by emitting NaN value/score tensors.
    """

    manifest = load_lower_triangular_lgssm_contract() if contract is None else contract
    materialized = materialize_lower_triangular_lgssm_with_first_derivatives(
        raw_parameters,
        manifest,
    )
    if materialized.derivatives is None:
        raise ValueError("first derivatives are required for score evaluation")
    result = tf_svd_linear_gaussian_score_first_order_graph_status(
        tf.convert_to_tensor(observations, dtype=tf.float64),
        materialized.model,
        materialized.derivatives,
        jitter=tf.constant(jitter, dtype=tf.float64),
        singular_floor=tf.constant(singular_floor, dtype=tf.float64),
    )
    prior_value, prior_score = gaussian_raw_prior_log_prob_and_score(
        materialized.raw_parameters,
        manifest,
    )
    status = _svd_graph_status_payload(result)
    finite_prior = tf.logical_and(
        tf.math.is_finite(prior_value),
        tf.reduce_all(tf.math.is_finite(prior_score)),
    )
    finite_likelihood = tf.logical_and(
        tf.math.is_finite(result.log_likelihood),
        tf.reduce_all(tf.math.is_finite(result.score)),
    )
    status_valid = tf.logical_and(
        tf.equal(
            status["status_code"],
            tf.constant(SVD_LINEAR_SCORE_STATUS_VALID_PRE_REGULARIZED, dtype=tf.int32),
        ),
        status["valid_pre_regularized_score"],
    )
    hmc_gradient_valid = tf.logical_and(status_valid, tf.logical_and(finite_prior, finite_likelihood))
    nan = tf.constant(float("nan"), dtype=tf.float64)
    likelihood_value = tf.where(hmc_gradient_valid, result.log_likelihood, nan)
    likelihood_score = tf.where(
        hmc_gradient_valid,
        result.score,
        tf.fill(tf.shape(result.score), nan),
    )
    posterior_value = tf.where(hmc_gradient_valid, prior_value + likelihood_value, nan)
    posterior_score = tf.where(
        hmc_gradient_valid,
        prior_score + likelihood_score,
        tf.fill(tf.shape(prior_score), nan),
    )
    status["valid_pre_regularized_score"] = hmc_gradient_valid
    return posterior_value, posterior_score, likelihood_value, likelihood_score, status


def _svd_graph_status_payload(result: Any) -> dict[str, tf.Tensor]:
    extra = result.diagnostics.extra
    return {
        "status_code": tf.convert_to_tensor(extra["status_code"], dtype=tf.int32),
        "valid_pre_regularized_score": tf.convert_to_tensor(
            extra["valid_pre_regularized_score"],
            dtype=tf.bool,
        ),
        "floor_count_value": tf.convert_to_tensor(
            result.diagnostics.regularization.floor_count,
            dtype=tf.int32,
        ),
        "min_innovation_eigenvalue": tf.convert_to_tensor(
            extra["min_innovation_eigenvalue"],
            dtype=tf.float64,
        ),
        "innovation_condition_estimate": tf.convert_to_tensor(
            extra["innovation_condition_estimate"],
            dtype=tf.float64,
        ),
    }


def _kron_tf(left: tf.Tensor, right: tf.Tensor) -> tf.Tensor:
    product = tf.einsum("ab,cd->acbd", left, right)
    return tf.reshape(
        product,
        (
            int(left.shape[0]) * int(right.shape[0]),
            int(left.shape[1]) * int(right.shape[1]),
        ),
    )


def _symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + tf.transpose(matrix))
