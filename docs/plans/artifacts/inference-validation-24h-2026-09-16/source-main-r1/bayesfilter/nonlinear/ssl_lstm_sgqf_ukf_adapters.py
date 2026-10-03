"""SSL-LSTM analytic SGQF and SVD-UKF adapter helpers.

This module wires the Phase-1 Gaussian additive SSL-LSTM target into existing
analytic first-order filter score engines.  It deliberately stays narrow:
diagonal covariance only, no public package export, and no HMC or benchmark
claim.  The target score derivatives are hand-coded TensorFlow expressions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import tensorflow as tf

from bayesfilter.nonlinear.fixed_sgqf_derivatives_tf import (
    TFFixedSGQFDerivatives,
    TFFixedSGQFScoreResult,
    tf_fixed_sgqf_score,
)
from bayesfilter.nonlinear.fixed_sgqf_tf import (
    TFFixedSGQFBranchConfig,
    TFFixedSGQFCloud,
    TFFixedSGQFNonlinearModel,
    tf_fixed_sgqf_cloud,
)
from bayesfilter.nonlinear.ssl_lstm_protocol import (
    SSLLSTMAdapterProtocol,
    SSLLSTMStaticConfig,
    build_expected_ssl_lstm_adapter_protocol,
    validate_ssl_lstm_value_score_artifact,
)
from bayesfilter.nonlinear.svd_sigma_point_derivatives_tf import (
    TFStructuralFirstDerivatives,
    tf_svd_ukf_score,
)
from bayesfilter.results_tf import TFFilterDerivativeResult
from bayesfilter.structural import StatePartition, StructuralFilterConfig
from bayesfilter.structural_tf import TFStructuralStateSpace


GateName = Literal["input", "forget", "output", "candidate"]

_GATE_INDEX: Mapping[GateName, int] = {
    "input": 0,
    "forget": 1,
    "output": 2,
    "candidate": 3,
}


@dataclass(frozen=True)
class SSLLSTMParameterSlices:
    """Contiguous parameter block offsets matching ``SSLLSTMStaticConfig``."""

    lstm_input_start: int
    lstm_recurrent_start: int
    lstm_bias_start: int
    latent_weight_start: int
    latent_bias_start: int
    observation_weight_start: int
    observation_bias_start: int
    initial_mean_start: int
    initial_std_start: int
    process_std_start: int
    observation_std_start: int
    parameter_dim: int


@dataclass(frozen=True)
class SSLLSTMConstrainedParameters:
    """Constrained SSL-LSTM parameter blocks used by filter score engines."""

    config: SSLLSTMStaticConfig
    unconstrained: tf.Tensor
    slices: SSLLSTMParameterSlices
    lstm_input: tf.Tensor
    lstm_recurrent: tf.Tensor
    lstm_bias: tf.Tensor
    latent_weight: tf.Tensor
    latent_bias: tf.Tensor
    observation_weight: tf.Tensor
    observation_bias: tf.Tensor
    initial_mean: tf.Tensor
    initial_std: tf.Tensor
    process_std: tf.Tensor
    observation_std: tf.Tensor
    d_initial_mean: tf.Tensor
    d_initial_covariance: tf.Tensor
    d_sgqf_process_covariance: tf.Tensor
    d_ukf_innovation_covariance: tf.Tensor
    d_observation_covariance: tf.Tensor
    initial_covariance: tf.Tensor
    sgqf_process_covariance: tf.Tensor
    ukf_innovation_covariance: tf.Tensor
    observation_covariance: tf.Tensor
    std_floor: float


@dataclass(frozen=True)
class SSLLSTMSGQFAdapterComponents:
    """Fixed-SGQF model, derivatives, and fixed branch objects."""

    parameters: SSLLSTMConstrainedParameters
    model: TFFixedSGQFNonlinearModel
    derivatives: TFFixedSGQFDerivatives
    cloud: TFFixedSGQFCloud
    branch_config: TFFixedSGQFBranchConfig
    protocol: SSLLSTMAdapterProtocol


@dataclass(frozen=True)
class SSLLSTMUKFAdapterComponents:
    """SVD-UKF structural model, derivatives, and protocol metadata."""

    parameters: SSLLSTMConstrainedParameters
    model: TFStructuralStateSpace
    derivatives: TFStructuralFirstDerivatives
    protocol: SSLLSTMAdapterProtocol


def ssl_lstm_parameter_slices(config: SSLLSTMStaticConfig) -> SSLLSTMParameterSlices:
    """Return contiguous block starts for the Phase-1 parameter order."""

    k = int(config.latent_dim)
    h = int(config.hidden_dim)
    d = int(config.observation_dim)
    n = int(config.augmented_state_dim)
    cursor = 0
    lstm_input_start = cursor
    cursor += 4 * h * k
    lstm_recurrent_start = cursor
    cursor += 4 * h * h
    lstm_bias_start = cursor
    cursor += 4 * h
    latent_weight_start = cursor
    cursor += k * h
    latent_bias_start = cursor
    cursor += k
    observation_weight_start = cursor
    cursor += d * k
    observation_bias_start = cursor
    cursor += d
    initial_mean_start = cursor
    cursor += n
    initial_std_start = cursor
    cursor += n
    process_std_start = cursor
    cursor += k
    observation_std_start = cursor
    cursor += d
    if cursor != config.parameter_dim:
        raise ValueError("SSL-LSTM slice layout does not match config parameter names")
    return SSLLSTMParameterSlices(
        lstm_input_start=lstm_input_start,
        lstm_recurrent_start=lstm_recurrent_start,
        lstm_bias_start=lstm_bias_start,
        latent_weight_start=latent_weight_start,
        latent_bias_start=latent_bias_start,
        observation_weight_start=observation_weight_start,
        observation_bias_start=observation_bias_start,
        initial_mean_start=initial_mean_start,
        initial_std_start=initial_std_start,
        process_std_start=process_std_start,
        observation_std_start=observation_std_start,
        parameter_dim=cursor,
    )


def unpack_ssl_lstm_parameters(
    theta: tf.Tensor,
    config: SSLLSTMStaticConfig,
    *,
    std_floor: float = 1.0e-4,
    derivative_parameter_indices: tuple[int, ...] | None = None,
) -> SSLLSTMConstrainedParameters:
    """Map an unconstrained vector to SSL-LSTM blocks and covariance scores."""

    values = tf.convert_to_tensor(theta, dtype=tf.float64)
    if values.shape.rank != 1:
        raise ValueError("SSL-LSTM theta must be a rank-one vector")
    if values.shape[0] is not None and int(values.shape[0]) != config.parameter_dim:
        raise ValueError("SSL-LSTM theta length does not match static config")
    k = int(config.latent_dim)
    h = int(config.hidden_dim)
    d = int(config.observation_dim)
    n = int(config.augmented_state_dim)
    slices = ssl_lstm_parameter_slices(config)

    def take(start: int, size: int) -> tf.Tensor:
        return values[start : start + size]

    lstm_input = tf.reshape(take(slices.lstm_input_start, 4 * h * k), [4, h, k])
    lstm_recurrent = tf.reshape(take(slices.lstm_recurrent_start, 4 * h * h), [4, h, h])
    lstm_bias = tf.reshape(take(slices.lstm_bias_start, 4 * h), [4, h])
    latent_weight = tf.reshape(take(slices.latent_weight_start, k * h), [k, h])
    latent_bias = take(slices.latent_bias_start, k)
    observation_weight = tf.reshape(take(slices.observation_weight_start, d * k), [d, k])
    observation_bias = take(slices.observation_bias_start, d)
    initial_mean = take(slices.initial_mean_start, n)
    raw_initial_std = take(slices.initial_std_start, n)
    raw_process_std = take(slices.process_std_start, k)
    raw_observation_std = take(slices.observation_std_start, d)

    floor = tf.constant(float(std_floor), dtype=tf.float64)
    initial_std = tf.nn.softplus(raw_initial_std) + floor
    process_std = tf.nn.softplus(raw_process_std) + floor
    observation_std = tf.nn.softplus(raw_observation_std) + floor
    d_initial_variance = 2.0 * initial_std * tf.math.sigmoid(raw_initial_std)
    d_process_variance = 2.0 * process_std * tf.math.sigmoid(raw_process_std)
    d_observation_variance = 2.0 * observation_std * tf.math.sigmoid(raw_observation_std)

    full_parameter_dim = int(config.parameter_dim)
    selected = (
        tuple(range(full_parameter_dim))
        if derivative_parameter_indices is None
        else tuple(int(index) for index in derivative_parameter_indices)
    )
    if len(set(selected)) != len(selected) or any(
        index < 0 or index >= full_parameter_dim for index in selected
    ):
        raise ValueError("derivative_parameter_indices must be unique full-chart indices")
    parameter_dim = len(selected)
    local_index = {index: row for row, index in enumerate(selected)}
    d_initial_mean = _scatter_rank2(
        [parameter_dim, n],
        [
            (local_index[slices.initial_mean_start + row], row)
            for row in range(n)
            if slices.initial_mean_start + row in local_index
        ],
        tf.ones(
            [sum(slices.initial_mean_start + row in local_index for row in range(n))],
            dtype=tf.float64,
        ),
    )
    d_initial_covariance = _scatter_rank3(
        [parameter_dim, n, n],
        [
            (local_index[slices.initial_std_start + row], row, row)
            for row in range(n)
            if slices.initial_std_start + row in local_index
        ],
        tf.stack(
            [d_initial_variance[row] for row in range(n) if slices.initial_std_start + row in local_index]
        )
        if any(slices.initial_std_start + row in local_index for row in range(n))
        else tf.zeros([0], dtype=tf.float64),
    )
    d_sgqf_process_covariance = _scatter_rank3(
        [parameter_dim, n, n],
        [
            (local_index[slices.process_std_start + row], row, row)
            for row in range(k)
            if slices.process_std_start + row in local_index
        ],
        tf.stack(
            [d_process_variance[row] for row in range(k) if slices.process_std_start + row in local_index]
        )
        if any(slices.process_std_start + row in local_index for row in range(k))
        else tf.zeros([0], dtype=tf.float64),
    )
    d_ukf_innovation_covariance = _scatter_rank3(
        [parameter_dim, k, k],
        [
            (local_index[slices.process_std_start + row], row, row)
            for row in range(k)
            if slices.process_std_start + row in local_index
        ],
        tf.stack(
            [d_process_variance[row] for row in range(k) if slices.process_std_start + row in local_index]
        )
        if any(slices.process_std_start + row in local_index for row in range(k))
        else tf.zeros([0], dtype=tf.float64),
    )
    d_observation_covariance = _scatter_rank3(
        [parameter_dim, d, d],
        [
            (local_index[slices.observation_std_start + row], row, row)
            for row in range(d)
            if slices.observation_std_start + row in local_index
        ],
        tf.stack(
            [d_observation_variance[row] for row in range(d) if slices.observation_std_start + row in local_index]
        )
        if any(slices.observation_std_start + row in local_index for row in range(d))
        else tf.zeros([0], dtype=tf.float64),
    )
    sgqf_process_variance = tf.concat(
        [
            tf.square(process_std),
            tf.zeros([2 * h], dtype=tf.float64),
        ],
        axis=0,
    )
    return SSLLSTMConstrainedParameters(
        config=config,
        unconstrained=values,
        slices=slices,
        lstm_input=lstm_input,
        lstm_recurrent=lstm_recurrent,
        lstm_bias=lstm_bias,
        latent_weight=latent_weight,
        latent_bias=latent_bias,
        observation_weight=observation_weight,
        observation_bias=observation_bias,
        initial_mean=initial_mean,
        initial_std=initial_std,
        process_std=process_std,
        observation_std=observation_std,
        d_initial_mean=d_initial_mean,
        d_initial_covariance=d_initial_covariance,
        d_sgqf_process_covariance=d_sgqf_process_covariance,
        d_ukf_innovation_covariance=d_ukf_innovation_covariance,
        d_observation_covariance=d_observation_covariance,
        initial_covariance=tf.linalg.diag(tf.square(initial_std)),
        sgqf_process_covariance=tf.linalg.diag(sgqf_process_variance),
        ukf_innovation_covariance=tf.linalg.diag(tf.square(process_std)),
        observation_covariance=tf.linalg.diag(tf.square(observation_std)),
        std_floor=float(std_floor),
    )


def make_ssl_lstm_fixed_sgqf_components(
    theta: tf.Tensor,
    config: SSLLSTMStaticConfig,
    *,
    evidence_path: str,
    sparse_level: int = 2,
    std_floor: float = 1.0e-4,
    branch_config: TFFixedSGQFBranchConfig | None = None,
) -> SSLLSTMSGQFAdapterComponents:
    """Build Fixed-SGQF SSL-LSTM components with analytic derivatives."""

    params = unpack_ssl_lstm_parameters(theta, config, std_floor=std_floor)
    model = TFFixedSGQFNonlinearModel(
        initial_mean=params.initial_mean,
        initial_covariance=params.initial_covariance,
        process_covariance=params.sgqf_process_covariance,
        observation_covariance=params.observation_covariance,
        transition_fn=lambda points: ssl_lstm_transition(params, points),
        observation_fn=lambda points: ssl_lstm_observation(params, points),
        name="ssl_lstm_fixed_sgqf_model",
    )
    derivatives = TFFixedSGQFDerivatives(
        d_initial_mean=params.d_initial_mean,
        d_initial_covariance=params.d_initial_covariance,
        d_process_covariance=params.d_sgqf_process_covariance,
        d_observation_covariance=params.d_observation_covariance,
        transition_state_jacobian_fn=lambda points: ssl_lstm_transition_state_jacobian(params, points),
        d_transition_fn=lambda points: ssl_lstm_transition_parameter_derivative(params, points),
        observation_state_jacobian_fn=lambda points: ssl_lstm_observation_state_jacobian(params, points),
        d_observation_fn=lambda points: ssl_lstm_observation_parameter_derivative(params, points),
        name="ssl_lstm_fixed_sgqf_hand_derivatives",
        transition_jvp_fn=lambda points, tangents: ssl_lstm_transition_state_jvp(
            params,
            points,
            tangents,
        ),
        observation_jvp_fn=lambda points, tangents: ssl_lstm_observation_state_jvp(
            params,
            points,
            tangents,
        ),
    )
    cloud = tf_fixed_sgqf_cloud(dim=config.augmented_state_dim, sparse_level=sparse_level)
    protocol = build_expected_ssl_lstm_adapter_protocol(
        config,
        filter_name="fixed_sgqf",
        evidence_path=evidence_path,
        target_scope="ssl_lstm_filter_hmc:fixed_sgqf:phase3",
        nonclaims=(
            "Phase 3 adapter-admission evidence only",
            "no SGQF sufficiency claim",
            "no HMC convergence claim",
            "no SSL-LSTM estimation success claim",
        ),
    )
    return SSLLSTMSGQFAdapterComponents(
        parameters=params,
        model=model,
        derivatives=derivatives,
        cloud=cloud,
        branch_config=branch_config or TFFixedSGQFBranchConfig(),
        protocol=protocol,
    )


def make_ssl_lstm_svd_ukf_components(
    theta: tf.Tensor,
    config: SSLLSTMStaticConfig,
    *,
    evidence_path: str,
    std_floor: float = 1.0e-4,
    derivative_parameter_indices: tuple[int, ...] | None = None,
) -> SSLLSTMUKFAdapterComponents:
    """Build structural SVD-UKF SSL-LSTM components with analytic derivatives."""

    if derivative_parameter_indices is not None:
        slices = ssl_lstm_parameter_slices(config)
        supported = frozenset(
            (
                slices.latent_weight_start,
                slices.latent_bias_start,
                slices.observation_weight_start,
                slices.observation_bias_start,
            )
        )
        unsupported = sorted(set(derivative_parameter_indices) - supported)
        if unsupported:
            raise ValueError(
                "selected SSL-LSTM score currently supports only the four "
                "state-complexity target coordinates; unsupported indices: "
                + ", ".join(str(index) for index in unsupported)
            )
    params = unpack_ssl_lstm_parameters(
        theta,
        config,
        std_floor=std_floor,
        derivative_parameter_indices=derivative_parameter_indices,
    )
    k = int(config.latent_dim)
    h = int(config.hidden_dim)
    n = int(config.augmented_state_dim)
    partition = StatePartition(
        state_names=tuple(
            [f"z.{idx}" for idx in range(k)]
            + [f"a.{idx}" for idx in range(h)]
            + [f"c.{idx}" for idx in range(h)]
        ),
        stochastic_indices=tuple(range(k)),
        deterministic_indices=tuple(range(k, n)),
        innovation_dim=k,
    )

    def transition(previous_state: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        deterministic = ssl_lstm_transition(params, previous_state)
        eps = _as_points(innovation)
        return tf.concat(
            [
                deterministic[:, :k] + eps,
                deterministic[:, k:],
            ],
            axis=1,
        )

    def deterministic_residual(
        previous_state: tf.Tensor,
        innovation: tf.Tensor,
        next_state: tf.Tensor,
    ) -> tf.Tensor:
        del innovation
        expected = ssl_lstm_transition(params, previous_state)
        return _as_points(next_state)[:, k:] - expected[:, k:]

    model = TFStructuralStateSpace(
        partition=partition,
        config=StructuralFilterConfig(
            integration_space="innovation",
            deterministic_completion="required",
        ),
        initial_mean=params.initial_mean,
        initial_covariance=params.initial_covariance,
        innovation_covariance=params.ukf_innovation_covariance,
        observation_covariance=params.observation_covariance,
        transition_fn=transition,
        observation_fn=lambda points: ssl_lstm_observation(params, points),
        deterministic_residual_fn=deterministic_residual,
        name="ssl_lstm_svd_ukf_structural_model",
    )

    def transition_innovation_jacobian(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        del innovation
        point_count = tf.shape(_as_points(previous))[0]
        top = tf.eye(k, dtype=tf.float64)
        bottom = tf.zeros([2 * h, k], dtype=tf.float64)
        matrix = tf.concat([top, bottom], axis=0)
        return tf.broadcast_to(matrix[tf.newaxis, :, :], [point_count, n, k])

    d_transition_fn = (
        (lambda previous, innovation: ssl_lstm_transition_parameter_derivative(params, previous))
        if derivative_parameter_indices is None
        else (
            lambda previous, innovation: ssl_lstm_transition_parameter_derivative_selected(
                params, previous, derivative_parameter_indices
            )
        )
    )
    d_observation_fn = (
        ssl_lstm_observation_parameter_derivative
        if derivative_parameter_indices is None
        else (
            lambda selected_params, points: ssl_lstm_observation_parameter_derivative_selected(
                selected_params, points, derivative_parameter_indices
            )
        )
    )
    derivatives = TFStructuralFirstDerivatives(
        d_initial_mean=params.d_initial_mean,
        d_initial_covariance=params.d_initial_covariance,
        d_innovation_covariance=params.d_ukf_innovation_covariance,
        d_observation_covariance=params.d_observation_covariance,
        transition_state_jacobian_fn=lambda previous, innovation: ssl_lstm_transition_state_jacobian(params, previous),
        transition_innovation_jacobian_fn=transition_innovation_jacobian,
        d_transition_fn=d_transition_fn,
        observation_state_jacobian_fn=lambda points: ssl_lstm_observation_state_jacobian(params, points),
        d_observation_fn=lambda points: d_observation_fn(params, points),
        name="ssl_lstm_svd_ukf_hand_derivatives",
        transition_jvp_fn=lambda previous, innovation, state_tangents, innovation_tangents: (
            ssl_lstm_transition_state_jvp(params, previous, state_tangents)
            + tf.concat(
                [
                    innovation_tangents,
                    tf.zeros(
                        [
                            tf.shape(innovation_tangents)[0],
                            tf.shape(innovation_tangents)[1],
                            2 * h,
                        ],
                        dtype=tf.float64,
                    ),
                ],
                axis=2,
            )
        ),
        observation_jvp_fn=lambda points, tangents: ssl_lstm_observation_state_jvp(
            params,
            points,
            tangents,
        ),
    )
    protocol = build_expected_ssl_lstm_adapter_protocol(
        config,
        filter_name="svd_ukf",
        evidence_path=evidence_path,
        target_scope="ssl_lstm_filter_hmc:svd_ukf:phase3",
        nonclaims=(
            "Phase 3 adapter-admission evidence only",
            "no UKF sufficiency claim",
            "no HMC convergence claim",
            "no SSL-LSTM estimation success claim",
        ),
    )
    return SSLLSTMUKFAdapterComponents(
        parameters=params,
        model=model,
        derivatives=derivatives,
        protocol=protocol,
    )


def tf_ssl_lstm_fixed_sgqf_score(
    observations: tf.Tensor,
    theta: tf.Tensor,
    config: SSLLSTMStaticConfig,
    *,
    evidence_path: str,
    sparse_level: int = 2,
    std_floor: float = 1.0e-4,
) -> tuple[TFFixedSGQFScoreResult, SSLLSTMSGQFAdapterComponents]:
    """Evaluate the SSL-LSTM Fixed-SGQF analytic score."""

    components = make_ssl_lstm_fixed_sgqf_components(
        theta,
        config,
        evidence_path=evidence_path,
        sparse_level=sparse_level,
        std_floor=std_floor,
    )
    result = tf_fixed_sgqf_score(
        observations,
        components.model,
        components.derivatives,
        cloud=components.cloud,
        branch_config=components.branch_config,
    )
    return result, components


def tf_ssl_lstm_svd_ukf_score(
    observations: tf.Tensor,
    theta: tf.Tensor,
    config: SSLLSTMStaticConfig,
    *,
    evidence_path: str,
    std_floor: float = 1.0e-4,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
    spectral_gap_tolerance: tf.Tensor | float = 1.0e-8,
    derivative_parameter_indices: tuple[int, ...] | None = None,
) -> tuple[TFFilterDerivativeResult, SSLLSTMUKFAdapterComponents]:
    """Evaluate the SSL-LSTM SVD-UKF analytic score."""

    components = make_ssl_lstm_svd_ukf_components(
        theta,
        config,
        evidence_path=evidence_path,
        std_floor=std_floor,
        derivative_parameter_indices=derivative_parameter_indices,
    )
    result = tf_svd_ukf_score(
        observations,
        components.model,
        components.derivatives,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        innovation_floor=tf.constant(1.0e-12, dtype=tf.float64),
        spectral_gap_tolerance=spectral_gap_tolerance,
    )
    return result, components


def ssl_lstm_transition(
    params: SSLLSTMConstrainedParameters,
    points: tf.Tensor,
) -> tf.Tensor:
    """Evaluate the deterministic SSL-LSTM transition mean."""

    z_prev, a_prev, c_prev = _split_state(params, points)
    gates = _gate_values(params, z_prev, a_prev, c_prev)
    c_next = gates["forget"] * c_prev + gates["input"] * gates["candidate"]
    a_next = gates["output"] * tf.math.tanh(c_next)
    z_mean = (
        a_next @ tf.transpose(params.latent_weight)
        + params.latent_bias[tf.newaxis, :]
    )
    return tf.concat([z_mean, a_next, c_next], axis=1)


def ssl_lstm_observation(
    params: SSLLSTMConstrainedParameters,
    points: tf.Tensor,
) -> tf.Tensor:
    """Evaluate the linear SSL-LSTM emission mean."""

    values = _as_points(points)
    z = values[:, : params.config.latent_dim]
    return (
        z @ tf.transpose(params.observation_weight)
        + params.observation_bias[tf.newaxis, :]
    )


def ssl_lstm_transition_state_jacobian(
    params: SSLLSTMConstrainedParameters,
    points: tf.Tensor,
) -> tf.Tensor:
    """Return ``d transition_mean / d previous_state`` for each point."""

    z_prev, a_prev, c_prev = _split_state(params, points)
    del z_prev
    gates = _gate_values(params, _as_points(points)[:, : params.config.latent_dim], a_prev, c_prev)
    derivatives = _gate_activation_derivatives(gates)
    w = params.lstm_input
    u = params.lstm_recurrent
    k = int(params.config.latent_dim)
    h = int(params.config.hidden_dim)

    coeff_i = gates["candidate"] * derivatives["input"]
    coeff_f = c_prev * derivatives["forget"]
    coeff_r = gates["input"] * derivatives["candidate"]
    coeff_o = tf.math.tanh(gates["cell"]) * derivatives["output"]
    coeff_c_to_a = gates["output"] * (1.0 - tf.square(tf.math.tanh(gates["cell"])))

    dc_dz = (
        coeff_i[:, :, tf.newaxis] * w[_GATE_INDEX["input"]][tf.newaxis, :, :]
        + coeff_f[:, :, tf.newaxis] * w[_GATE_INDEX["forget"]][tf.newaxis, :, :]
        + coeff_r[:, :, tf.newaxis] * w[_GATE_INDEX["candidate"]][tf.newaxis, :, :]
    )
    dc_da = (
        coeff_i[:, :, tf.newaxis] * u[_GATE_INDEX["input"]][tf.newaxis, :, :]
        + coeff_f[:, :, tf.newaxis] * u[_GATE_INDEX["forget"]][tf.newaxis, :, :]
        + coeff_r[:, :, tf.newaxis] * u[_GATE_INDEX["candidate"]][tf.newaxis, :, :]
    )
    dc_dc = tf.linalg.diag(gates["forget"])

    da_dz = (
        coeff_o[:, :, tf.newaxis] * w[_GATE_INDEX["output"]][tf.newaxis, :, :]
        + coeff_c_to_a[:, :, tf.newaxis] * dc_dz
    )
    da_da = (
        coeff_o[:, :, tf.newaxis] * u[_GATE_INDEX["output"]][tf.newaxis, :, :]
        + coeff_c_to_a[:, :, tf.newaxis] * dc_da
    )
    da_dc = coeff_c_to_a[:, :, tf.newaxis] * dc_dc

    dz_dz = tf.einsum("kh,rhj->rkj", params.latent_weight, da_dz)
    dz_da = tf.einsum("kh,rhj->rkj", params.latent_weight, da_da)
    dz_dc = tf.einsum("kh,rhj->rkj", params.latent_weight, da_dc)

    z_rows = tf.concat([dz_dz, dz_da, dz_dc], axis=2)
    a_rows = tf.concat([da_dz, da_da, da_dc], axis=2)
    c_rows = tf.concat([dc_dz, dc_da, dc_dc], axis=2)
    jacobian = tf.concat([z_rows, a_rows, c_rows], axis=1)
    expected_cols = k + 2 * h
    jacobian.set_shape([None, expected_cols, expected_cols])
    return jacobian


def ssl_lstm_transition_state_jvp(
    params: SSLLSTMConstrainedParameters,
    points: tf.Tensor,
    point_tangents: tf.Tensor,
) -> tf.Tensor:
    """Apply the transition state derivative without forming its Jacobian.

    ``point_tangents`` has layout ``[direction, point, state]`` and the return
    value has the same leading axes.
    """

    values = _as_points(points)
    tangents = tf.convert_to_tensor(point_tangents, dtype=tf.float64)
    if tangents.shape.rank != 3:
        raise ValueError("SSL-LSTM point tangents must be rank three")
    k = int(params.config.latent_dim)
    h = int(params.config.hidden_dim)
    n = int(params.config.augmented_state_dim)
    if tangents.shape[-1] is not None and int(tangents.shape[-1]) != n:
        raise ValueError("SSL-LSTM point tangent state dimension mismatch")

    z_prev, a_prev, c_prev = _split_state(params, values)
    d_z_prev = tangents[:, :, :k]
    d_a_prev = tangents[:, :, k : k + h]
    d_c_prev = tangents[:, :, k + h :]
    gates = _gate_values(params, z_prev, a_prev, c_prev)
    activation_derivatives = _gate_activation_derivatives(gates)
    d_preactivation = (
        tf.einsum("ghk,prk->prgh", params.lstm_input, d_z_prev)
        + tf.einsum("ghj,prj->prgh", params.lstm_recurrent, d_a_prev)
    )
    d_input = (
        activation_derivatives["input"][tf.newaxis, :, :]
        * d_preactivation[:, :, _GATE_INDEX["input"], :]
    )
    d_forget = (
        activation_derivatives["forget"][tf.newaxis, :, :]
        * d_preactivation[:, :, _GATE_INDEX["forget"], :]
    )
    d_output = (
        activation_derivatives["output"][tf.newaxis, :, :]
        * d_preactivation[:, :, _GATE_INDEX["output"], :]
    )
    d_candidate = (
        activation_derivatives["candidate"][tf.newaxis, :, :]
        * d_preactivation[:, :, _GATE_INDEX["candidate"], :]
    )
    d_cell = (
        d_forget * c_prev[tf.newaxis, :, :]
        + gates["forget"][tf.newaxis, :, :] * d_c_prev
        + d_input * gates["candidate"][tf.newaxis, :, :]
        + gates["input"][tf.newaxis, :, :] * d_candidate
    )
    tanh_cell = tf.math.tanh(gates["cell"])
    d_hidden = (
        d_output * tanh_cell[tf.newaxis, :, :]
        + gates["output"][tf.newaxis, :, :]
        * (1.0 - tf.square(tanh_cell))[tf.newaxis, :, :]
        * d_cell
    )
    d_latent = tf.einsum("kh,prh->prk", params.latent_weight, d_hidden)
    result = tf.concat([d_latent, d_hidden, d_cell], axis=2)
    result.set_shape([tangents.shape[0], tangents.shape[1], n])
    return result


def ssl_lstm_transition_parameter_derivative(
    params: SSLLSTMConstrainedParameters,
    points: tf.Tensor,
) -> tf.Tensor:
    """Return direct ``d transition_mean / d theta`` with points held fixed."""

    values = _as_points(points)
    z_prev, a_prev, c_prev = _split_state(params, values)
    gates = _gate_values(params, z_prev, a_prev, c_prev)
    derivatives = _gate_activation_derivatives(gates)
    k = int(params.config.latent_dim)
    h = int(params.config.hidden_dim)
    n = int(params.config.augmented_state_dim)
    one_h = tf.eye(h, dtype=tf.float64)
    one_k = tf.eye(k, dtype=tf.float64)
    zeros = tf.zeros([tf.shape(values)[0], n], dtype=tf.float64)
    pieces: list[tf.Tensor] = []

    def from_gate(gate: GateName, row: int, values_for_pre: tf.Tensor) -> tf.Tensor:
        direct_pre = tf.convert_to_tensor(values_for_pre, dtype=tf.float64)
        if gate == "input":
            dc_row = gates["candidate"][:, row] * derivatives["input"][:, row] * direct_pre
            da_row = _cell_to_hidden_coeff(gates)[:, row] * dc_row
        elif gate == "forget":
            dc_row = c_prev[:, row] * derivatives["forget"][:, row] * direct_pre
            da_row = _cell_to_hidden_coeff(gates)[:, row] * dc_row
        elif gate == "candidate":
            dc_row = gates["input"][:, row] * derivatives["candidate"][:, row] * direct_pre
            da_row = _cell_to_hidden_coeff(gates)[:, row] * dc_row
        elif gate == "output":
            dc_row = tf.zeros_like(direct_pre)
            da_row = tf.math.tanh(gates["cell"])[:, row] * derivatives["output"][:, row] * direct_pre
        else:
            raise ValueError(f"unknown gate: {gate}")
        dc = dc_row[:, tf.newaxis] * one_h[row][tf.newaxis, :]
        da = da_row[:, tf.newaxis] * one_h[row][tf.newaxis, :]
        dz = da_row[:, tf.newaxis] * params.latent_weight[:, row][tf.newaxis, :]
        return tf.concat([dz, da, dc], axis=1)

    for gate in ("input", "forget", "output", "candidate"):
        for row in range(h):
            for col in range(k):
                pieces.append(from_gate(gate, row, z_prev[:, col]))
    for gate in ("input", "forget", "output", "candidate"):
        for row in range(h):
            for col in range(h):
                pieces.append(from_gate(gate, row, a_prev[:, col]))
    for gate in ("input", "forget", "output", "candidate"):
        for row in range(h):
            pieces.append(from_gate(gate, row, tf.ones([tf.shape(values)[0]], dtype=tf.float64)))
    transition = ssl_lstm_transition(params, values)
    a_next = transition[:, k : k + h]
    for row in range(k):
        for col in range(h):
            dz = a_next[:, col, tf.newaxis] * one_k[row][tf.newaxis, :]
            pieces.append(tf.concat([dz, tf.zeros([tf.shape(values)[0], 2 * h], dtype=tf.float64)], axis=1))
    for row in range(k):
        dz = tf.ones([tf.shape(values)[0], 1], dtype=tf.float64) * one_k[row][tf.newaxis, :]
        pieces.append(tf.concat([dz, tf.zeros([tf.shape(values)[0], 2 * h], dtype=tf.float64)], axis=1))
    for _row in range(params.config.observation_dim):
        for _col in range(k):
            pieces.append(zeros)
    for _row in range(params.config.observation_dim):
        pieces.append(zeros)
    for _row in range(n):
        pieces.append(zeros)
    for _row in range(n):
        pieces.append(zeros)
    for _row in range(k):
        pieces.append(zeros)
    for _row in range(params.config.observation_dim):
        pieces.append(zeros)
    stacked = tf.stack(pieces, axis=0)
    stacked.set_shape([params.config.parameter_dim, None, n])
    return stacked


def ssl_lstm_transition_parameter_derivative_selected(
    params: SSLLSTMConstrainedParameters,
    points: tf.Tensor,
    parameter_indices: tuple[int, ...],
) -> tf.Tensor:
    """Return direct transition derivatives for a selected parameter subset.

    This avoids materialising the full ``parameter_dim`` derivative tensor for
    targets that deliberately estimate only a small free-coordinate block.
    The selected indices retain the source chart's order.
    """

    values = _as_points(points)
    z_prev, _a_prev, _c_prev = _split_state(params, values)
    transition = ssl_lstm_transition(params, values)
    k = int(params.config.latent_dim)
    h = int(params.config.hidden_dim)
    n = int(params.config.augmented_state_dim)
    slices = params.slices
    zeros = tf.zeros([tf.shape(values)[0], n], dtype=tf.float64)
    one_k = tf.eye(k, dtype=tf.float64)
    rows: list[tf.Tensor] = []
    for index in parameter_indices:
        index = int(index)
        if slices.latent_weight_start <= index < slices.latent_weight_start + k * h:
            row, col = divmod(index - slices.latent_weight_start, h)
            derivative = transition[:, k + col, tf.newaxis] * one_k[row][tf.newaxis, :]
            rows.append(tf.concat([derivative, tf.zeros([tf.shape(values)[0], 2 * h], tf.float64)], axis=1))
        elif slices.latent_bias_start <= index < slices.latent_bias_start + k:
            row = index - slices.latent_bias_start
            derivative = tf.broadcast_to(one_k[row][tf.newaxis, :], [tf.shape(values)[0], k])
            rows.append(tf.concat([derivative, tf.zeros([tf.shape(values)[0], 2 * h], tf.float64)], axis=1))
        else:
            rows.append(zeros)
    if not rows:
        return tf.zeros([0, tf.shape(values)[0], n], dtype=tf.float64)
    result = tf.stack(rows, axis=0)
    result.set_shape([len(parameter_indices), None, n])
    return result


def ssl_lstm_observation_state_jacobian(
    params: SSLLSTMConstrainedParameters,
    points: tf.Tensor,
) -> tf.Tensor:
    """Return ``d observation_mean / d state`` for each point."""

    values = _as_points(points)
    k = int(params.config.latent_dim)
    h = int(params.config.hidden_dim)
    point_count = tf.shape(values)[0]
    matrix = tf.concat(
        [
            params.observation_weight,
            tf.zeros([params.config.observation_dim, 2 * h], dtype=tf.float64),
        ],
        axis=1,
    )
    return tf.broadcast_to(
        matrix[tf.newaxis, :, :],
        [point_count, params.config.observation_dim, k + 2 * h],
    )


def ssl_lstm_observation_state_jvp(
    params: SSLLSTMConstrainedParameters,
    points: tf.Tensor,
    point_tangents: tf.Tensor,
) -> tf.Tensor:
    """Apply the linear observation state derivative matrix-free."""

    values = _as_points(points)
    tangents = tf.convert_to_tensor(point_tangents, dtype=tf.float64)
    if tangents.shape.rank != 3:
        raise ValueError("SSL-LSTM point tangents must be rank three")
    if (
        tangents.shape[1] is not None
        and values.shape[0] is not None
        and int(tangents.shape[1]) != int(values.shape[0])
    ):
        raise ValueError("SSL-LSTM point tangent point count mismatch")
    k = int(params.config.latent_dim)
    return tf.einsum(
        "dk,prk->prd",
        params.observation_weight,
        tangents[:, :, :k],
    )


def ssl_lstm_observation_parameter_derivative(
    params: SSLLSTMConstrainedParameters,
    points: tf.Tensor,
) -> tf.Tensor:
    """Return direct ``d observation_mean / d theta`` with points held fixed."""

    values = _as_points(points)
    z = values[:, : params.config.latent_dim]
    d = int(params.config.observation_dim)
    k = int(params.config.latent_dim)
    pieces: list[tf.Tensor] = []
    zeros = tf.zeros([tf.shape(values)[0], d], dtype=tf.float64)
    one_d = tf.eye(d, dtype=tf.float64)
    prefix_count = (
        4 * params.config.hidden_dim * k
        + 4 * params.config.hidden_dim * params.config.hidden_dim
        + 4 * params.config.hidden_dim
        + k * params.config.hidden_dim
        + k
    )
    for _ in range(prefix_count):
        pieces.append(zeros)
    for row in range(d):
        for col in range(k):
            pieces.append(z[:, col, tf.newaxis] * one_d[row][tf.newaxis, :])
    for row in range(d):
        pieces.append(tf.ones([tf.shape(values)[0], 1], dtype=tf.float64) * one_d[row][tf.newaxis, :])
    tail_count = (
        params.config.augmented_state_dim
        + params.config.augmented_state_dim
        + params.config.latent_dim
        + params.config.observation_dim
    )
    for _ in range(tail_count):
        pieces.append(zeros)
    stacked = tf.stack(pieces, axis=0)
    stacked.set_shape([params.config.parameter_dim, None, d])
    return stacked


def ssl_lstm_observation_parameter_derivative_selected(
    params: SSLLSTMConstrainedParameters,
    points: tf.Tensor,
    parameter_indices: tuple[int, ...],
) -> tf.Tensor:
    """Return observation derivatives for a selected parameter subset only."""

    values = _as_points(points)
    z = values[:, : params.config.latent_dim]
    d = int(params.config.observation_dim)
    k = int(params.config.latent_dim)
    slices = params.slices
    zeros = tf.zeros([tf.shape(values)[0], d], dtype=tf.float64)
    one_d = tf.eye(d, dtype=tf.float64)
    rows: list[tf.Tensor] = []
    for index in parameter_indices:
        index = int(index)
        if slices.observation_weight_start <= index < slices.observation_weight_start + d * k:
            row, col = divmod(index - slices.observation_weight_start, k)
            rows.append(z[:, col, tf.newaxis] * one_d[row][tf.newaxis, :])
        elif slices.observation_bias_start <= index < slices.observation_bias_start + d:
            row = index - slices.observation_bias_start
            rows.append(tf.ones([tf.shape(values)[0], 1], tf.float64) * one_d[row][tf.newaxis, :])
        else:
            rows.append(zeros)
    if not rows:
        return tf.zeros([0, tf.shape(values)[0], d], dtype=tf.float64)
    result = tf.stack(rows, axis=0)
    result.set_shape([len(parameter_indices), None, d])
    return result


def build_ssl_lstm_debug_value_score_artifact(
    *,
    protocol: SSLLSTMAdapterProtocol,
    log_likelihood: tf.Tensor,
    score: tf.Tensor,
    finite_difference_max_abs_error: float,
    artifact_role: str = "debug_reference",
    target_scope: str | None = None,
    compile_mode: str = "eager",
    jit_compile: bool = False,
    device: str = "CPU-hidden debug",
    tf32_enabled: bool = False,
    nonclaims: tuple[str, ...] | None = None,
) -> Mapping[str, object]:
    """Build a Phase-3 debug/reference artifact and validate its schema."""

    score_tensor = tf.convert_to_tensor(score, dtype=tf.float64)
    value_tensor = tf.convert_to_tensor(log_likelihood, dtype=tf.float64)
    score_finite = bool(
        bool(tf.reduce_all(tf.math.is_finite(score_tensor)))
        and bool(tf.math.is_finite(value_tensor))
    )
    artifact = {
        "schema_version": protocol.artifact_schema_version,
        "artifact_role": artifact_role,
        "target_scope": target_scope or protocol.contract.value_score.target_scope,
        "filter_name": protocol.filter_name,
        "gradient_path": protocol.gradient_path,
        "value_score_authority": protocol.contract.value_score.value_score_authority,
        "compile_mode": compile_mode,
        "jit_compile": bool(jit_compile),
        "device": device,
        "tf32_enabled": bool(tf32_enabled),
        "seed_policy": protocol.contract.seed_policy,
        "branch_or_randomness_policy": protocol.branch_or_randomness_policy,
        "log_likelihood": float(value_tensor),
        "score": [float(item) for item in tf.reshape(score_tensor, [-1])],
        "score_finite": score_finite,
        "finite_difference_check": {
            "max_abs_error": float(finite_difference_max_abs_error),
            "role": "promotion_veto_for_adapter_admission",
        },
        "diagnostic_roles": {
            "score_finite": "promotion_veto",
            "finite_difference_check": "promotion_veto_for_adapter_admission",
            "runtime": "explanatory",
            "score_norm": "explanatory",
        },
        "nonclaims": nonclaims
        or (
            "debug/reference Phase 3 adapter artifact",
            "not production GPU/XLA evidence",
            "no SGQF or UKF sufficiency claim",
            "no HMC convergence claim",
        ),
    }
    return validate_ssl_lstm_value_score_artifact(artifact, protocol=protocol)


def _gate_values(
    params: SSLLSTMConstrainedParameters,
    z_prev: tf.Tensor,
    a_prev: tf.Tensor,
    c_prev: tf.Tensor,
) -> Mapping[str, tf.Tensor]:
    pre = (
        tf.einsum("ghk,rk->rgh", params.lstm_input, z_prev)
        + tf.einsum("ghl,rl->rgh", params.lstm_recurrent, a_prev)
        + params.lstm_bias[tf.newaxis, :, :]
    )
    input_gate = tf.math.sigmoid(pre[:, _GATE_INDEX["input"], :])
    forget_gate = tf.math.sigmoid(pre[:, _GATE_INDEX["forget"], :])
    output_gate = tf.math.sigmoid(pre[:, _GATE_INDEX["output"], :])
    candidate = tf.math.tanh(pre[:, _GATE_INDEX["candidate"], :])
    cell = forget_gate * c_prev + input_gate * candidate
    return {
        "input": input_gate,
        "forget": forget_gate,
        "output": output_gate,
        "candidate": candidate,
        "cell": cell,
    }


def _gate_activation_derivatives(gates: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    return {
        "input": gates["input"] * (1.0 - gates["input"]),
        "forget": gates["forget"] * (1.0 - gates["forget"]),
        "output": gates["output"] * (1.0 - gates["output"]),
        "candidate": 1.0 - tf.square(gates["candidate"]),
    }


def _cell_to_hidden_coeff(gates: Mapping[str, tf.Tensor]) -> tf.Tensor:
    tanh_cell = tf.math.tanh(gates["cell"])
    return gates["output"] * (1.0 - tf.square(tanh_cell))


def _split_state(
    params: SSLLSTMConstrainedParameters,
    points: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    values = _as_points(points)
    k = int(params.config.latent_dim)
    h = int(params.config.hidden_dim)
    return values[:, :k], values[:, k : k + h], values[:, k + h : k + 2 * h]


def _as_points(points: tf.Tensor) -> tf.Tensor:
    values = tf.convert_to_tensor(points, dtype=tf.float64)
    if values.shape.rank == 1:
        return values[tf.newaxis, :]
    if values.shape.rank == 2:
        return values
    raise ValueError("SSL-LSTM points must be one- or two-dimensional")


def _scatter_rank2(
    shape: list[int],
    indices: list[tuple[int, int]],
    updates: tf.Tensor,
) -> tf.Tensor:
    if not indices:
        return tf.zeros(shape, dtype=tf.float64)
    return tf.tensor_scatter_nd_update(
        tf.zeros(shape, dtype=tf.float64),
        tf.constant(indices, dtype=tf.int32),
        tf.convert_to_tensor(updates, dtype=tf.float64),
    )


def _scatter_rank3(
    shape: list[int],
    indices: list[tuple[int, int, int]],
    updates: tf.Tensor,
) -> tf.Tensor:
    if not indices:
        return tf.zeros(shape, dtype=tf.float64)
    return tf.tensor_scatter_nd_update(
        tf.zeros(shape, dtype=tf.float64),
        tf.constant(indices, dtype=tf.int32),
        tf.convert_to_tensor(updates, dtype=tf.float64),
    )


__all__ = [
    "SSLLSTMConstrainedParameters",
    "SSLLSTMSGQFAdapterComponents",
    "SSLLSTMParameterSlices",
    "SSLLSTMUKFAdapterComponents",
    "build_ssl_lstm_debug_value_score_artifact",
    "make_ssl_lstm_fixed_sgqf_components",
    "make_ssl_lstm_svd_ukf_components",
    "ssl_lstm_observation",
    "ssl_lstm_observation_parameter_derivative",
    "ssl_lstm_observation_state_jvp",
    "ssl_lstm_observation_state_jacobian",
    "ssl_lstm_parameter_slices",
    "ssl_lstm_transition",
    "ssl_lstm_transition_parameter_derivative",
    "ssl_lstm_transition_state_jvp",
    "ssl_lstm_transition_state_jacobian",
    "tf_ssl_lstm_fixed_sgqf_score",
    "tf_ssl_lstm_svd_ukf_score",
    "unpack_ssl_lstm_parameters",
]
