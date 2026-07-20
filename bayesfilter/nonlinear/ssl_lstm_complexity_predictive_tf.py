"""Q-general principal-root predictive moments for the SSL-LSTM ladder."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable

import tensorflow as tf

from bayesfilter.nonlinear.sigma_points_tf import tf_svd_sigma_point_filter
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (
    SSLLSTMComplexityPosteriorTarget,
    complexity_posterior_target,
)
from bayesfilter.nonlinear.ssl_lstm_sgqf_ukf_adapters import (
    ssl_lstm_observation,
    ssl_lstm_transition,
    unpack_ssl_lstm_parameters,
)
from bayesfilter.structural import StatePartition, StructuralFilterConfig
from bayesfilter.structural_tf import TFStructuralStateSpace


FORECAST_HORIZON = 10
FORECAST_REPLICATION_COUNT = 2
CALIBRATION_DRAWS_PER_CHAIN = 256
CALIBRATION_CHAIN_COUNT = 4
FORECAST_CHUNK_SIZE = 32
ROOT_SEED = 20260719
TERMINAL_FAMILY = 5101
PROCESS_FAMILY = 5102
OBSERVATION_FAMILY = 5103
FLOAT64_EPSILON = 2.0**-52
COVARIANCE_TOLERANCE_MULTIPLIER = 128.0


class ComplexityPredictiveError(ValueError):
    """Raised when a q-general forecast contract is invalid."""


@dataclass(frozen=True)
class ComplexityConditionalForecast:
    conditional_means: tf.Tensor
    conditional_variances: tf.Tensor
    observations: tf.Tensor
    terminal_states: tf.Tensor
    status: tf.Tensor
    q: int
    draw_count: int
    replication_count: int
    horizon: int
    seed: tuple[int, int]
    target_signature: str
    construction_signature: str


@dataclass(frozen=True)
class ComplexityCalibrationScale:
    center: tf.Tensor
    scale: tf.Tensor
    q: int
    chain_count: int
    draw_count_per_chain: int
    replication_count: int
    seed_roots: tuple[tuple[int, int], ...]
    target_signature: str
    calibration_signature: str


class ComplexityForecastWorker:
    """Persistent scalar forecast surface for spawned CPU workers."""

    def __init__(self, q: int) -> None:
        self.q = int(q)
        self.target = complexity_posterior_target(self.q, jit_compile=True)
        self.program = complexity_forecast_compiled_program(
            self.target,
            draw_count=1,
            replication_count=FORECAST_REPLICATION_COUNT,
        )
        self.state_dim = int(self.target.config.static_config.augmented_state_dim)

    def evaluate(self, free: Any, seed: Any) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        values = tf.ensure_shape(tf.convert_to_tensor(free, tf.float64), [4])
        root = tf.ensure_shape(tf.convert_to_tensor(seed, tf.int32), [2])
        terminal = tf.random.stateless_normal(
            (1, FORECAST_REPLICATION_COUNT, self.state_dim),
            _fold(root, TERMINAL_FAMILY),
            dtype=tf.float64,
            alg="philox",
        )
        process = tf.random.stateless_normal(
            (1, FORECAST_REPLICATION_COUNT, FORECAST_HORIZON, self.q),
            _fold(root, PROCESS_FAMILY),
            dtype=tf.float64,
            alg="philox",
        )
        observation = tf.random.stateless_normal(
            (1, FORECAST_REPLICATION_COUNT, FORECAST_HORIZON),
            _fold(root, OBSERVATION_FAMILY),
            dtype=tf.float64,
            alg="philox",
        )
        means, variances, observations, _terminal, status = self.program(
            values[tf.newaxis, :], terminal, process, observation
        )
        if not bool(status[0].numpy()):
            raise ComplexityPredictiveError("worker forecast failed validity gate")
        return means[0], variances[0], observations[0]


def complexity_forecast_worker_factory(config: dict[str, Any]) -> ComplexityForecastWorker:
    if not isinstance(config, dict) or "q" not in config:
        raise ComplexityPredictiveError("forecast worker config requires q")
    worker = ComplexityForecastWorker(int(config["q"]))
    worker.evaluate(worker.target.config.prior_center, (ROOT_SEED, 49000 + worker.q))
    return worker


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_seed(seed: Any) -> tuple[tf.Tensor, tuple[int, int]]:
    tensor = tf.convert_to_tensor(seed, tf.int32)
    if tensor.shape != (2,):
        raise ComplexityPredictiveError("forecast seed must have static shape (2,)")
    values = tuple(int(value) for value in tensor.numpy().tolist())
    return tensor, (values[0], values[1])


def _fold(seed: tf.Tensor, value: int) -> tf.Tensor:
    return tf.random.experimental.stateless_fold_in(
        seed,
        tf.constant(int(value), tf.int32),
        alg="philox",
    )


def _covariance_factor(covariance: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    values = tf.convert_to_tensor(covariance, tf.float64)
    symmetric = 0.5 * (values + tf.transpose(values))
    scale = tf.maximum(tf.constant(1.0, tf.float64), tf.linalg.norm(symmetric))
    tolerance = (
        tf.constant(COVARIANCE_TOLERANCE_MULTIPLIER * FLOAT64_EPSILON, tf.float64)
        * scale
    )
    factor = tf.linalg.cholesky(symmetric)
    reconstruction = tf.linalg.norm(factor @ tf.transpose(factor) - symmetric)
    valid = tf.logical_and(
        tf.reduce_all(tf.math.is_finite(symmetric)),
        tf.logical_and(
            tf.reduce_all(tf.math.is_finite(factor)),
            tf.logical_and(
                tf.reduce_all(tf.linalg.diag_part(factor) > 0.0),
                tf.logical_and(
                    tf.math.is_finite(reconstruction),
                    reconstruction <= tf.constant(16.0, tf.float64) * tolerance,
                ),
            ),
        ),
    )
    return factor, valid


def _value_model(
    full: tf.Tensor,
    target: SSLLSTMComplexityPosteriorTarget,
) -> tuple[Any, TFStructuralStateSpace]:
    config = target.config.static_config
    params = unpack_ssl_lstm_parameters(
        full,
        config,
        derivative_parameter_indices=(),
    )
    k = int(config.latent_dim)
    h = int(config.hidden_dim)
    n = int(config.augmented_state_dim)
    partition = StatePartition(
        state_names=tuple(
            [f"z.{index}" for index in range(k)]
            + [f"a.{index}" for index in range(h)]
            + [f"c.{index}" for index in range(h)]
        ),
        stochastic_indices=tuple(range(k)),
        deterministic_indices=tuple(range(k, n)),
        innovation_dim=k,
    )

    def transition(previous_state: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        deterministic = ssl_lstm_transition(params, previous_state)
        return tf.concat(
            (
                deterministic[:, :k] + tf.convert_to_tensor(innovation, tf.float64),
                deterministic[:, k:],
            ),
            axis=1,
        )

    def deterministic_residual(
        previous_state: tf.Tensor,
        innovation: tf.Tensor,
        next_state: tf.Tensor,
    ) -> tf.Tensor:
        del innovation
        expected = ssl_lstm_transition(params, previous_state)
        return tf.convert_to_tensor(next_state, tf.float64)[:, k:] - expected[:, k:]

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
        name="ssl_lstm_complexity_principal_root_forecast_model",
    )
    return params, model


def _single_forecast_core(
    free: tf.Tensor,
    terminal_standard_normal: tf.Tensor,
    process_standard_normal: tf.Tensor,
    observation_standard_normal: tf.Tensor,
    target: SSLLSTMComplexityPosteriorTarget,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    full = target.full_theta(free)
    params, model = _value_model(full, target)
    filtered = tf_svd_sigma_point_filter(
        target.config.observations,
        model,
        backend="tf_principal_sqrt_ukf",
        innovation_floor=tf.constant(1.0e-12, tf.float64),
        return_filtered=True,
    )
    if filtered.filtered_means is None or filtered.filtered_covariances is None:
        raise RuntimeError("principal-root terminal filter must return history")
    terminal_mean = filtered.filtered_means[-1]
    terminal_factor, covariance_valid = _covariance_factor(
        filtered.filtered_covariances[-1]
    )
    terminal_states = (
        terminal_mean[tf.newaxis, :]
        + terminal_standard_normal @ tf.transpose(terminal_factor)
    )
    state = terminal_states
    conditional_rows = []
    observation_rows = []
    variance = tf.square(params.observation_std[0])
    for horizon in range(FORECAST_HORIZON):
        deterministic = ssl_lstm_transition(params, state)
        state = tf.concat(
            (
                deterministic[:, : target.q]
                + process_standard_normal[:, horizon, :]
                * params.process_std[tf.newaxis, :],
                deterministic[:, target.q :],
            ),
            axis=1,
        )
        means = tf.squeeze(
            ssl_lstm_observation(params, state), axis=-1
        )
        observations = (
            means
            + params.observation_std[0]
            * observation_standard_normal[:, horizon]
        )
        conditional_rows.append(means)
        observation_rows.append(observations)
    conditional_means = tf.stack(conditional_rows, axis=-1)
    conditional_variances = tf.fill(tf.shape(conditional_means), variance)
    observations = tf.stack(observation_rows, axis=-1)
    finite = tf.reduce_all(
        tf.math.is_finite(
            tf.concat(
                (
                    tf.reshape(conditional_means, [-1]),
                    tf.reshape(conditional_variances, [-1]),
                    tf.reshape(observations, [-1]),
                    tf.reshape(terminal_states, [-1]),
                    tf.reshape(filtered.log_likelihood, [-1]),
                ),
                axis=0,
            )
        )
    )
    positive_variance = tf.reduce_all(conditional_variances > 0.0)
    status = tf.logical_and(covariance_valid, tf.logical_and(finite, positive_variance))
    return (
        conditional_means,
        conditional_variances,
        observations,
        terminal_states,
        status,
    )


_PROGRAM_CACHE: dict[tuple[int, int, int, str], Callable[..., tuple[tf.Tensor, ...]]] = {}


def complexity_forecast_compiled_program(
    target: SSLLSTMComplexityPosteriorTarget,
    *,
    draw_count: int,
    replication_count: int = FORECAST_REPLICATION_COUNT,
) -> Callable[..., tuple[tf.Tensor, ...]]:
    count = int(draw_count)
    replications = int(replication_count)
    if count <= 0 or replications <= 0:
        raise ComplexityPredictiveError("draw and replication counts must be positive")
    state_dim = int(target.config.static_config.augmented_state_dim)
    key = (target.q, count, replications, target.target_signature())
    cached = _PROGRAM_CACHE.get(key)
    if cached is not None:
        return cached

    def program(
        free_draws: tf.Tensor,
        terminal_standard_normal: tf.Tensor,
        process_standard_normal: tf.Tensor,
        observation_standard_normal: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        return tf.map_fn(
            lambda rows: _single_forecast_core(
                rows[0], rows[1], rows[2], rows[3], target
            ),
            (
                free_draws,
                terminal_standard_normal,
                process_standard_normal,
                observation_standard_normal,
            ),
            fn_output_signature=(
                tf.TensorSpec((replications, FORECAST_HORIZON), tf.float64),
                tf.TensorSpec((replications, FORECAST_HORIZON), tf.float64),
                tf.TensorSpec((replications, FORECAST_HORIZON), tf.float64),
                tf.TensorSpec((replications, state_dim), tf.float64),
                tf.TensorSpec((), tf.bool),
            ),
            parallel_iterations=1,
        )

    compiled = tf.function(
        program,
        input_signature=(
            tf.TensorSpec((count, 4), tf.float64),
            tf.TensorSpec((count, replications, state_dim), tf.float64),
            tf.TensorSpec((count, replications, FORECAST_HORIZON, target.q), tf.float64),
            tf.TensorSpec((count, replications, FORECAST_HORIZON), tf.float64),
        ),
        jit_compile=True,
        reduce_retracing=True,
    )
    _PROGRAM_CACHE[key] = compiled
    return compiled


def forecast_complexity_conditional_moments(
    free_draws: Any,
    *,
    q: int,
    seed: Any,
    replication_count: int = FORECAST_REPLICATION_COUNT,
) -> ComplexityConditionalForecast:
    values = tf.convert_to_tensor(free_draws, tf.float64)
    if values.shape.rank != 2 or values.shape[-1] != 4 or values.shape[0] is None:
        raise ComplexityPredictiveError("free_draws must have static shape [draw,4]")
    count = int(values.shape[0])
    replications = int(replication_count)
    target = complexity_posterior_target(int(q), jit_compile=True)
    state_dim = int(target.config.static_config.augmented_state_dim)
    root, seed_values = _require_seed(seed)
    terminal = tf.random.stateless_normal(
        (count, replications, state_dim),
        _fold(root, TERMINAL_FAMILY),
        dtype=tf.float64,
        alg="philox",
    )
    process = tf.random.stateless_normal(
        (count, replications, FORECAST_HORIZON, int(q)),
        _fold(root, PROCESS_FAMILY),
        dtype=tf.float64,
        alg="philox",
    )
    observation = tf.random.stateless_normal(
        (count, replications, FORECAST_HORIZON),
        _fold(root, OBSERVATION_FAMILY),
        dtype=tf.float64,
        alg="philox",
    )
    program = complexity_forecast_compiled_program(
        target,
        draw_count=count,
        replication_count=replications,
    )
    means, variances, observations, terminal_states, status = program(
        values, terminal, process, observation
    )
    if not bool(tf.reduce_all(status).numpy()):
        raise ComplexityPredictiveError("q-general forecast failed finite/covariance gate")
    signature = _canonical_hash(
        {
            "schema": "bayesfilter.ssl_lstm.complexity_predictive.v1",
            "q": int(q),
            "draw_count": count,
            "replication_count": replications,
            "horizon": FORECAST_HORIZON,
            "seed": list(seed_values),
            "target_signature": target.target_signature(),
            "filter_backend": "tf_principal_sqrt_ukf",
            "terminal_covariance_factor": "cholesky_same_gaussian_law",
            "conditional_variance": "observation_std**2",
        }
    )
    return ComplexityConditionalForecast(
        conditional_means=means,
        conditional_variances=variances,
        observations=observations,
        terminal_states=terminal_states,
        status=status,
        q=int(q),
        draw_count=count,
        replication_count=replications,
        horizon=FORECAST_HORIZON,
        seed=seed_values,
        target_signature=target.target_signature(),
        construction_signature=signature,
    )


def calibration_seed_roots(q: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (ROOT_SEED, 40000 + 10000 * int(q) + 1000 * index)
        for index in range(4)
    )


def complexity_calibration_signature(
    *,
    q: int,
    chain_count: int,
    draw_count_per_chain: int,
    replication_count: int,
    seed_roots: tuple[tuple[int, int], ...],
    target_signature: str,
    forecast_signatures: tuple[str, ...],
) -> str:
    """Return the replayable signature for one frozen calibration contract."""

    return _canonical_hash(
        {
            "schema": "bayesfilter.ssl_lstm.complexity_calibration.v1",
            "q": int(q),
            "chain_count": int(chain_count),
            "draw_count_per_chain": int(draw_count_per_chain),
            "replication_count": int(replication_count),
            "seed_roots": [list(row) for row in seed_roots],
            "target_signature": str(target_signature),
            "forecast_signatures": list(forecast_signatures),
        }
    )


def calibration_from_observation_banks(
    observation_banks: tuple[tf.Tensor, ...],
    *,
    q: int,
    seed_roots: tuple[tuple[int, int], ...],
    target_signature: str,
    forecast_signatures: tuple[str, ...],
) -> ComplexityCalibrationScale:
    if not observation_banks or len(observation_banks) != len(seed_roots):
        raise ComplexityPredictiveError("one calibration bank is required per seed root")
    if len(forecast_signatures) != len(observation_banks):
        raise ComplexityPredictiveError("one forecast signature is required per bank")
    shapes = {tuple(tf.convert_to_tensor(row).shape) for row in observation_banks}
    if len(shapes) != 1:
        raise ComplexityPredictiveError("calibration observation-bank shapes differ")
    shape = next(iter(shapes))
    if len(shape) != 3 or shape[1:] != (
        FORECAST_REPLICATION_COUNT,
        FORECAST_HORIZON,
    ):
        raise ComplexityPredictiveError(
            "calibration banks require [draw,replication,horizon]"
        )
    observations = tf.concat(
        [tf.convert_to_tensor(row, tf.float64) for row in observation_banks], axis=0
    )
    path_count = int(observations.shape[0] * observations.shape[1])
    center = tf.reduce_mean(observations, axis=(0, 1))
    centered = observations - center
    scale = tf.sqrt(
        tf.reduce_sum(tf.square(centered), axis=(0, 1))
        / tf.cast(path_count - 1, tf.float64)
    )
    if not bool(
        tf.reduce_all(tf.math.is_finite(center)).numpy()
        and tf.reduce_all(tf.math.is_finite(scale)).numpy()
        and tf.reduce_all(scale > 0.0).numpy()
    ):
        raise ComplexityPredictiveError("calibration center/scale is invalid")
    signature = complexity_calibration_signature(
        q=q,
        chain_count=len(seed_roots),
        draw_count_per_chain=int(shape[0]),
        replication_count=FORECAST_REPLICATION_COUNT,
        seed_roots=seed_roots,
        target_signature=target_signature,
        forecast_signatures=forecast_signatures,
    )
    return ComplexityCalibrationScale(
        center=center,
        scale=scale,
        q=int(q),
        chain_count=len(seed_roots),
        draw_count_per_chain=int(shape[0]),
        replication_count=FORECAST_REPLICATION_COUNT,
        seed_roots=seed_roots,
        target_signature=target_signature,
        calibration_signature=signature,
    )


def calibrate_complexity_horizon_scales(
    *,
    q: int,
    draw_count_per_chain: int = CALIBRATION_DRAWS_PER_CHAIN,
) -> ComplexityCalibrationScale:
    count = int(draw_count_per_chain)
    if count <= 0:
        raise ComplexityPredictiveError("calibration draw count must be positive")
    target = complexity_posterior_target(int(q), jit_compile=True)
    if count % FORECAST_CHUNK_SIZE:
        raise ComplexityPredictiveError(
            "calibration draw count must be divisible by forecast chunk size"
        )
    truth_chunk = tf.repeat(
        target.config.prior_center[tf.newaxis, :], FORECAST_CHUNK_SIZE, axis=0
    )
    roots = calibration_seed_roots(q)
    banks = []
    signatures = []
    for chain_index, root in enumerate(roots):
        chunks = []
        chunk_signatures = []
        root_tensor = tf.constant(root, tf.int32)
        for chunk_index in range(count // FORECAST_CHUNK_SIZE):
            chunk_seed = tf.random.experimental.stateless_fold_in(
                root_tensor,
                tf.constant(chunk_index, tf.int32),
                alg="philox",
            )
            forecast = forecast_complexity_conditional_moments(
                truth_chunk,
                q=q,
                seed=chunk_seed,
                replication_count=FORECAST_REPLICATION_COUNT,
            )
            chunks.append(forecast.observations)
            chunk_signatures.append(forecast.construction_signature)
        banks.append(tf.concat(chunks, axis=0))
        signatures.append(
            _canonical_hash(
                {
                    "chain_index": chain_index,
                    "root": list(root),
                    "chunk_signatures": chunk_signatures,
                }
            )
        )
    return calibration_from_observation_banks(
        tuple(banks),
        q=q,
        seed_roots=roots,
        target_signature=target.target_signature(),
        forecast_signatures=tuple(signatures),
    )


__all__ = [
    "CALIBRATION_CHAIN_COUNT",
    "CALIBRATION_DRAWS_PER_CHAIN",
    "ComplexityCalibrationScale",
    "ComplexityConditionalForecast",
    "ComplexityForecastWorker",
    "ComplexityPredictiveError",
    "FORECAST_HORIZON",
    "FORECAST_CHUNK_SIZE",
    "FORECAST_REPLICATION_COUNT",
    "calibrate_complexity_horizon_scales",
    "calibration_from_observation_banks",
    "calibration_seed_roots",
    "complexity_forecast_compiled_program",
    "complexity_forecast_worker_factory",
    "forecast_complexity_conditional_moments",
]
